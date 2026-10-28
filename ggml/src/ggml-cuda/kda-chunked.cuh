#pragma once
// Chunked (parallel-over-tokens) KDA prefill: the FLA chunk_kda formulation with 64-token chunks.
//
// Per head the recurrent kernel computes, for t = 1..T, with S = [K][V] and a_t = exp(g_t) per key channel:
//   S_t = Diag(a_t) S_{t-1} + k_t u_t^T,   u_t = beta_t (v_t - (Diag(a_t) S_{t-1})^T k_t),   o_t = S_t^T (q_t / sqrt(K))
// Inside a chunk, with G_t = cumsum(g) from the chunk start, Gamma = exp(G) and S_0 the state at the chunk start:
//   A[t][s]  = beta_t k_t . (k_s * exp(G_t - G_s))                (s < t, strictly lower triangular)
//   w        = (I + A)^-1 (beta * Gamma * k)                      [C x K]
//   u_wy     = (I + A)^-1 (beta * v)                              [C x V]
//   u        = u_wy - w S_0                                       (the update rows the recurrence would produce)
//   o        = (q~ * Gamma) S_0 + Aqk u,  Aqk[t][s] = q~_t . (k_s * exp(G_t - G_s))  (s <= t),  q~ = q / sqrt(K)
//   S_C      = Diag(Gamma_C) S_0 + k_dec^T u,  k_dec[s] = k_s * exp(G_C - G_s)
// Factoring exp(G_t - G_s) = exp(G_t - G_b) exp(G_b - G_s) through a sub-block start b with s < b <= t keeps every
// exponent <= 0 for g <= 0, so nothing overflows; inside a diagonal sub-block the exps are computed directly.
//
// kda_chunk_intra: one block per (chunk, head, seq), fully parallel. Builds A and Aqk (16-token sub-blocks: the
//   off-diagonal blocks as small matmuls on decayed keys, the diagonal blocks elementwise), solves the two triangular
//   systems by blocked forward substitution, writes w^T | q~Gamma^T, u_wy, k_dec, Aqk^T, Gamma_C to scratch.
// kda_chunk_seq: one block per (head, seq). Walks the chunks of a slab with the 128x128 state in shared memory:
//   three small matmuls per chunk (3.3M FMA per chunk and head, the inherent state-touching work). The slab loop in the host
//   launcher bounds the scratch to CS chunks per head.
// The final state is the same quantity the recurrent kernel produces, so decode continues unchanged.

#include "common.cuh"

namespace kda_chunk {

static constexpr int C  = 64;        // tokens per chunk
static constexpr int HD = 128;       // head dim handled by the chunked path
static constexpr int CS = 16;        // chunks per slab (scratch = CS * n_heads * n_seqs chunk-heads)
static constexpr int TW = 16;        // key-dim tile width of the pair phase
static constexpr int NT_INTRA = 256;
static constexpr int NT_SEQ   = 256;

// per chunk-head scratch, in floats
static constexpr int SZ_WQ  = HD * 2 * C;   // [HD][2C]: columns 0..C-1 = w^T, C..2C-1 = (q~ Gamma)^T
static constexpr int SZ_UWY = C * HD;       // [C][HD]
static constexpr int SZ_KD  = C * HD;       // [C][HD]  k_dec
static constexpr int SZ_AQK = C * C;        // [C][C]   stored transposed: AQK[s][t], zero for s > t
static constexpr int SZ_DEC = HD;           // Gamma_C
static constexpr int SZ_RW  = C * HD;       // [C][HD]  beta Gamma k (intra-kernel temporary)
static constexpr int SZ_CH  = SZ_WQ + SZ_UWY + SZ_KD + SZ_AQK + SZ_DEC + SZ_RW;

static constexpr int SMEM_SEQ   = (HD * HD + C * HD) * (int) sizeof(float);                   // 98,304 B

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Triangular pair index p in [0, C*(C+1)/2) -> (t, s) with s <= t, row-major over t
__device__ __forceinline__ void pair_ts(int p, int & t, int & s) {
    t = (int) floorf((sqrtf(8.0f * (float) p + 1.0f) - 1.0f) * 0.5f);
    while (t * (t + 1) / 2 > p) --t;
    while ((t + 1) * (t + 2) / 2 <= p) ++t;
    s = p - t * (t + 1) / 2;
}

// Intra-chunk kernel: 256 threads, <= 50 KB of shared memory so two blocks share an SM (one computes while the
// other sits in a serial phase). Key dim processed in 16-wide tiles; per tile the cumulative gate G (log2 units),
// the raw k and scaled q tiles, the transposed decayed keys KR^T/QR^T (k_t, q~_t scaled by exp(G_t - G_b(t)),
// b(t) = start of the 16-token sub-block of t) and KC^T (k_s scaled by exp(G_b - G_s) for every (sub-block b,
// s < b) pair: 16 + 32 + 48 = 96 entries) are built in shared memory. Warps 0-3 accumulate the four diagonal
// 16x16 sub-blocks elementwise (exponents computed directly); warps 4-6 accumulate the off-diagonal sub-blocks
// as 4x4 register-tiled products KR_i KC_i^T and QR_i KC_i^T. Every exponent used is <= 0 for g <= 0.
// Then T = (I + A)^-1 is built explicitly (diagonal 16x16 blocks column-wise without barriers, two levels of
// small matmuls for the off-diagonal blocks), stored transposed, and both right-hand sides are solved as plain
// matmuls W = T (beta Gamma k), U = T (beta v).
static constexpr int AP   = C + 1;         // sA row stride
static constexpr int GTP  = TW + 4;        // G / k / q tile row stride (float4 rows, conflict-free)
static constexpr int KP   = C + 4;         // KR^T / QR^T row stride
static constexpr int KCN  = 16 + 32 + 48;  // KC entries: (sub-block i, s < 16 i), i = 1..3
static constexpr int KCP  = KCN + 4;       // KC^T row stride
static constexpr int TTP  = C + 4;         // T^T row stride
static constexpr int OFF_SGT = 0;                         // [C][GTP]  G tile
static constexpr int OFF_SKT = OFF_SGT + C * GTP;         // [C][GTP]  k tile
static constexpr int OFF_SQT = OFF_SKT + C * GTP;         // [C][GTP]  q tile (scaled)
static constexpr int OFF_SKR = OFF_SQT + C * GTP;         // [TW][KP]  KR^T
static constexpr int OFF_SQR = OFF_SKR + TW * KP;         // [TW][KP]  QR^T
static constexpr int OFF_SKC = OFF_SQR + TW * KP;         // [TW][KCP] KC^T
static constexpr int OFF_SQS = OFF_SKC + TW * KCP;        // [16][16]  cumsum partials
static constexpr int OFF_SA  = OFF_SQS + 16 * 16;         // [C][AP]   A  (7872..12032)
static constexpr int OFF_STT = 0;                         // [C][TTP]  T^T (after the pair phase)
static constexpr int OFF_SM  = OFF_STT + C * TTP;         // [32][32]  matmul scratch for the T build
static constexpr int OFF_SX  = OFF_STT + C * TTP;         // [C][HD]   solve buffer (overlaps A, which is dead by then)
static constexpr int OFF_SB  = OFF_SX + C * HD;           // [C]       beta
static constexpr int SMEM_INTRA = (OFF_SB + C) * (int) sizeof(float);   // 50,432 B
static_assert(OFF_SA + C * AP <= OFF_SB, "A must fit below beta");
static_assert(OFF_SM + 32 * 32 <= OFF_SA, "T-build scratch must not touch A");

__global__ void __launch_bounds__(NT_INTRA, 2) kda_chunk_intra(
        const float * __restrict__ q,       // [HD, n_tokens, n_heads_kq, n_seqs] contiguous
        const float * __restrict__ k,
        const float * __restrict__ v,       // strides vnb1 (token), vnb2 (head), vnb3 (seq); rows 16 B aligned
        const float * __restrict__ g,       // strides gnb0 (token), gnb1 (channel), gnb2 (head), gnb3 (seq)
        const float * __restrict__ beta,    // strides bnb1 (token), bnb2 (head), bnb3 (seq)
        float * __restrict__ scratch,
        const int n_tokens, const int tok0, const int n_heads, const int n_heads_kq, const int gqa_ratio, const int repeat_type,
        const size_t vnb1, const size_t vnb2, const size_t vnb3,
        const size_t gnb0, const size_t gnb1, const size_t gnb2, const size_t gnb3,
        const size_t bnb1, const size_t bnb2, const size_t bnb3) {
    constexpr int NT = NT_INTRA;
    constexpr float LOG2E = 1.4426950408889634f;

    const int c    = blockIdx.x;
    const int head = blockIdx.y;
    const int seq  = blockIdx.z;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int chunk_start = tok0 + c * C;
    const int nvalid = min(C, n_tokens - chunk_start);
    const int head_kq = repeat_type == 0 ? head / gqa_ratio : head % n_heads_kq;
    const float scale = rsqrtf((float) HD);

    const float * q_p = q + ((size_t) seq * n_heads_kq + head_kq) * (size_t) HD * n_tokens + (size_t) chunk_start * HD;
    const float * k_p = k + ((size_t) seq * n_heads_kq + head_kq) * (size_t) HD * n_tokens + (size_t) chunk_start * HD;
    const float * v_p = v + seq * vnb3 + head * vnb2 + chunk_start * vnb1;
    const float * g_p = g + seq * gnb3 + head * gnb2 + chunk_start * gnb0;
    const float * b_p = beta + seq * bnb3 + head * bnb2 + chunk_start * bnb1;

    const size_t ci = ((size_t) seq * n_heads + head) * CS + c;
    float * WQ  = scratch + ci * SZ_CH;
    float * UWY = WQ + SZ_WQ;
    float * KD  = UWY + SZ_UWY;
    float * AQK = KD + SZ_KD;
    float * DEC = AQK + SZ_AQK;
    float * RW  = DEC + SZ_DEC;      // [C][HD] beta Gamma k (right-hand side of the w solve)

    extern __shared__ float smem[];
    float * sGt   = smem + OFF_SGT;
    float * sKt   = smem + OFF_SKT;
    float * sQt   = smem + OFF_SQT;
    float * sKR   = smem + OFF_SKR;
    float * sQR   = smem + OFF_SQR;
    float * sKC   = smem + OFF_SKC;
    float * sQS   = smem + OFF_SQS;
    float * sA    = smem + OFF_SA;
    float * sTT   = smem + OFF_STT;
    float * sM    = smem + OFF_SM;
    float * sX    = smem + OFF_SX;
    float * sBeta = smem + OFF_SB;

    if (tid < C) {
        sBeta[tid] = tid < nvalid ? sigmoid_f(b_p[tid * bnb1]) : 0.0f;
    }
    for (int p = tid; p < C * C; p += NT) {
        if ((p >> 6) > (p & (C - 1))) AQK[p] = 0.0f;   // Aqk^T upper triangle
    }

    // roles in the pair phase
    const int blk = warp;                   // diagonal role (warps 0-3): sub-block
    const int rp  = (lane >> 2) & 7;        //   row pair (blk*16 + rp, blk*16 + 15 - rp)
    const int dq  = lane & 3;               //   channel quarter of the tile (4 channels)
    float da[16], db[16], ea[16], eb[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) { da[j] = db[j] = ea[j] = eb[j] = 0.0f; }
    const int tau = tid - 128;              // off-diagonal role (tid 128..223): 4x4 tile of (t, s)
    int oi, ot0, os0, obase;
    if (tau < 16)      { oi = 1; ot0 = 16 + 4 * (tau >> 2);        os0 = 4 * (tau & 3);        obase = 0;  }
    else if (tau < 48) { oi = 2; ot0 = 32 + 4 * ((tau - 16) >> 3); os0 = 4 * ((tau - 16) & 7); obase = 16; }
    else               { oi = 3; ot0 = 48 + 4 * ((tau - 48) / 12); os0 = 4 * ((tau - 48) % 12); obase = 48; }
    (void) oi;
    float oa[4][4], ob[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) { oa[i][0] = oa[i][1] = oa[i][2] = oa[i][3] = 0.0f; ob[i][0] = ob[i][1] = ob[i][2] = ob[i][3] = 0.0f; }

    const int bt = tid >> 2;                // build: token row (0..63)
    const int bf = tid & 3;                 // build: float4 index in the 16-wide tile
    const int cd = tid & 15;                // cumsum: channel in tile
    const int cp = tid >> 4;                // cumsum: part (4 tokens)

    for (int tile = 0; tile < HD / TW; ++tile) {
        const int d0 = tile * TW;
        __syncthreads();   // previous tile's readers done
        {
            // cumulative gate of this tile: 16 parts of 4 tokens, then a prefix over the parts
            float gv[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int tt = cp * 4 + i;
                gv[i] = tt < nvalid ? fminf(g_p[tt * gnb0 + (d0 + cd) * gnb1], 50.0f) * LOG2E : 0.0f;
            }
            float run = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { run += gv[i]; gv[i] = run; }
            sQS[cp * 16 + cd] = run;
            __syncthreads();
            float off = 0.0f;
            for (int j = 0; j < cp; ++j) off += sQS[j * 16 + cd];
#pragma unroll
            for (int i = 0; i < 4; ++i) sGt[(cp * 4 + i) * GTP + cd] = gv[i] + off;
        }
        __syncthreads();
        if (tid < TW) {
            DEC[d0 + tid] = ex2(sGt[(C - 1) * GTP + tid]);
        }
        {
            // k / q tiles, KR^T, QR^T, k_dec, beta Gamma k, q~Gamma^T
            const int dd = bf * 4;
            float4 kk = make_float4(0.0f, 0.0f, 0.0f, 0.0f), qq = kk;
            if (bt < nvalid) {
                kk = __ldg((const float4 *) (k_p + bt * HD + d0 + dd));
                qq = __ldg((const float4 *) (q_p + bt * HD + d0 + dd));
                qq.x *= scale; qq.y *= scale; qq.z *= scale; qq.w *= scale;
            }
            const float4 gt = *(const float4 *) (sGt + bt * GTP + dd);
            const float4 gb = *(const float4 *) (sGt + (bt & ~15) * GTP + dd);
            const float4 gc = *(const float4 *) (sGt + (C - 1) * GTP + dd);
            const float e0 = ex2(gt.x - gb.x), e1 = ex2(gt.y - gb.y), e2 = ex2(gt.z - gb.z), e3 = ex2(gt.w - gb.w);
            const float f0 = ex2(gt.x), f1 = ex2(gt.y), f2 = ex2(gt.z), f3 = ex2(gt.w);
            const float btv = sBeta[bt];
            *(float4 *) (sKt + bt * GTP + dd) = kk;
            *(float4 *) (sQt + bt * GTP + dd) = qq;
            sKR[(dd + 0) * KP + bt] = kk.x * e0; sKR[(dd + 1) * KP + bt] = kk.y * e1;
            sKR[(dd + 2) * KP + bt] = kk.z * e2; sKR[(dd + 3) * KP + bt] = kk.w * e3;
            sQR[(dd + 0) * KP + bt] = qq.x * e0; sQR[(dd + 1) * KP + bt] = qq.y * e1;
            sQR[(dd + 2) * KP + bt] = qq.z * e2; sQR[(dd + 3) * KP + bt] = qq.w * e3;
            *(float4 *) (KD + bt * HD + d0 + dd) =
                make_float4(kk.x * ex2(gc.x - gt.x), kk.y * ex2(gc.y - gt.y), kk.z * ex2(gc.z - gt.z), kk.w * ex2(gc.w - gt.w));
            *(float4 *) (RW + bt * HD + d0 + dd) = make_float4(btv * kk.x * f0, btv * kk.y * f1, btv * kk.z * f2, btv * kk.w * f3);
            float * wq = WQ + (size_t) (d0 + dd) * (2 * C) + C + bt;
            wq[0 * 2 * C] = qq.x * f0; wq[1 * 2 * C] = qq.y * f1; wq[2 * 2 * C] = qq.z * f2; wq[3 * 2 * C] = qq.w * f3;
        }
        __syncthreads();
        for (int idx = tid; idx < KCN * 4; idx += NT) {
            // KC^T entry e = (sub-block i, s): k_s * exp(G_{16 i} - G_s)
            const int e = idx >> 2, dd = (idx & 3) * 4;
            const int i = e < 16 ? 1 : (e < 48 ? 2 : 3);
            const int s = e - (i == 1 ? 0 : (i == 2 ? 16 : 48));
            const float4 kk = *(const float4 *) (sKt + s * GTP + dd);
            const float4 gs = *(const float4 *) (sGt + s * GTP + dd);
            const float4 gb = *(const float4 *) (sGt + (16 * i) * GTP + dd);
            sKC[(dd + 0) * KCP + e] = kk.x * ex2(gb.x - gs.x); sKC[(dd + 1) * KCP + e] = kk.y * ex2(gb.y - gs.y);
            sKC[(dd + 2) * KCP + e] = kk.z * ex2(gb.z - gs.z); sKC[(dd + 3) * KCP + e] = kk.w * ex2(gb.w - gs.w);
        }
        __syncthreads();
        if (warp < 4) {
            const int t1 = blk * 16 + rp, t2 = blk * 16 + 15 - rp;
            const int dd = dq * 4;
            const float4 g1v = *(const float4 *) (sGt + t1 * GTP + dd), g2v = *(const float4 *) (sGt + t2 * GTP + dd);
            const float4 k1v = *(const float4 *) (sKt + t1 * GTP + dd), k2v = *(const float4 *) (sKt + t2 * GTP + dd);
            const float4 q1v = *(const float4 *) (sQt + t1 * GTP + dd), q2v = *(const float4 *) (sQt + t2 * GTP + dd);
            const float g1[4] = { g1v.x, g1v.y, g1v.z, g1v.w }, g2[4] = { g2v.x, g2v.y, g2v.z, g2v.w };
            const float k1[4] = { k1v.x, k1v.y, k1v.z, k1v.w }, k2[4] = { k2v.x, k2v.y, k2v.z, k2v.w };
            const float q1[4] = { q1v.x, q1v.y, q1v.z, q1v.w }, q2[4] = { q2v.x, q2v.y, q2v.z, q2v.w };
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int s = blk * 16 + j;
                const float4 gsv = *(const float4 *) (sGt + s * GTP + dd);
                const float4 ksv = *(const float4 *) (sKt + s * GTP + dd);
                const float gs[4] = { gsv.x, gsv.y, gsv.z, gsv.w }, ks[4] = { ksv.x, ksv.y, ksv.z, ksv.w };
                if (j <= rp) {
                    float a = da[j], b = db[j];
#pragma unroll
                    for (int m = 0; m < 4; ++m) { const float kse = ks[m] * ex2(g1[m] - gs[m]); a = fmaf(k1[m], kse, a); b = fmaf(q1[m], kse, b); }
                    da[j] = a; db[j] = b;
                }
                if (j <= 15 - rp) {
                    float a = ea[j], b = eb[j];
#pragma unroll
                    for (int m = 0; m < 4; ++m) { const float kse = ks[m] * ex2(g2[m] - gs[m]); a = fmaf(k2[m], kse, a); b = fmaf(q2[m], kse, b); }
                    ea[j] = a; eb[j] = b;
                }
            }
        } else if (tau < 96) {
#pragma unroll
            for (int dl = 0; dl < TW; ++dl) {
                const float4 kr = *(const float4 *) (sKR + dl * KP + ot0);
                const float4 qr = *(const float4 *) (sQR + dl * KP + ot0);
                const float4 kc = *(const float4 *) (sKC + dl * KCP + obase + os0);
                const float krv[4] = { kr.x, kr.y, kr.z, kr.w }, qrv[4] = { qr.x, qr.y, qr.z, qr.w }, kcv[4] = { kc.x, kc.y, kc.z, kc.w };
#pragma unroll
                for (int i = 0; i < 4; ++i) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) { oa[i][j] = fmaf(krv[i], kcv[j], oa[i][j]); ob[i][j] = fmaf(qrv[i], kcv[j], ob[i][j]); }
                }
            }
        }
    }
    __syncthreads();   // tiles dead; write A (beta-scaled, s < t) and Aqk^T (s <= t)
    if (warp < 4) {
        const int t1 = blk * 16 + rp, t2 = blk * 16 + 15 - rp;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            float a = da[j], b = db[j], e = ea[j], f = eb[j];
            a += __shfl_xor_sync(0xffffffff, a, 1); a += __shfl_xor_sync(0xffffffff, a, 2);
            b += __shfl_xor_sync(0xffffffff, b, 1); b += __shfl_xor_sync(0xffffffff, b, 2);
            e += __shfl_xor_sync(0xffffffff, e, 1); e += __shfl_xor_sync(0xffffffff, e, 2);
            f += __shfl_xor_sync(0xffffffff, f, 1); f += __shfl_xor_sync(0xffffffff, f, 2);
            const int s = blk * 16 + j;
            if (dq == 0) {
                if (j <= rp)      { if (j < rp)      sA[t1 * AP + s] = sBeta[t1] * a; AQK[s * C + t1] = b; }
                if (j <= 15 - rp) { if (j < 15 - rp) sA[t2 * AP + s] = sBeta[t2] * e; AQK[s * C + t2] = f; }
            }
        }
    } else if (tau < 96) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int tt = ot0 + i, ss = os0 + j;
                sA[tt * AP + ss] = sBeta[tt] * oa[i][j];
                AQK[ss * C + tt] = ob[i][j];
            }
        }
    }
    // T = (I + A)^-1, stored transposed: sTT[s][t] = T[t][s]
    for (int p = tid; p < C * TTP; p += NT) sTT[p] = 0.0f;
    __syncthreads();
    if (tid < C) {
        // column j of diagonal block i: T[16i+r][16i+j] = -sum_{s=j}^{r-1} A[16i+r][16i+s] T[16i+s][16i+j]
        const int i = tid >> 4, j = tid & 15;
        float * col = sTT + (16 * i + j) * TTP + 16 * i;    // col[r] = T[16i+r][16i+j]
        col[j] = 1.0f;
#pragma unroll
        for (int r = 1; r < 16; ++r) {
            if (r > j) {
                const float * arow = sA + (16 * i + r) * AP + 16 * i;
                float acc = 0.0f;
#pragma unroll
                for (int s = 0; s < 15; ++s) {
                    if (s >= j && s < r) acc = fmaf(arow[s], col[s], acc);
                }
                col[r] = -acc;
            }
        }
    }
    __syncthreads();
    {
        // level 1: T_10 = -T_11 A_10 T_00 and T_32 = -T_33 A_32 T_22 (M = A_bb' T_b'b' first)
        const int pr = tid >> 7;                       // pair: 0 -> blocks (1,0), 1 -> blocks (3,2)
        const int bA = pr ? 48 : 16, bB = pr ? 32 : 0; // row block of A / T target, column block
        const int o  = tid & 127;                      // 2 outputs per thread: (t, u) and (t, u + 8)
        const int tt = o >> 3, uu = o & 7;
        float m0 = 0.0f, m1 = 0.0f;
#pragma unroll
        for (int s = 0; s < 16; ++s) {
            const float a = sA[(bA + tt) * AP + bB + s];
            m0 = fmaf(a, sTT[(bB + uu) * TTP + bB + s],     m0);
            m1 = fmaf(a, sTT[(bB + uu + 8) * TTP + bB + s], m1);
        }
        float * M = sM + pr * 256;
        M[tt * 16 + uu] = m0; M[tt * 16 + uu + 8] = m1;
        __syncthreads();
        float r0 = 0.0f, r1 = 0.0f;
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            const float tb = sTT[(bA + m) * TTP + bA + tt];   // T_BB[tt][m]
            r0 = fmaf(tb, M[m * 16 + uu],     r0);
            r1 = fmaf(tb, M[m * 16 + uu + 8], r1);
        }
        sTT[(bB + uu) * TTP + bA + tt]     = -r0;
        sTT[(bB + uu + 8) * TTP + bA + tt] = -r1;
    }
    __syncthreads();
    {
        // level 2: T_L (rows 32..63, cols 0..31) = -T_BB A_BL T_AA, 32x32 blocks; 4 outputs per thread
        const int tt = tid >> 3, u0 = (tid & 7) * 4;
        float m[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll 4
        for (int s = 0; s < 32; ++s) {
            const float a = sA[(32 + tt) * AP + s];
#pragma unroll
            for (int j = 0; j < 4; ++j) m[j] = fmaf(a, sTT[(u0 + j) * TTP + s], m[j]);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) sM[tt * 32 + u0 + j] = m[j];
        __syncthreads();
        float r[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll 4
        for (int mm = 0; mm < 32; ++mm) {
            const float tb = sTT[(32 + mm) * TTP + 32 + tt];  // T_BB[tt][mm]
#pragma unroll
            for (int j = 0; j < 4; ++j) r[j] = fmaf(tb, sM[mm * 32 + u0 + j], r[j]);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) sTT[(u0 + j) * TTP + 32 + tt] = -r[j];
    }
    __syncthreads();   // T^T complete; A is dead: X may be filled

    // w = T (beta Gamma k): X <- RW, W = T X, written transposed into WQ columns 0..C-1
#pragma unroll
    for (int i = 0; i < C * HD / 4 / NT; ++i) {
        const int p = tid + i * NT;
        *(float4 *) (sX + p * 4) = __ldg((const float4 *) (RW + p * 4));
    }
    __syncthreads();
    {
        // rows tg*8..+7 (warp-uniform), channels dg*4..+3; T is unit lower triangular, so s <= tg*8 + 7 suffices
        const int tg = tid >> 5, dg = tid & 31;
        float acc[8][4];
#pragma unroll
        for (int i = 0; i < 8; ++i) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f; }
#pragma unroll 4
        for (int s = 0; s < tg * 8 + 8; ++s) {
            const float4 t0 = *(const float4 *) (sTT + s * TTP + tg * 8);
            const float4 t1 = *(const float4 *) (sTT + s * TTP + tg * 8 + 4);
            const float4 x  = *(const float4 *) (sX + s * HD + dg * 4);
            const float tv[8] = { t0.x, t0.y, t0.z, t0.w, t1.x, t1.y, t1.z, t1.w };
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[i][0] = fmaf(tv[i], x.x, acc[i][0]); acc[i][1] = fmaf(tv[i], x.y, acc[i][1]);
                acc[i][2] = fmaf(tv[i], x.z, acc[i][2]); acc[i][3] = fmaf(tv[i], x.w, acc[i][3]);
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float * wq = WQ + (size_t) (dg * 4 + j) * (2 * C) + tg * 8;
            *(float4 *) wq       = make_float4(acc[0][j], acc[1][j], acc[2][j], acc[3][j]);
            *(float4 *) (wq + 4) = make_float4(acc[4][j], acc[5][j], acc[6][j], acc[7][j]);
        }
    }
    __syncthreads();
    // u_wy = T (beta v): X <- beta v, U = T X, natural layout
#pragma unroll
    for (int i = 0; i < C * HD / 4 / NT; ++i) {
        const int p = tid + i * NT;
        const int tt = p >> 5, dd = (p & 31) * 4;
        float4 vv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (tt < nvalid) {
            vv = __ldg((const float4 *) (v_p + tt * vnb1 + dd));
            const float b = sBeta[tt];
            vv.x *= b; vv.y *= b; vv.z *= b; vv.w *= b;
        }
        *(float4 *) (sX + tt * HD + dd) = vv;
    }
    __syncthreads();
    {
        const int tg = tid >> 5, dg = tid & 31;        // rows tg*8..+7, channels dg*4..+3
        float acc[8][4];
#pragma unroll
        for (int i = 0; i < 8; ++i) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f; }
#pragma unroll 4
        for (int s = 0; s < tg * 8 + 8; ++s) {
            const float4 t0 = *(const float4 *) (sTT + s * TTP + tg * 8);
            const float4 t1 = *(const float4 *) (sTT + s * TTP + tg * 8 + 4);
            const float4 x  = *(const float4 *) (sX + s * HD + dg * 4);
            const float tv[8] = { t0.x, t0.y, t0.z, t0.w, t1.x, t1.y, t1.z, t1.w };
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[i][0] = fmaf(tv[i], x.x, acc[i][0]); acc[i][1] = fmaf(tv[i], x.y, acc[i][1]);
                acc[i][2] = fmaf(tv[i], x.z, acc[i][2]); acc[i][3] = fmaf(tv[i], x.w, acc[i][3]);
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            *(float4 *) (UWY + (tg * 8 + i) * HD + dg * 4) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        }
    }
}

// One block per (head, seq): walks the chunks of a slab with the full 128x128 state in shared memory.
// 256 threads; thread (tr = tid & 15, tc = tid >> 4) owns an 8x8 register tile: phase 1 rows tr*4..+3 of the
// w part and of the q~Gamma part, phase 2b key rows tr*8..+7, columns tc*8..+7 in both. The 16 lanes of a
// half-warp share tc, so the global operand loads of a warp cover two contiguous 256 B row pieces.
__global__ void __launch_bounds__(NT_SEQ, 1) kda_chunk_seq(
        const float * __restrict__ scratch,
        const float * state_src,            // may alias state_dst (fused state copy)
        float * state_dst,
        float * __restrict__ dst,           // [HD, n_heads, n_tokens, n_seqs]
        const int n_tokens, const int tok0, const int n_chunks, const int n_heads) {
    constexpr int NT = NT_SEQ;
    const int head = blockIdx.x;
    const int seq  = blockIdx.y;
    const int tid  = threadIdx.x;
    const int tr   = tid & 15;
    const int tc   = tid >> 4;

    extern __shared__ float smem[];
    float * sS = smem;              // [HD][HD], float4 slot f of row r stored at f ^ ((r >> 3) & 7)
    float * sU = sS + HD * HD;      // [C][HD],  float4 slot f of row r stored at f ^ ((r >> 2) & 7)
#define KDA_SS(R, F) (sS + (R) * HD + (((F) ^ (((R) >> 3) & 7)) << 2))
#define KDA_SU(R, F) (sU + (R) * HD + (((F) ^ (((R) >> 2) & 7)) << 2))

    const size_t st_off = ((size_t) seq * n_heads + head) * (HD * HD);
    for (int i = tid; i < HD * HD / 4; i += NT) {
        *(float4 *) KDA_SS(i >> 5, i & 31) = *(const float4 *) (state_src + st_off + i * 4);
    }
    __syncthreads();

    const size_t ci0 = ((size_t) seq * n_heads + head) * CS;
    const size_t out_tok_stride = (size_t) HD * n_heads;

#define KDA_FMA8(ACC, A, B0, B1) \
    ACC[0] = fmaf(A, B0.x, ACC[0]); ACC[1] = fmaf(A, B0.y, ACC[1]); ACC[2] = fmaf(A, B0.z, ACC[2]); ACC[3] = fmaf(A, B0.w, ACC[3]); \
    ACC[4] = fmaf(A, B1.x, ACC[4]); ACC[5] = fmaf(A, B1.y, ACC[5]); ACC[6] = fmaf(A, B1.z, ACC[6]); ACC[7] = fmaf(A, B1.w, ACC[7]);

    for (int c = 0; c < n_chunks; ++c) {
        const float * WQ  = scratch + (ci0 + c) * SZ_CH;
        const float * UWY = WQ + SZ_WQ;
        const float * KD  = UWY + SZ_UWY;
        const float * AQK = KD + SZ_KD;
        const float * DEC = AQK + SZ_AQK;

        // phase 1: P = [w ; q~Gamma] S
        float acc[8][8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;
        }
        float4 uw[4][2];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            uw[i][0] = __ldg((const float4 *) (UWY + (tr * 4 + i) * HD + tc * 8));
            uw[i][1] = __ldg((const float4 *) (UWY + (tr * 4 + i) * HD + tc * 8 + 4));
        }
        {
            // operand ring: groups of 4 k-rows, loads issued two groups (512 FMAs per warp) ahead of use
            const float4 * wq0 = (const float4 *) (WQ + tr * 4);
            const float4 * wq1 = (const float4 *) (WQ + C + tr * 4);
            float4 ra[3][4], rb[3][4];
#define KDA_LD1(SLOT, G) { _Pragma("unroll") for (int j = 0; j < 4; ++j) { \
                ra[SLOT][j] = __ldg(wq0 + ((G) * 4 + j) * (2 * C / 4)); rb[SLOT][j] = __ldg(wq1 + ((G) * 4 + j) * (2 * C / 4)); } }
#define KDA_CP1(SLOT, G) { _Pragma("unroll") for (int j = 0; j < 4; ++j) { \
                const float4 b0 = *(const float4 *) KDA_SS((G) * 4 + j, tc * 2); const float4 b1 = *(const float4 *) KDA_SS((G) * 4 + j, tc * 2 + 1); \
                KDA_FMA8(acc[0], ra[SLOT][j].x, b0, b1); KDA_FMA8(acc[1], ra[SLOT][j].y, b0, b1); \
                KDA_FMA8(acc[2], ra[SLOT][j].z, b0, b1); KDA_FMA8(acc[3], ra[SLOT][j].w, b0, b1); \
                KDA_FMA8(acc[4], rb[SLOT][j].x, b0, b1); KDA_FMA8(acc[5], rb[SLOT][j].y, b0, b1); \
                KDA_FMA8(acc[6], rb[SLOT][j].z, b0, b1); KDA_FMA8(acc[7], rb[SLOT][j].w, b0, b1); } }
            KDA_LD1(0, 0); KDA_LD1(1, 1);
            for (int g = 0; g + 3 <= HD / 4; g += 3) {   // 32 groups: 10 x 3 + 2
                KDA_LD1(2, g + 2); KDA_CP1(0, g);
                if (g + 3 < HD / 4) KDA_LD1(0, g + 3);
                KDA_CP1(1, g + 1);
                if (g + 4 < HD / 4) KDA_LD1(1, g + 4);
                KDA_CP1(2, g + 2);
            }
            KDA_CP1(0, 30); KDA_CP1(1, 31);
#undef KDA_LD1
#undef KDA_CP1
        }
        float oq[4][8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int t = tr * 4 + i;
            const float4 uw0 = uw[i][0], uw1 = uw[i][1];
            *(float4 *) KDA_SU(t, tc * 2)     = make_float4(uw0.x - acc[i][0], uw0.y - acc[i][1], uw0.z - acc[i][2], uw0.w - acc[i][3]);
            *(float4 *) KDA_SU(t, tc * 2 + 1) = make_float4(uw1.x - acc[i][4], uw1.y - acc[i][5], uw1.z - acc[i][6], uw1.w - acc[i][7]);
#pragma unroll
            for (int j = 0; j < 8; ++j) oq[i][j] = acc[4 + i][j];
        }
        __syncthreads();

        // phase 2a: o = q~Gamma S + Aqk u  (AQK[s][t] is zero for s > t, so the loop is uniform)
        {
            const float4 * aqk = (const float4 *) (AQK + tr * 4);
            float4 rq[3][8];
#define KDA_LD2(SLOT, G) { _Pragma("unroll") for (int j = 0; j < 8; ++j) rq[SLOT][j] = __ldg(aqk + ((G) * 8 + j) * (C / 4)); }
#define KDA_CP2(SLOT, G) { _Pragma("unroll") for (int j = 0; j < 8; ++j) { \
                const float4 u0 = *(const float4 *) KDA_SU((G) * 8 + j, tc * 2); const float4 u1 = *(const float4 *) KDA_SU((G) * 8 + j, tc * 2 + 1); \
                KDA_FMA8(oq[0], rq[SLOT][j].x, u0, u1); KDA_FMA8(oq[1], rq[SLOT][j].y, u0, u1); \
                KDA_FMA8(oq[2], rq[SLOT][j].z, u0, u1); KDA_FMA8(oq[3], rq[SLOT][j].w, u0, u1); } }
            KDA_LD2(0, 0); KDA_LD2(1, 1);
            KDA_LD2(2, 2); KDA_CP2(0, 0); KDA_LD2(0, 3); KDA_CP2(1, 1); KDA_LD2(1, 4); KDA_CP2(2, 2);
            KDA_LD2(2, 5); KDA_CP2(0, 3); KDA_LD2(0, 6); KDA_CP2(1, 4); KDA_LD2(1, 7); KDA_CP2(2, 5);
            KDA_CP2(0, 6); KDA_CP2(1, 7);
#undef KDA_LD2
#undef KDA_CP2
        }
        {
            const int chunk_start = tok0 + c * C;
            const int nvalid = min(C, n_tokens - chunk_start);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int t = tr * 4 + i;
                if (t < nvalid) {
                    float * o = dst + ((size_t) seq * n_tokens + chunk_start + t) * out_tok_stride + (size_t) head * HD + tc * 8;
                    *(float4 *) o       = make_float4(oq[i][0], oq[i][1], oq[i][2], oq[i][3]);
                    *(float4 *) (o + 4) = make_float4(oq[i][4], oq[i][5], oq[i][6], oq[i][7]);
                }
            }
        }

        // phase 2b: S = Diag(Gamma_C) S + k_dec^T u  (key rows tr*8..+7, columns tc*8..+7)
        const float4 d0 = __ldg((const float4 *) (DEC + tr * 8));
        const float4 d1 = __ldg((const float4 *) (DEC + tr * 8 + 4));
#pragma unroll
        for (int i = 0; i < 8; ++i) {
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;
        }
        {
            const float4 * kd0 = (const float4 *) (KD + tr * 8);
            const float4 * kd1 = (const float4 *) (KD + tr * 8 + 4);
            float4 ra[3][4], rb[3][4];
#define KDA_LD3(SLOT, G) { _Pragma("unroll") for (int j = 0; j < 4; ++j) { \
                ra[SLOT][j] = __ldg(kd0 + ((G) * 4 + j) * (HD / 4)); rb[SLOT][j] = __ldg(kd1 + ((G) * 4 + j) * (HD / 4)); } }
#define KDA_CP3(SLOT, G) { _Pragma("unroll") for (int j = 0; j < 4; ++j) { \
                const float4 u0 = *(const float4 *) KDA_SU((G) * 4 + j, tc * 2); const float4 u1 = *(const float4 *) KDA_SU((G) * 4 + j, tc * 2 + 1); \
                KDA_FMA8(acc[0], ra[SLOT][j].x, u0, u1); KDA_FMA8(acc[1], ra[SLOT][j].y, u0, u1); \
                KDA_FMA8(acc[2], ra[SLOT][j].z, u0, u1); KDA_FMA8(acc[3], ra[SLOT][j].w, u0, u1); \
                KDA_FMA8(acc[4], rb[SLOT][j].x, u0, u1); KDA_FMA8(acc[5], rb[SLOT][j].y, u0, u1); \
                KDA_FMA8(acc[6], rb[SLOT][j].z, u0, u1); KDA_FMA8(acc[7], rb[SLOT][j].w, u0, u1); } }
            KDA_LD3(0, 0); KDA_LD3(1, 1);
            for (int g = 0; g + 3 <= C / 4; g += 3) {   // 16 groups: 5 x 3 + 1
                KDA_LD3(2, g + 2); KDA_CP3(0, g);
                if (g + 3 < C / 4) KDA_LD3(0, g + 3);
                KDA_CP3(1, g + 1);
                if (g + 4 < C / 4) KDA_LD3(1, g + 4);
                KDA_CP3(2, g + 2);
            }
            KDA_CP3(0, 15);
#undef KDA_LD3
#undef KDA_CP3
        }
        {
            const float dec[8] = { d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w };
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                float4 * sp0 = (float4 *) KDA_SS(tr * 8 + r, tc * 2);
                float4 * sp1 = (float4 *) KDA_SS(tr * 8 + r, tc * 2 + 1);
                float4 s0 = *sp0, s1 = *sp1;
                s0.x = fmaf(dec[r], s0.x, acc[r][0]); s0.y = fmaf(dec[r], s0.y, acc[r][1]);
                s0.z = fmaf(dec[r], s0.z, acc[r][2]); s0.w = fmaf(dec[r], s0.w, acc[r][3]);
                s1.x = fmaf(dec[r], s1.x, acc[r][4]); s1.y = fmaf(dec[r], s1.y, acc[r][5]);
                s1.z = fmaf(dec[r], s1.z, acc[r][6]); s1.w = fmaf(dec[r], s1.w, acc[r][7]);
                *sp0 = s0; *sp1 = s1;
            }
        }
        __syncthreads();
    }
#undef KDA_FMA8

    for (int i = tid; i < HD * HD / 4; i += NT) {
        *(float4 *) (state_dst + st_off + i * 4) = *(const float4 *) KDA_SS(i >> 5, i & 31);
    }
#undef KDA_SS
#undef KDA_SU
}

static inline size_t scratch_floats(int n_heads, int n_seqs) {
    return (size_t) CS * n_heads * n_seqs * SZ_CH;
}

// head_dim 128 only; per-step saved states (src6) need the recurrent kernel; q/k/v rows are read as float4
static inline bool supported(int head_dim, int n_tokens, const float * saved_states, const float * q, const float * k,
        const float * v, size_t vnb1, size_t vnb2, size_t vnb3) {
    if (head_dim != HD || saved_states != nullptr || n_tokens < 1) return false;
    if (((uintptr_t) q & 15) || ((uintptr_t) k & 15) || ((uintptr_t) v & 15)) return false;
    if ((vnb1 & 3) || (vnb2 & 3) || (vnb3 & 3)) return false;
    return true;
}

// Runs the chunked prefill for every sequence and head. scratch holds scratch_floats() floats.
static inline void launch(
        const float * q, const float * k, const float * v, const float * g, const float * beta,
        const float * state_in, float * dst, float * state_out, float * scratch,
        int n_tokens, int n_heads, int gqa_ratio, int repeat_type, int n_seqs,
        size_t vnb1, size_t vnb2, size_t vnb3,
        size_t gnb0, size_t gnb1, size_t gnb2, size_t gnb3,
        size_t bnb1, size_t bnb2, size_t bnb3,
        cudaStream_t stream) {
    static bool attr_done = false;
    if (!attr_done) {
        CUDA_CHECK(cudaFuncSetAttribute(kda_chunk_intra, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_INTRA));
        CUDA_CHECK(cudaFuncSetAttribute(kda_chunk_seq,   cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SEQ));
        CUDA_CHECK(cudaFuncSetAttribute(kda_chunk_intra, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        CUDA_CHECK(cudaFuncSetAttribute(kda_chunk_seq,   cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        attr_done = true;
    }
    const int n_heads_kq = n_heads / gqa_ratio;
    const int n_chunks = (n_tokens + C - 1) / C;
    for (int c0 = 0; c0 < n_chunks; c0 += CS) {
        const int nc = min(CS, n_chunks - c0);
        const int tok0 = c0 * C;
        kda_chunk_intra<<<dim3(nc, n_heads, n_seqs), NT_INTRA, SMEM_INTRA, stream>>>(
                q, k, v, g, beta, scratch, n_tokens, tok0, n_heads, n_heads_kq, gqa_ratio, repeat_type,
                vnb1, vnb2, vnb3, gnb0, gnb1, gnb2, gnb3, bnb1, bnb2, bnb3);
        kda_chunk_seq<<<dim3(n_heads, n_seqs), NT_SEQ, SMEM_SEQ, stream>>>(
                scratch, c0 == 0 ? state_in : state_out, state_out, dst, n_tokens, tok0, nc, n_heads);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kda_chunk
