// Fused scoring + per-query top-k selection for the n_vis variant of GGML_OP_INDEXER_TOPK (glm5next KPool-DSA,
// patch p9). Used by ggml_cuda_op_indexer_topk in indexer_topk.cu; GGML_DSA_INDEXER_FUSED=0 restores the old
// cuBLAS GEMM + fused read-back pass + cub segmented sort path for A/B.
//
// 1. k_dsa_score: score[pool, query] = sum_h w[query, h] * relu(q[query, h] . k[pool]) for D = 128 channels and
//    NH = 32 indexer heads, from the f16 pooled keys and the f16-converted queries with mma.sync m16n8k16 (f32
//    accumulate, whole K extent per tile). A block owns 4 queries (128 GEMM columns, resident in shared memory)
//    and loops over 64-pool tiles of its slab (cp.async double buffer). The epilogue stages each warp's 32 x 32
//    (pool x head) f32 tile in shared memory and one lane per pool applies relu / weight / head-sum in the same
//    sequential f32 FMA order as the old k_fused_relu_mul_sum_rows_nvis pass, so the scores can only differ from
//    the old path by the GEMM's accumulation order. Visibility: pool < n_vis[query] and col_ok[pool] (else -inf);
//    pools at or beyond the block's largest n_vis are not computed (the select never reads them). Only the
//    [n_pool, rows] f32 score is written, never the [n_pool, NH, rows] product.
//
// 2. k_dsa_select: one block per query. Radix select over the sortable 32-bit key of the score in 12 / 12 / 8-bit
//    digits (histogram + descending scan per level; the candidates of the first level's bin are compacted into
//    shared memory in pool order when they fit, otherwise the later levels re-scan the row), giving the exact
//    k-th score and the number of ties to take. Final pass: every pool above the threshold plus the lowest-index
//    ties, then a bitonic sort of the k entries by (score descending, pool ascending), which is the order the old
//    path's stable segmented sort produced. Invisible scores (-inf) are skipped; unfilled slots get the sentinel.
#pragma once

#include "common.cuh"

namespace dsa_idx {

constexpr int D   = 128;    // indexer channels per head
constexpr int NH  = 32;     // indexer heads
constexpr int QT  = 4;      // queries per block
constexpr int BN  = QT*NH;  // GEMM columns per block (128)
constexpr int BM  = 64;     // pools per tile
constexpr int NWARPS   = 8;
constexpr int NTHREADS = NWARPS*32;
constexpr int SLAB = 2048;  // pools per block (BM tiles looped)
constexpr int ROWB = D*2;   // bytes per f16 row

constexpr int SMEM_B = BN*ROWB;         // 32 KB: the block's 128 query rows
constexpr int SMEM_A = BM*ROWB;         // 16 KB per pool-tile buffer (x2)
constexpr int SMEM_C = NWARPS*32*32*4;  // 32 KB: per-warp 32x32 f32 staging
constexpr int SMEM_SCORE = SMEM_B + 2*SMEM_A + SMEM_C;   // 96 KB

// f16 rows of 256 bytes = 16 chunks of 16 bytes; chunk c of row r lives at chunk c ^ (r & 7) so that the 8 rows
// an ldmatrix touches hit 8 different 16-byte bank groups
__device__ __forceinline__ unsigned swz(int row, int chunk) {
    return (unsigned) (row*ROWB + ((chunk ^ (row & 7)) << 4));
}

__device__ __forceinline__ void cp_async_16_zfill(unsigned dst, const void * src, bool pred) {
#ifdef CP_ASYNC_AVAILABLE
    const int sz = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(sz));
#else
    GGML_UNUSED(dst); GGML_UNUSED(src); GGML_UNUSED(pred);
    NO_DEVICE_CODE;
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;\n");
#endif
}
__device__ __forceinline__ void cp_async_wait_0() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void ldsm_x4(unsigned addr, unsigned & r0, unsigned & r1, unsigned & r2, unsigned & r3) {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
#else
    GGML_UNUSED(addr); GGML_UNUSED(r0); GGML_UNUSED(r1); GGML_UNUSED(r2); GGML_UNUSED(r3);
    NO_DEVICE_CODE;
#endif
}
__device__ __forceinline__ void mma_16816_f32(float * c, const unsigned * a, const unsigned * b) {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    GGML_UNUSED(c); GGML_UNUSED(a); GGML_UNUSED(b);
    NO_DEVICE_CODE;
#endif
}
// f16 accumulate: c[0] = (row g: cols 2t, 2t+1), c[1] = (row g+8: cols 2t, 2t+1) as half2
__device__ __forceinline__ void mma_16816_f16(unsigned * c, const unsigned * a, const unsigned * b) {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    GGML_UNUSED(c); GGML_UNUSED(a); GGML_UNUSED(b);
    NO_DEVICE_CODE;
#endif
}

// accumulator tile of one m16n8 mma: f32 (4 floats) or f16 (2 half2), read back as the (row, col pair) floats
template <bool F16ACC> struct acc_t;
template <> struct acc_t<false> {
    float v[4];
    __device__ __forceinline__ void zero() { v[0] = v[1] = v[2] = v[3] = 0.0f; }
    __device__ __forceinline__ void mma(const unsigned * a, const unsigned * b) { mma_16816_f32(v, a, b); }
    __device__ __forceinline__ float2 get(int h) const { return make_float2(v[2*h], v[2*h + 1]); }
};
template <> struct acc_t<true> {
    unsigned v[2];
    __device__ __forceinline__ void zero() { v[0] = v[1] = 0u; }
    __device__ __forceinline__ void mma(const unsigned * a, const unsigned * b) { mma_16816_f16(v, a, b); }
    __device__ __forceinline__ float2 get(int h) const { return __half22float2(*(const half2 *) &v[h]); }
};

// kp: [n_pool][D] f16 pooled keys; qh: [rows][NH][D] f16 queries; w: [rows][NH]; n_vis: [rows] visible pool count;
// vis_lo: [rows] first visible pool or null (0); col_ok: [n_pool] or null; score: [rows][n_pool] f32 (pools outside
// the block's visible window are left untouched). F16ACC: f16 accumulation in the GEMM (2x the tensor rate on
// GA10x, scores differ from the f32 path at the 1e-3 level).
template <bool F16ACC>
__global__ void __launch_bounds__(NTHREADS, 1) k_dsa_score(
        const half  * __restrict__ kp,
        const half  * __restrict__ qh,
        const float * __restrict__ w,
        const int   * __restrict__ n_vis,
        const int   * __restrict__ vis_lo,
        const int   * __restrict__ col_ok,
        float       * __restrict__ score,
        const int n_pool, const int rows) {
    extern __shared__ __align__(128) unsigned char smem[];
    const unsigned sB  = (unsigned) __cvta_generic_to_shared(smem);
    const unsigned sA0 = sB + SMEM_B;
    float * sC = (float *) (smem + SMEM_B + 2*SMEM_A);

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm   = warp & 1;    // 32-pool half of the tile
    const int wn   = warp >> 1;   // query of the block (0..3)
    const int q0   = blockIdx.x*QT;
    const int slab0 = blockIdx.y*SLAB;

    // the union of the block's visible windows [lo, lo + n_vis): tiles outside it are not computed
    int vmin = n_pool, vmax = 0;
#pragma unroll
    for (int j = 0; j < QT; ++j) {
        if (q0 + j < rows) {
            const int lo = vis_lo ? max(0, vis_lo[q0 + j]) : 0;
            vmin = min(vmin, lo);
            vmax = max(vmax, lo + n_vis[q0 + j]);
        }
    }
    vmax = min(vmax, n_pool);
    if (slab0 >= vmax || slab0 + SLAB <= vmin) {
        return;
    }
    const int t0     = max(0, (vmin - slab0)/BM);   // first tile of the slab that touches the window
    const int ntiles = (min(slab0 + SLAB, vmax) - slab0 + BM - 1)/BM;

    // the block's 128 query rows (query*NH + head), zero-filled past the last row
#pragma unroll
    for (int i = 0; i < (BN*16)/NTHREADS; ++i) {
        const int idx = tid + i*NTHREADS;
        const int r = idx >> 4, c = idx & 15;
        const bool ok = q0 + r/NH < rows;
        cp_async_16_zfill(sB + swz(r, c), qh + (ok ? (size_t) (q0*NH + r)*D : 0) + c*8, ok);
    }
    auto load_a = [&](int tile, int buf) {
        const int p0 = slab0 + tile*BM;
#pragma unroll
        for (int i = 0; i < (BM*16)/NTHREADS; ++i) {
            const int idx = tid + i*NTHREADS;
            const int r = idx >> 4, c = idx & 15;
            const bool ok = p0 + r < n_pool;
            cp_async_16_zfill(sA0 + buf*SMEM_A + swz(r, c), kp + (ok ? (size_t) (p0 + r)*D : 0) + c*8, ok);
        }
    };
    load_a(t0, t0 & 1);
    cp_async_commit();

    const int qw = q0 + wn;
    const bool qok = qw < rows;
    float wreg[NH];
#pragma unroll
    for (int i = 0; i < NH/4; ++i) {
        const float4 v = qok ? *(const float4 *) (w + (size_t) qw*NH + 4*i) : make_float4(0.f, 0.f, 0.f, 0.f);
        wreg[4*i + 0] = v.x; wreg[4*i + 1] = v.y; wreg[4*i + 2] = v.z; wreg[4*i + 3] = v.w;
    }
    const int lo = qok && vis_lo ? max(0, vis_lo[qw]) : 0;
    const int hi = qok ? min(lo + n_vis[qw], n_pool) : 0;
    float * sCw = sC + warp*(32*32);
    const int g = lane >> 2, tq = lane & 3;

    for (int t = t0; t < ntiles; ++t) {
        cp_async_wait_0();
        __syncthreads();   // tile t (and B) landed for every warp; buffer (t+1)&1 is no longer read
        if (t + 1 < ntiles) {
            load_a(t + 1, (t + 1) & 1);
            cp_async_commit();
        }
        const unsigned sA = sA0 + (t & 1)*SMEM_A;

        acc_t<F16ACC> acc[2][4];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                acc[mt][nt].zero();
            }
        }
#pragma unroll
        for (int ks = 0; ks < D/16; ++ks) {
            unsigned a[2][4], b[4][2];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int r = wm*32 + mt*16 + (lane & 15);
                const int c = ks*2 + (lane >> 4);
                ldsm_x4(sA + swz(r, c), a[mt][0], a[mt][1], a[mt][2], a[mt][3]);
            }
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                const int r = wn*32 + np*16 + (lane & 7) + ((lane >> 4) << 3);
                const int c = ks*2 + ((lane >> 3) & 1);
                ldsm_x4(sB + swz(r, c), b[2*np][0], b[2*np][1], b[2*np + 1][0], b[2*np + 1][1]);
            }
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    acc[mt][nt].mma(a[mt], b[nt]);
                }
            }
        }

        // stage the warp's 32 pools x 32 heads (float4 chunk c of row r at chunk c ^ (r & 7)), then lane p sums
        // pool p's heads 0..31 in order
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = mt*16 + g + 8*h;
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const int chunk = nt*2 + (tq >> 1);
                    *(float2 *) (sCw + row*32 + ((chunk ^ (row & 7)) << 2) + (tq & 1)*2) = acc[mt][nt].get(h);
                }
            }
        }
        __syncwarp();
        float sum = 0.0f;
#pragma unroll
        for (int ch = 0; ch < 8; ++ch) {
            const float4 v = *(const float4 *) (sCw + lane*32 + ((ch ^ (lane & 7)) << 2));
            sum = fmaf(fmaxf(v.x, 0.0f), wreg[4*ch + 0], sum);
            sum = fmaf(fmaxf(v.y, 0.0f), wreg[4*ch + 1], sum);
            sum = fmaf(fmaxf(v.z, 0.0f), wreg[4*ch + 2], sum);
            sum = fmaf(fmaxf(v.w, 0.0f), wreg[4*ch + 3], sum);
        }
        const int pool = slab0 + t*BM + wm*32 + lane;
        if (qok && pool < n_pool) {
            const bool vis = pool >= lo && pool < hi && (col_ok == nullptr || col_ok[pool] != 0);
            score[(size_t) qw*n_pool + pool] = vis ? sum : -INFINITY;
        }
        __syncwarp();   // the next tile's staging overwrites sCw
    }
}

// ---------------------------------------------------------------------------------------------------------------
// per-query top-k select

constexpr int SEL_NT   = 512;
constexpr int SEL_NBIN = 4096;   // 12-bit digits
constexpr int SEL_CAP  = 2048;   // first-level candidates kept in shared memory
constexpr int SEL_KMAX = 1024;   // largest k (power of two)

// monotonic unsigned key: larger score -> larger key (-inf is excluded before keying)
__device__ __forceinline__ unsigned score_key(float s) {
    const unsigned u = __float_as_uint(s);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ unsigned long long sel_pack(unsigned key, int idx) {
    return ((unsigned long long) key << 32) | (unsigned long long) (0xffffffffu - (unsigned) idx);
}
__device__ __forceinline__ int sel_unpack_idx(unsigned long long v) {
    return (int) (0xffffffffu - (unsigned) (v & 0xffffffffull));
}

// exclusive prefix of one int per thread over the block (one __syncthreads); total in `total`
__device__ __forceinline__ int sel_block_scan(int v, int * warp_sums, int & total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int incl = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int t = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += t;
    }
    if (lane == 31) warp_sums[warp] = incl;
    __syncthreads();
    int prefix = 0;
    total = 0;
#pragma unroll
    for (int i = 0; i < SEL_NT/32; ++i) {
        const int s = warp_sums[i];
        prefix += i < warp ? s : 0;
        total  += s;
    }
    return prefix + incl - v;
}

// descending walk over hist[0..nbins): the bin b with (count in bins > b) < need <= (count in bins >= b).
// s_res[0] = b (-1 when the total is below need; the caller sets it to -1 first), s_res[1] = count above b.
__device__ __forceinline__ void sel_find_bin(const unsigned * hist, int nbins, int need, int * warp_sums, int * s_res) {
    const int t = threadIdx.x;
    unsigned v[8];
    int cnt = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int b = nbins - 1 - (t*8 + j);
        v[j] = b >= 0 ? hist[b] : 0u;
        cnt += (int) v[j];
    }
    int total;
    int cum = sel_block_scan(cnt, warp_sums, total);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int b = nbins - 1 - (t*8 + j);
        if (b >= 0 && cum < need && cum + (int) v[j] >= need) {
            s_res[0] = b;
            s_res[1] = cum;
        }
        cum += (int) v[j];
    }
    __syncthreads();
}

// 4 consecutive scores of the window [lo, hi) starting at i0 (a multiple of 4); outside the window: -inf
__device__ __forceinline__ void sel_load4(const float * __restrict__ row, int i0, int lo, int hi, float * s) {
    if (i0 >= lo && i0 + 3 < hi) {
        const float4 v = *(const float4 *) (row + i0);
        s[0] = v.x; s[1] = v.y; s[2] = v.z; s[3] = v.w;
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) s[j] = i0 + j >= lo && i0 + j < hi ? row[i0 + j] : -INFINITY;
    }
}

// one ordered pass over the visible row: keys above thr go to out[] through the shared counter (any order), keys
// equal to thr are ranked in pool order and the first n_eq go to out[base_eq + rank]
__device__ __forceinline__ void sel_pass_global(const float * __restrict__ row, int lo, int hi, unsigned thr, int n_eq, int base_eq,
        unsigned long long * out, int * s_cnt, int (*warp_sums)[SEL_NT/32]) {
    const int tid = threadIdx.x;
    int base = 0, parity = 0;
    for (int c0 = lo & ~3; c0 < hi; c0 += SEL_NT*4) {
        const int i0 = c0 + tid*4;
        float s[4];
        sel_load4(row, i0, lo, hi, s);
        bool eq[4];
        int cnt = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            eq[j] = false;
            if (s[j] != -INFINITY) {
                const unsigned key = score_key(s[j]);
                if (key > thr) {
                    const int slot = atomicAdd(s_cnt, 1);
                    out[slot] = sel_pack(key, i0 + j);
                } else if (key == thr) {
                    eq[j] = true;
                    ++cnt;
                }
            }
        }
        if (n_eq > 0) {
            int total;
            int r = base + sel_block_scan(cnt, warp_sums[parity], total);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (eq[j]) {
                    if (r < n_eq) out[base_eq + r] = sel_pack(thr, i0 + j);
                    ++r;
                }
            }
            base += total;
            parity ^= 1;
        }
    }
}

__global__ void __launch_bounds__(SEL_NT, 2) k_dsa_select(
        const float * __restrict__ score,
        const int   * __restrict__ n_vis,
        const int   * __restrict__ vis_lo,
        int         * __restrict__ dst,
        const int n_pool, const int k, const int sentinel) {
    __shared__ unsigned hist[SEL_NBIN];
    __shared__ unsigned ckey[SEL_CAP];
    __shared__ int      cidx[SEL_CAP];
    __shared__ unsigned long long out[SEL_KMAX];
    __shared__ int warp_sums[2][SEL_NT/32];
    __shared__ int s_res[2];
    __shared__ int s_cnt;

    const int tid = threadIdx.x;
    const int q   = blockIdx.x;
    const float * row = score + (size_t) q*n_pool;
    int * drow = dst + (size_t) q*k;
    const int lo = vis_lo ? max(0, vis_lo[q]) : 0;
    const int hi = min(lo + n_vis[q], n_pool);   // the visible window is [lo, hi)

    // level 1: top 12 bits
    for (int i = tid; i < SEL_NBIN; i += SEL_NT) hist[i] = 0u;
    if (tid == 0) { s_res[0] = -1; s_cnt = 0; }
    __syncthreads();
    for (int i = lo + tid; i < hi; i += SEL_NT) {
        const float s = row[i];
        if (s != -INFINITY) atomicAdd(&hist[score_key(s) >> 20], 1u);
    }
    __syncthreads();
    sel_find_bin(hist, SEL_NBIN, k, warp_sums[0], s_res);
    const int b1 = s_res[0];
    const int above1 = s_res[1];

    unsigned thr = 0u;   // final: take keys > thr, plus the first n_eq keys == thr in pool order
    int n_eq = 0;
    int n_sel = k;
    bool done = false;   // the output list is complete (no final pass over the row needed)
    if (b1 < 0) {
        // fewer than k visible pools: take them all
        sel_pass_global(row, lo, hi, 0u, 0, 0, out, &s_cnt, warp_sums);
        __syncthreads();
        n_sel = s_cnt;
        done = true;
    } else {
        const int cnt1  = (int) hist[b1];
        const int need1 = k - above1;
        __syncthreads();   // hist is reused below
        if (cnt1 == need1) {
            thr = b1 > 0 ? ((unsigned) b1 << 20) - 1u : 0u;
            n_eq = 0;
        } else if (cnt1 <= SEL_CAP) {
            // compact the bin's entries into shared memory in pool order; entries above the bin are final
            {
                int base = 0, parity = 0;
                for (int c0 = lo & ~3; c0 < hi; c0 += SEL_NT*4) {
                    const int i0 = c0 + tid*4;
                    float s[4];
                    sel_load4(row, i0, lo, hi, s);
                    unsigned key[4];
                    bool in[4];
                    int cnt = 0;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        in[j] = false;
                        key[j] = 0u;
                        if (s[j] != -INFINITY) {
                            key[j] = score_key(s[j]);
                            const int b = (int) (key[j] >> 20);
                            if (b > b1) {
                                const int slot = atomicAdd(&s_cnt, 1);
                                out[slot] = sel_pack(key[j], i0 + j);
                            } else if (b == b1) {
                                in[j] = true;
                                ++cnt;
                            }
                        }
                    }
                    int total;
                    int r = base + sel_block_scan(cnt, warp_sums[parity], total);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        if (in[j]) { ckey[r] = key[j]; cidx[r] = i0 + j; ++r; }
                    }
                    base += total;
                    parity ^= 1;
                }
            }
            __syncthreads();
            // level 2 over the candidates: bits 8..19
            for (int i = tid; i < SEL_NBIN; i += SEL_NT) hist[i] = 0u;
            if (tid == 0) s_res[0] = -1;
            __syncthreads();
            for (int i = tid; i < cnt1; i += SEL_NT) atomicAdd(&hist[(ckey[i] >> 8) & 0xfffu], 1u);
            __syncthreads();
            sel_find_bin(hist, SEL_NBIN, need1, warp_sums[0], s_res);
            const int b2 = s_res[0], above2 = s_res[1];
            const int cnt2 = (int) hist[b2];
            const int need2 = need1 - above2;
            const unsigned pre2 = ((unsigned) b1 << 20) | ((unsigned) b2 << 8);
            __syncthreads();
            if (cnt2 == need2) {
                thr = pre2 > 0u ? pre2 - 1u : 0u;
                n_eq = 0;
            } else {
                // level 3: the low byte
                for (int i = tid; i < 256; i += SEL_NT) hist[i] = 0u;
                if (tid == 0) s_res[0] = -1;
                __syncthreads();
                for (int i = tid; i < cnt1; i += SEL_NT) {
                    const unsigned key = ckey[i];
                    if ((key >> 8) == (pre2 >> 8)) atomicAdd(&hist[key & 0xffu], 1u);
                }
                __syncthreads();
                sel_find_bin(hist, 256, need2, warp_sums[0], s_res);
                const int b3 = s_res[0], above3 = s_res[1];
                thr  = pre2 | (unsigned) b3;
                n_eq = need2 - above3;
                __syncthreads();
            }
            // final over the candidates (pool order): above thr -> counter slots, ties -> ranked slots
            {
                const int base_eq = k - n_eq;
                int cnt = 0;
                bool eq[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int i = tid*4 + j;
                    eq[j] = false;
                    if (i < cnt1) {
                        const unsigned key = ckey[i];
                        if (key > thr) {
                            const int slot = atomicAdd(&s_cnt, 1);
                            out[slot] = sel_pack(key, cidx[i]);
                        } else if (key == thr) {
                            eq[j] = true;
                            ++cnt;
                        }
                    }
                }
                int total;
                int r = sel_block_scan(cnt, warp_sums[0], total);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (eq[j]) {
                        if (r < n_eq) out[base_eq + r] = sel_pack(thr, cidx[tid*4 + j]);
                        ++r;
                    }
                }
            }
            __syncthreads();
            done = true;
        } else {
            // too many first-level candidates for shared memory: refine from the row
            for (int i = tid; i < SEL_NBIN; i += SEL_NT) hist[i] = 0u;
            if (tid == 0) s_res[0] = -1;
            __syncthreads();
            for (int i = lo + tid; i < hi; i += SEL_NT) {
                const float s = row[i];
                if (s != -INFINITY) {
                    const unsigned key = score_key(s);
                    if ((int) (key >> 20) == b1) atomicAdd(&hist[(key >> 8) & 0xfffu], 1u);
                }
            }
            __syncthreads();
            sel_find_bin(hist, SEL_NBIN, need1, warp_sums[0], s_res);
            const int b2 = s_res[0], above2 = s_res[1];
            const int cnt2 = (int) hist[b2];
            const int need2 = need1 - above2;
            const unsigned pre2 = ((unsigned) b1 << 20) | ((unsigned) b2 << 8);
            __syncthreads();
            if (cnt2 == need2) {
                thr = pre2 > 0u ? pre2 - 1u : 0u;
                n_eq = 0;
            } else {
                for (int i = tid; i < 256; i += SEL_NT) hist[i] = 0u;
                if (tid == 0) s_res[0] = -1;
                __syncthreads();
                for (int i = lo + tid; i < hi; i += SEL_NT) {
                    const float s = row[i];
                    if (s != -INFINITY) {
                        const unsigned key = score_key(s);
                        if ((key >> 8) == (pre2 >> 8)) atomicAdd(&hist[key & 0xffu], 1u);
                    }
                }
                __syncthreads();
                sel_find_bin(hist, 256, need2, warp_sums[0], s_res);
                const int b3 = s_res[0], above3 = s_res[1];
                thr  = pre2 | (unsigned) b3;
                n_eq = need2 - above3;
                __syncthreads();
            }
        }
    }
    if (!done) {
        sel_pass_global(row, lo, hi, thr, n_eq, k - n_eq, out, &s_cnt, warp_sums);
        __syncthreads();
    }

    // sort the n_sel entries descending (score, then ascending pool); pad to a power of two
    int npow = 1;
    while (npow < k) npow <<= 1;
    for (int i = n_sel + tid; i < npow; i += SEL_NT) out[i] = 0ull;
    for (int size = 2; size <= npow; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            __syncthreads();
            for (int i = tid; i < npow; i += SEL_NT) {
                const int j = i ^ stride;
                if (j > i) {
                    const unsigned long long a = out[i], b = out[j];
                    const bool up = (i & size) == 0;
                    if (up ? (a < b) : (a > b)) {
                        out[i] = b;
                        out[j] = a;
                    }
                }
            }
        }
    }
    __syncthreads();
    for (int i = tid; i < k; i += SEL_NT) {
        drow[i] = i < n_sel ? sel_unpack_idx(out[i]) : sentinel;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// host side

static inline bool fused_supported(int d, int nh, int k, int cc) {
    return d == D && nh == NH && k >= 1 && k <= SEL_KMAX && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= CC_AMPERE;
}

// rows of the score buffer per launch: keep [n_pool x rows_chunk] f32 at or under max_bytes
static inline int rows_per_chunk(int n_pool, int rows, size_t max_bytes) {
    int r = (int) std::min<size_t>((size_t) rows, max_bytes/((size_t) n_pool*sizeof(float)));
    r = std::max(QT, (r/QT)*QT);
    return std::min(r, rows);
}

// kp: [n_pool][D] f16; qh: [rows][NH][D] f16; w: [rows][NH] f32; n_vis: [rows]; vis_lo: [rows] or null;
// col_ok: [n_pool] or null; dst: [rows][k] i32; score_buf: n_pool*rows_chunk floats; f16acc: f16 GEMM accumulation
// ms_score / ms_select: when given, the phases are timed per chunk with synchronizing events (diagnostics only)
static void topk_fused(const half * kp, int n_pool, const half * qh, const float * w, const int * n_vis, const int * vis_lo,
        const int * col_ok, int * dst, int rows, int k, int sentinel, float * score_buf, int rows_chunk, bool f16acc,
        int device, cudaStream_t stream, float * ms_score = nullptr, float * ms_select = nullptr) {
    static bool attr_set[GGML_CUDA_MAX_DEVICES] = { false };
    if (!attr_set[device]) {
        CUDA_CHECK(cudaFuncSetAttribute(k_dsa_score<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SCORE));
        CUDA_CHECK(cudaFuncSetAttribute(k_dsa_score<true>,  cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SCORE));
        attr_set[device] = true;
    }
    const bool timing = ms_score != nullptr && ms_select != nullptr;
    cudaEvent_t ev[3];
    if (timing) {
        for (int i = 0; i < 3; ++i) CUDA_CHECK(cudaEventCreate(&ev[i]));
    }
    for (int r0 = 0; r0 < rows; r0 += rows_chunk) {
        const int nr = std::min(rows_chunk, rows - r0);
        const dim3 grid((nr + QT - 1)/QT, (n_pool + SLAB - 1)/SLAB, 1);
        if (timing) CUDA_CHECK(cudaEventRecord(ev[0], stream));
        if (f16acc) {
            k_dsa_score<true><<<grid, NTHREADS, SMEM_SCORE, stream>>>(kp, qh + (size_t) r0*NH*D, w + (size_t) r0*NH,
                    n_vis + r0, vis_lo ? vis_lo + r0 : nullptr, col_ok, score_buf, n_pool, nr);
        } else {
            k_dsa_score<false><<<grid, NTHREADS, SMEM_SCORE, stream>>>(kp, qh + (size_t) r0*NH*D, w + (size_t) r0*NH,
                    n_vis + r0, vis_lo ? vis_lo + r0 : nullptr, col_ok, score_buf, n_pool, nr);
        }
        CUDA_CHECK(cudaGetLastError());
        if (timing) CUDA_CHECK(cudaEventRecord(ev[1], stream));
        k_dsa_select<<<nr, SEL_NT, 0, stream>>>(score_buf, n_vis + r0, vis_lo ? vis_lo + r0 : nullptr, dst + (size_t) r0*k, n_pool, k, sentinel);
        CUDA_CHECK(cudaGetLastError());
        if (timing) {
            CUDA_CHECK(cudaEventRecord(ev[2], stream));
            CUDA_CHECK(cudaEventSynchronize(ev[2]));
            float a = 0.0f, b = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&a, ev[0], ev[1]));
            CUDA_CHECK(cudaEventElapsedTime(&b, ev[1], ev[2]));
            *ms_score  += a;
            *ms_select += b;
        }
    }
    if (timing) {
        for (int i = 0; i < 3; ++i) CUDA_CHECK(cudaEventDestroy(ev[i]));
    }
}

} // namespace dsa_idx
