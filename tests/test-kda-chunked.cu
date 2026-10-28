// Standalone numeric and timing check of the chunked KDA prefill against the recurrent kernel.
// Build (from the repo root, inside the CUDA dev image):
//   nvcc -O3 -std=c++17 -arch=sm_86 -DGGML_CUDA_FUSION=1 -DGGML_CUDA_MIN_BATCH_OFFLOAD=32 \
//        -I ggml/include -I ggml/src -I ggml/src/ggml-cuda tests/test-kda-chunked.cu -o /tmp/test-kda-chunked -lcublas
// Run: /tmp/test-kda-chunked [n_tokens ...]
#include "../ggml/src/ggml-cuda/kda.cu"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <vector>

void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "CUDA error: %s\n  %s at %s:%d: %s\n", msg, stmt, file, line, func);
    exit(1);
}
std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int) { abort(); }
extern "C" void ggml_abort(const char * file, int line, const char * fmt, ...) { fprintf(stderr, "ggml_abort %s:%d %s\n", file, line, fmt); abort(); }
void ggml_cuda_set_device(int device) { CUDA_CHECK(cudaSetDevice(device)); }

struct cmp_result { double max_abs_err, max_abs_ref, rms_err, rms_ref, max_rel; size_t worst; };

static cmp_result compare(const std::vector<float> & a, const std::vector<float> & ref) {
    cmp_result r{0, 0, 0, 0, 0, 0};
    double se = 0, sr = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double e = fabs((double) a[i] - ref[i]);
        if (e > r.max_abs_err) { r.max_abs_err = e; r.worst = i; }
        r.max_abs_ref = std::max(r.max_abs_ref, (double) fabs(ref[i]));
        se += e * e; sr += (double) ref[i] * ref[i];
    }
    r.rms_err = sqrt(se / a.size());
    r.rms_ref = sqrt(sr / a.size());
    r.max_rel = r.max_abs_err / std::max(r.max_abs_ref, 1e-30);
    return r;
}

template <typename T> static T * to_dev(const std::vector<T> & h) {
    T * d; CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <typename T> static std::vector<T> to_host(const T * d, size_t n) {
    std::vector<T> h(n); CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h;
}

static int run_case(int n_tokens, int n_heads, int n_seqs, int gqa_ratio, int reps, bool quiet_timing) {
    const int HD = 128;
    const int n_heads_kq = n_heads / gqa_ratio;
    std::mt19937 rng(1234 + n_tokens);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    // q, k: [HD, n_tokens, n_heads_kq, n_seqs] contiguous, L2-normalized per token (as the model does)
    const size_t qk_n = (size_t) HD * n_tokens * n_heads_kq * n_seqs;
    std::vector<float> hq(qk_n), hk(qk_n);
    for (size_t row = 0; row < qk_n / HD; ++row) {
        float sq = 0, sk = 0;
        for (int d = 0; d < HD; ++d) { hq[row*HD+d] = nrm(rng); hk[row*HD+d] = nrm(rng); sq += hq[row*HD+d]*hq[row*HD+d]; sk += hk[row*HD+d]*hk[row*HD+d]; }
        for (int d = 0; d < HD; ++d) { hq[row*HD+d] /= sqrtf(sq); hk[row*HD+d] /= sqrtf(sk); }
    }
    // v, g in the model's [HD, n_heads, n_tokens, n_seqs] layouts (then viewed with the permuted strides)
    const size_t vg_n = (size_t) HD * n_heads * n_tokens * n_seqs;
    std::vector<float> hv(vg_n), hg(vg_n);
    for (size_t i = 0; i < vg_n; ++i) { hv[i] = nrm(rng); const float u = uni(rng); hg[i] = -(0.01f + 1.5f * u * u); }
    const size_t vnb1 = (size_t) HD * n_heads, vnb2 = HD, vnb3 = (size_t) HD * n_heads * n_tokens;
    const size_t gnb0 = (size_t) HD * n_heads, gnb1 = 1, gnb2 = HD, gnb3 = (size_t) HD * n_heads * n_tokens;
    // beta: [n_heads, 1, n_tokens, n_seqs] -> permuted strides bnb1 = n_heads (token), bnb2 = 1 (head)
    const size_t b_n = (size_t) n_heads * n_tokens * n_seqs;
    std::vector<float> hb(b_n);
    for (size_t i = 0; i < b_n; ++i) hb[i] = 1.5f * nrm(rng);
    const size_t bnb1 = n_heads, bnb2 = 1, bnb3 = (size_t) n_heads * n_tokens;
    // state: [HD][HD] per head, [k][v]
    const size_t st_n = (size_t) HD * HD * n_heads * n_seqs;
    std::vector<float> hs(st_n);
    for (size_t i = 0; i < st_n; ++i) hs[i] = 0.3f * nrm(rng);
    const size_t out_n = (size_t) HD * n_heads * n_tokens * n_seqs;

    float * dq = to_dev(hq); float * dk = to_dev(hk); float * dv = to_dev(hv); float * dg = to_dev(hg); float * db = to_dev(hb);
    float * ds = to_dev(hs);
    float * out_r; float * st_r; float * out_c; float * st_c; float * st_alias;
    CUDA_CHECK(cudaMalloc(&out_r, out_n * sizeof(float))); CUDA_CHECK(cudaMalloc(&st_r, st_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out_c, out_n * sizeof(float))); CUDA_CHECK(cudaMalloc(&st_c, st_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st_alias, st_n * sizeof(float)));
    float * scratch; CUDA_CHECK(cudaMalloc(&scratch, kda_chunk::scratch_floats(n_heads, n_seqs) * sizeof(float)));
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));
    cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));

    auto run_rec = [&]() {
        kda_f32_cuda(dq, dk, dv, dg, db, ds, out_r, st_r, nullptr, HD, n_tokens, n_heads, gqa_ratio, 0, n_seqs,
                     vnb1, vnb2, vnb3, gnb0, gnb1, gnb2, gnb3, bnb1, bnb2, bnb3, stream);
    };
    auto run_chunk = [&](const float * sin, float * sout) {
        kda_chunk::launch(dq, dk, dv, dg, db, sin, out_c, sout, scratch, n_tokens, n_heads, gqa_ratio, 0, n_seqs,
                          vnb1, vnb2, vnb3, gnb0, gnb1, gnb2, gnb3, bnb1, bnb2, bnb3, stream);
    };

    run_rec(); CUDA_CHECK(cudaStreamSynchronize(stream));
    run_chunk(ds, st_c); CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto ho_r = to_host(out_r, out_n), ho_c = to_host(out_c, out_n);
    const auto hs_r = to_host(st_r, st_n), hs_c = to_host(st_c, st_n);
    const cmp_result co = compare(ho_c, ho_r), cs = compare(hs_c, hs_r);

    // aliased state (fused state copy): must match the non-aliased result bit for bit
    CUDA_CHECK(cudaMemcpy(st_alias, ds, st_n * sizeof(float), cudaMemcpyDeviceToDevice));
    run_chunk(st_alias, st_alias); CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto hs_a = to_host(st_alias, st_n);
    size_t alias_diff = 0;
    for (size_t i = 0; i < st_n; ++i) alias_diff += hs_a[i] != hs_c[i];

    // timing
    float ms_rec = 0, ms_chunk = 0;
    if (reps > 0) {
        run_rec(); run_chunk(ds, st_c); CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(e0, stream)); for (int i = 0; i < reps; ++i) run_rec(); CUDA_CHECK(cudaEventRecord(e1, stream));
        CUDA_CHECK(cudaEventSynchronize(e1)); CUDA_CHECK(cudaEventElapsedTime(&ms_rec, e0, e1)); ms_rec /= reps;
        CUDA_CHECK(cudaEventRecord(e0, stream)); for (int i = 0; i < reps; ++i) run_chunk(ds, st_c); CUDA_CHECK(cudaEventRecord(e1, stream));
        CUDA_CHECK(cudaEventSynchronize(e1)); CUDA_CHECK(cudaEventElapsedTime(&ms_chunk, e0, e1)); ms_chunk /= reps;
    }

    // per-kernel split timing (same launch sequence, events around each kernel kind)
    if (reps > 0 && getenv("KDA_TEST_SPLIT")) {
        using namespace kda_chunk;
        const int n_chunks = (n_tokens + C - 1) / C;
        float ms_intra = 0, ms_seq = 0;
        for (int rep = 0; rep < reps; ++rep) {
            for (int c0 = 0; c0 < n_chunks; c0 += CS) {
                const int nc = std::min(CS, n_chunks - c0), tok0 = c0 * C;
                float ms;
                CUDA_CHECK(cudaEventRecord(e0, stream));
                kda_chunk_intra<<<dim3(nc, n_heads, n_seqs), NT_INTRA, SMEM_INTRA, stream>>>(dq, dk, dv, dg, db, scratch, n_tokens, tok0, n_heads, n_heads_kq, gqa_ratio, 0,
                        vnb1, vnb2, vnb3, gnb0, gnb1, gnb2, gnb3, bnb1, bnb2, bnb3);
                CUDA_CHECK(cudaEventRecord(e1, stream)); CUDA_CHECK(cudaEventSynchronize(e1)); CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1)); ms_intra += ms;
                CUDA_CHECK(cudaEventRecord(e0, stream));
                kda_chunk_seq<<<dim3(n_heads, n_seqs), NT_SEQ, SMEM_SEQ, stream>>>(scratch, c0 == 0 ? ds : st_c, st_c, out_c, n_tokens, tok0, nc, n_heads);
                CUDA_CHECK(cudaEventRecord(e1, stream)); CUDA_CHECK(cudaEventSynchronize(e1)); CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1)); ms_seq += ms;
            }
        }
        printf("   split: intra %.3f ms, seq %.3f ms per call (%d chunks, %d slabs)\n", ms_intra / reps, ms_seq / reps, n_chunks, (n_chunks + CS - 1) / CS);
    }
    if (getenv("KDA_TEST_CHECKSUM")) {
        // bitwise fingerprint of the chunked output and state (to confirm two kernel versions agree bit for bit)
        uint64_t h = 1469598103934665603ull;
        auto mix = [&](const std::vector<float> & x) { for (float f : x) { uint32_t u; memcpy(&u, &f, 4); h ^= u; h *= 1099511628211ull; } };
        mix(ho_c); mix(hs_c);
        printf("   checksum %016llx\n", (unsigned long long) h);
    }
    const bool ok = co.max_rel < 2e-4 && cs.max_rel < 2e-4 && alias_diff == 0;
    printf("| %6d | %3d | %d | %d | out: max|err| %.3e (max|ref| %.3e, rel %.2e, rms %.2e) | state: max|err| %.3e (max|ref| %.3e, rel %.2e, rms %.2e) | alias %s |",
           n_tokens, n_heads, n_seqs, gqa_ratio, co.max_abs_err, co.max_abs_ref, co.max_rel, co.rms_err / co.rms_ref,
           cs.max_abs_err, cs.max_abs_ref, cs.max_rel, cs.rms_err / cs.rms_ref, alias_diff == 0 ? "ok" : "DIFF");
    if (reps > 0 && !quiet_timing) printf(" rec %.3f ms, chunk %.3f ms (%.1fx) |", ms_rec, ms_chunk, ms_rec / ms_chunk);
    printf(" %s\n", ok ? "PASS" : "FAIL");
    if (!ok) {
        printf("   worst out idx %zu: chunk %.6f ref %.6f; worst state idx %zu: chunk %.6f ref %.6f\n",
               co.worst, ho_c[co.worst], ho_r[co.worst], cs.worst, hs_c[cs.worst], hs_r[cs.worst]);
    }

    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dg); cudaFree(db); cudaFree(ds);
    cudaFree(out_r); cudaFree(st_r); cudaFree(out_c); cudaFree(st_c); cudaFree(st_alias); cudaFree(scratch);
    cudaStreamDestroy(stream); cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    std::vector<int> toks;
    for (int i = 1; i < argc; ++i) toks.push_back(atoi(argv[i]));
    if (toks.empty()) toks = {64, 65, 1000, 8192};
    const int reps = getenv("KDA_TEST_REPS") ? atoi(getenv("KDA_TEST_REPS")) : 3;
    int fails = 0;
    printf("| tokens | heads | seqs | gqa | output error | state error | aliasing | timing per call | result |\n");
    for (int n : toks) fails += run_case(n, 64, 1, 1, reps, false);
    if (!getenv("KDA_TEST_NO_EXTRA")) {
        fails += run_case(200, 64, 2, 1, 0, true);   // two sequences
        fails += run_case(130, 64, 1, 2, 0, true);   // gqa: 32 q/k heads for 64 v heads
    }
    printf("%s\n", fails == 0 ? "ALL PASS" : "FAILURES");
    return fails;
}
