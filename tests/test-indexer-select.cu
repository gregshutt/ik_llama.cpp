// Standalone A/B of the p9 fused indexer kernels (indexer_select.cuh) against the old n_vis path of
// ggml_cuda_op_indexer_topk (cuBLAS f16 GEMM + k_fused_relu_mul_sum_rows_nvis + cub segmented sort +
// k_copy_topk_sentinel), on random inputs shaped like a glm5next ubatch.
// Build (from the repo root, inside the CUDA dev image):
//   nvcc -O3 -std=c++17 -arch=sm_86 -DGGML_CUDA_USE_CUB=1 -DGGML_CUDA_FUSION=1 -DGGML_CUDA_MIN_BATCH_OFFLOAD=32 \
//        -I ggml/include -I ggml/src -I ggml/src/ggml-cuda tests/test-indexer-select.cu -o /tmp/test-indexer-select -lcublas
// Run: /tmp/test-indexer-select <n_pool> <rows> [k=512] [iters=3] [mode=0] [f16acc=0]
//   mode 0: random scores; 1: duplicated pools (exact ties); 2: w = 0 for every other query (all-zero scores);
//   3: a per-query visible window [lo, lo + n_vis) (vis_lo; the old path scores [0, lo + n_vis) and the index
//   lists are checked against a host top-k over the window); f16acc=1 runs the f16-accumulate GEMM variant
#include "../ggml/src/ggml-cuda/indexer_topk.cu"
#include "../ggml/src/ggml-cuda/argsort.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <vector>

void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "CUDA error: %s\n  %s at %s:%d: %s\n", msg, stmt, file, line, func);
    exit(1);
}
std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int) { abort(); }
extern "C" void ggml_abort(const char * file, int line, const char * fmt, ...) { fprintf(stderr, "ggml_abort %s:%d %s\n", file, line, fmt); abort(); }
void ggml_cuda_set_device(int device) { CUDA_CHECK(cudaSetDevice(device)); }
to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type) { return nullptr; }
to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type) { return nullptr; }
void ggml_cuda_op_mul_mat_q(ggml_backend_cuda_context &, ggml_type, const mmq_args &) { abort(); }
void quantize_mmq_q8_1_cuda(const float *, void *, int64_t, int64_t, int64_t, int64_t, ggml_type, cudaStream_t) { abort(); }
extern "C" int64_t ggml_nrows(const struct ggml_tensor * t) { return t->ne[1]*t->ne[2]*t->ne[3]; }
extern "C" bool ggml_is_contiguous(const struct ggml_tensor *) { return true; }
extern "C" bool ggml_are_same_shape(const struct ggml_tensor *, const struct ggml_tensor *) { return true; }
extern "C" bool ggml_is_quantized(enum ggml_type) { return false; }
int ggml_cuda_get_device() { int d = 0; CUDA_CHECK(cudaGetDevice(&d)); return d; }
static ggml_cuda_device_info g_info;
const ggml_cuda_device_info & ggml_cuda_info() { return g_info; }

struct simple_pool : ggml_cuda_pool {
    void * alloc(size_t size, size_t * actual) override { void * p = nullptr; CUDA_CHECK(cudaMalloc(&p, size)); *actual = size; return p; }
    void free(void * p, size_t) override { CUDA_CHECK(cudaFree(p)); }
};

static __global__ void k_f32_to_f16(const float * x, half * y, size_t n) {
    const size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

int main(int argc, char ** argv) {
    const int n_pool = argc > 1 ? atoi(argv[1]) : 8192;
    const int rows   = argc > 2 ? atoi(argv[2]) : 8192;
    const int k      = argc > 3 ? atoi(argv[3]) : 512;
    const int iters  = argc > 4 ? atoi(argv[4]) : 3;
    const int mode   = argc > 5 ? atoi(argv[5]) : 0;
    const bool f16acc = argc > 6 && atoi(argv[6]) != 0;
    const int D = dsa_idx::D, NH = dsa_idx::NH;
    const int sentinel = n_pool;
    printf("n_pool %d rows %d k %d iters %d mode %d f16acc %d\n", n_pool, rows, k, iters, mode, (int) f16acc);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    g_info.devices[0].cc = 100*prop.major + 10*prop.minor;
    printf("device %s cc %d, %d SMs\n", prop.name, g_info.devices[0].cc, prop.multiProcessorCount);

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> h_k((size_t) n_pool*D), h_q((size_t) rows*NH*D), h_w((size_t) rows*NH);
    for (auto & v : h_k) v = 0.25f*nd(rng);
    for (auto & v : h_q) v = 0.5f*nd(rng);
    for (auto & v : h_w) v = 0.5f + 0.5f*nd(rng);
    if (mode == 1) {
        // every pool is one of 64 prototypes: many exact ties
        for (int p = 64; p < n_pool; ++p) std::copy(h_k.begin() + (size_t) (p % 64)*D, h_k.begin() + (size_t) (p % 64 + 1)*D, h_k.begin() + (size_t) p*D);
    }
    if (mode == 2) {
        for (int r = 0; r < rows; r += 2) std::fill(h_w.begin() + (size_t) r*NH, h_w.begin() + (size_t) (r + 1)*NH, 0.0f);
    }
    // visibility like the last ubatch of a context of n_pool*4 cells: query r sits at position depth + r
    const int depth = std::max(0, n_pool*4 - rows);
    std::vector<int> h_nvis(rows), h_ok(n_pool, 1), h_lo(rows, 0), h_nvis_old(rows);
    for (int r = 0; r < rows; ++r) h_nvis[r] = std::min(n_pool, (depth + r + 1)/4);
    if (mode == 3) {
        // sliding window: query r sees [lo, lo + n_vis) with lo growing by 1 pool per 3 queries (not 4-aligned)
        for (int r = 0; r < rows; ++r) { h_lo[r] = std::min(n_pool - 1, r/3 + 5); h_nvis[r] = std::min(n_pool - h_lo[r], (depth + r + 1)/4); }
    }
    for (int r = 0; r < rows; ++r) h_nvis_old[r] = h_lo[r] + h_nvis[r];   // the old path has no window: it scores [0, lo + n_vis)
    h_ok[n_pool - 1] = 0;                        // the incomplete last pool
    for (int p = 7; p < n_pool; p += 997) h_ok[p] = 0;

    half * d_k; float * d_q; half * d_q16; float * d_w; int * d_nvis; int * d_nvis_old; int * d_lo; int * d_ok; int * d_old; int * d_new;
    CUDA_CHECK(cudaMalloc(&d_k, h_k.size()*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_q, h_q.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q16, h_q.size()*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w, h_w.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nvis, rows*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nvis_old, rows*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lo, rows*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ok, n_pool*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_old, (size_t) rows*k*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_new, (size_t) rows*k*sizeof(int)));
    {
        std::vector<half> hk(h_k.size());
        for (size_t i = 0; i < h_k.size(); ++i) hk[i] = __float2half(h_k[i]);
        CUDA_CHECK(cudaMemcpy(d_k, hk.data(), hk.size()*sizeof(half), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), h_q.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), h_w.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nvis, h_nvis.data(), rows*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nvis_old, h_nvis_old.data(), rows*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lo, h_lo.data(), rows*sizeof(int), cudaMemcpyHostToDevice));
    const int * d_lo_arg = mode == 3 ? d_lo : nullptr;
    CUDA_CHECK(cudaMemcpy(d_ok, h_ok.data(), n_pool*sizeof(int), cudaMemcpyHostToDevice));
    k_f32_to_f16<<<(unsigned) ((h_q.size() + 255)/256), 256>>>(d_q, d_q16, h_q.size());
    CUDA_CHECK(cudaGetLastError());

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetStream(cublas, stream));
    simple_pool pool;
    cudaEvent_t e0, e1, e2;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1)); CUDA_CHECK(cudaEventCreate(&e2));

    // ---- old path, chunked exactly like ggml_cuda_op_indexer_topk ----
    std::vector<float> score_old((size_t) n_pool*rows);
    float ms_old = 0.0f;
    {
        constexpr size_t k_max_buf_size = 1 << 28;
        const size_t per_row = size_t(n_pool)*(NH*sizeof(half) + sizeof(int) + sizeof(float)) + D*NH*sizeof(half);
        int max_rows = (int) ((k_max_buf_size + per_row - 1)/per_row);
        max_rows = std::min(max_rows, rows);
        const int nstep = (rows + max_rows - 1)/max_rows;
        printf("old path: %d rows per step, %d steps\n", max_rows, nstep);
        float * kq; float * score; int * sorted;
        CUDA_CHECK(cudaMalloc(&kq, (size_t) n_pool*NH*max_rows*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&score, (size_t) n_pool*max_rows*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&sorted, (size_t) n_pool*max_rows*sizeof(int)));
        const float alpha = 1.0f, beta = 0.0f;
        for (int it = 0; it < iters; ++it) {
            CUDA_CHECK(cudaEventRecord(e0, stream));
            for (int istep = 0; istep < nstep; ++istep) {
                const int first_row = max_rows*istep;
                const int nr = std::min(max_rows, rows - first_row);
                CUBLAS_CHECK(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, n_pool, NH*nr, D,
                        &alpha, d_k, CUDA_R_16F, D, d_q16 + (size_t) first_row*NH*D, CUDA_R_16F, D,
                        &beta, kq, CUDA_R_32F, n_pool, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                const dim3 grid(nr, (n_pool + 255)/256, 1);
                k_fused_relu_mul_sum_rows_nvis<float><<<grid, 256, 0, stream>>>(kq, d_w + (size_t) first_row*NH, d_nvis_old + first_row, d_ok, score, n_pool, NH);
                CUDA_CHECK(cudaGetLastError());
                if (it == 0) {
                    CUDA_CHECK(cudaMemcpyAsync(score_old.data() + (size_t) first_row*n_pool, score, (size_t) nr*n_pool*sizeof(float), cudaMemcpyDeviceToHost, stream));
                }
                argsort_f32_i32_cuda_cub(pool, score, sorted, n_pool, nr, GGML_SORT_ORDER_DESC, stream);
                CUDA_CHECK(cudaGetLastError());
                k_copy_topk_sentinel<<<nr, 256, 0, stream>>>(sorted, score, d_old + (size_t) first_row*k, n_pool, k, sentinel);
                CUDA_CHECK(cudaGetLastError());
            }
            CUDA_CHECK(cudaEventRecord(e1, stream));
            CUDA_CHECK(cudaEventSynchronize(e1));
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            if (it == 0 || ms < ms_old) ms_old = ms;
        }
        CUDA_CHECK(cudaFree(kq)); CUDA_CHECK(cudaFree(score)); CUDA_CHECK(cudaFree(sorted));
    }

    // ---- fused path ----
    std::vector<float> score_new((size_t) n_pool*rows, -INFINITY);
    float ms_new = 0.0f, ms_score = 0.0f, ms_select = 0.0f;
    {
        const int rows_chunk = dsa_idx::rows_per_chunk(n_pool, rows, (size_t) 256 << 20);
        printf("fused path: %d rows per chunk, %d chunks\n", rows_chunk, (rows + rows_chunk - 1)/rows_chunk);
        float * score;
        CUDA_CHECK(cudaMalloc(&score, (size_t) n_pool*rows_chunk*sizeof(float)));
        for (int it = 0; it < iters; ++it) {
            float a = 0.0f, b = 0.0f;
            CUDA_CHECK(cudaEventRecord(e0, stream));
            dsa_idx::topk_fused(d_k, n_pool, d_q16, d_w, d_nvis, d_lo_arg, d_ok, d_new, rows, k, sentinel, score, rows_chunk, f16acc, 0, stream, &a, &b);
            CUDA_CHECK(cudaEventRecord(e1, stream));
            CUDA_CHECK(cudaEventSynchronize(e1));
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            if (it == 0 || ms < ms_new) { ms_new = ms; ms_score = a; ms_select = b; }
        }
        // the score of the last chunk only is still in the buffer: recompute chunk by chunk for the comparison
        for (int r0 = 0; r0 < rows; r0 += rows_chunk) {
            const int nr = std::min(rows_chunk, rows - r0);
            CUDA_CHECK(cudaMemsetAsync(score, 0xff, (size_t) n_pool*nr*sizeof(float), stream));   // NaN pattern: "not written"
            const dim3 grid((nr + dsa_idx::QT - 1)/dsa_idx::QT, (n_pool + dsa_idx::SLAB - 1)/dsa_idx::SLAB, 1);
            if (f16acc) dsa_idx::k_dsa_score<true><<<grid, dsa_idx::NTHREADS, dsa_idx::SMEM_SCORE, stream>>>(d_k, d_q16 + (size_t) r0*NH*D, d_w + (size_t) r0*NH, d_nvis + r0, d_lo_arg ? d_lo_arg + r0 : nullptr, d_ok, score, n_pool, nr);
            else        dsa_idx::k_dsa_score<false><<<grid, dsa_idx::NTHREADS, dsa_idx::SMEM_SCORE, stream>>>(d_k, d_q16 + (size_t) r0*NH*D, d_w + (size_t) r0*NH, d_nvis + r0, d_lo_arg ? d_lo_arg + r0 : nullptr, d_ok, score, n_pool, nr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(score_new.data() + (size_t) r0*n_pool, score, (size_t) nr*n_pool*sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(score));
    }

    // ---- compare scores on the visible pools ----
    size_t n_cmp = 0, n_diff = 0, n_unwritten = 0;
    double max_abs = 0, max_rel = 0;
    for (int r = 0; r < rows; ++r) {
        for (int p = h_lo[r]; p < h_lo[r] + h_nvis[r]; ++p) {
            const float a = score_old[(size_t) r*n_pool + p], b = score_new[(size_t) r*n_pool + p];
            ++n_cmp;
            if (std::isnan(b)) { ++n_unwritten; continue; }
            if (a != b) {
                ++n_diff;
                const double d = fabs((double) a - b);
                max_abs = std::max(max_abs, d);
                max_rel = std::max(max_rel, d/std::max(1e-30, (double) fabs(a)));
            }
        }
    }
    printf("scores: %zu visible compared, %zu unwritten, %zu differ (max abs %.3g, max rel %.3g)\n", n_cmp, n_unwritten, n_diff, max_abs, max_rel);

    // ---- compare the index lists ----
    std::vector<int> h_old((size_t) rows*k), h_new((size_t) rows*k), h_ref((size_t) rows*k);
    CUDA_CHECK(cudaMemcpy(h_old.data(), d_old, h_old.size()*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_new.data(), d_new, h_new.size()*sizeof(int), cudaMemcpyDeviceToHost));
    // host reference from the old scores: top k of the window [lo, lo + n_vis) by (score desc, pool asc), -inf excluded
    {
        std::vector<std::pair<float, int>> cand;
        for (int r = 0; r < rows; ++r) {
            cand.clear();
            for (int p = h_lo[r]; p < h_lo[r] + h_nvis[r]; ++p) {
                const float s = score_old[(size_t) r*n_pool + p];
                if (s != -INFINITY) cand.push_back({s, p});
            }
            const size_t kk = std::min<size_t>(k, cand.size());
            std::partial_sort(cand.begin(), cand.begin() + kk, cand.end(), [](const std::pair<float, int> & a, const std::pair<float, int> & b) {
                return a.first > b.first || (a.first == b.first && a.second < b.second); });
            for (int i = 0; i < k; ++i) h_ref[(size_t) r*k + i] = (size_t) i < kk ? cand[i].second : sentinel;
        }
        int ref_old = 0, ref_new = 0;
        for (int r = 0; r < rows; ++r) {
            if (!std::equal(h_ref.begin() + (size_t) r*k, h_ref.begin() + (size_t) (r + 1)*k, h_old.begin() + (size_t) r*k)) ++ref_old;
            if (!std::equal(h_ref.begin() + (size_t) r*k, h_ref.begin() + (size_t) (r + 1)*k, h_new.begin() + (size_t) r*k)) ++ref_new;
        }
        printf("host reference (old scores, window, score desc / pool asc): old path differs in %d rows, fused in %d rows\n", ref_old, ref_new);
        if (mode == 3) h_old = h_ref;   // the old GPU path cannot express the window
    }
    int rows_seq_diff = 0, rows_set_diff = 0, rows_tie_only = 0, shown = 0;
    for (int r = 0; r < rows; ++r) {
        const int * a = h_old.data() + (size_t) r*k;
        const int * b = h_new.data() + (size_t) r*k;
        if (std::equal(a, a + k, b)) continue;
        ++rows_seq_diff;
        std::set<int> sa(a, a + k), sb(b, b + k);
        if (sa == sb) {
            // same set, different order: must be an order change among equal scores
            bool ok = true;
            for (int i = 0; i < k; ++i) {
                if (a[i] != b[i] && (a[i] == sentinel || b[i] == sentinel || score_old[(size_t) r*n_pool + a[i]] != score_old[(size_t) r*n_pool + b[i]])) ok = false;
            }
            if (ok) { ++rows_tie_only; continue; }
        }
        ++rows_set_diff;
        if (shown < 5) {
            ++shown;
            printf("row %d (n_vis %d): ", r, h_nvis[r]);
            std::vector<int> only_a, only_b;
            std::set_difference(sa.begin(), sa.end(), sb.begin(), sb.end(), std::back_inserter(only_a));
            std::set_difference(sb.begin(), sb.end(), sa.begin(), sa.end(), std::back_inserter(only_b));
            printf("%zu only old, %zu only new;", only_a.size(), only_b.size());
            for (size_t i = 0; i < std::min<size_t>(3, only_a.size()); ++i) printf(" old %d (%.8g/%.8g)", only_a[i], only_a[i] < n_pool ? score_old[(size_t) r*n_pool + only_a[i]] : NAN, only_a[i] < n_pool ? score_new[(size_t) r*n_pool + only_a[i]] : NAN);
            for (size_t i = 0; i < std::min<size_t>(3, only_b.size()); ++i) printf(" new %d (%.8g/%.8g)", only_b[i], only_b[i] < n_pool ? score_old[(size_t) r*n_pool + only_b[i]] : NAN, only_b[i] < n_pool ? score_new[(size_t) r*n_pool + only_b[i]] : NAN);
            printf("\n");
        }
    }
    printf("index lists: %d of %d rows differ in sequence, %d of those only reorder equal scores, %d differ in set\n",
            rows_seq_diff, rows, rows_tie_only, rows_set_diff);
    // sanity of the new lists: descending scores, no duplicates, sentinel only after the last visible
    int bad_rows = 0;
    for (int r = 0; r < rows; ++r) {
        const int * b = h_new.data() + (size_t) r*k;
        bool bad = false;
        std::set<int> seen;
        float prev = INFINITY;
        for (int i = 0; i < k; ++i) {
            if (b[i] == sentinel) { for (int j = i; j < k; ++j) if (b[j] != sentinel) bad = true; break; }
            if (b[i] < h_lo[r] || b[i] >= h_lo[r] + h_nvis[r] || !h_ok[b[i]] || !seen.insert(b[i]).second) { bad = true; break; }
            const float s = score_old[(size_t) r*n_pool + b[i]];
            if (s > prev) bad = true;
            prev = s;
        }
        if (bad) { ++bad_rows; if (bad_rows <= 3) printf("new list row %d malformed\n", r); }
    }
    printf("new lists malformed: %d rows\n", bad_rows);

    const double flops = 2.0*n_pool*(double) rows*NH*D;
    printf("time: old %.2f ms, fused %.2f ms (score %.2f ms = %.1f TFLOPS, select %.2f ms), speedup %.2fx\n",
            ms_old, ms_new, ms_score, flops/(ms_score*1e-3)/1e12, ms_select, ms_old/ms_new);
    return (n_unwritten == 0 && rows_set_diff == 0 && bad_rows == 0) ? 0 : 1;
}
