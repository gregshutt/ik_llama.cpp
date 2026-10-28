#include "../llama-build-context.h"
#include "../llama-context.h"
#include "../llama-delta-net.h"
#include "../llama-model.h"
#include "../llama-spec-features-dflash.h"
#include "../llama-impl.h"

#include <cinttypes>

// Score-chunk size for the k-pool indexer: bounds the [n_pool, n_head, Tc] score tensor so it
// stays under the int32 element-count limit of the RELU kernel and the GPU compute buffer.
// Mirrors OPENPANGU_IDX_SCORE_CHUNK.
static constexpr int64_t GLM5NEXT_IDX_SCORE_CHUNK = 256;

// GLM-5.3-Flash (glm5next): hybrid trunk that alternates KDA linear attention (34 of 45
// layers) with absorbed MLA + DSA lightning k-pool indexer (11 layers), every block wrapped
// in hyper-connection streams (mHC, Sinkhorn). NoPE-only (rope.dimension_count = 0).
// The NextN/MTP block is wired through the GLM5NEXT MTP graph.

// mHC pre (copied from build_deepseek4.cpp)

static ggml_tensor * glm5next_build_hc_pre(
        ggml_context * ctx0,
        llm_build_context & llm,
        const llama_hparams & hparams,
        int64_t n_embd,
        float norm_rms_eps,
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        ggml_tensor ** post_out,
        ggml_tensor ** comb_out,
        const llm_build_cb & cb, int il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx0, x);
    }
    auto flat   = ggml_reshape_2d(ctx0, x, n_embd * hc, nt);
    auto normed = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    cb(normed, "hc_pre", il);
    auto mixes = ggml_mul_mat(ctx0, hc_fn, normed);
    cb(mixes, "hc_pre_mixes", il);

    auto all = ggml_hc_pre(ctx0, mixes, hc_scale, hc_base, hc, hparams.dsv4_hc_sinkhorn_iters, hparams.dsv4_hc_eps);

    auto pre  = ggml_view_2d(ctx0, all, hc, nt, hc * sizeof(float), 0);
    auto post = ggml_view_2d(ctx0, all, hc, nt, hc * sizeof(float), hc * nt * sizeof(float));
    auto comb = ggml_view_3d(ctx0, all, hc, hc, nt, hc * sizeof(float), hc * hc * sizeof(float), 2 * hc * nt * sizeof(float));

    *post_out = post;
    *comb_out = comb;

    return llm.build_mhc_weighted_sum(x, pre, n_embd, hc);
}

// Host-filled k-pool DSA inputs shared by the MLA layers of one graph (filled in llama_set_inputs)
struct glm5next_kpool_inputs {
    ggml_tensor * cells     = nullptr; // I32 [kpool*n_pool (+kpool: sentinel row in direct mode)]
    ggml_tensor * pools     = nullptr; // the same as [kpool, n_pool(+1)]: one shared view, because every view of a
                                       // host input counts as a new cross-backend input of the split that uses it
    ggml_tensor * bias      = nullptr; // F32 [n_pool, n_tokens]      masked path: 0 if pool visible, else -inf
    ggml_tensor * tail      = nullptr; // I32 [n_tail, n_tokens]      trailing incomplete pool cells (+ -1 padding in direct mode)
    ggml_tensor * ape_slots = nullptr; // I32 [kpool]                 identity [0..kpool-1]
    ggml_tensor * nvis      = nullptr; // I32 [n_tokens]              direct path: leading pools visible to each query
    ggml_tensor * ok        = nullptr; // I32 [n_pool]                direct path: 1 if the pool is complete
    ggml_tensor * lo        = nullptr; // I32 [n_tokens]              direct path: first column visible to each query (its sequence's pools)
    ggml_tensor * row       = nullptr; // I32 [n_pool]                kp_l row of each column (the pools of every sequence)
    ggml_tensor * win_pools = nullptr; // I32 [n_win]                 kp_l window: pools re-pooled by this graph
    ggml_tensor * win_cells = nullptr; // I32 [kpool*n_win]
    bool    direct = false;            // maskless index-list selection (CUDA sparse kernels only)
    int64_t n_kv   = 0;                // cache view the pool tables cover (padded last used cell)
    int64_t n_pool = 0;                // score columns: max(n_kv / kpool, the pools of every sequence)
    int64_t n_sel  = 0;                // pools picked per query
    int64_t n_idx  = 0;                // index-list length in direct mode (multiple of 256)
};

// The direct path hands FLASH_ATTN_EXT an index list and no mask, so it may only be used when one of the
// CUDA sparse kernels will accept the op (ggml_cuda_flash_attn_ext_mma_new_sparse / ggml_cuda_dsa_attn_ext);
// this mirrors their conditions. LLAMA_DSA_MASK=1 forces the masked path (also for non-CUDA backends).
static bool glm5next_dsa_direct_ok(const llama_context & lctx, int64_t n_kv, int64_t n_tokens, int64_t n_idx) {
    static const bool force_mask = getenv("LLAMA_DSA_MASK") != nullptr;
    if (force_mask || !lctx.cparams.flash_attn) {
        return false;
    }
    static const bool force_direct = getenv("LLAMA_DSA_FORCE_DIRECT") != nullptr;   // testing: sparse kernels below the threshold
    if (n_kv <= n_idx && !force_direct) {
        return false;          // every key fits the list: dense attention over the few keys
    }
    const ggml_type tk = lctx.kv_self.type_k;
    if (tk != GGML_TYPE_F16 && tk != GGML_TYPE_Q8_0) {
        return false;
    }
    const bool quant   = tk != GGML_TYPE_F16;
    const bool mma_ok  = n_tokens >= 8 && (!quant || 2*n_tokens*n_idx >= n_kv);
    const bool gath_ok = n_tokens <= 16 ? n_idx < n_kv : n_kv >= 4*n_idx;
    return mma_ok || gath_ok;
}

// Allocates the host-filled inputs for one graph (null when DSA is off or the cache has no pool yet).
static void glm5next_kpool_setup(llm_build_context & llm, glm5next_kpool_inputs & kp, bool allow_direct) {
    auto & lctx    = llm.lctx;
    auto & ctx0    = llm.ctx0;
    auto & hparams = llm.hparams;
    auto & cb      = llm.cb;

    const bool use_dsa = lctx.cparams.dsa
        && hparams.indexer_head_size > 0
        && hparams.indexer_block_size > 0
        && !lctx.kv_self.kr_l.empty();
    if (!use_dsa) {
        return;
    }
    const int64_t r        = hparams.indexer_block_size;
    const int64_t n_tokens = llm.n_tokens;
    // the pool tables follow this ubatch's plan (llama_kpool_plan_update): the K view covers the used cells
    // (this hybrid model keeps kv_self.n at the full cache size, so the pool tables, the indexer scoring and
    // the K view scale with the context in use, not with -c) and the columns are the pools of every sequence
    int64_t n_kv   = llm.n_kv;
    int64_t n_pool = n_kv / r;
    if (!llm.worst_case) {
        auto & plan = lctx.kpool_plan;
        if (!plan.valid || plan.kv_head != llm.kv_head || plan.n_tokens != n_tokens) {
            llama_kpool_plan_update(lctx, llm.batch);
        }
        GGML_ASSERT(plan.valid && "glm5next kpool: no pool plan for this ubatch");
        GGML_ASSERT(plan.n_kv <= llm.n_kv);
        n_kv   = plan.n_kv;
        n_pool = plan.n_pool;
    }
    kp.n_kv   = n_kv;
    kp.n_pool = n_pool;
    lctx.kpool_n_kv   = n_kv;
    lctx.kpool_n_pool = n_pool;
    if (n_pool <= 0) {
        return;
    }
    const int64_t tail_cnt = r > 1 ? r - 1 : 0;
    const int64_t top_k    = lctx.cparams.dsa_top_k > 0 ? lctx.cparams.dsa_top_k : hparams.indexer_top_k;
    // direct mode: the index list is padded, so it need not fit the cache; every visible pool can be picked
    // while n_kv <= top_k (sparse attention then equals dense attention). Masked mode keeps the stock bound
    // (kpool*n_sel + tail cells <= n_kv so the cell list fits the mask).
    const int64_t n_sel_direct = std::min<int64_t>(n_pool, top_k / r);
    const int64_t n_idx_direct = GGML_PAD(r * n_sel_direct + tail_cnt, 256);
    kp.direct = allow_direct && n_sel_direct >= 1 && !lctx.kv_self.kp_l.empty() &&
                glm5next_dsa_direct_ok(lctx, n_kv, n_tokens, n_idx_direct);
    kp.n_sel  = kp.direct ? n_sel_direct : std::min<int64_t>({n_pool, top_k / r, (n_kv - tail_cnt) / r});
    kp.n_idx  = kp.direct ? n_idx_direct : 0;

    static int last_mode = -1;
    static const bool dbg = getenv("LLAMA_DSA_DEBUG") != nullptr;
    const int mode = kp.direct ? 2 : 1;
    if (mode != last_mode || dbg) {
        LLAMA_LOG_INFO("%s: glm5next DSA selection path: %s (n_kv %" PRId64 ", n_pool %" PRId64 " holding %" PRId64 " pools, n_tokens %" PRId64 ", n_sel %" PRId64 ", n_idx %" PRId64 ", re-pool %d)\n",
                __func__, kp.direct ? "direct index list, no mask" : "masked", n_kv, n_pool,
                llm.worst_case ? n_pool : lctx.kpool_plan.n_seq_pools, n_tokens, kp.n_sel, kp.n_idx,
                llm.worst_case ? 0 : (int) lctx.kpool_plan.repool_rows.size());
        last_mode = mode;
    }

    kp.cells     = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, r * (n_pool + (kp.direct ? 1 : 0)));
    kp.ape_slots = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, r);
    ggml_set_input(kp.cells);
    ggml_set_input(kp.ape_slots);
    kp.pools     = ggml_reshape_2d(ctx0, kp.cells, r, n_pool + (kp.direct ? 1 : 0));
    cb(kp.cells,     "inp_kpool_cells", -1);
    cb(kp.ape_slots, "inp_kpool_ape",   -1);
    if (kp.direct) {
        kp.nvis = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        kp.lo   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        kp.ok   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pool);
        kp.tail = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kp.n_idx - r * kp.n_sel, n_tokens);
        ggml_set_input(kp.nvis);
        ggml_set_input(kp.lo);
        ggml_set_input(kp.ok);
        ggml_set_input(kp.tail);
        cb(kp.nvis, "inp_kpool_nvis", -1);
        cb(kp.lo,   "inp_kpool_lo",   -1);
        cb(kp.ok,   "inp_kpool_ok",   -1);
        cb(kp.tail, "inp_kpool_tail", -1);
    } else {
        kp.bias = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_pool, n_tokens);
        ggml_set_input(kp.bias);
        cb(kp.bias, "inp_kpool_bias", -1);
        if (r > 1) {
            kp.tail = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r - 1, n_tokens);
            ggml_set_input(kp.tail);
            cb(kp.tail, "inp_kpool_tail", -1);
        }
    }
    if (allow_direct && !lctx.kv_self.kp_l.empty()) {
        // a run of n consecutive positions spans n/r pools plus one when it straddles a pool boundary, and an
        // ubatch holds at most one run per sequence; a stale cache (state restore, defrag) needs every pool
        const int64_t n_win = lctx.qsa_pooled_stale ? n_pool
            : std::min<int64_t>(n_pool, (n_tokens + r - 1)/r + std::max<int64_t>(1, lctx.cparams.n_seq_max) + 1);
        kp.win_pools = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_win);
        kp.win_cells = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, r * n_win);
        kp.row       = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pool);
        ggml_set_input(kp.win_pools);
        ggml_set_input(kp.win_cells);
        ggml_set_input(kp.row);
        cb(kp.win_pools, "inp_kpool_win_pools", -1);
        cb(kp.win_cells, "inp_kpool_win_cells", -1);
        cb(kp.row,       "inp_kpool_row",       -1);
    }

    lctx.kpool_direct        = kp.direct;
    lctx.inp_kpool_cells     = kp.cells;
    lctx.inp_kpool_bias      = kp.bias;
    lctx.inp_kpool_tail      = kp.tail;
    lctx.inp_kpool_ape_slots = kp.ape_slots;
    lctx.inp_kpool_nvis      = kp.nvis;
    lctx.inp_kpool_ok        = kp.ok;
    lctx.inp_kpool_win_pools = kp.win_pools;
    lctx.inp_kpool_win_cells = kp.win_cells;
    lctx.inp_kpool_lo        = kp.lo;
    lctx.inp_kpool_row       = kp.row;
}

// DSA k-pool indexer: writes this ubatch's indexer [key; gate] rows to kr_l, (re)pools the pools this
// ubatch touches into kp_l (or every pool from kr_l when there is no pooled cache), scores the pools
// (a pool key is a learned convex mix of its member keys; score = sum_h w_h relu(q_h . k_pool)) and
// picks the top n_sel pools per query. Returns
//   direct mode: I32 [n_idx, n_tokens] KV cell list for FLASH_ATTN_EXT src[5] (-1 = no key, holes allowed);
//   masked mode: I32 [kpool*n_sel + kpool-1, n_tokens] cell list for ggml_indexer_mask;
//   nullptr: dense fallback (the cache writes above still happen).
static ggml_tensor * build_glm5next_dsa_select(
        llm_build_context & llm,
        ggml_cgraph * gf, int il,
        ggml_tensor * cur,            // [n_embd, n_tokens] — pre-norm input to the attention block
        ggml_tensor * qr,             // [q_lora_rank, n_tokens] — q_lora latent (post wq_a + norm)
        const glm5next_kpool_inputs & kp) {
    auto & lctx    = llm.lctx;
    auto & ctx0    = llm.ctx0;
    auto & hparams = llm.hparams;
    auto & model   = llm.model;
    auto & kv_self = lctx.kv_self;
    auto & cb      = llm.cb;
    auto & layer   = model.layers[il];

    const int64_t d      = hparams.indexer_head_size;   // 128
    const int64_t nh     = hparams.indexer_n_head;       // 32
    const int64_t r      = hparams.indexer_block_size;   // 4 (kpool)
    const int64_t n_kv   = kp.n_kv;                      // view covering the used cells (see glm5next_kpool_setup)
    const int64_t n_tok  = llm.n_tokens;
    const int64_t n_pool = kp.n_pool;                    // the pools of every sequence (llama_kpool_plan_update)

    if (n_pool == 0) {
        return nullptr;
    }

    // ---- indexer key + gate (per token) ----
    ggml_tensor * k = ggml_mul_mat(ctx0, layer.indexer_attn_k, cur);   // [d, n_tok]
    k = llm_build_context::llm_build_norm(ctx0, k, hparams, layer.indexer_k_norm, layer.indexer_k_norm_b, LLM_NORM, cb, il);
    cb(k, "dsa_indexer_k", il);

    ggml_tensor * g = ggml_mul_mat(ctx0, layer.indexer_comp_wgate, cur);  // [d, n_tok]
    cb(g, "dsa_indexer_gate", il);

    // pack [key; gate] into one cache row: [2*d, n_tok]
    ggml_tensor * packed = ggml_reshape_2d(ctx0, ggml_concat(ctx0, k, g, 0), 2 * d, n_tok);
    cb(packed, "dsa_indexer_packed", il);

    // write to kr_l at kv_head (key+gate packed, row = 2*d)
    const auto kr_row = ggml_row_size(kv_self.kr_l[il]->type, 2 * d);
    auto kr_view = ggml_view_2d(ctx0, kv_self.kr_l[il], 2 * d, n_tok, kr_row, kr_row * llm.kv_head);
    auto kr_cpy = ggml_cpy(ctx0, packed, kr_view);
    if ((size_t) il < lctx.dsa_cache_copies.size()) {
        lctx.dsa_cache_copies[il].cpy  = kr_cpy;
        lctx.dsa_cache_copies[il].step = kv_self.kr_l[il]->nb[1];
    }
    ggml_build_forward_expand(gf, kr_cpy);

    // ---- read back the full cached key+gate set: [2*d, n_kv] ----
    auto all = ggml_view_2d(ctx0, kv_self.kr_l[il], 2 * d, n_kv, kr_row, 0);
    cb(all, "dsa_cached_all", il);

    // with the pooled-key cache only the pools this ubatch writes into are pooled; the rest are read
    // back from kp_l. Without it every pool is pooled from kr_l on every graph.
    ggml_tensor * kp_cache = (size_t) il < kv_self.kp_l.size() ? kv_self.kp_l[il] : nullptr;
    const bool use_kp_cache = kp_cache && kp.win_pools && kp.win_cells;
    GGML_ASSERT(!kp.direct || use_kp_cache);
    ggml_tensor * pool_idx  = use_kp_cache ? kp.win_cells : kp.cells;
    const int64_t n_pooled  = use_kp_cache ? kp.win_pools->ne[0] : n_pool;
    if (!use_kp_cache && kp.direct) {
        GGML_ABORT("glm5next direct DSA path needs the pooled-key cache");
    }
    GGML_ASSERT(use_kp_cache || kp.cells->ne[0] == r * n_pool);

    // ---- gather pool members: pool_idx indexes which cells belong to each pool ----
    // membership depends on cache positions, shared across all query tokens
    auto members = ggml_get_rows(ctx0, all, pool_idx);       // [2*d, r*n_pooled]
    members = ggml_reshape_3d(ctx0, members, 2 * d, r, n_pooled);
    cb(members, "dsa_pool_members", il);

    // split into keys [d, r, n_pooled] and gates [d, r, n_pooled] (packed along dim 0)
    auto m_k = ggml_cont(ctx0, ggml_view_3d(ctx0, members, d, r, n_pooled,
            members->nb[1], members->nb[2], 0));
    auto m_g = ggml_cont(ctx0, ggml_view_3d(ctx0, members, d, r, n_pooled,
            members->nb[1], members->nb[2], ggml_row_size(members->type, d)));
    cb(m_k, "dsa_pool_k", il);
    cb(m_g, "dsa_pool_g", il);

    // pool key = softmax(gate + ape) . keys over the r members, channel by channel.
    // ape is an intra-pool position bias
    m_g = ggml_add(ctx0, m_g, ggml_get_rows(ctx0, layer.indexer_comp_ape, kp.ape_slots));

    // softmax normalizes ne[0], so bring the member axis there
    auto w = ggml_soft_max(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, m_g, 1, 0, 2, 3)));
    auto v = ggml_cont(ctx0, ggml_permute(ctx0, m_k, 1, 0, 2, 3));
    auto pooled = ggml_sum_rows(ctx0, ggml_mul(ctx0, v, w));
    pooled = ggml_cont(ctx0, ggml_permute(ctx0, pooled, 1, 0, 2, 3));
    pooled = ggml_reshape_2d(ctx0, pooled, d, n_pooled);
    cb(pooled, "dsa_indexer_k_pooled", il);

    if (use_kp_cache) {
        // score against the scatter's result, not the cache, so the read depends on the write. The rows are
        // keyed by cell (llama_kpool_plan_update), so the columns are gathered through kp.row: inside the fused
        // op on the direct path, with get_rows on the masked one
        GGML_ASSERT(kp.row);
        auto kp_all = ggml_set_rows(ctx0, kp_cache, pooled, kp.win_pools);
        cb(kp_all, "dsa_indexer_k_scatter", il);
        ggml_build_forward_expand(gf, kp_all);   // the dense fallback below returns before scoring
        if (kp.direct) {
            pooled = ggml_view_2d(ctx0, kp_all, d, kp_cache->ne[1], ggml_row_size(kp_cache->type, d), 0);
        } else {
            pooled = ggml_get_rows(ctx0, kp_all, kp.row);                   // f32 [d, n_pool]
        }
        cb(pooled, "dsa_indexer_k_cached", il);
    }

    const int64_t n_sel = kp.n_sel;
    if (n_sel < 1) {
        return nullptr;  // cache too small for the indexer — dense attention over the few keys
    }

    // ---- indexer query (nope-only: no rope) ----
    auto q = ggml_mul_mat(ctx0, layer.indexer_attn_q_b, qr);  // [d*nh, n_tok]
    q = ggml_reshape_3d(ctx0, q, d, nh, n_tok);
    cb(q, "dsa_indexer_q", il);

    // relu(x*s) == s*relu(x) for s > 0, so both positive scalars fold into the weights
    auto wts = ggml_mul_mat(ctx0, layer.indexer_proj, cur);  // [nh, n_tok]
    wts = ggml_scale(ctx0, wts, 1.0f / sqrtf(float(d * nh)));
    cb(wts, "dsa_indexer_weights", il);

    if (kp.direct) {
        // fused score + top-k over the visible pools: sorted by score, unfilled slots name the sentinel
        // pool (n_pool), whose cells are -1
        auto sel = ggml_indexer_topk_nvis(ctx0, pooled, q, wts, kp.nvis, kp.ok, kp.lo, kp.row, GGML_UNARY_OP_RELU,
                (int) n_sel, (int) n_pool);                            // [n_sel, n_tok]
        cb(sel, "dsa_top_k_pools", il);

        auto cells = ggml_get_rows(ctx0, kp.pools,                      // [r, n_pool + 1]
                ggml_reshape_2d(ctx0, sel, n_sel * n_tok, 1));         // [r, n_sel*n_tok]
        cells = ggml_reshape_2d(ctx0, cells, r * n_sel, n_tok);        // [r*n_sel, n_tok]

        // the trailing incomplete pool (index_kpool_always_select_tail) plus -1 padding to n_idx
        auto idx = ggml_concat(ctx0, cells, kp.tail, 0);               // [n_idx, n_tok]
        GGML_ASSERT(idx->ne[0] == kp.n_idx);
        cb(idx, "dsa_kv_idx", il);
        return idx;
    }

    // ---- masked path: score with plain ops, chunked over the query dim ----
    wts = ggml_reshape_3d(ctx0, wts, nh, 1, n_tok);
    const bool chunk_scores = GLM5NEXT_IDX_SCORE_CHUNK > 0 && n_tok > GLM5NEXT_IDX_SCORE_CHUNK;
    ggml_tensor * sel = nullptr;
    if (!chunk_scores) {
        // small batch (decode or short prefill): no chunking needed
        auto score = ggml_mul_mat(ctx0, pooled, q);                  // [n_pool, nh, n_tok]
        score = ggml_relu(ctx0, score);
        score = ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3));  // [nh, n_pool, n_tok]
        score = ggml_sum_rows(ctx0, ggml_mul(ctx0, score, wts));     // [1, n_pool, n_tok]
        score = ggml_reshape_2d(ctx0, score, n_pool, n_tok);         // [n_pool, n_tok]
        score = ggml_add(ctx0, score, kp.bias);
        cb(score, "dsa_indexer_score", il);
        sel = ggml_cont(ctx0, ggml_top_k(ctx0, score, n_sel));       // [n_sel, n_tok]
        cb(sel, "dsa_top_k_pools", il);
    } else {
        for (int64_t c0 = 0; c0 < n_tok; c0 += GLM5NEXT_IDX_SCORE_CHUNK) {
            const int64_t tc = std::min<int64_t>(GLM5NEXT_IDX_SCORE_CHUNK, n_tok - c0);
            auto q_c = ggml_cont(ctx0, ggml_view_3d(ctx0, q, d, nh, tc,
                    q->nb[1], q->nb[2], (size_t) c0 * q->nb[2]));
            auto wts_c = ggml_view_3d(ctx0, wts, nh, 1, tc,
                    wts->nb[1], wts->nb[2], (size_t) c0 * wts->nb[2]);
            auto pb_c = ggml_cont(ctx0, ggml_view_2d(ctx0, kp.bias, n_pool, tc,
                    kp.bias->nb[1], (size_t) c0 * kp.bias->nb[1]));

            auto sc = ggml_mul_mat(ctx0, pooled, q_c);               // [n_pool, nh, tc]
            sc = ggml_relu(ctx0, sc);
            sc = ggml_cont(ctx0, ggml_permute(ctx0, sc, 1, 0, 2, 3));    // [nh, n_pool, tc]
            sc = ggml_sum_rows(ctx0, ggml_mul(ctx0, sc, wts_c));     // [1, n_pool, tc]
            sc = ggml_reshape_2d(ctx0, sc, n_pool, tc);              // [n_pool, tc]
            sc = ggml_add(ctx0, sc, pb_c);

            auto sel_c = ggml_cont(ctx0, ggml_top_k(ctx0, sc, n_sel));   // [n_sel, tc]
            sel = sel == nullptr ? sel_c : ggml_concat(ctx0, sel, sel_c, 1);
        }
        cb(sel, "dsa_top_k_pools", il);
    }

    // ---- expand each selected pool into its r member cells ----
    auto top_k = ggml_get_rows(ctx0, kp.pools,                       // [r, n_pool]
            ggml_reshape_2d(ctx0, sel, n_sel * n_tok, 1));      // [r, n_sel*n_tok]
    top_k = ggml_reshape_2d(ctx0, top_k, r * n_sel, n_tok);     // [r*n_sel, n_tok]

    // index_kpool_always_select_tail: the trailing incomplete pool has no pool key and can
    // never be picked above, so its cells are appended instead of taking pool budget
    if (kp.tail) {
        top_k = ggml_concat(ctx0, top_k, kp.tail, 0);
    }
    cb(top_k, "dsa_top_k", il);

    return top_k;
}

// q = wq_b(wq_a(x))                        → [n_head * head_dim, n_tokens]
// Qcur = wk_b(permute(q))                  → [kv_lora_rank, n_tokens, n_head]  (absorbed)
// kv_cmpr = norm(wkv_a_mqa(x))             → [kv_lora_rank, n_tokens]
// store kv_cmpr in k_l, transposed in v_l
// Flash (-fa):  kqv_cmpr = flash_attn_ext(Qcur, K=V=K_cache, mask, scale)  (fused, no score matrix)
// Non-flash:    kq = K_cache @ Qcur → soft_max(kq, mask) → kqv_cmpr = V_cache_trans @ kq
// out = wo @ (wv_b @ kqv_cmpr)             → [n_embd, n_tokens]
//
// No rope, no q_rope/k_rope split. Returns attention output WITHOUT residual (mHC handles it).
// When the DSA k-pool indexer is active (--dsa + indexer tensors + kpool > 0), the flash path
// uses a sparse top-k mask instead of the dense KQ_mask.

static ggml_tensor * build_glm5next_mla_attention(
        llm_build_context & llm,
        ggml_cgraph * gf, int il,
        ggml_tensor * inp,             // [n_embd, n_tokens] — pre-norm
        ggml_tensor * KQ_mask,               // may be null when kp.direct (no layer needs it)
        float kq_scale,
        const glm5next_kpool_inputs & kp) {
    auto & lctx    = llm.lctx;
    auto & ctx0    = llm.ctx0;
    auto & hparams = llm.hparams;
    auto & model   = llm.model;
    auto & kv_self = lctx.kv_self;
    auto & cb      = llm.cb;
    auto & layer   = model.layers[il];

    const int64_t n_head       = llm.n_head;
    const int64_t head_dim     = hparams.n_embd_head_k_full;   // 256 (key_length_mla)
    const int64_t kv_lora_rank = hparams.n_lora_kv;            // 512
    const int64_t n_embd_head_v = hparams.n_embd_head_v_full;  // 256 (value_length_mla)

    // -sm graph: each device computes its head slice of the absorbed MLA attention and the row-split output
    // projection; the partial outputs are all-reduced. Mirrors build_deepseek2's absorb path without rope.
    if (model.split_mode == LLAMA_SPLIT_MODE_GRAPH && layer.wo->extra) {
        auto wo_split = (const ggml_split_tensor_t *)layer.wo->extra;
        const int n_device = wo_split->n_device;
        if (!lctx.cparams.flash_attn) {
            GGML_ABORT("-sm graph for glm5next requires -fa on");
        }
        if (lctx.cparams.dsa && kp.cells) {
            GGML_ABORT("-sm graph for glm5next does not support --dsa yet");
        }
        GGML_ASSERT(KQ_mask);
        auto cache_repl = (const ggml_split_tensor_t *)kv_self.k_l[il]->extra;
        GGML_ASSERT(cache_repl && cache_repl->n_device == n_device);
        auto sp = [](ggml_tensor * t, int id) {
            auto data = (const ggml_split_tensor_t *)t->extra;
            GGML_ASSERT(data && data->splits[id]);
            return data->splits[id];
        };
        std::vector<ggml_tensor *> partials(n_device, nullptr);
        for (int id = 0; id < n_device; ++id) {
            if (!wo_split->splits[id]) continue;
            const int il_id = 1000 * il + id;

            auto input  = llm_build_context::get_input_tensor_sm_graph(ctx0, inp, id);
            auto cur_id = llm_build_context::llm_build_norm(ctx0, input, hparams, sp(layer.attn_norm, id), nullptr, LLM_NORM_RMS, cb, il_id);

            auto qr_id = ggml_mul_mat(ctx0, sp(layer.wq_a, id), cur_id);
            ggml_build_forward_expand(gf, qr_id);
            qr_id = llm_build_context::llm_build_norm(ctx0, qr_id, hparams, sp(layer.attn_q_a_norm, id), nullptr, LLM_NORM_RMS, cb, il_id);
            auto q_id = ggml_mul_mat(ctx0, sp(layer.wq_b, id), qr_id);          // [n_head_local*head_dim, n_tokens]
            cb(q_id, "q", il_id);
            const int64_t n_head_local = q_id->ne[0] / head_dim;
            q_id = ggml_reshape_3d(ctx0, q_id, head_dim, n_head_local, llm.n_tokens);
            q_id = ggml_permute(ctx0, q_id, 0, 2, 1, 3);                         // [head_dim, n_tokens, n_head_local]
            auto Qcur_id = ggml_mul_mat(ctx0, sp(layer.wk_b, id), q_id);         // [kv_lora_rank, n_tokens, n_head_local]
            cb(Qcur_id, "Qcur", il_id);

            auto kv_id = ggml_mul_mat(ctx0, sp(layer.wkv_a_mqa, id), cur_id);    // [kv_lora_rank, n_tokens]
            ggml_build_forward_expand(gf, kv_id);
            kv_id = llm_build_context::llm_build_norm(ctx0, kv_id, hparams, sp(layer.attn_kv_a_norm, id), nullptr, LLM_NORM_RMS, cb, il_id);
            cb(kv_id, "kv_compressed", il_id);

            ggml_tensor * cache_local = cache_repl->splits[id];
            const auto row_size_l = ggml_row_size(cache_local->type, kv_lora_rank);
            auto write_view = ggml_view_2d(ctx0, cache_local, kv_lora_rank, llm.n_tokens, row_size_l, row_size_l * llm.kv_head);
            const int cc_idx = 2 * n_device * il + 2 * id;
            GGML_ASSERT(cc_idx + 1 < (int)lctx.cache_copies.size());
            lctx.cache_copies[cc_idx + 0].cpy  = ggml_cpy(ctx0, kv_id, write_view);
            lctx.cache_copies[cc_idx + 0].step = row_size_l;
            ggml_build_forward_expand(gf, lctx.cache_copies[cc_idx + 0].cpy);

            auto K_local = ggml_view_2d(ctx0, cache_local, kv_lora_rank, llm.n_kv, row_size_l, 0);
            cb(K_local, "kv_cache", il_id);

            auto kqv_id = ggml_flash_attn_ext(ctx0, Qcur_id, K_local, K_local, KQ_mask, kq_scale, hparams.f_max_alibi_bias, 0.f);
            if (K_local->ne[1] < 256) {
                ggml_flash_attn_ext_set_prec(kqv_id, GGML_PREC_F32);
            }
            cb(kqv_id, "kqv_compressed", il_id);
            kqv_id = ggml_permute(ctx0, kqv_id, 0, 2, 1, 3);                     // [kv_lora_rank, n_tokens, n_head_local]

            auto out_id = ggml_mul_mat(ctx0, sp(layer.wv_b, id), kqv_id);        // [n_embd_head_v, n_tokens, n_head_local]
            out_id = ggml_cont(ctx0, ggml_permute(ctx0, out_id, 0, 2, 1, 3));    // [n_embd_head_v, n_head_local, n_tokens]
            out_id = ggml_reshape_2d(ctx0, out_id, n_embd_head_v * n_head_local, llm.n_tokens);

            auto partial = llm_build_context::llm_build_lora_mm(lctx, ctx0, wo_split->splits[id], out_id);
            if (partial->ne[1] > 32 && lctx.cparams.reduce_type != GGML_TYPE_F32) {
                partial = ggml_cast(ctx0, partial, lctx.cparams.reduce_type);
            }
            ggml_build_forward_expand(gf, partial);
            partials[id] = partial;
        }
        auto combined = ggml_reduce(ctx0, partials.data(), n_device, GGML_OP_ADD);
        ggml_build_forward_expand(gf, combined);
        cb(combined, "kqv_out", il);
        return combined;
    }


    // norm
    auto cur = llm_build_context::llm_build_norm(ctx0, inp, hparams, layer.attn_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(cur, "attn_norm", il);

    // q = wq_b(wq_a(x)) → [n_head * head_dim, n_tokens]
    auto qr = ggml_mul_mat(ctx0, layer.wq_a, cur);
    cb(qr, "q_a", il);
    qr = llm_build_context::llm_build_norm(ctx0, qr, hparams, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(qr, "q_a_norm", il);

    // DSA k-pool indexer: score k-pools and pick top_k cells (null = dense fallback).
    // Gate: --dsa flag + indexer tensors present + kpool > 0 + pool metadata created.
    ggml_tensor * top_k = nullptr;
    if (lctx.cparams.dsa && layer.indexer_attn_q_b && hparams.indexer_block_size > 0 && kp.cells) {
        top_k = build_glm5next_dsa_select(llm, gf, il, cur, qr, kp);
    }
    const bool direct_idx = top_k && kp.direct;   // index list goes to FLASH_ATTN_EXT src[5], no mask
    GGML_ASSERT((KQ_mask || direct_idx) && "glm5next: dense attention needs the KQ mask");

    auto q = ggml_mul_mat(ctx0, layer.wq_b, qr);
    cb(q, "q", il);

    // absorbed: Qcur = wk_b(permute(q)) → [kv_lora_rank, n_tokens, n_head]
    q = ggml_reshape_3d(ctx0, q, head_dim, n_head, llm.n_tokens);
    q = ggml_permute(ctx0, q, 0, 2, 1, 3);                         // [head_dim, n_tokens, n_head]
    auto Qcur = ggml_mul_mat(ctx0, layer.wk_b, q);                 // [kv_lora_rank, n_tokens, n_head]
    cb(Qcur, "Qcur", il);

    // kv_cmpr = norm(wkv_a_mqa(x)) → [kv_lora_rank, n_tokens]
    auto kv_cmpr = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    cb(kv_cmpr, "kv_a_mqa", il);
    kv_cmpr = llm_build_context::llm_build_norm(ctx0, kv_cmpr, hparams, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(kv_cmpr, "kv_compressed", il);

    // store kv_cmpr in k_l (K cache) — no rope part, just kv_lora_rank
    const auto row_size = ggml_row_size(kv_self.k_l[il]->type, kv_lora_rank);
    auto k_cache_view = ggml_view_2d(ctx0, kv_self.k_l[il], kv_lora_rank, llm.n_tokens,
            row_size, row_size * llm.kv_head);
    lctx.cache_copies[2 * il + 0].cpy  = ggml_cpy(ctx0, kv_cmpr, k_cache_view);
    lctx.cache_copies[2 * il + 0].step = row_size;
    ggml_build_forward_expand(gf, lctx.cache_copies[2 * il + 0].cpy);

    // read K cache (the index list only names used cells, so the direct path views the used part)
    const int64_t n_kv_attn = direct_idx ? kp.n_kv : llm.n_kv;
    auto K_cache = ggml_view_2d(ctx0, kv_self.k_l[il], kv_lora_rank, n_kv_attn, row_size, 0);
    cb(K_cache, "kv_cache", il);

    // Build the sparse attention mask when the indexer returned top_k cell indices.
    // ggml_indexer_mask takes the dense KQ_mask + I32 cell-index topk and unmask only those
    // cells (keeping future/padding masked via the base causal mask). Same op deepseek2 uses.
    ggml_tensor * attn_mask = KQ_mask;
    if (top_k && !direct_idx) {
        attn_mask = ggml_indexer_mask(ctx0, KQ_mask, top_k);
        cb(attn_mask, "dsa_sparse_mask", il);
    }

    // Attention: flash (fused, avoids materializing [n_kv, n_tokens, n_head]) vs. soft_max path.
    // Flash is required for large contexts — the soft_max path materializes the full score
    // matrix [n_kv, n_tokens, n_head], whose compute buffer scales linearly with ctx-size.
    // Absorbed NoPE MLA: K = V = the compressed latent (kv_lora_rank, no rope part), MQA.
    ggml_tensor * kqv_cmpr = nullptr;
    if (lctx.cparams.flash_attn) {
        kqv_cmpr = ggml_flash_attn_ext(ctx0, Qcur, K_cache, K_cache, direct_idx ? nullptr : attn_mask,
                kq_scale, hparams.f_max_alibi_bias, 0.f);
        if (direct_idx) {
            kqv_cmpr->src[5] = top_k;
        } else if (top_k) {
            int n_selected_padded = GGML_PAD(top_k->ne[0], 256);
            if (n_selected_padded < attn_mask->ne[0]) {
                auto selected = ggml_mask_to_index(ctx0, attn_mask, n_selected_padded);
                cb(selected, "mask_to_idx", il);
                kqv_cmpr->src[5] = selected;
            }
        }
        if (K_cache->ne[1] < 256) {
            ggml_flash_attn_ext_set_prec(kqv_cmpr, GGML_PREC_F32);
        }
        cb(kqv_cmpr, "kqv_compressed", il);
        // ggml_flash_attn_ext returns {v->ne[0], q->ne[2], q->ne[1], q->ne[3]} =
        // [kv_lora_rank, n_head, n_tokens, 1] — n_head and n_tokens are swapped vs. the
        // non-flash kqv_cmpr layout [kv_lora_rank, n_tokens, n_head]. Permute to match,
        // otherwise the downstream mul_mat(wv_b, kqv_cmpr) fails ggml_can_mul_mat for
        // n_tokens > 1 (batch dim n_head vs n_tokens mismatch). Mirrors build_deepseek2.
        kqv_cmpr = ggml_permute(ctx0, kqv_cmpr, 0, 2, 1, 3);
        cb(kqv_cmpr, "kqv_compressed_perm", il);
    } else {
        // V_cache_trans: with mla_attn == 1, store transposed kv_cmpr in v_l and read it back.
        // With mla_attn > 1, v_l is not allocated — transpose k_l on the fly.
        ggml_tensor * V_cache_trans = nullptr;
        if (lctx.cparams.mla_attn == 1 && kv_self.v_l[il]) {
            auto v_cache_trans_view = ggml_view_2d(ctx0, kv_self.v_l[il], llm.n_tokens, kv_lora_rank,
                    ggml_row_size(kv_self.v_l[il]->type, kv_self.size),
                    ggml_row_size(kv_self.v_l[il]->type, llm.kv_head));
            cb(v_cache_trans_view, "kv_cache_trans_view", il);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_transpose(ctx0, kv_cmpr), v_cache_trans_view));
            V_cache_trans = ggml_view_2d(ctx0, kv_self.v_l[il], llm.n_kv, kv_lora_rank,
                    ggml_row_size(kv_self.v_l[il]->type, kv_self.size), 0);
        } else {
            V_cache_trans = ggml_cont(ctx0, ggml_transpose(ctx0, K_cache));
        }
        cb(V_cache_trans, "kv_cache_trans", il);

        // kq = K_cache @ Qcur → [n_kv, n_tokens, n_head]
        auto kq = ggml_mul_mat(ctx0, K_cache, Qcur);
        if (K_cache->ne[1] < 256) {
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        }
        cb(kq, "kq", il);

        // soft_max with mask
        kq = ggml_soft_max_ext(ctx0, kq, attn_mask, kq_scale, hparams.f_max_alibi_bias);
        cb(kq, "kq_soft_max", il);

        // kqv_cmpr = V_cache_trans @ kq → [kv_lora_rank, n_tokens, n_head]
        kqv_cmpr = ggml_mul_mat(ctx0, V_cache_trans, kq);
        cb(kqv_cmpr, "kqv_compressed", il);
    }

    // kqv = wv_b @ kqv_cmpr → [n_embd_head_v, n_tokens, n_head]
    auto wv_b = layer.wv_b;
    if (wv_b->ne[1] != n_embd_head_v) {
        wv_b = ggml_reshape_3d(ctx0, wv_b, kv_lora_rank, n_embd_head_v, n_head);
    }
    auto kqv = ggml_mul_mat(ctx0, wv_b, kqv_cmpr);
    cb(kqv, "kqv", il);

    // reshape → [n_embd_head_v * n_head, n_tokens]
    kqv = ggml_permute(ctx0, kqv, 0, 2, 1, 3);   // [n_embd_head_v, n_head, n_tokens]
    kqv = ggml_cont(ctx0, kqv);
    kqv = ggml_reshape_2d(ctx0, kqv, n_embd_head_v * n_head, llm.n_tokens);
    cb(kqv, "kqv_2d", il);

    // out = wo @ kqv
    cur = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.wo, kqv);
    cb(cur, "kqv_out", il);

    return cur;
}

ggml_cgraph * llm_build_context::build_glm5next_mtp() {
    ggml_cgraph * gf = new_graph_custom();

    const bool emit_logits = cparams.mtp_op_type == MTP_OP_DRAFT_GEN ||
            cparams.mtp_op_type == MTP_OP_UPDATE_ACCEPTED;

    GGML_ASSERT(hparams.nextn_predict_layers == 1);

    const int il = hparams.n_layer - hparams.nextn_predict_layers;
    const auto & mtp_layer = model.layers[il];
    const float kq_scale = 1.0f / sqrtf(float(hparams.n_embd_head_k_full));

    GGML_ASSERT(mtp_layer.nextn.eh_proj != nullptr);
    GGML_ASSERT(mtp_layer.nextn.enorm != nullptr);
    GGML_ASSERT(mtp_layer.nextn.hnorm != nullptr);

    auto hidden_state = build_inp_mtp_states(n_embd);
    auto token_emb = build_inp_embd_mtp(model.tok_embd);
    auto inp_pos = build_inp_pos();
    auto pos_mask = ggml_cast(ctx0, inp_pos, GGML_TYPE_F32);
    pos_mask = ggml_clamp(ctx0, pos_mask, 0.0f, 1.0f);
    pos_mask = ggml_reshape_2d(ctx0, pos_mask, 1, n_tokens);
    pos_mask = ggml_repeat(ctx0, pos_mask, token_emb);
    if (token_emb->type != GGML_TYPE_F32) {
        token_emb = ggml_cast(ctx0, token_emb, GGML_TYPE_F32);
    }
    token_emb = ggml_mul(ctx0, token_emb, pos_mask);
    cb(token_emb, "mtp_token_embd_pos_masked", -1);
    auto KQ_mask = build_inp_KQ_mask();
    auto inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    // DSA selection for the NextN MLA layer (masked path: the tail layer has no pooled-key cache).
    glm5next_kpool_inputs kp;
    if (il < (int) lctx.kv_self.kr_l.size() && lctx.kv_self.kr_l[il] != nullptr) {
        glm5next_kpool_setup(*this, kp, /*allow_direct=*/false);
    }

    auto cur = build_mtp_input(mtp_layer, hidden_state, token_emb, il);
    auto inpSA = cur;
    cur = build_glm5next_mla_attention(*this, gf, il, cur, KQ_mask, kq_scale, kp);

    auto ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "mtp_ffn_inp", il);
    if (inp_out_ids) {
        ffn_inp = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
    }

    cur = llm_build_norm(ctx0, ffn_inp, hparams, mtp_layer.ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(cur, "ffn_norm", il);

    auto moe_out = llm_build_moe_ffn(ctx0, lctx, cur,
            mtp_layer.ffn_gate_inp, nullptr,
            mtp_layer.ffn_up_exps, nullptr,
            mtp_layer.ffn_gate_exps, nullptr,
            mtp_layer.ffn_down_exps, nullptr,
            mtp_layer.ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SILU, hparams.expert_weights_norm,
            true, hparams.expert_weights_scale,
            (llm_expert_gating_func_type) hparams.expert_gating_func,
            cb, il, gf, false, mtp_layer.ffn_up_gate_exps, nullptr, nullptr, nullptr, nullptr);
    ggml_build_forward_expand(gf, moe_out);
    cb(moe_out, "ffn_moe_out", il);

    auto ffn_shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
            mtp_layer.ffn_up_shexp, nullptr, nullptr,
            mtp_layer.ffn_gate_shexp, nullptr, nullptr,
            mtp_layer.ffn_down_shexp, nullptr, nullptr,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf);
    cb(ffn_shexp, "ffn_shexp", il);

    cur = ggml_add(ctx0, moe_out, ffn_shexp);
    cur = ggml_add(ctx0, cur, ffn_inp);
    cur = lctx.cvec.apply_to(ctx0, cur, il);
    cb(cur, "mtp_ffn_out_resid", il);

    if (!emit_logits) {
        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    ggml_tensor * head_norm = mtp_layer.nextn.shared_head_norm
        ? mtp_layer.nextn.shared_head_norm : model.output_norm;
    GGML_ASSERT(head_norm != nullptr);

    cur = llm_build_norm(ctx0, cur, hparams, head_norm,
            nullptr, LLM_NORM_RMS, cb, il);
    cb(cur, "result_norm", -1);

    cur = build_output(lctx, ctx0, cur, model.output, nullptr, cb);
    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

ggml_cgraph * llm_build_context::build_glm5next() {
    if (lctx.cparams.mtp_op_type != MTP_OP_NONE) {
        return build_glm5next_mtp();
    }

    ggml_cgraph * gf = new_graph_custom();

    const int64_t hc    = hparams.dsv4_hc_mult;

    // NoPE: no YaRN mscale correction
    const float kq_scale = 1.0f / sqrtf(float(hparams.n_embd_head_k_full));

    delta_net delta(lctx, batch);

    auto inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    auto inp_out_ids = (!lctx.cparams.mtp || n_outputs > 0) ? build_inp_out_ids() : nullptr;

    // KDA recurrent state slot routing
    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);

    // DSA k-pool indexer pool metadata: built once, shared across all MLA layers, host-filled in
    // llama_set_inputs. With the direct (index list) path no MLA layer reads the dense KQ mask, so the
    // O(n_kv x n_tokens) host fill and upload are skipped altogether.
    glm5next_kpool_inputs kp;
    glm5next_kpool_setup(*this, kp, /*allow_direct=*/true);
    auto KQ_mask = kp.direct ? nullptr : build_inp_KQ_mask();

    // expand embedding to hc streams: [n_embd, n_tokens] → [n_embd, hc, n_tokens]
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    const int n_trunk_layers = hparams.n_layer_kv_from_start;

    for (int il = 0; il < n_trunk_layers; ++il) {
        // attention block (hc_pre → attn → mhc_post)
        ggml_tensor * post_attn, * comb_attn;
        auto residual_attn = inpL;

        auto cur = glm5next_build_hc_pre(ctx0, *this, hparams, n_embd, hparams.f_norm_rms_eps,
                inpL,
                model.layers[il].hc_attn_fn,
                model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base,
                &post_attn, &comb_attn, cb, il);
        cb(cur, "hc_attn_pre", il);

        if (hparams.is_recurrent(il)) {
            // KDA linear attention (applies attn_norm internally; no residual — mHC handles it)
            cur = delta.build_layer_attn_kda(ctx0, gf, cur, nullptr, il, cb, /*add_residual = */ false);
        } else {
            // NoPE absorbed MLA (applies attn_norm internally; no residual — mHC handles it).
            // When --dsa is active, the indexer builds a sparse top-k mask; else dense.
            cur = build_glm5next_mla_attention(*this, gf, il, cur, KQ_mask, kq_scale, kp);
        }

        if (cur->type != GGML_TYPE_F32) {
            cur = ggml_cast(ctx0, cur, GGML_TYPE_F32); // -sm graph: attention partials are reduced in reduce_type
        }
        inpL = build_mhc_post(cur, post_attn, residual_attn, comb_attn, n_embd, hc, true);
        cb(inpL, "attn_out", il);

        // FFN block (hc_pre → ffn_norm → ffn/moe → mhc_post)
        ggml_tensor * post_ffn, * comb_ffn;
        auto residual_ffn = inpL;

        cur = glm5next_build_hc_pre(ctx0, *this, hparams, n_embd, hparams.f_norm_rms_eps,
                inpL,
                model.layers[il].hc_ffn_fn,
                model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base,
                &post_ffn, &comb_ffn, cb, il);
        cb(cur, "hc_ffn_pre", il);

        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            // dense FFN: no residual (mHC owns it); ffn_norm already applied above
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf);
        } else {
            // MoE: routed experts (add_input=false, mHC owns the residual) + shared expert
            auto moe_out = llm_build_moe_ffn(ctx0, lctx, cur,
                        model.layers[il].ffn_gate_inp, nullptr,
                        model.layers[il].ffn_up_exps, nullptr,
                        model.layers[il].ffn_gate_exps, nullptr,
                        model.layers[il].ffn_down_exps, nullptr,
                        model.layers[il].ffn_exp_probs_b,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, hparams.expert_weights_norm,
                        true, hparams.expert_weights_scale,
                        (llm_expert_gating_func_type) hparams.expert_gating_func,
                        cb, il, gf, false, model.layers[il].ffn_up_gate_exps, nullptr, nullptr, nullptr, nullptr);
            ggml_build_forward_expand(gf, moe_out);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up_shexp,   nullptr, nullptr,
                    model.layers[il].ffn_gate_shexp, nullptr, nullptr,
                    model.layers[il].ffn_down_shexp, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf);
            cb(ffn_shexp, "ffn_shexp", il);

            if (moe_out->type != GGML_TYPE_F32) {
                moe_out = ggml_cast(ctx0, moe_out, GGML_TYPE_F32);
            }
            if (ffn_shexp->type != GGML_TYPE_F32) {
                ffn_shexp = ggml_cast(ctx0, ffn_shexp, GGML_TYPE_F32);
            }
            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            ggml_build_forward_expand(gf, cur);
        }
        cb(cur, "ffn_out", il);

        if (cur->type != GGML_TYPE_F32) {
            cur = ggml_cast(ctx0, cur, GGML_TYPE_F32); // -sm graph: FFN partials are reduced in reduce_type
        }
        inpL = build_mhc_post(cur, post_ffn, residual_ffn, comb_ffn, n_embd, hc, true);
        inpL = lctx.cvec.apply_to(ctx0, inpL, il);
        if (lctx.dflash.capture && llama_dflash_capture_layer_requested(lctx.dflash.capture->layer_ids, il)) {
            ggml_tensor * capture = llama_dflash_hc_mean_for_capture(ctx0, inpL, hparams.dsv4_hc_mult);
            cb(capture, "l_out", il);
            ggml_build_forward_expand(gf, capture);
        } else {
            cb(inpL, "l_out", il);
        }
    }

    // Collapse the mHC streams before the shared output head.
    {
        auto flat = ggml_reshape_2d(ctx0, inpL, n_embd * hc, n_tokens);
        if (!lctx.cparams.mtp && n_outputs != n_tokens) {
            flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        }
        const int64_t n_out = flat->ne[1];
        // sum the hc streams
        ggml_tensor * summed = nullptr;
        for (int64_t s = 0; s < hc; ++s) {
            auto stream = ggml_view_2d(ctx0, flat, n_embd, n_out,
                    flat->nb[1], s * n_embd * ggml_element_size(flat));
            summed = summed ? ggml_add(ctx0, summed, stream) : stream;
        }
        inpL = ggml_scale(ctx0, summed, 1.0f / hc);
        cb(inpL, "hc_collapse", -1);

        if (lctx.cparams.mtp) {
            // MTP consumes the full post-output-norm target state.
            inpL = llm_build_norm(ctx0, inpL, hparams, model.output_norm,
                    nullptr, LLM_NORM_RMS, cb, -1);
            cb(inpL, "result_mtp_embd", -1);
            ggml_set_output(inpL);
            ggml_build_forward_expand(gf, inpL);
            if (n_outputs == 0) {
                return gf;
            }
        }
    }

    if (lctx.cparams.mtp && n_outputs != n_tokens) {
        inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
    }

    auto cur = build_output(lctx, ctx0, inpL, model.output,
            lctx.cparams.mtp ? nullptr : model.output_norm, cb);
    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}
