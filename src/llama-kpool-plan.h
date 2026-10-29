#pragma once

// glm5next KPool-DSA with several sequences: the per-ubatch host plan of the indexer's pool tables.
//
//   * every sequence's pools are enumerated by position: pool b of sequence s holds the cells at positions
//     [b*kpool, (b+1)*kpool) that carry s, in slot order; sequence after sequence they fill the score columns
//     [seq_lo, seq_lo + (max_pos + 1)/kpool), so a query at position q of s sees [seq_lo, seq_lo + (q+1)/kpool).
//     A sequence's trailing incomplete pool is not a column (its cells go to the queries' tail block). Columns
//     beyond the enumerated pools (up to n_kv/kpool, the single-sequence count) are empty.
//   * a complete pool's key lives in one kv_self.kp_l row, keyed by the pool's last member cell: the row stays
//     while that cell is the last member of some complete pool, so a sequence's pools keep their rows across
//     ubatches and a pool shared by two sequences (a copied prefix) uses one row. A host shadow of the members
//     each row was pooled from decides what a graph pools again: rows with changed members, rows with a member
//     written by this ubatch, and every row after a state restore or defrag (stale).
//
// The plan is a pure function of the cache cells, the ubatch and the row allocator's state, so it is a template
// over the cell type (llama_kv_cell in the library, a mock in tests/test-kpool-plan.cpp): a Cell has a
// `llama_pos pos` (-1 = free), an iterable `seq_id` of llama_seq_id and is_empty(). llama_set_inputs copies the
// plan into the graph's inputs and commits the shadows of the rows the graph pools again.

#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <vector>

struct llama_kpool_plan {
    bool    valid       = false;
    int32_t kv_head     = -1;     // the ubatch the plan belongs to
    int32_t n_tokens    = 0;
    int64_t n_kv        = 0;      // K view covering every used cell (padded to 256)
    int64_t n_pool      = 0;      // score columns: max(n_kv / kpool, pools of all sequences)
    int64_t n_seq_pools = 0;      // columns that hold a pool
    int64_t n_rows      = 0;      // rows of kp_l
    std::vector<int32_t> cells;        // [kpool * n_pool] member cell of each slot (0 where the pool is incomplete)
    std::vector<int32_t> ok;           // [n_pool] 1 if the pool is complete and has a kp_l row
    std::vector<int32_t> row;          // [n_pool] its kp_l row (-1: none)
    std::vector<int32_t> vis_lo;       // [n_tokens] first column visible to the query
    std::vector<int32_t> vis_n;        // [n_tokens] number of visible columns
    std::vector<int32_t> tail;         // [(kpool - 1) * n_tokens] cells of the query's own incomplete pool (-1: none)
    std::vector<int32_t> repool_rows;  // rows this graph pools again (never empty when rows exist)
    std::vector<int32_t> repool_cells; // [kpool * repool_rows.size()] their member cells
};

struct llama_kpool_rows {
    std::vector<int32_t> row_of_cell;  // [kv_size] row of the pool whose last member is this cell (-1: none)
    std::vector<int32_t> owner;        // [n_rows] that cell (-1: free row)
    std::vector<int32_t> members;      // [kpool * n_rows] the members the row was pooled from (-1: never)
    std::vector<int32_t> free_rows;
};

// Builds the plan for the ubatch at cells [kv_head, kv_head + n_tokens) whose tokens have positions pos[] and
// sequences seq_id[][0]; r = kpool, n_rows = rows of kp_l (0: no pooled-key cache, every complete pool is ok and
// nothing is pooled again), stale = pool every row again (the caller clears its flag once the window is filled).
// Returns the number of complete pools that got no row (left out of the selection).
template <typename Cell>
int llama_kpool_plan_compute(llama_kpool_plan & plan, llama_kpool_rows & rows,
        const Cell * cells, uint32_t kv_size, uint32_t kv_head, int64_t r, int64_t n_rows, uint32_t n_seq_max,
        bool stale, int32_t n_tokens, const llama_pos * pos, llama_seq_id * const * seq_id) {
    plan.valid = false;
    if (r <= 0) {
        return 0;
    }

    // the K view: every used cell, padded like the FA kernels want it
    int64_t cell_max = 0;
    for (int64_t i = kv_size; i > 0; --i) {
        const auto & c = cells[i - 1];
        if (c.pos >= 0 && !c.is_empty()) {
            cell_max = i;
            break;
        }
    }
    const int64_t n_kv = std::min<int64_t>(kv_size, std::max<int64_t>(256, GGML_PAD(cell_max, 256)));

    // highest position per sequence
    std::vector<llama_pos> seq_max(std::max<uint32_t>(1, n_seq_max), -1);
    for (int64_t i = 0; i < cell_max; ++i) {
        const auto & c = cells[i];
        if (c.pos < 0) continue;
        for (const llama_seq_id s : c.seq_id) {
            if (s < 0) continue;
            if ((size_t) s >= seq_max.size()) seq_max.resize(s + 1, -1);
            seq_max[s] = std::max(seq_max[s], c.pos);
        }
    }
    const size_t n_seq = seq_max.size();
    std::vector<int64_t> seq_lo(n_seq, 0), seq_np(n_seq, 0);
    int64_t n_seq_pools = 0;
    for (size_t s = 0; s < n_seq; ++s) {
        seq_lo[s] = n_seq_pools;
        seq_np[s] = seq_max[s] >= 0 ? (seq_max[s] + 1) / r : 0;
        n_seq_pools += seq_np[s];
    }
    // padded to a multiple of 64 (n_kv / r already is): the fused indexer kernel works in 64-pool tiles
    const int64_t n_pool = GGML_PAD(std::max<int64_t>(n_kv / r, n_seq_pools), 64);

    // member cells per (sequence, pool) slot; a sequence's trailing incomplete pool is kept for the tails
    plan.cells.assign(r * n_pool, -1);
    std::vector<int32_t> filled(n_pool, 0);
    std::vector<int32_t> seq_tail(n_seq * r, -1);
    for (int64_t i = 0; i < cell_max; ++i) {
        const auto & c = cells[i];
        if (c.pos < 0) continue;
        const int64_t b = c.pos / r, t = c.pos % r;
        for (const llama_seq_id s : c.seq_id) {
            if (s < 0) continue;
            if (b < seq_np[s]) {
                auto & slot = plan.cells[(seq_lo[s] + b) * r + t];
                if (slot < 0) { slot = (int32_t) i; filled[seq_lo[s] + b]++; }
            } else if (b == seq_np[s]) {
                seq_tail[s * r + t] = (int32_t) i;
            }
        }
    }

    // per query: the visible column range and the cells of its own incomplete pool (index_kpool_always_select_tail)
    plan.vis_lo.assign(n_tokens, 0);
    plan.vis_n.assign(n_tokens, 0);
    plan.tail.assign((r - 1) * n_tokens, -1);
    for (int32_t j = 0; j < n_tokens; ++j) {
        const llama_seq_id s = seq_id[j][0];
        const llama_pos    q = pos[j];
        if (s < 0 || (size_t) s >= n_seq || q < 0) continue;
        const int64_t np = seq_np[s];
        const int64_t bt = (q + 1) / r;                 // the query's own pool, the one ending at or after q
        plan.vis_lo[j] = (int32_t) seq_lo[s];
        plan.vis_n[j]  = (int32_t) std::min<int64_t>(np, bt);
        for (int64_t t = 0; t < r - 1 && bt * r + t <= q; ++t) {
            int32_t cell = -1;
            if (bt < np) {
                cell = plan.cells[(seq_lo[s] + bt) * r + t];   // a pool the sequence has already completed
            } else if (bt == np) {
                cell = seq_tail[s * r + t];
            }
            plan.tail[j * (r - 1) + t] = cell;
        }
    }

    // kp_l rows: free the rows whose pool is gone, give every complete pool a row, decide what to pool again
    plan.row.assign(n_pool, -1);
    plan.ok.assign(n_pool, 0);
    plan.repool_rows.clear();
    plan.repool_cells.clear();
    int n_dropped = 0;
    if (n_rows <= 0) {
        for (int64_t col = 0; col < n_pool; ++col) {
            if (filled[col] == r) {
                plan.ok[col] = 1;
            } else {
                std::fill_n(plan.cells.begin() + col * r, r, 0);
            }
        }
    } else {
        if ((int64_t) rows.owner.size() != n_rows || rows.row_of_cell.size() != kv_size || stale) {
            rows.owner.assign(n_rows, -1);
            rows.members.assign(n_rows * r, -1);
            rows.row_of_cell.assign(kv_size, -1);
            rows.free_rows.clear();
            for (int64_t row = n_rows; row > 0; --row) {
                rows.free_rows.push_back((int32_t) (row - 1));   // row 0 is handed out first
            }
        }
        std::vector<uint8_t> is_last(kv_size, 0);
        for (int64_t col = 0; col < n_pool; ++col) {
            if (filled[col] == r) is_last[plan.cells[col * r + r - 1]] = 1;
        }
        for (int64_t row = 0; row < n_rows; ++row) {
            const int32_t owner = rows.owner[row];
            if (owner >= 0 && !is_last[owner]) {
                rows.row_of_cell[owner] = -1;
                rows.owner[row] = -1;
                std::fill_n(rows.members.begin() + row * r, r, -1);
                rows.free_rows.push_back((int32_t) row);
            }
        }
        std::vector<uint8_t> marked(n_rows, 0);
        for (int64_t col = 0; col < n_pool; ++col) {
            if (filled[col] != r) {
                std::fill_n(plan.cells.begin() + col * r, r, 0);   // cell 0 keeps the gather in range; ok = 0 masks it
                continue;
            }
            const int32_t * m = plan.cells.data() + col * r;
            int32_t row = rows.row_of_cell[m[r - 1]];
            if (row < 0) {
                if (rows.free_rows.empty()) {
                    n_dropped++;
                    std::fill_n(plan.cells.begin() + col * r, r, 0);
                    continue;
                }
                row = rows.free_rows.back();
                rows.free_rows.pop_back();
                rows.owner[row] = m[r - 1];
                rows.row_of_cell[m[r - 1]] = row;
                std::fill_n(rows.members.begin() + row * r, r, -1);
            }
            plan.row[col] = row;
            plan.ok[col]  = 1;
            if (marked[row]) continue;
            bool again = false;
            for (int64_t t = 0; t < r; ++t) {
                if (rows.members[row * r + t] != m[t] || (m[t] >= (int32_t) kv_head && m[t] < (int32_t) kv_head + n_tokens)) again = true;
            }
            if (again) {
                marked[row] = 1;
                plan.repool_rows.push_back(row);
                plan.repool_cells.insert(plan.repool_cells.end(), m, m + r);
            }
        }
        if (plan.repool_rows.empty()) {
            // the window is never empty: pool one live pool again (the same value) or, without any, write a free row
            int64_t col = 0;
            while (col < n_pool && !plan.ok[col]) ++col;
            if (col < n_pool) {
                plan.repool_rows.push_back(plan.row[col]);
                plan.repool_cells.insert(plan.repool_cells.end(), plan.cells.begin() + col * r, plan.cells.begin() + (col + 1) * r);
            } else {
                plan.repool_rows.push_back(rows.free_rows.empty() ? 0 : rows.free_rows.back());
                plan.repool_cells.insert(plan.repool_cells.end(), r, 0);
            }
        }
    }

    plan.valid       = true;
    plan.kv_head     = (int32_t) kv_head;
    plan.n_tokens    = n_tokens;
    plan.n_kv        = n_kv;
    plan.n_pool      = n_pool;
    plan.n_seq_pools = n_seq_pools;
    plan.n_rows      = n_rows;
    return n_dropped;
}
