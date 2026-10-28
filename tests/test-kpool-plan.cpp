// Unit test of the glm5next KPool-DSA pool plan (src/llama-kpool-plan.h) on a mock KV cache: single and several
// sequences, mixed ubatches, decode, a copied prefix, context shift, slot reuse, a stale cache. Checks, for every
// ubatch, that each query's visible columns are exactly its sequence's complete pools below its own pool, that the
// tail holds its own pool's cells, that together they cover every attendable cell, that kp_l rows are unique per
// member set and stable across ubatches, and that the re-pool list is exactly the rows whose members changed or
// were written by the ubatch. Build (no library needed):
//   g++ -std=c++17 -O1 -Iinclude -Iggml/include tests/test-kpool-plan.cpp -o test-kpool-plan && ./test-kpool-plan
#include "../src/llama-kpool-plan.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

static int n_checks = 0, n_fail = 0;
#define CHECK(cond, ...) do { n_checks++; if (!(cond)) { n_fail++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

struct cell {
    llama_pos pos = -1;
    std::set<llama_seq_id> seq_id;
    bool is_empty() const { return seq_id.empty(); }
    bool has(llama_seq_id s) const { return seq_id.count(s) > 0; }
};

struct tok { llama_seq_id s; llama_pos p; };

struct mock_cache {
    std::vector<cell> cells;
    uint32_t head = 0;
    const int64_t r = 4;
    int64_t n_rows;
    uint32_t n_seq_max;
    bool stale = false;
    llama_kpool_plan plan;
    llama_kpool_rows rows;
    std::map<std::vector<int32_t>, int32_t> row_of_members_prev;   // row stability across ubatches
    std::string name;

    mock_cache(const char * name_, uint32_t size, uint32_t n_seq_max_ = 4)
        : cells(size), n_rows(size / 4 + 4 * n_seq_max_ + 64), n_seq_max(n_seq_max_), name(name_) {}

    // llama_kv_cache_find_slot: the first run of n free cells at or after head (wrapping once)
    void find_slot(size_t n) {
        for (int pass = 0; pass < 2; ++pass) {
            for (uint32_t h = head; h + n <= cells.size(); ++h) {
                bool ok = true;
                for (size_t i = 0; i < n; ++i) if (cells[h + i].pos >= 0) { ok = false; break; }
                if (ok) { head = h; return; }
            }
            head = 0;
        }
        printf("FAIL %s: no slot of %zu cells\n", name.c_str(), n); exit(1);
    }

    void seq_rm(llama_seq_id s, llama_pos p0, llama_pos p1) {
        for (auto & c : cells) if (c.pos >= p0 && c.pos < p1 && c.has(s)) { c.seq_id.erase(s); if (c.is_empty()) c.pos = -1; }
    }
    void seq_add(llama_seq_id s, llama_pos p0, llama_pos p1, llama_pos d) {
        for (auto & c : cells) if (c.pos >= p0 && c.pos < p1 && c.has(s)) c.pos += d;
    }
    void seq_cp(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1) {
        for (auto & c : cells) if (c.pos >= p0 && c.pos < p1 && c.has(src)) c.seq_id.insert(dst);
    }

    // one ubatch: place the tokens, build the plan, check it, commit the shadows (unbounded window)
    void decode(const std::vector<tok> & toks) {
        find_slot(toks.size());
        std::vector<llama_pos> pos; std::vector<llama_seq_id> sid; std::vector<llama_seq_id *> sidp;
        for (size_t i = 0; i < toks.size(); ++i) {
            cells[head + i].pos = toks[i].p; cells[head + i].seq_id = { toks[i].s };
            pos.push_back(toks[i].p); sid.push_back(toks[i].s);
        }
        for (size_t i = 0; i < toks.size(); ++i) sidp.push_back(&sid[i]);
        // the shadows before the plan runs (for the re-pool check)
        const llama_kpool_rows before = rows;
        const bool was_stale = stale;
        const int dropped = llama_kpool_plan_compute(plan, rows, cells.data(), (uint32_t) cells.size(), head, r, n_rows,
                n_seq_max, stale, (int32_t) toks.size(), pos.data(), sidp.data());
        CHECK(dropped == 0, "%s: %d pools without a row", name.c_str(), dropped);
        CHECK(plan.valid, "%s: plan invalid", name.c_str());
        verify(toks, before, was_stale);
        // commit as llama_set_inputs does with a window that holds every entry
        for (size_t e = 0; e < plan.repool_rows.size(); ++e) {
            std::copy_n(plan.repool_cells.data() + e * r, r, rows.members.begin() + (int64_t) plan.repool_rows[e] * r);
        }
        stale = false;
        row_of_members_prev.clear();
        for (int64_t col = 0; col < plan.n_pool; ++col) {
            if (plan.ok[col]) row_of_members_prev[std::vector<int32_t>(plan.cells.begin() + col * r, plan.cells.begin() + (col + 1) * r)] = plan.row[col];
        }
        head += toks.size();
    }

    int n_cells_of(llama_seq_id s, llama_pos p0, llama_pos p1) const {
        int n = 0;
        for (auto & c : cells) if (c.pos >= p0 && c.pos < p1 && c.has(s)) n++;
        return n;
    }

    void verify(const std::vector<tok> & toks, const llama_kpool_rows & before, bool was_stale) {
        const int64_t n_pool = plan.n_pool;
        // the K view and the column count
        int64_t cell_max = 0;
        for (int64_t i = cells.size(); i > 0; --i) if (cells[i - 1].pos >= 0) { cell_max = i; break; }
        CHECK(plan.n_kv == std::min<int64_t>(cells.size(), std::max<int64_t>(256, GGML_PAD(cell_max, 256))), "%s: n_kv %lld", name.c_str(), (long long) plan.n_kv);
        CHECK(plan.n_pool >= plan.n_kv / r && plan.n_pool >= plan.n_seq_pools, "%s: n_pool %lld", name.c_str(), (long long) plan.n_pool);
        // every ok column is a complete pool of one position range; rows unique per member set
        std::map<int32_t, std::vector<int32_t>> members_of_row;
        for (int64_t col = 0; col < n_pool; ++col) {
            std::vector<int32_t> m(plan.cells.begin() + col * r, plan.cells.begin() + (col + 1) * r);
            if (!plan.ok[col]) {
                CHECK(plan.row[col] == -1 && m == std::vector<int32_t>(r, 0), "%s: column %lld not ok but has row/cells", name.c_str(), (long long) col);
                continue;
            }
            if (n_rows <= 0) {
                // without a cache ok means complete; the position check below still applies
            }
            const llama_pos p0 = cells[m[0]].pos;
            CHECK(p0 >= 0 && p0 % r == 0, "%s: column %lld first member at pos %d", name.c_str(), (long long) col, p0);
            for (int64_t t = 0; t < r; ++t) CHECK(cells[m[t]].pos == p0 + t, "%s: column %lld slot %lld pos %d != %d", name.c_str(), (long long) col, (long long) t, cells[m[t]].pos, p0 + (int) t);
            if (n_rows <= 0) {
                CHECK(plan.row[col] == -1, "%s: column %lld has a row without a cache", name.c_str(), (long long) col);
                continue;
            }
            CHECK(plan.row[col] >= 0 && plan.row[col] < n_rows, "%s: column %lld row %d", name.c_str(), (long long) col, plan.row[col]);
            auto it = members_of_row.find(plan.row[col]);
            if (it == members_of_row.end()) members_of_row[plan.row[col]] = m;
            else CHECK(it->second == m, "%s: row %d shared by two different pools", name.c_str(), plan.row[col]);
            auto pr = row_of_members_prev.find(m);
            if (pr != row_of_members_prev.end() && !was_stale) CHECK(pr->second == plan.row[col], "%s: pool at column %lld moved from row %d to %d", name.c_str(), (long long) col, pr->second, plan.row[col]);
            // the allocator agrees
            CHECK(rows.owner[plan.row[col]] == m[r - 1] && rows.row_of_cell[m[r - 1]] == plan.row[col], "%s: allocator mismatch at column %lld", name.c_str(), (long long) col);
        }
        // per query
        for (size_t j = 0; j < toks.size(); ++j) {
            const llama_seq_id s = toks[j].s; const llama_pos q = toks[j].p;
            const int64_t lo = plan.vis_lo[j], nv = plan.vis_n[j], bt = (q + 1) / r;
            CHECK(nv == bt || (nv < bt), "%s: query %zu n_vis %lld > (q+1)/r %lld", name.c_str(), j, (long long) nv, (long long) bt);
            // the sequence's highest position decides how many pools it has
            llama_pos smax = -1; for (auto & c : cells) if (c.has(s)) smax = std::max(smax, c.pos);
            CHECK(nv == std::min<int64_t>((smax + 1) / r, bt), "%s: query %zu n_vis %lld", name.c_str(), j, (long long) nv);
            std::set<int32_t> attended;
            for (int64_t b = 0; b < nv; ++b) {
                const int64_t col = lo + b;
                CHECK(col < n_pool, "%s: query %zu column %lld out of range", name.c_str(), j, (long long) col);
                if (plan.ok[col]) {
                    for (int64_t t = 0; t < r; ++t) {
                        const int32_t c = plan.cells[col * r + t];
                        CHECK(cells[c].has(s) && cells[c].pos == b * r + t, "%s: query %zu (seq %d pos %d) visible column %lld slot %lld is cell %d (pos %d, seq? %d)",
                                name.c_str(), j, s, q, (long long) b, (long long) t, c, cells[c].pos, (int) cells[c].has(s));
                        attended.insert(c);
                    }
                } else {
                    CHECK(n_cells_of(s, b * r, (b + 1) * r) < r, "%s: query %zu pool %lld complete but not ok", name.c_str(), j, (long long) b);
                }
            }
            for (int64_t t = 0; t < r - 1; ++t) {
                const int32_t c = plan.tail[j * (r - 1) + t];
                const llama_pos p = bt * r + t;
                int32_t want = -1;
                if (p <= q) for (size_t i = 0; i < cells.size(); ++i) if (cells[i].pos == p && cells[i].has(s)) want = (int32_t) i;
                CHECK(c == want, "%s: query %zu (seq %d pos %d) tail slot %lld cell %d, want %d", name.c_str(), j, s, q, (long long) t, c, want);
                if (c >= 0) attended.insert(c);
            }
            // coverage: every cell of the sequence at pos <= q, except those in an incomplete pool below the query's own
            std::set<int32_t> want;
            for (size_t i = 0; i < cells.size(); ++i) {
                const auto & c = cells[i];
                if (!c.has(s) || c.pos < 0 || c.pos > q) continue;
                const int64_t b = c.pos / r;
                if (b < bt && n_cells_of(s, b * r, (b + 1) * r) < r) continue;
                want.insert((int32_t) i);
            }
            CHECK(attended == want, "%s: query %zu (seq %d pos %d) attends %zu cells, want %zu", name.c_str(), j, s, q, attended.size(), want.size());
        }
        // re-pool: exactly the rows whose shadow differs from the members or with a member in this ubatch
        if (n_rows <= 0) {
            CHECK(plan.repool_rows.empty(), "%s: re-pool entries without a cache", name.c_str());
            return;
        }
        std::set<int32_t> want_rows, got_rows(plan.repool_rows.begin(), plan.repool_rows.end());
        for (int64_t col = 0; col < n_pool; ++col) {
            if (!plan.ok[col]) continue;
            const int32_t row = plan.row[col];
            bool again = was_stale || (size_t) row >= before.members.size() / r || before.owner.size() != (size_t) n_rows;
            for (int64_t t = 0; t < r && !again; ++t) {
                const int32_t m = plan.cells[col * r + t];
                if (before.members[row * r + t] != m || (m >= (int32_t) head && m < (int32_t) (head + toks.size()))) again = true;
                // a row that was freed and handed out again this ubatch starts with a cleared shadow
                if (before.owner[row] != m) {}
            }
            if (!again && before.owner[row] != plan.cells[col * r + r - 1]) again = true;   // newly allocated row
            if (again) want_rows.insert(row);
        }
        if (want_rows.empty()) {
            CHECK(got_rows.size() == 1, "%s: fallback window entry missing", name.c_str());
        } else {
            CHECK(got_rows == want_rows, "%s: re-pool rows %zu, want %zu", name.c_str(), got_rows.size(), want_rows.size());
        }
        for (size_t e = 0; e < plan.repool_rows.size(); ++e) {
            const int32_t row = plan.repool_rows[e];
            if (rows.owner[row] >= 0) {
                CHECK(plan.repool_cells[e * r + r - 1] == rows.owner[row], "%s: window entry %zu cells do not belong to row %d", name.c_str(), e, row);
            }
        }
    }
};

static std::vector<tok> run(llama_seq_id s, llama_pos p0, int n) {
    std::vector<tok> v; for (int i = 0; i < n; ++i) v.push_back({s, p0 + i}); return v;
}
static std::vector<tok> cat(std::vector<tok> a, const std::vector<tok> & b) { a.insert(a.end(), b.begin(), b.end()); return a; }

int main() {
    {   // 1. one sequence: prefill in odd ubatches, then decode
        mock_cache c("single", 1024, 1);
        llama_pos p = 0;
        for (int n : {7, 9, 4, 13, 1, 1, 1, 2, 5}) { c.decode(run(0, p, n)); p += n; }
        for (int i = 0; i < 12; ++i) { c.decode(run(0, p, 1)); p++; }
        for (size_t j = 0; j < c.plan.vis_lo.size(); ++j) CHECK(c.plan.vis_lo[j] == 0, "single: lo != 0");
        CHECK(c.plan.n_seq_pools == (p) / 4, "single: %lld pools for %d positions", (long long) c.plan.n_seq_pools, p);
    }
    {   // 2. four sequences, mixed ubatches (prompts of different lengths, then decode of all four per ubatch)
        mock_cache c("four", 2048, 4);
        llama_pos p[4] = {0, 0, 0, 0};
        auto add = [&](llama_seq_id s, int n) { auto v = run(s, p[s], n); p[s] += n; return v; };
        c.decode(cat(add(0, 10), add(1, 5)));
        c.decode(cat(add(2, 17), add(0, 3)));
        c.decode(add(3, 1));
        c.decode(cat(cat(add(1, 9), add(3, 6)), add(2, 2)));
        for (int i = 0; i < 10; ++i) c.decode(cat(cat(cat(add(0, 1), add(1, 1)), add(2, 1)), add(3, 1)));
        c.decode(cat(add(1, 1), add(3, 1)));
        for (int i = 0; i < 5; ++i) c.decode(cat(add(3, 1), add(0, 1)));
        // sequence 1 ends, a new prompt reuses its cells (slot reuse: the pools of the new prompt must be pooled again)
        c.seq_rm(1, 0, 1 << 30);
        p[1] = 0; c.head = 0;
        c.decode(add(1, 12));
        c.decode(add(1, 4));
        for (int i = 0; i < 3; ++i) c.decode(cat(add(1, 1), add(2, 1)));
    }
    {   // 3. a copied prefix (llama_kv_cache_seq_cp): shared pools use one row, the partly shared pool its own
        mock_cache c("prefix", 1024, 3);
        c.decode(run(0, 0, 10));
        c.seq_cp(0, 1, 0, 10);
        c.decode(cat(run(1, 10, 3), run(0, 10, 2)));
        c.decode(cat(run(1, 13, 1), run(0, 12, 1)));
        c.seq_cp(0, 2, 0, 13);
        c.decode(run(2, 13, 6));
        for (int i = 0; i < 4; ++i) c.decode(cat(cat(run(0, 13 + i, 1), run(1, 14 + i, 1)), run(2, 19 + i, 1)));
    }
    {   // 4. context shift (seq_rm + seq_add): a shift by a multiple of kpool keeps the pools, any other re-pools
        mock_cache c("shift", 1024, 2);
        c.decode(run(0, 0, 40));
        c.decode(run(1, 0, 9));
        c.seq_rm(0, 4, 12); c.seq_add(0, 12, 40, -8);
        c.decode(cat(run(0, 32, 1), run(1, 9, 1)));
        c.seq_rm(0, 2, 5); c.seq_add(0, 5, 33, -3);
        c.decode(run(0, 30, 3));
        c.decode(run(1, 10, 2));
    }
    {   // 5. stale (state restore / defrag): every row pooled again, rows may move
        mock_cache c("stale", 512, 2);
        c.decode(cat(run(0, 0, 20), run(1, 0, 6)));
        c.stale = true;
        c.decode(cat(run(0, 20, 1), run(1, 6, 1)));
        c.decode(cat(run(0, 21, 1), run(1, 7, 1)));
    }
    {   // 6. gaps: a removed middle range leaves an incomplete pool below the query (not attendable, not ok)
        mock_cache c("gap", 512, 1);
        c.decode(run(0, 0, 20));
        c.seq_rm(0, 9, 11);
        c.decode(run(0, 20, 3));
    }
    {   // 7. no pooled-key cache (n_rows 0): every complete pool ok, nothing to pool again
        mock_cache c("norows", 512, 2);
        c.n_rows = 0;
        c.decode(cat(run(0, 0, 9), run(1, 0, 5)));
        c.decode(cat(run(0, 9, 1), run(1, 5, 1)));
    }
    printf("%d checks, %d failures\n", n_checks, n_fail);
    return n_fail ? 1 : 0;
}
