#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <map>
#include <string>
#include <tuple>
#include <vector>

/**
 * This the arbitrary data which will be passed to each callback.
 * Later on we can for example add operation or tensor name filter from the CLI arg, or a file descriptor to dump the tensor.
 */
struct callback_data {
    std::vector<uint8_t> data;
};

static std::string ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static float ggml_get_float_value(const uint8_t * data, ggml_type type, size_t i) {
    if (type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*(const ggml_fp16_t *) &data[i]);
    } else if (type == GGML_TYPE_F32) {
        return *(const float *) &data[i];
    } else if (type == GGML_TYPE_I32) {
        return (float) *(const int32_t *) &data[i];
    } else if (type == GGML_TYPE_I16) {
        return (float) *(const int16_t *) &data[i];
    } else if (type == GGML_TYPE_I8) {
        return (float) *(const int8_t *) &data[i];
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
    GGML_ASSERT(n > 0);
    // Compute the FULL tensor sum first (debug aid — was previously summing
    // only the displayed first/last elements, which made it useless for
    // comparing two binaries on the same tensor).
    double sum = 0.0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    sum += ggml_get_float_value(data, type, i);
                }
            }
        }
    }
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        printf("                                     [\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2*n) {
                printf("                                      ..., \n");
                i2 = ne[2] - n;
            }
            printf("                                      [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2*n) {
                    printf("                                       ..., \n");
                    i1 = ne[1] - n;
                }
                printf("                                       [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2*n) {
                        printf("..., ");
                        i0 = ne[0] - n;
                    }
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    printf("%12.4f", ggml_get_float_value(data, type, i));
                    if (i0 < ne[0] - 1) printf(", ");
                }
                printf("],\n");
            }
            printf("                                      ],\n");
        }
        printf("                                     ]\n");
        printf("                                     sum = %f\n", (float) sum);
    }
}

/**
 * GGML operations callback during the graph execution.
 *
 * @param t current tensor
 * @param ask when ask is true, the scheduler wants to know if we are interested in data from this tensor
 *            if we return true, a follow-up call will be made with ask=false in which we can do the actual collection.
 *            see ggml_backend_sched_eval_callback
 * @param user_data user data to pass at each call back
 * @return true to receive data or continue the graph, false otherwise
 */
// EVAL_CB_DUMP_DIR=<dir>: write every tensor whose name contains one of the comma-separated EVAL_CB_DUMP_FILTER
// substrings (default: all) as f32
// to <dir>/<name>.bin (header: int32 ne0, int32 n_rows; then rows). EVAL_CB_DUMP_COLS=n keeps only the last
// n rows (the last n tokens of a [ne0, n_tokens] activation). EVAL_CB_QUIET=1 suppresses the prints.
static const char * dump_dir    = getenv("EVAL_CB_DUMP_DIR");
static const char * dump_filter = getenv("EVAL_CB_DUMP_FILTER");
static const int    dump_cols   = getenv("EVAL_CB_DUMP_COLS") ? atoi(getenv("EVAL_CB_DUMP_COLS")) : 0;
static const bool   quiet       = getenv("EVAL_CB_QUIET") != nullptr;
static const bool   dump_seq    = getenv("EVAL_CB_DUMP_SEQ") != nullptr;   // name.<n>.bin per graph, not overwritten

static bool dump_wanted(const struct ggml_tensor * t) {
    if (!dump_dir || ggml_is_quantized(t->type)) {
        return false;
    }
    if (!dump_filter) {
        return true;
    }
    // comma-separated list of substrings
    std::string f(dump_filter);
    for (size_t i = 0; i < f.size();) {
        size_t j = f.find(',', i);
        if (j == std::string::npos) j = f.size();
        if (j > i && strstr(t->name, f.substr(i, j - i).c_str()) != nullptr) {
            return true;
        }
        i = j + 1;
    }
    return false;
}

static void dump_tensor(const struct ggml_tensor * t, const uint8_t * data) {
    const int64_t ne0 = t->ne[0];
    const int64_t nr  = ggml_nrows(t);
    const int64_t r0  = dump_cols > 0 && dump_cols < nr ? nr - dump_cols : 0;
    std::string path = std::string(dump_dir) + "/" + t->name + ".bin";
    if (dump_seq) {
        static std::map<std::string, int> seq;
        path = std::string(dump_dir) + "/" + t->name + "." + std::to_string(seq[t->name]++) + ".bin";
    }
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "dump: cannot open %s\n", path.c_str());
        return;
    }
    const int32_t hdr[2] = { (int32_t) ne0, (int32_t) (nr - r0) };
    fwrite(hdr, sizeof(hdr), 1, f);
    std::vector<float> row(ne0);
    for (int64_t r = r0; r < nr; ++r) {
        // row r of the tensor viewed as [ne0, nrows] (rows run over ne1, ne2, ne3 in order)
        int64_t i1 = r % t->ne[1], i2 = (r / t->ne[1]) % t->ne[2], i3 = r / (t->ne[1] * t->ne[2]);
        const uint8_t * p = data + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
        for (int64_t i = 0; i < ne0; ++i) {
            row[i] = ggml_get_float_value((uint8_t *) p, t->type, i * t->nb[0]);
        }
        fwrite(row.data(), sizeof(float), ne0, f);
    }
    fclose(f);
}

static int ggml_debug(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (callback_data *) user_data;

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    if (ask) {
        return dump_dir ? (dump_wanted(t) ? 1 : 0) : 1; // when dumping, only fetch what is dumped
    }
    if (dump_dir) {
        if (!dump_wanted(t)) {
            return 1;
        }
        const bool is_host = ggml_backend_buffer_is_host(t->buffer);
        if (!is_host) {
            auto n_bytes = ggml_nbytes(t);
            cb_data->data.resize(n_bytes);
            ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
        }
        dump_tensor(t, is_host ? (const uint8_t *) t->data : cb_data->data.data());
        if (quiet) {
            return 1;
        }
    }

    char src1_str[128] = {0};
    if (src1) {
        snprintf(src1_str, sizeof(src1_str), "%s{%s}", src1->name, ggml_ne_string(src1).c_str());
    }

    printf("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__,
           t->name, ggml_type_name(t->type), ggml_op_desc(t),
           src0->name, ggml_ne_string(src0).c_str(),
           src1 ? src1_str : "",
           ggml_ne_string(t).c_str());


    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    if (!ggml_is_quantized(t->type) && !quiet) {
        uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();
        ggml_print_tensor(data, t->type, t->ne, t->nb, 3);
    }

    return 1;
}

static bool run(llama_context * ctx, const gpt_params & params) {
    const bool add_bos = llama_should_add_bos_token(llama_get_model(ctx));

    std::vector<llama_token> tokens = ::common_tokenize(ctx, params.prompt, add_bos);

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size(), 0, 0))) {
        fprintf(stderr, "%s : failed to eval\n", __func__);
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    callback_data cb_data;

    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }

    print_build_info();

    std::mt19937 rng(params.seed);

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = ggml_debug;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    // init
    llama_init_result llama_init = llama_init_from_gpt_params(params);

    llama_model * model = llama_init.model;
    llama_context * ctx = llama_init.context;
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s\n", gpt_params_get_system_info(params).c_str());
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    llama_print_timings(ctx);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
