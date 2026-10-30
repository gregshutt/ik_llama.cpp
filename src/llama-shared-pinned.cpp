#include "llama-shared-pinned.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#include "ggml-alloc.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef __linux__
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// Protocol (all files under GGML_CUDA_SHARED_PINNED_DIR, default /dev/shm, named ggml-shared-<hash>.*):
//   .lock   flock(LOCK_EX) held by the creator from the check until mark_ready; attachers take it only for the check
//   .bin    the segment; every user holds flock(LOCK_SH) on it while it runs (unlink-if-last probes with LOCK_EX|NB)
//   .ready  exists once the fill is complete; absent (or the .bin size wrong) => the next locker recreates the segment
//   .info   what the segment holds (model files and tensor layout), for humans
// A creator that dies mid-fill releases .lock; whoever gets it next finds no .ready and refills from the file.

static uint64_t fnv1a(uint64_t h, const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
static uint64_t fnv1a_str(uint64_t h, const std::string & s) { return fnv1a(h, s.data(), s.size() + 1); }
template <typename T> static uint64_t fnv1a_val(uint64_t h, const T & v) { return fnv1a(h, &v, sizeof(v)); }

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool env_true(const char * name) {
    const char * e = getenv(name);
    return e != nullptr && *e != '\0' && *e != '0';
}

bool llama_shared_pinned_enabled(const llama_model_loader & ml) {
#if defined(GGML_USE_CUDA) && defined(__linux__)
    if (!env_true("GGML_CUDA_SHARED_PINNED_WEIGHTS")) {
        return false;
    }
    if (ml.repack_tensors) {
        LLAMA_LOG_WARN("%s: GGML_CUDA_SHARED_PINNED_WEIGHTS ignored: run-time repacking (-rtr) rewrites the weights after they are shared\n", __func__);
        return false;
    }
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr || getenv("GGML_CUDA_NO_PINNED_WEIGHTS") != nullptr) {
        LLAMA_LOG_WARN("%s: GGML_CUDA_SHARED_PINNED_WEIGHTS ignored: GGML_CUDA_NO_PINNED(_WEIGHTS) keeps the weights unpinned\n", __func__);
        return false;
    }
    return true;
#else
    GGML_UNUSED(ml);
    return false;
#endif
}

llama_shared_pinned_segment::llama_shared_pinned_segment(const std::string & dir, uint64_t hash, size_t size, ggml_context * ctx)
    : hash(hash), size(size), ctx(ctx) {
    char name[64];
    snprintf(name, sizeof(name), "/ggml-shared-%016llx", (unsigned long long) hash);
    data_path  = dir + name + ".bin";
    lock_path  = dir + name + ".lock";
    ready_path = dir + name + ".ready";
    info_path  = dir + name + ".info";
}

#if defined(__linux__)

bool llama_shared_pinned_segment::open() {
    lock_fd = ::open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0644);
    if (lock_fd < 0) {
        LLAMA_LOG_WARN("%s: cannot open %s: %s\n", __func__, lock_path.c_str(), strerror(errno));
        return false;
    }
    if (flock(lock_fd, LOCK_EX | LOCK_NB) != 0) {
        LLAMA_LOG_INFO("%s: another process is creating %s, waiting for it\n", __func__, data_path.c_str());
        const double t0 = now_s();
        while (flock(lock_fd, LOCK_EX) != 0) {
            if (errno != EINTR) {
                LLAMA_LOG_WARN("%s: flock(%s) failed: %s\n", __func__, lock_path.c_str(), strerror(errno));
                return false;
            }
        }
        LLAMA_LOG_INFO("%s: waited %.1f s\n", __func__, now_s() - t0);
    }

    struct stat st {};
    const bool complete = ::stat(data_path.c_str(), &st) == 0 && (size_t) st.st_size == size &&
                          ::access(ready_path.c_str(), F_OK) == 0;
    if (complete) {
        data_fd = ::open(data_path.c_str(), O_RDWR | O_CLOEXEC);
        if (data_fd < 0) {
            LLAMA_LOG_WARN("%s: cannot open %s: %s\n", __func__, data_path.c_str(), strerror(errno));
            return false;
        }
        created = false;
        ready   = true;
    } else {
        ::unlink(ready_path.c_str());
        data_fd = ::open(data_path.c_str(), O_CREAT | O_RDWR | O_TRUNC | O_CLOEXEC, 0644); // O_TRUNC frees a stale segment's pages
        if (data_fd < 0) {
            LLAMA_LOG_WARN("%s: cannot create %s: %s\n", __func__, data_path.c_str(), strerror(errno));
            return false;
        }
        if (ftruncate(data_fd, (off_t) size) != 0) {
            LLAMA_LOG_WARN("%s: cannot size %s to %.2f GiB: %s (is the tmpfs large enough?)\n", __func__,
                    data_path.c_str(), size/1024.0/1024.0/1024.0, strerror(errno));
            return false;
        }
        created = true;
        ready   = false;
    }

    addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, data_fd, 0);
    if (addr == MAP_FAILED) {
        addr = nullptr;
        LLAMA_LOG_WARN("%s: mmap of %s failed: %s\n", __func__, data_path.c_str(), strerror(errno));
        return false;
    }
    if (created) {
#ifdef MADV_POPULATE_WRITE
        const double t0 = now_s();
        if (madvise(addr, size, MADV_POPULATE_WRITE) != 0) {
            LLAMA_LOG_WARN("%s: madvise(MADV_POPULATE_WRITE) failed: %s (the segment is populated as it is filled)\n", __func__, strerror(errno));
        } else {
            LLAMA_LOG_INFO("%s: populated %.2f GiB of %s in %.1f s\n", __func__, size/1024.0/1024.0/1024.0, data_path.c_str(), now_s() - t0);
        }
#endif
    }
    // "in use" mark; the creation lock stays with the creator until mark_ready
    if (flock(data_fd, LOCK_SH) != 0) {
        LLAMA_LOG_WARN("%s: flock(%s, LOCK_SH) failed: %s\n", __func__, data_path.c_str(), strerror(errno));
    }
    if (!created) {
        flock(lock_fd, LOCK_UN);
    }
    return true;
}

void llama_shared_pinned_segment::mark_ready() {
    if (!created || ready) {
        return;
    }
    if (FILE * f = fopen(info_path.c_str(), "w")) {
        fputs(info.c_str(), f);
        fclose(f);
    }
    int fd = ::open(ready_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) {
        LLAMA_LOG_WARN("%s: cannot write %s: %s (other processes will refill the segment)\n", __func__, ready_path.c_str(), strerror(errno));
    } else {
        char line[80];
        snprintf(line, sizeof(line), "%016llx %zu\n", (unsigned long long) hash, size);
        if (write(fd, line, strlen(line)) < 0) { /* the marker's existence is what counts */ }
        close(fd);
        ready = true;
        LLAMA_LOG_INFO("%s: %s is complete (%.2f GiB); later processes attach to it\n", __func__, data_path.c_str(), size/1024.0/1024.0/1024.0);
    }
    flock(lock_fd, LOCK_UN);
}

llama_shared_pinned_segment::~llama_shared_pinned_segment() {
    if (addr != nullptr && !mapping_owned_by_buffer) {
        munmap(addr, size);
    }
    if (data_fd >= 0) {
        if (env_true("GGML_CUDA_SHARED_PINNED_UNLINK") && lock_fd >= 0) {
            // serialize with starters (they hold .lock from the ready check through their LOCK_SH), then probe for other users
            flock(lock_fd, LOCK_EX);
            if (flock(data_fd, LOCK_EX | LOCK_NB) == 0) {
                ::unlink(ready_path.c_str());
                ::unlink(info_path.c_str());
                ::unlink(data_path.c_str());
                LLAMA_LOG_INFO("%s: removed %s (last user)\n", __func__, data_path.c_str());
            }
            flock(lock_fd, LOCK_UN);
        }
        close(data_fd);
    }
    if (lock_fd >= 0) {
        flock(lock_fd, LOCK_UN);
        close(lock_fd);
    }
}

#else // !__linux__

bool llama_shared_pinned_segment::open() { return false; }
void llama_shared_pinned_segment::mark_ready() {}
llama_shared_pinned_segment::~llama_shared_pinned_segment() {}

#endif

ggml_backend_buffer_t llama_shared_pinned_alloc(llama_model_loader & ml, llama_model & model, ggml_context * ctx,
        ggml_backend_buffer_type_t buft) {
#if defined(GGML_USE_CUDA) && defined(__linux__)
    const char * dir_env = getenv("GGML_CUDA_SHARED_PINNED_DIR");
    const std::string dir = dir_env && *dir_env ? dir_env : "/dev/shm";

    // the layout: every unallocated non-view tensor of the context, in context order, exactly as
    // ggml_backend_alloc_ctx_tensors_from_buft would place it in a single buffer
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    struct placed { ggml_tensor * t; size_t offset; size_t size; };
    std::vector<placed> layout;
    size_t total = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (t->data == nullptr && t->view_src == nullptr) {
            const size_t sz = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
            layout.push_back({t, total, sz});
            total += sz;
        }
    }
    if (total == 0) {
        return nullptr;
    }

    // identity: the model files, the tensors' file positions and the layout
    uint64_t h = 1469598103934665603ull;
    std::string info;
    for (size_t i = 0; i < ml.files.size(); ++i) {
        struct stat st {};
        if (fstat(ml.files[i]->file_id(), &st) != 0) {
            LLAMA_LOG_WARN("%s: fstat of model file %zu failed: %s\n", __func__, i, strerror(errno));
            return nullptr;
        }
        const uint64_t sz = (uint64_t) st.st_size, mt = (uint64_t) st.st_mtim.tv_sec, ino = (uint64_t) st.st_ino;
        h = fnv1a_val(fnv1a_val(fnv1a_val(h, sz), mt), ino);
        char line[160];
        snprintf(line, sizeof(line), "file %zu size %llu mtime %llu inode %llu\n", i, (unsigned long long) sz, (unsigned long long) mt, (unsigned long long) ino);
        info += line;
    }
    for (const auto & p : layout) {
        const auto * w = ml.get_weight(ggml_get_name(p.t));
        const uint64_t offs = w ? w->offs : 0, idx = w ? w->idx : 0;
        h = fnv1a_str(h, ggml_get_name(p.t));
        h = fnv1a_val(h, p.t->type);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) h = fnv1a_val(h, p.t->ne[d]);
        h = fnv1a_val(fnv1a_val(fnv1a_val(fnv1a_val(h, idx), offs), p.offset), p.size);
        char line[256];
        snprintf(line, sizeof(line), "%s type %s file %llu@%llu offset %zu size %zu\n", ggml_get_name(p.t),
                ggml_type_name(p.t->type), (unsigned long long) idx, (unsigned long long) offs, p.offset, p.size);
        info += line;
    }

    auto seg = std::make_unique<llama_shared_pinned_segment>(dir, h, total, ctx);
    seg->info = info;
    if (!seg->open()) {
        LLAMA_LOG_WARN("%s: shared pinned segment unavailable, falling back to a private pinned copy\n", __func__);
        return nullptr;
    }
    ggml_backend_buffer_t buf = ggml_backend_cuda_host_buffer_from_ptr(seg->addr, total);
    if (buf == nullptr) {
        LLAMA_LOG_WARN("%s: could not pin the shared segment, falling back to a private pinned copy\n", __func__);
        return nullptr;
    }
    seg->mapping_owned_by_buffer = true;

    // place the tensors (same walk as ggml-alloc's alloc_tensor_range) and check the offsets against the layout
    struct ggml_tallocr tallocr = ggml_tallocr_new(buf);
    size_t ip = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (t->data == nullptr) {
            if (t->view_src == nullptr) {
                ggml_tallocr_alloc(&tallocr, t);
                GGML_ASSERT(ip < layout.size() && layout[ip].t == t);
                GGML_ASSERT((size_t) ((char *) t->data - (char *) seg->addr) == layout[ip].offset);
                ++ip;
            } else if (t->buffer == nullptr) {
                ggml_backend_view_init(t);
            }
        } else if (t->view_src != nullptr && t->buffer == nullptr) {
            ggml_backend_view_init(t);
        }
    }
    if (seg->created) {
        LLAMA_LOG_INFO("%s: created shared pinned segment %s (%.2f GiB, %zu tensors); filling it from the model\n", __func__,
                seg->data_path.c_str(), total/1024.0/1024.0/1024.0, layout.size());
    } else {
        ml.preloaded_bufs.push_back(buf);
        LLAMA_LOG_INFO("%s: attached to shared pinned segment %s (%.2f GiB, %zu tensors): already loaded\n", __func__,
                seg->data_path.c_str(), total/1024.0/1024.0/1024.0, layout.size());
    }
    model.shared_segments.push_back(std::move(seg));
    return buf;
#else
    GGML_UNUSED(ml); GGML_UNUSED(model); GGML_UNUSED(ctx); GGML_UNUSED(buft);
    return nullptr;
#endif
}

void llama_shared_pinned_mark_ready(llama_model & model, ggml_context * ctx) {
    for (auto & seg : model.shared_segments) {
        if (seg->ctx == ctx) {
            seg->mark_ready();
        }
    }
}
