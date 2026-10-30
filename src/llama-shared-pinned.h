#pragma once

// GGML_CUDA_SHARED_PINNED_WEIGHTS: the tensors of the CPU-side (CUDA_Host) context of a model are kept in one
// MAP_SHARED tmpfs file that every process loading the same model shares and pins with cudaHostRegister, instead of a
// private pinned copy per process. The first process creates and fills the segment; later ones attach to the finished
// segment and skip the load. See llama-shared-pinned.cpp for the protocol.

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct llama_model;
struct llama_model_loader;

struct llama_shared_pinned_segment {
    std::string    data_path, lock_path, ready_path, info_path;
    uint64_t       hash = 0;
    size_t         size = 0;
    void         * addr = nullptr;
    int            data_fd = -1;
    int            lock_fd = -1;
    bool           created = false;         // this process fills the segment (holds the creation lock until mark_ready)
    bool           ready = false;           // the fill is complete (ours, or found on attach)
    bool           mapping_owned_by_buffer = false; // the ggml buffer munmaps; the segment only closes its fds
    ggml_context * ctx = nullptr;
    std::string    info;                    // human-readable description written next to the segment

    llama_shared_pinned_segment(const std::string & dir, uint64_t hash, size_t size, ggml_context * ctx);
    ~llama_shared_pinned_segment();

    bool open();         // lock, create (truncate + populate) or attach (ready marker present), map
    void mark_ready();   // after the fill: write the info and ready marker, release the creation lock
};

// the env var is set and nothing in the loader's configuration conflicts with sharing
bool llama_shared_pinned_enabled(const llama_model_loader & ml);

// places every unallocated tensor of ctx in the model's shared segment (created or attached) and returns the pinned
// CUDA_Host buffer over it; tensors of an attached segment are added to ml.preloaded_bufs so load_all_data skips them.
// NULL when sharing is not possible (the caller then allocates a private buffer as usual).
ggml_backend_buffer_t llama_shared_pinned_alloc(llama_model_loader & ml, llama_model & model, ggml_context * ctx,
        ggml_backend_buffer_type_t buft);

// called after load_all_data(ctx): marks the segment of ctx ready if this process created it
void llama_shared_pinned_mark_ready(llama_model & model, ggml_context * ctx);
