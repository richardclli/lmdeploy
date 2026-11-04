// Copyright (c) OpenMMLab. All rights reserved.
//
// D6-P0 debug probe: env-gated dump of the speculative draft/target logits so
// top-k coverage can be measured offline. No effect unless TM_SPEC_DUMP_DIR is
// set; compiled into all builds but inert by default.
#pragma once

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace turbomind {

inline std::atomic<int>  g_spec_probe_round{0};
inline std::atomic<long> g_spec_probe_event{0};

inline bool SpecProbeEnabled()
{
    static const bool enabled = [] {
        const char* dir = std::getenv("TM_SPEC_DUMP_DIR");
        return dir != nullptr && *dir != '\0';
    }();
    return enabled;
}

// Append one logits block: header line "R <round> <pos> <rows> <cols> <dtype>"
// followed by rows*cols elements of `dtype` (raw bytes).
inline void SpecProbeDump(const char*         file,
                          int                 round,
                          int                 pos,
                          const void*         device,
                          int                 rows,
                          int                 cols,
                          size_t              element_bytes,
                          const char*         dtype,
                          cudaStream_t        stream)
{
    if (!SpecProbeEnabled() || rows <= 0 || cols <= 0) {
        return;
    }
    const char* dir = std::getenv("TM_SPEC_DUMP_DIR");
    std::string path = std::string(dir) + "/" + file;

    std::vector<char> host(static_cast<size_t>(rows) * cols * element_bytes);
    cudaMemcpyAsync(host.data(), device, host.size(), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    const long event = g_spec_probe_event.fetch_add(1);

    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    std::ofstream f(path, std::ios::binary | std::ios::app);
    char header[256];
    int  n = std::snprintf(header, sizeof(header), "R %ld %d %d %d %d %s\n", event, round, pos, rows, cols, dtype);
    f.write(header, n);
    f.write(host.data(), static_cast<std::streamsize>(host.size()));
}

}  // namespace turbomind
