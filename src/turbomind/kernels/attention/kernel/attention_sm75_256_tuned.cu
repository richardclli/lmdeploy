// Copyright (c) OpenMMLab. All rights reserved.
//
// SM75 (Turing) prefill attention variants for head_dim = 256, selected at
// registration time via the TM_ATTN_SM75 environment variable.
//
// Rationale: the stock `attention_sm75_256.cu` tile (CTA_Q=64, CTA_S=64) needs
// >255 registers/thread at head_dim 256 and spills ~2.8 KB (causal), capping
// warp occupancy and yielding only ~11 TFLOPS on the causal prefill path.
// Shrinking the KV tile to CTA_S=32 removes most of the spill (see below) and,
// with CTA_Q=64, halves the shared memory footprint.
//
// Registered *before* `attention_sm75_256.cu` in the CMake source order; the
// prefill path in Registry::Find returns the first match, so an active variant
// wins and the stock kernel is used otherwise. No `AttnDesc`/`Registry::Find`
// behavior is changed.

#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/cta_map.h"
#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/impl_1688.h"
#include "src/turbomind/kernels/attention/linear_iterator.h"
#include "src/turbomind/kernels/attention/mainloop_sm70.h"
#include "src/turbomind/kernels/attention/registrar.h"

#include <cstdlib>
#include <cstring>

namespace turbomind::attention {

namespace {

constexpr int kHeadDim = 256;
constexpr int kWarp_Q  = 16;

template<int CTA_Q, int CTA_S>
struct Tuned {
    template<class T, bool Causal = true>
    using KT = AttentionUniversal<arch::Sm75,
                                  Mainloop<arch::Sm70,
                                           Impl<MMA_1688, T, T, 1, CTA_Q, CTA_S, 1, kWarp_Q, CTA_S, kHeadDim, 2>>,
                                  LinearIteratorFactory<T, CTA_S, kHeadDim>,
                                  AttentionCtaMap,
                                  Causal>;
};

const char* selector()
{
    return std::getenv("TM_ATTN_SM75");
}

bool selected(const char* name)
{
    const char* s = selector();
    return s && std::strcmp(s, name) == 0;
}

}  // namespace

// CTA_Q=64, CTA_S=32: 32 KB smem (2 CTAs/SM by smem), ~0.75 KB causal spill.
Registrar reg_sm75_256_t32q64([](Collector& c) {
    if (selected("t32q64")) {
        c.add<Tuned<64, 32>::KT<half>>();
        c.add<Tuned<64, 32>::KT<half, false>>();
    }
});

// CTA_Q=128, CTA_S=32: 64 KB smem (1 CTA/SM), ~0.3 KB causal spill.
Registrar reg_sm75_256_t32q128([](Collector& c) {
    if (selected("t32q128")) {
        c.add<Tuned<128, 32>::KT<half>>();
        c.add<Tuned<128, 32>::KT<half, false>>();
    }
});

}  // namespace turbomind::attention
