// Copyright (c) OpenMMLab. All rights reserved.
//
// DSpine draft weight tree (TurboMind). Beyond the plain draft decoder layers:
//   * fc          : context projection (n_tap * hidden -> hidden)
//   * hidden_norm : context RMSNorm
//   * packed      : code-bridge [gate; predecessor_gate; prev] C -> 3R projection
//   * gate_bias   : code-bridge gate bias [R]
//   * codebook    : transfer-space codebook [vocab, C]
//   * tg / tsv    : refinement tables [vocab, R] and [vocab, 2R]
//
// Per-layer bridge read/write linears live on the draft's DecoderLayerWeight
// (`bridge_read` / `bridge_write`), like DFlash2's grouped convs.
#pragma once

#include "src/turbomind/core/module.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/norm_weight.h"

namespace turbomind::core {

struct DspineWeightConfig: ModuleConfig {
    DspineWeightConfig(): ModuleConfig{"DspineWeight"} {}

#define DSPINE_WEIGHT_FIELDS(X) X(DataType, data_type)

    DSPINE_WEIGHT_FIELDS(TM_MEMBER)
    TM_FOR_EACH(DspineWeightConfig, DSPINE_WEIGHT_FIELDS)

#undef DSPINE_WEIGHT_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

class DspineWeight: public core::Module {
public:
    const char* type() const override
    {
        return "DspineWeight";
    }

    DspineWeight() = default;
    explicit DspineWeight(const core::DspineWeightConfig&) {}

#define DSPINE_WEIGHT_CHILDREN(X)                                                                                      \
    X(LinearWeight, fc)                                                                                                \
    X(NormWeight, hidden_norm)                                                                                         \
    X(LinearWeight, packed)

#define DSPINE_WEIGHT_PARAMS(X) X(gate_bias) X(codebook) X(tg) X(tsv)

    TM_MODULE_DECLARE(DspineWeight, DSPINE_WEIGHT_CHILDREN, DSPINE_WEIGHT_PARAMS)
};

}  // namespace turbomind
