// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/norm_weight.h"

#include <vector>

namespace turbomind::core {

struct MtpWeightConfig: ModuleConfig {
    MtpWeightConfig(): ModuleConfig{"MtpWeight"} {}

#define MTP_WEIGHT_FIELDS(X)                                                                                           \
    X(int, tp_size)                                                                                                    \
    X(int, tp_rank)                                                                                                    \
    X(DataType, data_type)                                                                                             \
    X(int, hidden_units)

    MTP_WEIGHT_FIELDS(TM_MEMBER)
    TM_FOR_EACH(MtpWeightConfig, MTP_WEIGHT_FIELDS)

#undef MTP_WEIGHT_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

class DecoderLayerWeight;

/// MTP (multi-token-prediction) draft head weight tree (M4).
///
/// Shape (Qwen3.5 native MTP, 1 layer):
///   hidden = norm_embedding(embed(token)) ; prev = norm_hidden(prev_hidden)
///   h = fc(concat(hidden, prev)) -> decoder layer -> norm -> shared lm_head
///
/// Shares the target's `tok_embeddings` and `output` (lm_head); only `fc`, the
/// three norms and the one decoder layer are owned here.
class MtpWeight: public core::Module {
public:
    const char* type() const override
    {
        return "MtpWeight";
    }

    MtpWeight() = default;

    explicit MtpWeight(const core::MtpWeightConfig& cfg);

    void prepare() override;
    bool verify(std::vector<std::string>& missing) override;

#define MTP_WEIGHT_CHILDREN(X)                                                                                         \
    X(LinearWeight, fc)                                                                                                \
    X(NormWeight, pre_fc_norm_embedding)                                                                               \
    X(NormWeight, pre_fc_norm_hidden)                                                                                  \
    X(NormWeight, norm)                                                                                                \
    X(core::ModuleList, layers)

#define MTP_WEIGHT_PARAMS(X)

    TM_MODULE_DECLARE(MtpWeight, MTP_WEIGHT_CHILDREN, MTP_WEIGHT_PARAMS)

    DecoderLayerWeight* layer(int i) const;

    // --- Derived in prepare() ---
    DataType data_type{};
    int      hidden_units{};
    int      num_layer{};

    // --- From MtpWeightConfig at construction ---
    int tp_size{};
    int tp_rank{};
};

}  // namespace turbomind
