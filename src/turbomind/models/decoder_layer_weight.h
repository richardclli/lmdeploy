// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/module.h"

namespace turbomind::core {

struct DecoderLayerConfig: ModuleConfig {
    DecoderLayerConfig(): ModuleConfig{"DecoderLayerWeight"} {}

#define DECODER_LAYER_CONFIG_FIELDS(X) X(DataType, data_type)

    DECODER_LAYER_CONFIG_FIELDS(TM_MEMBER)
    TM_FOR_EACH(DecoderLayerConfig, DECODER_LAYER_CONFIG_FIELDS)

#undef DECODER_LAYER_CONFIG_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

class AttentionWeight;
class DeltaNetWeight;
class FfnWeight;
class MoeWeight;
class NormWeight;
class GroupedConvWeight;
class LinearWeight;

/// Architecture-independent decoder layer weight composite.
class DecoderLayerWeight: public core::Module {
public:
    const char* type() const override
    {
        return "DecoderLayerWeight";
    }

    DecoderLayerWeight() = default;
    DecoderLayerWeight(const core::ModuleConfig&);

    ~DecoderLayerWeight() override;  // defined in .cc where child types are complete

    bool verify(std::vector<std::string>& missing) override;

    // --- X-macro field lists ---
#define DECODER_LAYER_WEIGHT_CHILDREN(X)                                                                               \
    X(AttentionWeight, attention)                                                                                      \
    X(DeltaNetWeight, linear_attn)                                                                                     \
    X(FfnWeight, feed_forward)                                                                                         \
    X(MoeWeight, moe_ffn)                                                                                              \
    X(NormWeight, attention_norm)                                                                                      \
    X(NormWeight, ffn_norm)                                                                                            \
    X(GroupedConvWeight, attention_conv)                                                                              \
    X(GroupedConvWeight, mlp_conv)                                                                                    \
    X(LinearWeight, bridge_read)                                                                                      \
    X(LinearWeight, bridge_write)

#define DECODER_LAYER_WEIGHT_PARAMS(X)

    TM_MODULE_DECLARE(DecoderLayerWeight, DECODER_LAYER_WEIGHT_CHILDREN, DECODER_LAYER_WEIGHT_PARAMS)
};

}  // namespace turbomind
