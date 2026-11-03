// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/mtp_weight.h"
#include "src/turbomind/core/registry.h"
#include "src/turbomind/models/decoder_layer_weight.h"

namespace turbomind {

MtpWeight::MtpWeight(const core::MtpWeightConfig& cfg):
    tp_size(cfg.tp_size), tp_rank(cfg.tp_rank), data_type(cfg.data_type), hidden_units(cfg.hidden_units)
{
}

void MtpWeight::prepare()
{
    for_each_child([this](const char* /*name*/, Module* child) {
        if (child) {
            child->prepare();
        }
    });
    num_layer = layers ? layers->size() : 0;
}

DecoderLayerWeight* MtpWeight::layer(int i) const
{
    if (!layers) {
        return nullptr;
    }
    return static_cast<DecoderLayerWeight*>(layers->child(std::to_string(i)));
}

bool MtpWeight::verify(std::vector<std::string>& missing)
{
    Module::verify(missing);
    if (!fc) {
        missing.push_back(full_path() + ": missing fc");
    }
    if (!layers) {
        missing.push_back(full_path() + ": missing layers");
    }
    return missing.empty();
}

TM_MODULE_REGISTER(MtpWeight, core::MtpWeightConfig);

TM_MODULE_METHODS(MtpWeight, MTP_WEIGHT_CHILDREN, MTP_WEIGHT_PARAMS)

}  // namespace turbomind
