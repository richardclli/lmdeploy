// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/speculative/dspine/dspine_weight.h"

#include "src/turbomind/core/registry.h"

namespace turbomind {

TM_MODULE_REGISTER(DspineWeight, core::DspineWeightConfig);
TM_MODULE_METHODS(DspineWeight, DSPINE_WEIGHT_CHILDREN, DSPINE_WEIGHT_PARAMS)

}  // namespace turbomind
