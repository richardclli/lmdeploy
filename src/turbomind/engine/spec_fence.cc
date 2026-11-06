// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/engine/spec_fence.h"

#include "src/turbomind/core/context.h"
#include "src/turbomind/core/logger.h"

namespace turbomind {

void SpecFence(const char* name)
{
#ifdef TM_SPEC_FENCE_STRICT
    core::Context::stream().Sync();
    TM_LOG_INFO("[SpecFence] {}", name);
#else
    (void)name;
#endif
}

}  // namespace turbomind
