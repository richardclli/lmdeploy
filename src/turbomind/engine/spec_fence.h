// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

namespace turbomind {

// A speculative-round phase boundary (M2 fences F1-F7). Within a single stream
// FIFO order already orders the phases, so the production fence is a no-op
// marker; the `TM_SPEC_FENCE_STRICT` build turns it into a stream sync plus a
// trace line to bisect cross-stream races between round phases.
void SpecFence(const char* name);

}  // namespace turbomind
