// Copyright (c) OpenMMLab. All rights reserved.
//
// D6/M13 env-gated spike: process-global tree shape used by the verification
// attention. Set once by the spec model when TM_SPEC_TREE is present; read by
// UnifiedAttentionLayer::MakeVerificationArguments. Unset ⇒ null ⇒ chain path.
#pragma once

namespace turbomind {

// Device buffer [max_batch * node_stride]; node_parent[r*stride + p] = parent
// query row of node p (-1 = root/context). node_stride == node_count + 1.
inline const int* g_spec_node_parent = nullptr;
inline int        g_spec_node_stride = 0;

}  // namespace turbomind
