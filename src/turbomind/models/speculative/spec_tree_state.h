// Copyright (c) OpenMMLab. All rights reserved.
//
// D6/M13 env-gated spike: process-global tree shape used by the verification
// attention. Set once by the spec model when TM_SPEC_TREE is present; read by
// UnifiedAttentionLayer::MakeVerificationArguments. Unset ⇒ null ⇒ chain path.
#pragma once

namespace turbomind {

// Host parent map (node_parent[p] = parent node of p, -1 = root/context). Read by
// UnifiedAttentionLayer::MakeVerificationArguments, which copies it by value into
// the verification kernel arguments. node_stride == node_count + 1.
inline const int* g_spec_node_parent = nullptr;
inline int        g_spec_node_stride = 0;
inline int        g_spec_node_count  = 0;  // number of draft nodes (excludes bonus leaves)

// Whether the draft-side tree mask (mainloop_sm70.h ApplyMask) should be fed the
// per-request tree origin by the draft decode. Set by the spec model.
inline bool g_spec_draft_tree_mask = false;

}  // namespace turbomind
