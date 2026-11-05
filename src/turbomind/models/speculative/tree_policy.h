// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/models/speculative/speculative_policy.h"

#include <vector>

namespace turbomind {

/// Fixed-shape tree verification.
///
/// Two shapes, both env-selected:
///  * `DisjointChains(M, d)`: M chains of length d laid out side by side
///    (node `m*d + j`, parent `m*d + j - 1`, chain heads -> root).
///  * `TailBranch(depth, width)`: a length-`(depth-1)` chain (nodes
///    `0..depth-2`) with `width` sibling continuations at the last level
///    (nodes `depth-1 .. depth-2+width`, all children of `depth-2`). This is a
///    degenerate tree that needs **no** draft-side mask: the siblings are
///    leaves, so the draft still drafts one greedy chain and the extra
///    candidates are pure verification rows.
///
/// `parent()` is index-aligned with the verification query rows: query `p`
/// attends to the context plus its ancestors-or-self, matching the causal
/// chain when `parent[p] = p-1`.
class TreePolicy final: public SpeculativePolicy {
public:
    static TreePolicy DisjointChains(int chain_count, int depth);
    static TreePolicy TailBranch(int depth, int width);

    int max_proposals() const override;

    int token_row_tail() const override;

    RoundExtent Extent(const RoundRequest& request) const override;

    std::optional<BootstrapExtent> Bootstrap(int prompt_len) const override;

    const std::vector<int>& parent() const
    {
        return parent_;
    }

    int node_count() const
    {
        return node_count_;
    }

    int depth() const
    {
        return depth_;
    }

    int width() const
    {
        return width_;
    }

    bool is_tail_branch() const
    {
        return tail_branch_;
    }

private:
    TreePolicy() = default;

    int              node_count_{};
    int              depth_{};
    int              width_{};
    bool             tail_branch_{};
    std::vector<int> parent_;  // node_count_ + 1 entries (last = bonus row)
};

}  // namespace turbomind
