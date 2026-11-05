// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/models/speculative/speculative_policy.h"

#include <vector>

namespace turbomind {

/// Fixed-shape tree verification. The proposal "tree" is a shallow set of
/// disjoint chains of length `depth`; node `m*depth + j` is the j-th token of
/// chain m. Draft query node `p` attends to the context plus its ancestors
/// (block-diagonal), encoded by `parent()`.
///
/// The engine schedules `node_count()` query rows and consumes the accepted
/// path exactly as it does for `FixedChainPolicy`; `parent()` is handed to the
/// verification attention (`Arguments::node_parent`).
class TreePolicy final: public SpeculativePolicy {
public:
    TreePolicy(int chain_count, int depth);

    int max_proposals() const override;

    int token_row_tail() const override;

    RoundExtent Extent(const RoundRequest& request) const override;

    std::optional<BootstrapExtent> Bootstrap(int prompt_len) const override;

    /// node_count() = chain_count * depth; one parent per query row (0..V-1)
    /// plus the trailing bonus row (index V). -1 = root (context only).
    const std::vector<int>& parent() const
    {
        return parent_;
    }

    int node_count() const
    {
        return chain_count_ * depth_;
    }

    int depth() const
    {
        return depth_;
    }

    int chain_count() const
    {
        return chain_count_;
    }

private:
    int              chain_count_;
    int              depth_;
    std::vector<int> parent_;
};

}  // namespace turbomind
