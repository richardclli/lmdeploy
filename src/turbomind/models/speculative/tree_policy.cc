// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/speculative/tree_policy.h"

#include <stdexcept>

namespace turbomind {

TreePolicy::TreePolicy(int chain_count, int depth): chain_count_{chain_count}, depth_{depth}
{
    if (chain_count_ <= 0 || depth_ <= 0) {
        throw std::invalid_argument("TreePolicy: chain_count and depth must be positive");
    }
    const int v = node_count();
    parent_.resize(v + 1);
    for (int m = 0; m < chain_count_; ++m) {
        for (int j = 0; j < depth_; ++j) {
            parent_[m * depth_ + j] = (j == 0) ? -1 : m * depth_ + j - 1;
        }
    }
    // Trailing bonus query row: causal over the last chain (anchor-equivalent
    // to the fixed chain when chain_count_ == 1).
    parent_[v] = v - 1;
}

int TreePolicy::max_proposals() const
{
    return node_count();
}

int TreePolicy::token_row_tail() const
{
    return 2;
}

RoundExtent TreePolicy::Extent(const RoundRequest&) const
{
    const int v = node_count();
    return {v + 1, v - 1};
}

std::optional<BootstrapExtent> TreePolicy::Bootstrap(int prompt_len) const
{
    const int v = node_count();
    return BootstrapExtent{prompt_len + v - 1, prompt_len + 2 * v};
}

}  // namespace turbomind
