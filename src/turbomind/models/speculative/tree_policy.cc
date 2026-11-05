// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/speculative/tree_policy.h"

#include <stdexcept>

namespace turbomind {

TreePolicy TreePolicy::DisjointChains(int chain_count, int depth)
{
    if (chain_count <= 0 || depth <= 0) {
        throw std::invalid_argument("TreePolicy: chain_count and depth must be positive");
    }
    TreePolicy p;
    p.node_count_  = chain_count * depth;
    p.depth_       = depth;
    p.width_       = chain_count;
    p.tail_branch_ = false;
    // Layout: draft nodes `m*depth + j` (chain `m`, level `j`), followed by one
    // bonus leaf per chain `node_count + m` (child of that chain's last node).
    // The bonus leaf's row predicts the token after a fully-accepted chain, so
    // the bonus is path-conditioned instead of using a single shared row.
    // For chain_count == 1 this is exactly the linear chain (drafts 0..d-1 plus
    // one bonus row).
    const int positions = p.node_count_ + chain_count;
    p.parent_.resize(positions);
    for (int m = 0; m < chain_count; ++m) {
        for (int j = 0; j < depth; ++j) {
            p.parent_[m * depth + j] = (j == 0) ? -1 : m * depth + j - 1;
        }
    }
    for (int m = 0; m < chain_count; ++m) {
        p.parent_[p.node_count_ + m] = m * depth + (depth - 1);  // bonus leaf
    }
    return p;
}

TreePolicy TreePolicy::TailBranch(int depth, int width)
{
    if (depth <= 1 || width <= 0) {
        throw std::invalid_argument("TreePolicy: tail branch needs depth>1 and width>0");
    }
    TreePolicy p;
    const int prefix = depth - 1;  // chain nodes 0..prefix-1
    p.node_count_    = prefix + width;
    p.depth_         = depth;
    p.width_         = width;
    p.tail_branch_   = true;
    p.parent_.resize(p.node_count_ + 1);
    for (int j = 0; j < prefix; ++j) {
        p.parent_[j] = (j == 0) ? -1 : j - 1;
    }
    for (int k = 0; k < width; ++k) {
        p.parent_[prefix + k] = prefix - 1;  // siblings share the last prefix node
    }
    p.parent_[p.node_count_] = prefix - 1;  // bonus row (derived from winner in the walk)
    return p;
}

int TreePolicy::max_proposals() const
{
    return node_count_;
}

int TreePolicy::token_row_tail() const
{
    return 2;
}

RoundExtent TreePolicy::Extent(const RoundRequest&) const
{
    // query rows = draft nodes + one bonus leaf per chain; cache private tail = drafts - 1.
    return {node_count_ + width_, node_count_ - 1};
}

std::optional<BootstrapExtent> TreePolicy::Bootstrap(int prompt_len) const
{
    return BootstrapExtent{prompt_len + node_count_ - 1, prompt_len + 2 * node_count_};
}

}  // namespace turbomind
