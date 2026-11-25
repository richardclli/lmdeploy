// Copyright (c) OpenMMLab. All rights reserved.
//
// DSpine speculative method: a DFlash-family draft (5-layer Qwen3, causal
// sliding-window block attention) with a per-layer code bridge and a serial
// last-write refinement tail. Greedy, single-request, mirroring the in-tree
// DFlash2 path's contract.
#pragma once

#include "src/turbomind/models/language_model.h"
#include "src/turbomind/models/speculative/speculative_model.h"
#include "src/turbomind/models/speculative/speculative_policy.h"

#include <memory>
#include <vector>

namespace turbomind {

class DspineWeight;
class TargetHiddenProjection;
class Context;
class NormWeight;

/// Block policy: one block of `n_proposals + 1` query rows (`[next, mask x n]`),
/// no private cache tail.
class DspinePolicy final: public SpeculativePolicy {
public:
    explicit DspinePolicy(int n_proposals): n_{n_proposals} {}

    int max_proposals() const override;
    int token_row_tail() const override;
    RoundExtent Extent(const RoundRequest& request) const override;
    std::optional<BootstrapExtent> Bootstrap(int prompt_len) const override;

private:
    int n_{};
};

class DspineModel final: public SpeculativeModel {
public:
    explicit DspineModel(const SpeculativeModelArgs& args);
    ~DspineModel() override;

    const SpeculativePolicy& policy() const override;
    void Run(BatchOp op, int phase, TensorMap& env) override;
    HiddenStateTap* Tap(int phase) override;
    void RunDraft(int phase, const DraftContext& ctx, TensorMap& env) override;

private:
    const DspineWeight&     spec_weights_;
    DspinePolicy            policy_;
    LanguageModel           draft_;
    std::unique_ptr<TargetHiddenProjection> projection_;

    const Context&     ctx_;
    const EngineParam& param_;
    LanguageModel&     target_;
    int   hidden_{}, hd_{}, local_nh_{}, local_nkv_{}, total_heads_{}, inter_local_{};
    int   top_k_{}, rank_{}, code_size_{}, block_{}, mask_token_id_{}, window_{}, max_ctx_{};
    float rope_base_{};
    int   dtype_code_{};
    float bridge_eps_{};
    int   causal_{1};
    int   no_delta_{0};
    int   top1_{0};

    std::vector<Tensor> ctx_k_;
    std::vector<Tensor> ctx_v_;
    Buffer_<int>        ctx_len_;

    // Last bridge write's pre-write state, consumed by the refinement tail.
    Tensor last_raw_;
    Tensor last_packed_;
    Buffer_<float> last_rms_;

    void MaterializeContext(const Tensor& context_hidden, const Buffer_<int>& base_dev,
                            const Buffer_<int>& committed_dev, int rows, cudaStream_t stream);
    Tensor RunBlock(const Tensor& block_embeddings, const Tensor& positions, const Tensor& ids, int rows,
                    int ctx_rows, cudaStream_t stream, TensorMap& env);
};

}  // namespace turbomind
