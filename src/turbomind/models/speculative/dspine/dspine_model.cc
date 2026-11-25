// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/speculative/dspine/dspine_model.h"

#include <algorithm>
#include <cmath>

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/kernels/copy/copy.h"
#include "src/turbomind/kernels/dflash2/dflash2_kernels.h"
#include "src/turbomind/kernels/dspine/dspine_kernels.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/attention_weight.h"
#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/models/ffn_weight.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/model_weight.h"
#include "src/turbomind/models/norm_weight.h"
#include "src/turbomind/models/speculative/dspine/dspine_weight.h"
#include "src/turbomind/models/speculative/eagle3/target_hidden_projection.h"
#include "src/turbomind/models/speculative/registry.h"

namespace turbomind {

int DspinePolicy::max_proposals() const
{
    return n_;
}

int DspinePolicy::token_row_tail() const
{
    return n_ + 1;
}

RoundExtent DspinePolicy::Extent(const RoundRequest&) const
{
    return {n_ + 1, 0};
}

std::optional<BootstrapExtent> DspinePolicy::Bootstrap(int prompt_len) const
{
    return BootstrapExtent{prompt_len + n_, prompt_len + 2 * n_ + 1};
}

DspineModel::DspineModel(const SpeculativeModelArgs& args):
    spec_weights_{*TM_CHECK_NOTNULL(args.draft_weights.get<DspineWeight>("spec"))},
    policy_{args.param.spec_num_draft_tokens},
    draft_{args.registry, args.param, args.ctx, args.draft_weights, args.phases},
    projection_{std::make_unique<TargetHiddenProjection>(args.param,
                                                         args.ctx,
                                                         args.phases,
                                                         args.target.weights().num_layer,
                                                         args.target.weights().hidden_units,
                                                         args.target.weights().data_type,
                                                         args.param.spec_tap_layer_ids,
                                                         *spec_weights_.fc)},
    ctx_{args.ctx},
    param_{args.param},
    target_{args.target}
{
    const auto& dw = draft_.weights();
    hidden_        = dw.hidden_units;
    const auto* a0 = dw.layer(0)->attention.get();
    hd_            = a0->head_dim;
    local_nh_      = a0->head_num / a0->tp_size;
    local_nkv_     = a0->kv_head_num / a0->tp_size;
    total_heads_   = local_nh_ + 2 * local_nkv_;
    const auto* f0 = dw.layer(0)->feed_forward.get();
    if (f0->w1 && f0->w1->output_dim) {
        inter_local_ = f0->w1->output_dim;
    }
    else if (f0->w1w3 && f0->w1w3->output_dim) {
        inter_local_ = f0->w1w3->output_dim / 2;
    }
    else {
        inter_local_ = f0->inter_size / (f0->tp_size ? f0->tp_size : 1);
    }

    top_k_         = param_.spec_selector_top_k ? param_.spec_selector_top_k : 16;
    rank_          = param_.spec_selector_rank ? param_.spec_selector_rank : 512;
    code_size_     = param_.spec_code_size ? param_.spec_code_size : 1024;
    block_         = param_.spec_block_size ? param_.spec_block_size : 16;
    mask_token_id_ = param_.spec_mask_token_id;
    window_        = param_.spec_sliding_window ? param_.spec_sliding_window : 8192;
    rope_base_     = 1e7f;
    max_ctx_       = param_.session_len;
    bridge_eps_    = spec_weights_.hidden_norm->norm_eps_;

    if (dw.data_type == kFloat16) {
        dtype_code_ = 1;
    }
    else if (dw.data_type == kBfloat16) {
        dtype_code_ = 0;
    }
    else {
        dtype_code_ = 2;
    }
    TM_CHECK(dtype_code_ != 0) << "DSpine draft requires an fp16 or fp32 engine dtype in v1";

    const DataType dtype  = dw.data_type;
    ctx_k_.resize(dw.num_layer);
    ctx_v_.resize(dw.num_layer);
    const size_t bytes = (size_t)max_ctx_ * local_nkv_ * hd_ * byte_size(dtype);
    for (int l = 0; l < dw.num_layer; ++l) {
        ctx_k_[l] = Tensor{{max_ctx_, local_nkv_, hd_}, dtype, kDEVICE};
        ctx_v_[l] = Tensor{{max_ctx_, local_nkv_, hd_}, dtype, kDEVICE};
        cudaMemsetAsync(ctx_k_[l].raw_data(), 0, bytes, 0);
        cudaMemsetAsync(ctx_v_[l].raw_data(), 0, bytes, 0);
    }
    cudaDeviceSynchronize();
    ctx_len_ = {param_.max_batch_size, kDEVICE};
    cudaMemsetAsync(ctx_len_.data(), 0, param_.max_batch_size * sizeof(int));

    last_raw_    = Tensor{{block_, hidden_}, dtype, kDEVICE};
    last_packed_ = Tensor{{block_, 3 * rank_}, dtype, kDEVICE};
    last_rms_    = {block_, kDEVICE};
}

DspineModel::~DspineModel() = default;

const SpeculativePolicy& DspineModel::policy() const
{
    return policy_;
}

void DspineModel::Run(BatchOp, int, TensorMap&) {}

HiddenStateTap* DspineModel::Tap(int)
{
    return projection_.get();
}

void DspineModel::MaterializeContext(const Tensor& context_hidden, const Buffer_<int>& base_dev,
                                     const Buffer_<int>& committed_dev, int rows, cudaStream_t stream)
{
    const auto& dw  = draft_.weights();
    auto&       lin = *ctx_.linear;
    const float eps = spec_weights_.hidden_norm->norm_eps_;
    const bool  zc  = spec_weights_.hidden_norm->zero_centered_;

    Buffer_<int> pos_dev{rows, kDEVICE};
    invokeDflash2CountUp(pos_dev.data(), base_dev.data(), rows, stream);

    Tensor x{{rows, hidden_}, dw.data_type, kDEVICE};
    invokeRMSNorm(x, context_hidden, spec_weights_.hidden_norm->weight, eps, zc, stream);

    const int per_row = local_nkv_ * hd_;
    Tensor    qkv2{{rows, total_heads_ * hd_}, dw.data_type, kDEVICE};
    for (int l = 0; l < dw.num_layer; ++l) {
        const auto* a = dw.layer(l)->attention.get();
        lin.Forward(x, *a->w_qkv, qkv2);
        Tensor qkv = qkv2.view({rows, total_heads_, hd_});
        invokeQkRMSNorm(qkv,
                        a->q_norm->weight,
                        a->k_norm->weight,
                        local_nh_,
                        local_nkv_,
                        a->q_norm->norm_eps_,
                        a->q_norm->zero_centered_,
                        stream);
        Tensor krow = qkv.slice({0, local_nh_, 0}, {rows, local_nkv_, hd_});
        Tensor vrow = qkv.slice({0, local_nh_ + local_nkv_, 0}, {rows, local_nkv_, hd_});
        Tensor kc{{rows, local_nkv_, hd_}, dw.data_type, kDEVICE};
        Tensor vc{{rows, local_nkv_, hd_}, dw.data_type, kDEVICE};
        GenericCopy(krow, kc, stream);
        GenericCopy(vrow, vc, stream);
        invokeDflash2Rope(kc.raw_data(), pos_dev.data(), rows, local_nkv_, hd_, rope_base_, 16, stream);
        invokeDflash2KvScatter(ctx_k_[l].raw_data(), kc.raw_data(), base_dev.data(), committed_dev.data(), rows,
                               per_row, 16, stream);
        invokeDflash2KvScatter(ctx_v_[l].raw_data(), vc.raw_data(), base_dev.data(), committed_dev.data(), rows,
                               per_row, 16, stream);
    }
}

Tensor DspineModel::RunBlock(const Tensor& block_embeddings, const Tensor& positions, const Tensor& ids, int rows,
                             int ctx_rows, cudaStream_t stream, TensorMap& env)
{
    const auto&    dw    = draft_.weights();
    auto&          lin   = *ctx_.linear;
    const DataType dtype = dw.data_type;
    const float    qscale = 1.0f / std::sqrt((float)hd_);
    const float    sqrt_c = std::sqrt((float)code_size_);
    const int      R      = rank_;

    Tensor x{{rows, hidden_}, dtype, kDEVICE};
    Tensor red{{rows, hidden_}, dtype, kDEVICE};
    Tensor qkv2{{rows, total_heads_ * hd_}, dtype, kDEVICE};
    Tensor attn{{rows, local_nh_, hd_}, dtype, kDEVICE};
    Tensor o{{rows, hidden_}, dtype, kDEVICE};
    Tensor h{{rows, hidden_}, dtype, kDEVICE};
    Tensor gating{{rows, inter_local_}, dtype, kDEVICE};
    Tensor inter{{rows, inter_local_}, dtype, kDEVICE};
    Tensor down{{rows, hidden_}, dtype, kDEVICE};
    Tensor cur_out{{rows, hidden_}, dtype, kDEVICE};

    Tensor         normalized{{rows, hidden_}, dtype, kDEVICE};
    Tensor         projected{{rows, code_size_}, dtype, kDEVICE};
    Tensor         codes{{rows, code_size_}, dtype, kDEVICE};
    Tensor         packed{{rows, 3 * R}, dtype, kDEVICE};
    Tensor         message{{rows, R}, dtype, kDEVICE};
    Tensor         delta{{rows, hidden_}, dtype, kDEVICE};
    Buffer_<float> rms{rows, kDEVICE};

    auto allreduce = [&](Tensor& t) {
        if (ctx_.comm.d_comm) {
            ctx_.comm.d_comm->AllReduceSum(
                t.raw_data(), red.raw_data(), (size_t)rows * hidden_, dtype, ctx_.comm.d_tp_group, stream);
            Copy(red, t);
        }
    };

    const NormWeight* n0 = dw.layer(0)->attention_norm.get();
    invokeRMSNorm(x, block_embeddings, n0->weight, n0->norm_eps_, n0->zero_centered_, stream);
    Tensor residual = block_embeddings;

    for (int l = 0; l < dw.num_layer; ++l) {
        auto*       layer = dw.layer(l);
        const auto* a     = layer->attention.get();
        const auto* ff    = layer->feed_forward.get();

        lin.Forward(x, *a->w_qkv, qkv2);
        Tensor qkv = qkv2.view({rows, total_heads_, hd_});
        invokeQkRMSNorm(qkv, a->q_norm->weight, a->k_norm->weight, local_nh_, local_nkv_, a->q_norm->norm_eps_,
                        a->q_norm->zero_centered_, stream);
        Tensor qv = qkv.slice({0, 0, 0}, {rows, local_nh_, hd_});
        Tensor kv = qkv.slice({0, local_nh_, 0}, {rows, local_nkv_, hd_});
        Tensor vv = qkv.slice({0, local_nh_ + local_nkv_, 0}, {rows, local_nkv_, hd_});
        Tensor q2{{rows, local_nh_, hd_}, dtype, kDEVICE};
        Tensor k2{{rows, local_nkv_, hd_}, dtype, kDEVICE};
        Tensor v2{{rows, local_nkv_, hd_}, dtype, kDEVICE};
        GenericCopy(qv, q2, stream);
        GenericCopy(kv, k2, stream);
        GenericCopy(vv, v2, stream);
        invokeDflash2Rope(q2.raw_data(), positions.data<int>(), rows, local_nh_, hd_, rope_base_, 16, stream);
        invokeDflash2Rope(k2.raw_data(), positions.data<int>(), rows, local_nkv_, hd_, rope_base_, 16, stream);
        Tensor kctx = ctx_k_[l].slice({0, 0, 0}, {ctx_rows, local_nkv_, hd_});
        Tensor vctx = ctx_v_[l].slice({0, 0, 0}, {ctx_rows, local_nkv_, hd_});
        invokeDspineAttention(attn.raw_data(),
                              q2.raw_data(),
                              kctx.raw_data(),
                              vctx.raw_data(),
                              k2.raw_data(),
                              v2.raw_data(),
                              rows,
                              ctx_rows,
                              local_nh_,
                              local_nkv_,
                              hd_,
                              qscale,
                              window_,
                              dtype_code_,
                              stream);
        lin.Forward(attn.view({rows, local_nh_ * hd_}), *a->wo, o);
        allreduce(o);
        invokeDflash2Add(h.raw_data(), residual.raw_data(), o.raw_data(), (long long)rows * hidden_, 16, stream);

        invokeRMSNorm(x, h, layer->ffn_norm->weight, layer->ffn_norm->norm_eps_, layer->ffn_norm->zero_centered_,
                      stream);
        if (ff->w1 && ff->w3) {
            lin.Forward(x, *ff->w1, gating);
            lin.Forward(x, *ff->w3, inter);
            Activation(gating, inter, ff->act_type, stream);
            lin.Forward(gating, *ff->w2, down);
        }
        else if (ff->w1w3) {
            Tensor gate_up{{rows, 2 * inter_local_}, dtype, kDEVICE};
            lin.Forward(x, *ff->w1w3, gate_up);
            gating = gate_up.slice({0, 0}, {rows, inter_local_});
            inter  = gate_up.slice({0, inter_local_}, {rows, inter_local_});
            Activation(gating, inter, ff->act_type, stream);
            lin.Forward(gating, *ff->w2, down);
        }
        else {
            TM_CHECK(false) << "DSpine draft layer has no feed-forward weights";
        }
        allreduce(down);
        invokeDflash2Add(cur_out.raw_data(), h.raw_data(), down.raw_data(), (long long)rows * hidden_, 16, stream);

        const NormWeight* next_norm = (l + 1 < dw.num_layer) ? dw.layer(l + 1)->attention_norm.get() : dw.norm.get();

        invokeDspineRead(cur_out.raw_data(), normalized.raw_data(), rms.data(), rows, hidden_, bridge_eps_,
                         dtype_code_, stream);
        lin.Forward(normalized, *layer->bridge_read, projected);
        invokeDspineCodes(projected.raw_data(), codes.raw_data(), spec_weights_.codebook.raw_data(), ids.data<int>(),
                          rows, code_size_, block_, sqrt_c, bridge_eps_, dtype_code_, stream);
        lin.Forward(codes, *spec_weights_.packed, packed);
        invokeDspineMessage(packed.raw_data(), spec_weights_.gate_bias.raw_data(), message.raw_data(), rows, R, block_,
                            dtype_code_, stream);
        lin.Forward(message, *layer->bridge_write, delta);

        if (l == dw.num_layer - 1) {
            GenericCopy(cur_out, last_raw_, stream);
            GenericCopy(packed, last_packed_, stream);
            cudaMemcpyAsync(last_rms_.data(), rms.data(), rows * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        }

        invokeDspineApply(cur_out.raw_data(),
                          x.raw_data(),
                          cur_out.raw_data(),
                          delta.raw_data(),
                          rms.data(),
                          next_norm->weight.raw_data(),
                          rows,
                          hidden_,
                          block_,
                          next_norm->norm_eps_,
                          dtype_code_,
                          stream);
        residual = cur_out;
    }
    return x;
}

void DspineModel::RunDraft(int phase, const DraftContext& ctx, TensorMap& env)
{
    if (*ctx_.is_warm_up) {
        return;
    }
    if (ctx.batch_size != 1) {
        return;
    }
    const auto&    dw     = draft_.weights();
    const auto     stream = core::Context::stream().handle();
    const DataType dtype  = dw.data_type;

    Tensor context_hidden = projection_->ProjectAndGather(phase, stream);
    const int rows = context_hidden.shape(0);
    if (rows == 0) {
        return;
    }

    Buffer_<int> base_dev{1, kDEVICE};
    cudaMemcpyAsync(base_dev.data(), ctx_len_.data(), sizeof(int), cudaMemcpyDeviceToDevice, stream);
    Buffer_<int> committed_dev{1, kDEVICE};
    invokeDflash2CommitLen(committed_dev.data(),
                           ctx.sequence_length.data(),
                           ctx.accept_len.data(),
                           ctx.accept_len.data() != nullptr,
                           stream);

    MaterializeContext(context_hidden, base_dev, committed_dev, rows, stream);
    cudaMemcpyAsync(ctx_len_.data(), committed_dev.data(), sizeof(int), cudaMemcpyDeviceToDevice, stream);

    cudaStreamSynchronize(stream);
    int committed = 0;
    cudaMemcpy(&committed, committed_dev.data(), sizeof(int), cudaMemcpyDeviceToHost);

    Buffer_<int> ids{block_, kDEVICE};
    invokeDflash2BuildBlockIds(ids.data(), ctx.request_token_ids_ptrs, committed_dev.data(), block_, mask_token_id_,
                               stream);

    Tensor embed_buf{{block_, hidden_}, dtype, kDEVICE};
    Tensor embeds = target_.Embed(ids, embed_buf, env);

    Buffer_<int> bpos_dev{block_, kDEVICE};
    invokeDflash2CountUp(bpos_dev.data(), committed_dev.data(), block_, stream);

    Tensor normed = RunBlock(embeds, bpos_dev, ids, block_, committed, stream, env);

    const int L = block_ - 1;
    Tensor    mask_h = normed.slice({1, 0}, {L, hidden_});
    Tensor    logits = target_.Logits(mask_h, {}, env);
    const int vocab  = logits.shape(1);

    Buffer_<int>   cand{L * top_k_, kDEVICE};
    Buffer_<float> values{L * top_k_, kDEVICE};
    invokeDflash2TopK(cand.data(), values.data(), logits.raw_data(), L, vocab, top_k_, 16, stream);

    Buffer_<int> tokens{L, kDEVICE};
    invokeDspineRefine(tokens.data(),
                       cand.data(),
                       values.data(),
                       last_packed_.raw_data(),
                       spec_weights_.tg.raw_data(),
                       spec_weights_.tsv.raw_data(),
                       spec_weights_.gate_bias.raw_data(),
                       last_rms_.data() + 1,
                       ids.data(),
                       L,
                       top_k_,
                       rank_,
                       dtype_code_,
                       stream);

    Buffer_<int> sb{1, kDEVICE};
    Copy(committed_dev, sb);
    invokeDflash2StoreTokens(ctx.request_token_ids_ptrs, tokens.data(), sb.data(), L, stream);
}

TM_REGISTER_SPECULATIVE_MODEL("dspine", DspineModel);

}  // namespace turbomind
