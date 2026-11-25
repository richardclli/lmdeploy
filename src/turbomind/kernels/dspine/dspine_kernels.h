// Copyright (c) OpenMMLab. All rights reserved.
//
// DSpine draft kernels: causal block attention, the per-layer code bridge
// (read / codes / message / apply) and the serial last-write refinement tail.
#pragma once

#include <cuda_runtime.h>

namespace turbomind {

// dtype code shared by the kernels below: 0 = bf16, 1 = fp16, 2 = fp32.

/// Causal variant of the DFlash2 block attention: context keys within the
/// sliding window are all visible; block key j (j >= tc) is visible to query i
/// only when (j - tc) <= i.
void invokeDspineAttention(void*        out,
                           const void*  q,
                           const void*  k_ctx,
                           const void*  v_ctx,
                           const void*  k_blk,
                           const void*  v_blk,
                           int          tq,
                           int          tc,
                           int          nh,
                           int          nkv,
                           int          hd,
                           float        scale,
                           int          window,
                           int          causal,
                           int          dtype,
                           cudaStream_t stream);

/// read: rms[row] = sqrt(max(mean(raw^2), FLT_MIN)); normalized[row,:] =
/// raw[row,:] / (rms[row] + eps) in the compute dtype.
void invokeDspineRead(const void*  raw,
                      void*        normalized,
                      float*       rms,
                      int          rows,
                      int          hidden,
                      float        eps,
                      int          dtype,
                      cudaStream_t stream);

/// codes: per row, normalize the C-dim projection to sqrt(C) * p / (|p| + eps).
/// Rows where (row % block == 0) are replaced by sqrt(C) * codebook[anchor].
void invokeDspineCodes(const void*  projected,
                       void*        codes,
                       const void*  codebook,
                       const int*   anchor_ptr,
                       int          rows,
                       int          code_size,
                       int          block,
                       float        sqrt_code,
                       float        eps,
                       int          dtype,
                       cudaStream_t stream);

/// message: split packed [rows, 3R] into q=[0,R), s=[R,2R), v=[2R,3R).
///   message[row] = 0                              if row % block == 0
///                = sigmoid(q[row] + s[row-1] + bias) * v[row-1]  otherwise
void invokeDspineMessage(const void*  packed,
                         const void*  gate_bias,
                         void*        message,
                         int          rows,
                         int          rank,
                         int          block,
                         int          dtype,
                         cudaStream_t stream);

/// apply: out = raw + rms * delta on non-anchor rows; normed = rmsnorm(out) *
/// next_norm_weight. `out` and `normed` are written.
void invokeDspineApply(void*        out,
                       void*        normed,
                       const void*  raw,
                       const void*  delta,
                       const float* rms,
                       const void*  next_norm_weight,
                       int          rows,
                       int          hidden,
                       int          block,
                       float        eps,
                       int          dtype,
                       cudaStream_t stream);

/// refine: serial last-write refinement over proposal rows 1..n.
///   base[i,c]  = rms[i] * pre_logits[i, cand[i,c]]   (pre-write residual logit)
///   sid[i,p]   = anchor if i == 0 else cand[i-1,p]
///   m[r]       = sigmoid(q[i,r] + tsv[sid, r] + bias[r]) * tsv[sid, R+r]
///   score[p,c] = base[i,c] + rms[i] * sum_r m[r] * tg[cand[i,c], r]
///   idx[i,p]   = argmax_c score[p,c]
///   chain: j = idx[0,0]; tok[i] = cand[i, j]; j = idx[i+1, j]
/// `packed` is the last bridge write's [block, 3R] projection; proposal i reads
/// its q-part from row (i + 1). Writes `tokens[n]`.
void invokeDspineRefine(int*         tokens,
                        const int*   cand,
                        const void*  pre_logits,
                        const void*  packed,
                        const void*  tg,
                        const void*  tsv,
                        const void*  gate_bias,
                        const float* rms,
                        const int*   anchor_ptr,
                        int          vocab,
                        int          n,
                        int          top_k,
                        int          rank,
                        int          dtype,
                        cudaStream_t stream);

/// Deterministic top-1 proposal (debug): tokens[i] = cand[i, 0].
void invokeDspineTop1(int* tokens, const int* cand, int n, int top_k, cudaStream_t stream);

/// Causal int4 context-KV block attention (context K/V packed int4 with
/// per-(token,head) scale/zero; block K/V in the compute dtype).
void invokeDspineAttentionInt4(void*        out,
                               const void*  q,
                               const void*  k_ctx_data,
                               const void*  k_ctx_param,
                               const void*  v_ctx_data,
                               const void*  v_ctx_param,
                               const void*  k_blk,
                               const void*  v_blk,
                               int          tq,
                               int          tc,
                               int          nh,
                               int          nkv,
                               int          hd,
                               float        scale,
                               int          window,
                               int          causal,
                               int          dtype,
                               cudaStream_t stream);

}  // namespace turbomind
