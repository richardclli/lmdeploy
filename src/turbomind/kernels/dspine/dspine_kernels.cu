// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/dspine/dspine_kernels.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cfloat>

#include "src/turbomind/core/check.h"

namespace turbomind {

namespace {

template<typename T>
__device__ __forceinline__ float to_f(T x);
template<>
__device__ __forceinline__ float to_f<__half>(__half x)
{
    return __half2float(x);
}
template<>
__device__ __forceinline__ float to_f<float>(float x)
{
    return x;
}
template<>
__device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 x)
{
    return __bfloat162float(x);
}

template<typename T>
__device__ __forceinline__ void from_f(T& dst, float x);
template<>
__device__ __forceinline__ void from_f<__half>(__half& dst, float x)
{
    dst = __float2half_rn(x);
}
template<>
__device__ __forceinline__ void from_f<float>(float& dst, float x)
{
    dst = x;
}
template<>
__device__ __forceinline__ void from_f<__nv_bfloat16>(__nv_bfloat16& dst, float x)
{
    dst = __float2bfloat16_rn(x);
}

__device__ __forceinline__ float warp_reduce_sum(float x)
{
    for (int offset = 16; offset > 0; offset /= 2) {
        x += __shfl_down_sync(0xffffffff, x, offset);
    }
    return x;
}

// ---------------------------------------------------------------------------
// Causal block attention (context within window | causal block)
// ---------------------------------------------------------------------------

template<typename T, int HD>
__global__ void dspine_attn_kernel(T* __restrict__ out,
                                   const T* __restrict__ q,
                                   const T* __restrict__ k_ctx,
                                   const T* __restrict__ v_ctx,
                                   const T* __restrict__ k_blk,
                                   const T* __restrict__ v_blk,
                                   int  tq,
                                   int  tc,
                                   int  nh,
                                   int  nkv,
                                   float scale,
                                   int  window)
{
    constexpr int ND = HD / 32;
    constexpr int NW = 8;

    const int bh  = blockIdx.x;
    const int i   = bh / nh;
    const int h   = bh - i * nh;
    const int kvh = h * nkv / nh;
    const int qpos = tc + i;
    const int first_ctx = max(0, qpos - window + 1);
    const int nkeys = tc + i + 1;  // causal: block keys [tc, tc+i]
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;

    const T* qrow = q + ((long long)i * nh + h) * HD;
    float    qd[ND];
#pragma unroll
    for (int k = 0; k < ND; ++k) {
        qd[k] = to_f(qrow[lane + k * 32]);
    }

    float m = -CUDART_INF_F, l = 0.f, acc[ND];
#pragma unroll
    for (int k = 0; k < ND; ++k) {
        acc[k] = 0.f;
    }

    for (int j = first_ctx + wid; j < nkeys; j += NW) {
        const T* krow;
        const T* vrow;
        if (j < tc) {
            krow = k_ctx + ((long long)j * nkv + kvh) * HD;
            vrow = v_ctx + ((long long)j * nkv + kvh) * HD;
        }
        else {
            krow = k_blk + ((long long)(j - tc) * nkv + kvh) * HD;
            vrow = v_blk + ((long long)(j - tc) * nkv + kvh) * HD;
        }
        float part = 0.f;
#pragma unroll
        for (int k = 0; k < ND; ++k) {
            part += qd[k] * to_f(krow[lane + k * 32]);
        }
        const float s    = warp_reduce_sum(part) * scale;
        const float mn   = fmaxf(m, s);
        const float corr = (m == -CUDART_INF_F) ? 0.f : __expf(m - mn);
        const float e    = __expf(s - mn);
        l = l * corr + e;
#pragma unroll
        for (int k = 0; k < ND; ++k) {
            acc[k] = acc[k] * corr + e * to_f(vrow[lane + k * 32]);
        }
        m = mn;
    }

    __shared__ float sm[NW];
    __shared__ float sl[NW];
    __shared__ float sa[NW][32][ND];
    if (lane == 0) {
        sm[wid] = m;
        sl[wid] = l;
    }
#pragma unroll
    for (int k = 0; k < ND; ++k) {
        sa[wid][lane][k] = acc[k];
    }
    __syncthreads();

    float M = -CUDART_INF_F;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
        M = fmaxf(M, sm[w]);
    }
    float L = 0.f, A[ND];
#pragma unroll
    for (int k = 0; k < ND; ++k) {
        A[k] = 0.f;
    }
#pragma unroll
    for (int w = 0; w < NW; ++w) {
        const float f = __expf(sm[w] - M);
        L += sl[w] * f;
#pragma unroll
        for (int k = 0; k < ND; ++k) {
            A[k] += sa[w][lane][k] * f;
        }
    }
    T* orow = out + ((long long)i * nh + h) * HD;
#pragma unroll
    for (int k = 0; k < ND; ++k) {
        T r;
        from_f(r, A[k] / L);
        orow[lane + k * 32] = r;
    }
}

template<typename T>
void launch_attn(void*        out,
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
                 cudaStream_t stream)
{
    TM_CHECK_EQ(hd, 128) << "DSpine attention expects head_dim 128";
    const int grid  = tq * nh;
    const int block = 256;
    dspine_attn_kernel<T, 128><<<grid, block, 0, stream>>>(static_cast<T*>(out),
                                                           static_cast<const T*>(q),
                                                           static_cast<const T*>(k_ctx),
                                                           static_cast<const T*>(v_ctx),
                                                           static_cast<const T*>(k_blk),
                                                           static_cast<const T*>(v_blk),
                                                           tq,
                                                           tc,
                                                           nh,
                                                           nkv,
                                                           scale,
                                                           window);
}

// ---------------------------------------------------------------------------
// read: row-wise rms + normalize
// ---------------------------------------------------------------------------

template<typename T>
__global__ void dspine_read_kernel(const T* __restrict__ raw,
                                   T* __restrict__ normalized,
                                   float* __restrict__ rms_out,
                                   int    rows,
                                   int    H,
                                   float  eps)
{
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const T* r = raw + (long long)row * H;
    T*       o = normalized + (long long)row * H;

    __shared__ float red[256];
    float            ss = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float v = to_f(r[i]);
        ss += v * v;
    }
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            red[threadIdx.x] += red[threadIdx.x + s];
        }
        __syncthreads();
    }
    __shared__ float srms;
    if (threadIdx.x == 0) {
        srms = sqrtf(fmaxf(red[0] / (float)H, FLT_MIN));
        rms_out[row] = srms;
    }
    __syncthreads();
    const float denom = srms + eps;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        T v;
        from_f(v, to_f(r[i]) / denom);
        o[i] = v;
    }
}

// ---------------------------------------------------------------------------
// codes: normalize projection, anchor override
// ---------------------------------------------------------------------------

template<typename T>
__global__ void dspine_codes_kernel(const T* __restrict__ projected,
                                    T* __restrict__ codes,
                                    const T* __restrict__ codebook,
                                    const int* __restrict__ anchor_ptr,
                                    int    rows,
                                    int    C,
                                    int    block,
                                    float  sqrt_c,
                                    float  eps)
{
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const T* p = projected + (long long)row * C;
    T*       o = codes + (long long)row * C;

    if (block > 0 && (row % block) == 0) {
        const int  a    = anchor_ptr ? anchor_ptr[0] : 0;
        const T*   crow = codebook + (long long)a * C;
        for (int i = threadIdx.x; i < C; i += blockDim.x) {
            T v;
            from_f(v, sqrt_c * to_f(crow[i]));
            o[i] = v;
        }
        return;
    }

    __shared__ float red[256];
    float            ss = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        const float v = to_f(p[i]);
        ss += v * v;
    }
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            red[threadIdx.x] += red[threadIdx.x + s];
        }
        __syncthreads();
    }
    __shared__ float snorm;
    if (threadIdx.x == 0) {
        snorm = sqrtf(red[0]) + eps;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        T v;
        from_f(v, sqrt_c * to_f(p[i]) / snorm);
        o[i] = v;
    }
}

// ---------------------------------------------------------------------------
// message: adjacent predecessor injection
// ---------------------------------------------------------------------------

template<typename T>
__global__ void dspine_message_kernel(const T* __restrict__ packed,
                                      const T* __restrict__ gate_bias,
                                      T* __restrict__ message,
                                      int    rows,
                                      int    R,
                                      int    block)
{
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const T* pk = packed + (long long)row * 3 * R;
    T*       mo = message + (long long)row * R;

    if (block > 0 && (row % block) == 0) {
        for (int i = threadIdx.x; i < R; i += blockDim.x) {
            T z;
            from_f(z, 0.f);
            mo[i] = z;
        }
        return;
    }
    const T* prev = packed + (long long)(row - 1) * 3 * R;
    for (int i = threadIdx.x; i < R; i += blockDim.x) {
        const float q = to_f(pk[i]);
        const float s = to_f(prev[R + i]);
        const float v = to_f(prev[2 * R + i]);
        const float b = gate_bias ? to_f(gate_bias[i]) : 0.f;
        const float g = 1.f / (1.f + __expf(-(q + s + b)));
        T o;
        from_f(o, g * v);
        mo[i] = o;
    }
}

// ---------------------------------------------------------------------------
// apply: residual add of scaled delta + next norm
// ---------------------------------------------------------------------------

template<typename T>
__global__ void dspine_apply_kernel(T* __restrict__ out,
                                    T* __restrict__ normed,
                                    const T* __restrict__ raw,
                                    const T* __restrict__ delta,
                                    const float* __restrict__ rms,
                                    const T* __restrict__ next_w,
                                    int    rows,
                                    int    H,
                                    int    block,
                                    float  eps)
{
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const T* r = raw + (long long)row * H;
    const T* d = delta + (long long)row * H;
    T*       o = out + (long long)row * H;
    T*       n = normed + (long long)row * H;
    const bool anchor = block > 0 && (row % block) == 0;
    const float s = anchor ? 0.f : rms[row];

    __shared__ float red[256];
    float            ss = 0.f;
    // First materialize `out`, accumulating its sum of squares for the norm.
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float v = to_f(r[i]) + s * to_f(d[i]);
        T ov;
        from_f(ov, v);
        o[i] = ov;
        ss += v * v;
    }
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int t = blockDim.x / 2; t > 0; t >>= 1) {
        if (threadIdx.x < t) {
            red[threadIdx.x] += red[threadIdx.x + t];
        }
        __syncthreads();
    }
    __shared__ float snorm;
    if (threadIdx.x == 0) {
        snorm = rsqrtf(red[0] / (float)H + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float y = to_f(o[i]) * snorm;
        T v;
        from_f(v, y * to_f(next_w[i]));
        n[i] = v;
    }
}

// ---------------------------------------------------------------------------
// refine: serial last-write refinement (scores + greedy chain)
// ---------------------------------------------------------------------------

template<typename T>
__global__ void dspine_refine_scores(int* __restrict__ idx,
                                     const int* __restrict__ cand,
                                     const float* __restrict__ values,
                                     const T* __restrict__ packed,
                                     const T* __restrict__ tg,
                                     const T* __restrict__ tsv,
                                     const T* __restrict__ gate_bias,
                                     const float* __restrict__ rms,
                                     const int* __restrict__ anchor_ptr,
                                     int    K,
                                     int    R)
{
    extern __shared__ float sh[];  // [K*K]
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    const T*  pk  = packed + (long long)(i + 1) * 3 * R;  // q-part of row i+1
    const int* ci = cand + i * K;

    for (int e = tid; e < K * K; e += nth) {
        const int p = e / K;
        const int c = e - p * K;
        const int sid = (i == 0) ? anchor_ptr[0] : cand[(i - 1) * K + p];
        const T*  ts  = tsv + (long long)sid * 2 * R;
        const T*  tc  = tg + (long long)ci[c] * R;
        float s = 0.f;
        for (int r = 0; r < R; ++r) {
            const float b = gate_bias ? to_f(gate_bias[r]) : 0.f;
            const float g = 1.f / (1.f + __expf(-(to_f(pk[r]) + to_f(ts[r]) + b)));
            const float m = g * to_f(ts[R + r]);
            s += m * to_f(tc[r]);
        }
        const float base = rms[i] * values[i * K + c];
        sh[e] = base + rms[i] * s;
    }
    __syncthreads();

    for (int p = tid; p < K; p += nth) {
        float best = -CUDART_INF_F;
        int   bi   = 0;
        for (int c = 0; c < K; ++c) {
            const float v = sh[p * K + c];
            if (v > best) {
                best = v;
                bi   = c;
            }
        }
        idx[i * K + p] = bi;
    }
}

__global__ void dspine_chain_kernel(int* __restrict__ tokens,
                                    const int* __restrict__ cand,
                                    const int* __restrict__ idx,
                                    int    n,
                                    int    K)
{
    int j = idx[0];
    for (int i = 0; i < n; ++i) {
        tokens[i] = cand[i * K + j];
        if (i + 1 < n) {
            j = idx[(i + 1) * K + j];
        }
    }
}

}  // namespace

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
                           int          dtype,
                           cudaStream_t stream)
{
    if (dtype == 0) {
        launch_attn<__nv_bfloat16>(out, q, k_ctx, v_ctx, k_blk, v_blk, tq, tc, nh, nkv, hd, scale, window, stream);
    }
    else if (dtype == 1) {
        launch_attn<__half>(out, q, k_ctx, v_ctx, k_blk, v_blk, tq, tc, nh, nkv, hd, scale, window, stream);
    }
    else {
        launch_attn<float>(out, q, k_ctx, v_ctx, k_blk, v_blk, tq, tc, nh, nkv, hd, scale, window, stream);
    }
}

void invokeDspineRead(const void*  raw,
                      void*        normalized,
                      float*       rms,
                      int          rows,
                      int          hidden,
                      float        eps,
                      int          dtype,
                      cudaStream_t stream)
{
    const int block = 256;
    if (dtype == 0) {
        dspine_read_kernel<__nv_bfloat16><<<rows, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(raw),
            static_cast<__nv_bfloat16*>(normalized), rms, rows, hidden, eps);
    }
    else if (dtype == 1) {
        dspine_read_kernel<__half><<<rows, block, 0, stream>>>(static_cast<const __half*>(raw),
                                                               static_cast<__half*>(normalized), rms, rows, hidden,
                                                               eps);
    }
    else {
        dspine_read_kernel<float><<<rows, block, 0, stream>>>(
            static_cast<const float*>(raw), static_cast<float*>(normalized), rms, rows, hidden, eps);
    }
}

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
                       cudaStream_t stream)
{
    const int nthreads = 256;
    if (dtype == 0) {
        dspine_codes_kernel<__nv_bfloat16><<<rows, nthreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(projected), static_cast<__nv_bfloat16*>(codes),
            static_cast<const __nv_bfloat16*>(codebook), anchor_ptr, rows, code_size, block, sqrt_code, eps);
    }
    else if (dtype == 1) {
        dspine_codes_kernel<__half><<<rows, nthreads, 0, stream>>>(static_cast<const __half*>(projected),
                                                                  static_cast<__half*>(codes),
                                                                  static_cast<const __half*>(codebook), anchor_ptr,
                                                                  rows, code_size, block, sqrt_code, eps);
    }
    else {
        dspine_codes_kernel<float><<<rows, nthreads, 0, stream>>>(static_cast<const float*>(projected),
                                                                 static_cast<float*>(codes),
                                                                 static_cast<const float*>(codebook), anchor_ptr,
                                                                 rows, code_size, block, sqrt_code, eps);
    }
}

void invokeDspineMessage(const void*  packed,
                         const void*  gate_bias,
                         void*        message,
                         int          rows,
                         int          rank,
                         int          block,
                         int          dtype,
                         cudaStream_t stream)
{
    const int nthreads = 256;
    if (dtype == 0) {
        dspine_message_kernel<__nv_bfloat16><<<rows, nthreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(packed), static_cast<const __nv_bfloat16*>(gate_bias),
            static_cast<__nv_bfloat16*>(message), rows, rank, block);
    }
    else if (dtype == 1) {
        dspine_message_kernel<__half><<<rows, nthreads, 0, stream>>>(static_cast<const __half*>(packed),
                                                                    static_cast<const __half*>(gate_bias),
                                                                    static_cast<__half*>(message), rows, rank, block);
    }
    else {
        dspine_message_kernel<float><<<rows, nthreads, 0, stream>>>(static_cast<const float*>(packed),
                                                                   static_cast<const float*>(gate_bias),
                                                                   static_cast<float*>(message), rows, rank, block);
    }
}

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
                       cudaStream_t stream)
{
    const int nthreads = 256;
    if (dtype == 0) {
        dspine_apply_kernel<__nv_bfloat16><<<rows, nthreads, 0, stream>>>(
            static_cast<__nv_bfloat16*>(out), static_cast<__nv_bfloat16*>(normed),
            static_cast<const __nv_bfloat16*>(raw), static_cast<const __nv_bfloat16*>(delta), rms,
            static_cast<const __nv_bfloat16*>(next_norm_weight), rows, hidden, block, eps);
    }
    else if (dtype == 1) {
        dspine_apply_kernel<__half><<<rows, nthreads, 0, stream>>>(static_cast<__half*>(out),
                                                                  static_cast<__half*>(normed),
                                                                  static_cast<const __half*>(raw),
                                                                  static_cast<const __half*>(delta), rms,
                                                                  static_cast<const __half*>(next_norm_weight), rows,
                                                                  hidden, block, eps);
    }
    else {
        dspine_apply_kernel<float><<<rows, nthreads, 0, stream>>>(static_cast<float*>(out),
                                                                 static_cast<float*>(normed),
                                                                 static_cast<const float*>(raw),
                                                                 static_cast<const float*>(delta), rms,
                                                                 static_cast<const float*>(next_norm_weight), rows,
                                                                 hidden, block, eps);
    }
}


void invokeDspineRefine(int*         tokens,
                        const int*   cand,
                        const float* values,
                        const void*  packed,
                        const void*  tg,
                        const void*  tsv,
                        const void*  gate_bias,
                        const float* rms,
                        const int*   anchor_ptr,
                        int          n,
                        int          top_k,
                        int          rank,
                        int          dtype,
                        cudaStream_t stream)
{
    TM_CHECK(top_k > 0 && top_k * top_k <= 1024) << "DSpine refine expects 0 < top_k <= 32";
    int* idx = nullptr;
    TM_CHECK(cudaMallocAsync(&idx, (size_t)n * top_k * sizeof(int), stream) == cudaSuccess)
        << "DSpine refine: scratch allocation failed";
    const size_t shmem    = (size_t)top_k * top_k * sizeof(float);
    const int    nthreads = top_k * top_k;
    auto dispatch         = [&](auto type_tag) {
        using T = decltype(type_tag);
        dspine_refine_scores<T><<<n, nthreads, shmem, stream>>>(idx,
                                                                cand,
                                                                values,
                                                                static_cast<const T*>(packed),
                                                                static_cast<const T*>(tg),
                                                                static_cast<const T*>(tsv),
                                                                static_cast<const T*>(gate_bias),
                                                                rms,
                                                                anchor_ptr,
                                                                top_k,
                                                                rank);
    };
    if (dtype == 0) {
        dispatch(__nv_bfloat16{});
    }
    else if (dtype == 1) {
        dispatch(__half{});
    }
    else {
        dispatch(float{});
    }
    dspine_chain_kernel<<<1, 1, 0, stream>>>(tokens, cand, idx, n, top_k);
    cudaFreeAsync(idx, stream);
}

}  // namespace turbomind
