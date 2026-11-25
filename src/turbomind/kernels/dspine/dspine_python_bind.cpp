// Copyright (c) OpenMMLab. All rights reserved.
//
// Python bindings for the DSpine kernels, used by the reference-parity harness.
#include <cstdint>

#include <cuda_runtime.h>

#include <pybind11/pybind11.h>

#include "src/turbomind/kernels/dspine/dspine_kernels.h"
#include "src/turbomind/python/dlpack.h"
#include "src/turbomind/python/eagle3_component_bindings.h"
#include "src/turbomind/python/eagle3_dlpack_internal.h"

namespace py = pybind11;

namespace turbomind::python {
namespace {

int GetCudaOrdinal(py::handle tensor)
{
    return tensor.attr("__dlpack_device__")().cast<py::tuple>()[1].cast<int>();
}

int CodeOf(const detail::Tensor& t)
{
    auto dt = t.dtype();
    if (dt == kFloat16) {
        return 1;
    }
    if (dt == kBfloat16) {
        return 0;
    }
    if (dt == kFloat32) {
        return 2;
    }
    TM_CHECK(false) << "dspine bind: unsupported dtype";
    return -1;
}

}  // namespace

void BindDspineKernels(py::module_& module)
{
    module.def(
        "dspine_attention",
        [](py::handle out_obj, py::handle q_obj, py::handle kc_obj, py::handle vc_obj, py::handle kb_obj,
           py::handle vb_obj, float scale, int window, int causal, uintptr_t stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(q_obj)};
            auto out = detail::ConsumeDLPackWithStrides(out_obj, stream_ptr);
            auto q   = detail::ConsumeDLPackWithStrides(q_obj, stream_ptr);
            auto kc  = detail::ConsumeDLPackWithStrides(kc_obj, stream_ptr);
            auto vc  = detail::ConsumeDLPackWithStrides(vc_obj, stream_ptr);
            auto kb  = detail::ConsumeDLPackWithStrides(kb_obj, stream_ptr);
            auto vb  = detail::ConsumeDLPackWithStrides(vb_obj, stream_ptr);
            invokeDspineAttention(out.data_or(static_cast<void*>(nullptr)),
                                  q.data_or(static_cast<const void*>(nullptr)),
                                  kc.data_or(static_cast<const void*>(nullptr)),
                                  vc.data_or(static_cast<const void*>(nullptr)),
                                  kb.data_or(static_cast<const void*>(nullptr)),
                                  vb.data_or(static_cast<const void*>(nullptr)),
                                  static_cast<int>(q.shape(0)),
                                  static_cast<int>(kc.shape(0)),
                                  static_cast<int>(q.shape(1)),
                                  static_cast<int>(kc.shape(1)),
                                  static_cast<int>(q.shape(2)),
                                  scale,
                                  window,
                                  causal,
                                  CodeOf(q),
                                  reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("out"), py::arg("q"), py::arg("k_ctx"), py::arg("v_ctx"), py::arg("k_blk"), py::arg("v_blk"),
        py::arg("scale"), py::arg("window"), py::arg("causal"), py::arg("stream_ptr"));

    module.def(
        "dspine_read",
        [](py::handle raw_obj, py::handle norm_obj, py::handle rms_obj, float eps, uintptr_t stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(raw_obj)};
            auto raw  = detail::ConsumeDLPackWithStrides(raw_obj, stream_ptr);
            auto norm = detail::ConsumeDLPackWithStrides(norm_obj, stream_ptr);
            auto rms  = detail::ConsumeDLPackWithStrides(rms_obj, stream_ptr);
            invokeDspineRead(raw.data_or(static_cast<const void*>(nullptr)),
                             norm.data_or(static_cast<void*>(nullptr)),
                             reinterpret_cast<float*>(rms.data_or(static_cast<void*>(nullptr))),
                             static_cast<int>(raw.shape(0)),
                             static_cast<int>(raw.shape(1)),
                             eps,
                             CodeOf(raw),
                             reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("raw"), py::arg("normalized"), py::arg("rms"), py::arg("eps"), py::arg("stream_ptr"));

    module.def(
        "dspine_codes",
        [](py::handle proj_obj, py::handle codes_obj, py::handle cb_obj, py::handle anchor_obj, int block,
           float sqrt_c, float eps, uintptr_t stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(proj_obj)};
            auto proj   = detail::ConsumeDLPackWithStrides(proj_obj, stream_ptr);
            auto codes  = detail::ConsumeDLPackWithStrides(codes_obj, stream_ptr);
            auto cb     = detail::ConsumeDLPackWithStrides(cb_obj, stream_ptr);
            auto anchor = detail::ConsumeDLPackWithStrides(anchor_obj, stream_ptr);
            invokeDspineCodes(proj.data_or(static_cast<const void*>(nullptr)),
                              codes.data_or(static_cast<void*>(nullptr)),
                              cb.data_or(static_cast<const void*>(nullptr)),
                              reinterpret_cast<const int*>(anchor.data_or(static_cast<const void*>(nullptr))),
                              static_cast<int>(proj.shape(0)),
                              static_cast<int>(proj.shape(1)),
                              block,
                              sqrt_c,
                              eps,
                              CodeOf(proj),
                              reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("projected"), py::arg("codes"), py::arg("codebook"), py::arg("anchor"), py::arg("block"),
        py::arg("sqrt_code"), py::arg("eps"), py::arg("stream_ptr"));

    module.def(
        "dspine_message",
        [](py::handle packed_obj, py::handle bias_obj, py::handle msg_obj, int block, uintptr_t stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(packed_obj)};
            auto packed = detail::ConsumeDLPackWithStrides(packed_obj, stream_ptr);
            auto bias   = detail::ConsumeDLPackWithStrides(bias_obj, stream_ptr);
            auto msg    = detail::ConsumeDLPackWithStrides(msg_obj, stream_ptr);
            invokeDspineMessage(packed.data_or(static_cast<const void*>(nullptr)),
                                bias.data_or(static_cast<const void*>(nullptr)),
                                msg.data_or(static_cast<void*>(nullptr)),
                                static_cast<int>(packed.shape(0)),
                                static_cast<int>(packed.shape(1)) / 3,
                                block,
                                CodeOf(packed),
                                reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("packed"), py::arg("gate_bias"), py::arg("message"), py::arg("block"), py::arg("stream_ptr"));

    module.def(
        "dspine_apply",
        [](py::handle out_obj, py::handle norm_obj, py::handle raw_obj, py::handle delta_obj, py::handle rms_obj,
           py::handle nw_obj, int block, float eps, uintptr_t stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(raw_obj)};
            auto out   = detail::ConsumeDLPackWithStrides(out_obj, stream_ptr);
            auto norm  = detail::ConsumeDLPackWithStrides(norm_obj, stream_ptr);
            auto raw   = detail::ConsumeDLPackWithStrides(raw_obj, stream_ptr);
            auto delta = detail::ConsumeDLPackWithStrides(delta_obj, stream_ptr);
            auto rms   = detail::ConsumeDLPackWithStrides(rms_obj, stream_ptr);
            auto nw    = detail::ConsumeDLPackWithStrides(nw_obj, stream_ptr);
            invokeDspineApply(out.data_or(static_cast<void*>(nullptr)),
                              norm.data_or(static_cast<void*>(nullptr)),
                              raw.data_or(static_cast<const void*>(nullptr)),
                              delta.data_or(static_cast<const void*>(nullptr)),
                              reinterpret_cast<const float*>(rms.data_or(static_cast<const void*>(nullptr))),
                              nw.data_or(static_cast<const void*>(nullptr)),
                              static_cast<int>(raw.shape(0)),
                              static_cast<int>(raw.shape(1)),
                              block,
                              eps,
                              CodeOf(raw),
                              reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("out"), py::arg("normalized"), py::arg("raw"), py::arg("delta"), py::arg("rms"),
        py::arg("next_norm_weight"), py::arg("block"), py::arg("eps"), py::arg("stream_ptr"));
    module.def(
        "dspine_attention_int4",
        [](py::handle out_obj, py::handle q_obj, py::handle kd_obj, py::handle kp_obj, py::handle vd_obj,
           py::handle vp_obj, py::handle kb_obj, py::handle vb_obj, float scale, int window, int causal,
           uintptr_t stream_ptr) {
            CudaDeviceGuard guard{GetCudaOrdinal(q_obj)};
            auto out = detail::ConsumeDLPackWithStrides(out_obj, stream_ptr);
            auto q   = detail::ConsumeDLPackWithStrides(q_obj, stream_ptr);
            auto kd  = detail::ConsumeDLPackWithStrides(kd_obj, stream_ptr);
            auto kp  = detail::ConsumeDLPackWithStrides(kp_obj, stream_ptr);
            auto vd  = detail::ConsumeDLPackWithStrides(vd_obj, stream_ptr);
            auto vp  = detail::ConsumeDLPackWithStrides(vp_obj, stream_ptr);
            auto kb  = detail::ConsumeDLPackWithStrides(kb_obj, stream_ptr);
            auto vb  = detail::ConsumeDLPackWithStrides(vb_obj, stream_ptr);
            invokeDspineAttentionInt4(out.data_or(static_cast<void*>(nullptr)),
                                      q.data_or(static_cast<const void*>(nullptr)),
                                      kd.data_or(static_cast<const void*>(nullptr)),
                                      kp.data_or(static_cast<const void*>(nullptr)),
                                      vd.data_or(static_cast<const void*>(nullptr)),
                                      vp.data_or(static_cast<const void*>(nullptr)),
                                      kb.data_or(static_cast<const void*>(nullptr)),
                                      vb.data_or(static_cast<const void*>(nullptr)),
                                      static_cast<int>(q.shape(0)),
                                      static_cast<int>(kd.shape(0)),
                                      static_cast<int>(q.shape(1)),
                                      static_cast<int>(kd.shape(1)),
                                      static_cast<int>(q.shape(2)),
                                      scale,
                                      window,
                                      causal,
                                      CodeOf(q),
                                      reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("out"), py::arg("q"), py::arg("k_ctx_data"), py::arg("k_ctx_param"), py::arg("v_ctx_data"),
        py::arg("v_ctx_param"), py::arg("k_blk"), py::arg("v_blk"), py::arg("scale"), py::arg("window"),
        py::arg("causal"), py::arg("stream_ptr"));
}

}  // namespace turbomind::python
