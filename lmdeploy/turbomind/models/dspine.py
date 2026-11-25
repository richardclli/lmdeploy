# Copyright (c) OpenMMLab. All rights reserved.
"""DSpine draft weight model for TurboMind.

A 5-layer Qwen3 draft with causal sliding-window block attention, sharing the
target's input embeddings and output head. Beyond DFlash2 it carries a per-layer
*code bridge* (adjacent causal injection) and a serial last-write refinement
tail. The checkpoint has no convs and no DFlash2 candidate selector.

Weight layout (HF export, no ``model.`` prefix):
  fc.weight, hidden_norm.weight, norm.weight, layers.*, selected_prefix_bridge.*
"""

from __future__ import annotations

import math
import os

from ..builders import (
    DecoderLayerBuilder,
    DecoderLayerConfig,
    ModuleListBuilder,
    ModuleListConfig,
    TextModelBuilder,
)
from ..builders._base import _CPP_TO_TORCH
from ..builders.linear import Linear
from ..builders.dspine_weight import DspineWeightBuilder, DspineWeightConfig
from ..weight_format import TrivialFormat
from .base import INPUT_MODELS
from .dflash2 import Qwen3Dflash2TextModel
from .utils import make_model_weight_config

import torch


def _tm_linear(tm_weight: torch.Tensor, cpp_dtype, torch_dtype) -> Linear:
    """A Linear whose weight is already in TM layout (axis 0 = input dim)."""
    return Linear(tensors={'weight': tm_weight.to(torch_dtype).contiguous()},
                  weight_format=TrivialFormat(weight_dtype=cpp_dtype))


@INPUT_MODELS.register_module(name='qwen3-dspine')
class Qwen3DspineTextModel(Qwen3Dflash2TextModel):

    def __init__(self, cfg, *, resolver, prefix='', target_model_path=None):
        super().__init__(cfg, resolver=resolver, prefix=prefix)
        self._target_model_path = target_model_path
        dflash = dict(cfg.dflash_config)
        bridge = dict(dflash['selected_prefix_code_bridge'])
        self._code_size = int(bridge['code_size'])
        self._rank = int(bridge['rank'])
        self._bridge_eps = float(bridge['eps'])
        self._active_layers = [int(i) - 1 for i in dflash['selected_prefix_feature_layers']]
        self._b_read = None
        self._b_early = None
        self._b_write = None

    # ------------------------------------------------------------------
    def model(self, pfx):
        # Read every code-bridge tensor exactly once (the resolver consumes
        # popped keys) and stash what layers()/spec() both need.
        br = pfx + 'selected_prefix_bridge'
        self._b_codebook = br.pop('codebook')
        self._b_read = br.pop('read_weight')
        self._b_early = br.pop('early_read_weights')
        self._b_write = br.pop('feature_write_weights')
        gate = br + 'gate'
        self._b_gate_w = gate.pop('weight')
        self._b_gate_b = gate.pop('bias')
        self._b_pred_w = (br + 'predecessor_gate').pop('weight')
        self._b_prev_w = (br + 'prev').pop('weight')
        # out.weight / terminal_disabled_version / version tensors are unused
        # by the greedy terminal-disabled path; drop them so they are marked
        # consumed.
        for extra in ('out', 'independent_write_version', 'terminal_disabled_version'):
            sub = br + extra
            for key in ('weight', ''):
                if sub.has(key):
                    sub.pop(key)
        # Root-level version scalars (causal-layout / refinement) are unused.
        for scalar in ('draft_block_causal_version', 'last_write_refine_version'):
            sp = pfx + scalar
            if sp.has(''):
                sp.pop('')

        # Non-consuming read: the norm module below consumes it.
        self._b_norm_w = (pfx + 'norm').get('weight')

        root_cfg = make_model_weight_config(self.cfg)
        root_cfg.decoder_only = True
        builder = TextModelBuilder(
            root_cfg,
            self._ctx,
            root_handles=self._root_handles,
            tp=self._model_tp,
            vocab_size=self.cfg.vocab_size,
            root_child='draft_model')
        builder.norm = self.norm(pfx + 'norm')
        builder.layers = self.layers(pfx + 'layers')
        builder.spec = self.spec(pfx)
        builder.build()

    # ------------------------------------------------------------------
    def layers(self, pfx):
        cpp_dtype = self._ctx.data_type
        torch_dtype = _CPP_TO_TORCH[cpp_dtype]
        L = self.cfg.num_hidden_layers
        layers = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for i, p in pfx.slices(0, L):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._ctx)
            d.attention_norm = self.norm(p + 'input_layernorm')
            d.attention = self.attn(p + 'self_attn')
            d.ffn_norm = self.norm(p + 'post_attention_layernorm')
            d.feed_forward = self.ffn(p + 'mlp', tp=self._mlp_tp)
            # read: [C,H] -> TM [H,C]; write: [H,R] -> TM [R,H]
            read_w = self._b_read if i == L - 1 else self._b_early[i]
            d._add_linear('bridge_read', _tm_linear(read_w.t(), cpp_dtype, torch_dtype), split_side=None)
            d._add_linear('bridge_write', _tm_linear(self._b_write[i].t(), cpp_dtype, torch_dtype), split_side=None)
            layers[i] = d.build()
        return layers.build()

    # ------------------------------------------------------------------
    def spec(self, pfx):
        cpp_dtype = self._ctx.data_type
        torch_dtype = _CPP_TO_TORCH[cpp_dtype]

        packed = torch.cat([self._b_gate_w, self._b_pred_w, self._b_prev_w], dim=0)  # [3R,C]

        tg, tsv = self._build_tables(torch_dtype)

        spec = DspineWeightBuilder(DspineWeightConfig(), self._ctx)
        spec.tp = self._model_tp
        spec.add_fc(self._linear(pfx + 'fc'))
        spec.hidden_norm = self.norm(pfx + 'hidden_norm')
        spec.add_packed(_tm_linear(packed.t(), cpp_dtype, torch_dtype))
        spec.add_tensor('gate_bias', self._b_gate_b.to(torch_dtype).contiguous())
        spec.add_tensor('codebook', self._b_codebook.to(torch_dtype).contiguous())
        spec.add_tensor('tg', tg.to(self._b_norm_w.device))
        spec.add_tensor('tsv', tsv.to(self._b_norm_w.device))
        return spec.build()

    # ------------------------------------------------------------------
    def _build_tables(self, torch_dtype):
        """Vocabulary-wide greedy refinement tables TG [V,R] and TSV [V,2R].

        TG = ((lm_head * norm_w) @ feature_write[-1]); TSV =
        (sqrt(C) * codebook) @ [predecessor_gate; prev]^T. Computed in
        vocabulary chunks on the rank's device (fp32 matmul, cast to the
        engine dtype) so the peak working set stays small during export.
        """
        C = self._code_size
        dev = self._b_norm_w.device
        head_cpu = self._load_target_head()  # [V,H] on CPU
        V = head_cpu.shape[0]

        nw = self._b_norm_w.to(dev, torch.float32)  # [H]
        wb = self._b_write[-1].to(dev, torch.float32)  # [H,R]
        cb = self._b_codebook  # [V,C] on device
        pv = torch.cat([self._b_pred_w.to(dev), self._b_prev_w.to(dev)], dim=0).to(torch.float32)  # [2R,C]
        pvt = pv.t().contiguous()  # [C,2R]
        sqrt_c = math.sqrt(C)

        chunk = 8192
        tg_parts, tsv_parts = [], []
        for s0 in range(0, V, chunk):
            e0 = min(s0 + chunk, V)
            hc = head_cpu[s0:e0].to(dev, torch.float32)  # [n,H]
            tg_parts.append(((hc * nw) @ wb).to(torch_dtype).to('cpu'))
            tsv_parts.append(((sqrt_c * cb[s0:e0].to(torch.float32)) @ pvt).to(torch_dtype).to('cpu'))
            del hc
        tg = torch.cat(tg_parts, dim=0).contiguous()
        tsv = torch.cat(tsv_parts, dim=0).contiguous()
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        return tg, tsv

    def _load_target_head(self):
        from safetensors import safe_open
        root = self._target_model_path
        for name in sorted(os.listdir(root)):
            if not name.endswith('.safetensors'):
                continue
            path = os.path.join(root, name)
            try:
                with safe_open(path, 'pt') as f:
                    if 'lm_head.weight' in f.keys():
                        return f.get_tensor('lm_head.weight')
            except Exception:
                continue
        raise FileNotFoundError(f'lm_head.weight not found under {root}')
