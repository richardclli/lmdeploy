# Copyright (c) OpenMMLab. All rights reserved.
"""Builder for the DSpine method weight tree (ModelWeight.spec).

The bridge read/write linears live on each draft decoder layer
(``bridge_read`` / ``bridge_write``); this tree holds the context projection,
hidden norm, the folded gate/predecessor/prev projection, the gate bias, the
transfer codebook and the precomputed refinement tables.
"""

import _turbomind as _tm

from ._base import Builder

DspineWeightConfig = _tm.DspineWeightConfig


class DspineWeightBuilder(Builder):
    """fc + hidden_norm + packed projection + gate_bias/codebook/tg/tsv."""

    def add_fc(self, linear):
        self._add_linear('fc', linear, split_side=None)

    def add_packed(self, linear):
        self._add_linear('packed', linear, split_side=None)

    def add_tensor(self, name, tensor):
        self._add_tensor(name, tensor, split_side=None)
