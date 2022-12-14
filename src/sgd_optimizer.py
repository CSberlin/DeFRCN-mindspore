# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import mindspore as ms
import mindspore.nn as nn
from typing import Any, Dict, List, Set
import mindspore.ops as ops

# class MySGD(nn.SGD):
#     def __init__(self, cfg, model, weight_lr, bias_lr):
#         params: List[Dict[str, Any]] = []
#         memo: Set[ms.Parameter] = set()
#         for value in model.trainable_params():
#             if "weight" in value.name or "bias" in value.name:
#                 if not value.requires_grad:
#                     continue
#                 # Avoid duplicating parameters
#                 if value in memo:
#                     continue
#                 memo.add(value)
#                 lr = weight_lr
#                 weight_decay = cfg.weight_decay
#                 if "bias" in value.name:
#                     lr = bias_lr
#                     weight_decay = cfg.weight_decay_bias
#                 params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
#         super().__init__(params, cfg.base_lr , momentum=cfg.sgd_momentum, nesterov=cfg.nesterov, weight_decay=cfg.weight_decay)
#         self._original_construct = super().construct
#         self.histogram_summary = ops.HistogramSummary()
#         self.gradient_names = [param.name + ".gradient" for param in self.parameters]

#     def construct(self, grads):
#         # Record gradient.
#         l = len(self.gradient_names)
#         for i in range(l):
#             self.histogram_summary(self.gradient_names[i], grads[i])
#         return self._original_construct(grads)

def build_optimizer(cfg, model, weight_lr, bias_lr):
    """
    Build an optimizer from config.
    """
    params: List[Dict[str, Any]] = []
    memo: Set[ms.Parameter] = set()
    for value in model.trainable_params():
        if "weight" in value.name or "bias" in value.name:
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = weight_lr
            weight_decay = cfg.weight_decay
            if "bias" in value.name:
                lr = bias_lr
                weight_decay = cfg.weight_decay_bias
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = nn.SGD(
        params, cfg.base_lr , momentum=cfg.sgd_momentum, nesterov=cfg.nesterov, 
        weight_decay=cfg.weight_decay, loss_scale=cfg.loss_scale
    )
    return optimizer