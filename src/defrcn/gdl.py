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

import mindspore.nn as nn
import mindspore as ms
from mindspore.common.initializer import One, Zero
import mindspore.ops as ops
from mindspore.ops import GradOperation
import numpy as np

class GradientDecoupleLayer(nn.Cell):
    def __init__(self, _lambda):
        super(GradientDecoupleLayer, self).__init__()
        self._lambda = _lambda

    def construct(self, x):
        return x

    def bprop(self, input, output, grad_output):
        grad_input = grad_output*1
        dtype = grad_output.dtype
        grad_input = grad_input * ops.Cast()(self._lambda, dtype)
        return (ops.Cast()(grad_input, dtype),)


class AffineLayer(nn.Cell):
    def __init__(self, num_channels, bias=False):
        super(AffineLayer, self).__init__()
        weight = ms.Tensor(shape=(1, num_channels, 1, 1), dtype=ms.float32, init=One())
        self.weight = ms.Parameter(weight, requires_grad=True)
        self.bias = None
        if bias:
            bias = ms.Tensor(shape=(1, num_channels, 1, 1), dtype=ms.float32, init=Zero())
            self.bias = ms.Parameter(bias, requires_grad=True)

    def construct(self, x):
        out = x * self.weight.expand_as(x)
        if self.bias is not None:
            out = out + self.bias.expand_as(x)
        return out