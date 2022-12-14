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
"""Defrcn ROIAlign module."""

import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.nn import layer as L
from mindspore.common.tensor import Tensor

class ROIAlign(nn.Cell):
    """
    Extract RoI features from multiple feature map.

    Args:
        out_size_h (int) - RoI height.
        out_size_w (int) - RoI width.
        spatial_scale (int) - RoI spatial scale.
        sample_num (int) - RoI sample number.
    """
    def __init__(self,
                 out_size_h,
                 out_size_w,
                 spatial_scale,
                 sample_num=0):
        super(ROIAlign, self).__init__()

        self.out_size = (out_size_h, out_size_w)
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.align_op = ops.ROIAlign(self.out_size[0], self.out_size[1],
                                     self.spatial_scale, self.sample_num)

    def construct(self, features, rois):
        return self.align_op(features, rois)

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += '(out_size={}, spatial_scale={}, sample_num={}'.format(
            self.out_size, self.spatial_scale, self.sample_num)
        return format_str

class SingleRoIExtractor(nn.Cell):
    """
    Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        config (dict): Config
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        batch_size (int)： Batchsize.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self,
                 config,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 batch_size=1,
                 finest_scale=56):
        super(SingleRoIExtractor, self).__init__()
        cfg = config
        self.train_batch_size = batch_size
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.out_size = config.roi_layer.out_size
        self.sample_num = config.roi_layer.sample_num
        self.roi_layers = self.build_roi_layers(self.featmap_strides)

        self.sqrt = ops.Sqrt()
        self.log = ops.Log()
        self.finest_scale_ = finest_scale
        self.clamp = ops.clip_by_value

        self.cast = ops.Cast()
        self.equal = ops.Equal()
        self.select = ops.Select()

        _mode_16 = False
        self.dtype = np.float16 if _mode_16 else np.float32
        self.ms_dtype = ms.float16 if _mode_16 else ms.float32
        self.set_train_local(cfg, training=True)

    def set_train_local(self, config, training=True):
        """Set training flag."""
        self.training_local = training

        cfg = config
        # Init tensor
        self.batch_size = cfg.roi_sample_num if self.training_local else cfg.rpn_max_num
        self.batch_size = self.train_batch_size*self.batch_size \
            if self.training_local else cfg.test_batch_size*self.batch_size
       
    def num_inputs(self):
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def log2(self, value):
        return self.log(value) / self.log(self.twos)

    def build_roi_layers(self, featmap_strides):
        roi_layer = ROIAlign(self.out_size, self.out_size,
                                 spatial_scale=1 / featmap_strides,
                                 sample_num=self.sample_num)
        return roi_layer

    def construct(self, rois, feat):
        roi_feats_t = self.roi_layers(feat, rois)
        return roi_feats_t
