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
"""Resnet backbone."""

import numpy as np
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.common.tensor import Tensor


def weight_init_ones(shape):
    """Weight init."""
    return Tensor(np.full(shape, 0.01).astype(np.float32))


def conv3x3(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                    has_bias=False, pad_mode="pad")

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, has_bias=False)

def _BatchNorm2dInit(out_chls, momentum=0.9, affine=False, use_batch_statistics=False):
    """Batchnorm2D wrapper."""
    dtype = np.float32
    gamma_init = Tensor(np.array(np.ones(out_chls)).astype(dtype))
    beta_init = Tensor(np.array(np.ones(out_chls) * 0).astype(dtype))
    moving_mean_init = Tensor(np.array(np.ones(out_chls) * 0).astype(dtype))
    moving_var_init = Tensor(np.array(np.ones(out_chls)).astype(dtype))
    return nn.BatchNorm2d(out_chls, momentum=momentum, affine=affine, gamma_init=gamma_init,
                          beta_init=beta_init, moving_mean_init=moving_mean_init,
                          moving_var_init=moving_var_init, use_batch_statistics=use_batch_statistics)

class ResNetFea(nn.Cell):
    """
    ResNet architecture.

    Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        weights_update (bool): Weight update flag.
    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResNet(ResidualBlock,
        >>>        [3, 4, 23, 3],
        >>>        [layer1,layer2,...,layer4]
        >>>        ['layer3'])
    """

    def __init__(self,
                 stem,
                 layers,
                 out_features,
                 ):
        super(ResNetFea, self).__init__()
        self.stem = stem
        self.layer_names, self.layers = [], []
        self.out_features = out_features
        if out_features is not None:
            num_stages = max(
                [{"layer1": 1, "layer2": 2, "layer3": 3, "layer4": 4}.get(f, 0) for f in out_features]
            )
            layers = layers[:num_stages]
        for i, blocks in enumerate(layers):
            name = "layer" + str(i + 1)
            layer = nn.SequentialCell(*blocks)
            self.insert_child_to_cell(name, layer)
            self.layer_names.append(name)
            self.layers.append(layer)
        self.layer_names = tuple(self.layer_names)  
    
    # Make it static for scripting
    @staticmethod
    def make_layer(block, layer_num, in_channel, out_channel, stride, training=False, weights_update=False):
        """Make block layer."""
        layers = []
        down_sample = False
        if stride != 1 or in_channel != out_channel:
            down_sample = True
        resblk = block(in_channel,
                       out_channel,
                       stride=stride,
                       down_sample=down_sample,
                       training=training,
                       weights_update=weights_update)
        layers.append(resblk)

        for _ in range(1, layer_num):
            resblk = block(out_channel, out_channel, stride=1, training=training, weights_update=weights_update)
            layers.append(resblk)

        return nn.SequentialCell(layers)

    def construct(self, x):
        """
        construct the ResNet Network

        Args:
            x: input feature data.

        Returns:
        Tensor, output tensor.
        """
        x = self.stem(x)
        for name, layer in zip(self.layer_names, self.layers):
            x = layer(x)
        return x


class ResidualBlockUsing(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        in_channels (int) - Input channel.
        out_channels (int) - Output channel.
        stride (int) - Stride size for the initial convolutional layer. Default: 1.
        down_sample (bool) - If to do the downsample in block. Default: False.
        momentum (float) - Momentum for batchnorm layer. Default: 0.1.
        training (bool) - Training flag. Default: False.
        weights_updata (bool) - Weights update flag. Default: False.

    Returns:
        Tensor, output tensor.

    Examples:
        ResidualBlock(3,256,stride=2,down_sample=True)
    """
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 down_sample=False,
                 momentum=0.9,
                 training=False,
                 weights_update=False):
        super(ResidualBlockUsing, self).__init__()

        self.affine = False

        self.downsample = down_sample
        if self.downsample:
            self.conv_down_sample = conv1x1(in_channels, out_channels, stride=stride)
            self.bn_down_sample = _BatchNorm2dInit(out_channels, momentum=momentum, affine=self.affine,
                                                   use_batch_statistics=training)
            if training:
                self.bn_down_sample = self.bn_down_sample.set_train()
            if not weights_update:
                self.conv_down_sample.weight.requires_grad = False

        out_chls = out_channels // self.expansion
        self.conv1 = conv1x1(in_channels, out_chls, stride=stride)
        self.bn1 = _BatchNorm2dInit(out_chls, momentum=momentum, affine=self.affine, use_batch_statistics=training)

        self.conv2 = conv3x3(out_chls, out_chls, kernel_size=3, padding=1)
        self.bn2 = _BatchNorm2dInit(out_chls, momentum=momentum, affine=self.affine, use_batch_statistics=training)

        self.conv3 = conv1x1(out_chls, out_channels)
        self.bn3 = _BatchNorm2dInit(out_channels, momentum=momentum, affine=self.affine, use_batch_statistics=training)

        if not training:
            self.bn1 = self.bn1.set_train(False)
            self.bn2 = self.bn2.set_train(False)
            self.bn3 = self.bn3.set_train(False)

        if not weights_update:
            self.conv1.weight.requires_grad = False
            self.conv2.weight.requires_grad = False
            self.conv3.weight.requires_grad = False

        self.relu = ops.ReLU()
        self.add = ops.Add()

    def construct(self, x):
        """
        construct the ResNet V1 residual block

        Args:
            x: input feature data.

        Returns:
        Tensor, output tensor.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            identity = self.conv_down_sample(identity)
            identity = self.bn_down_sample(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class BasicStem(nn.Cell):
    """
    The standard ResNet stem (layers before the first residual block),
    with a conv, relu and max_pool.
    """

    def __init__(self, in_channels=3, out_channels=64, weights_update=False):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super(BasicStem, self).__init__()
        bn_training = False
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, has_bias=False, pad_mode="pad")
        self.bn1 = _BatchNorm2dInit(64, affine=bn_training, use_batch_statistics=bn_training)
        self.relu = ops.ReLU()
        self.maxpool = nn.SequentialCell([
            nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT"),
            nn.MaxPool2d(kernel_size=3, stride=2)])
        self.weights_update = weights_update

        if not self.weights_update:
            self.conv1.weight.requires_grad = False

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)
        return c1


def build_resnet_backbone(block, config, weights_update=False):
    stem = BasicStem()
    layer_nums = config.resnet_block
    out_features = config.resnet_out_features
    in_channels = config.resnet_in_channels
    out_channels = config.resnet_out_channels
    freeze_at = config.backbone_freeze_at
    layers = []
    # use FrozenBN
    bn_training = False
    for i in range(0, len(in_channels)):
        if i == 0:
            weights_update = False
            layer = ResNetFea.make_layer(block,
                                         layer_nums[i],
                                         in_channel=in_channels[i],
                                         out_channel=out_channels[i],
                                         stride=1,
                                         training=bn_training,
                                         weights_update=weights_update)
        elif i==len(in_channels)-1 or i==len(in_channels)-2:
            weights_update = True
            layer = ResNetFea.make_layer(block,
                                         layer_nums[i],
                                         in_channel=in_channels[i],
                                         out_channel=out_channels[i],
                                         stride=2,
                                         training=bn_training,
                                         weights_update=weights_update)
        else:
            weights_update = False
            layer = ResNetFea.make_layer(block,
                                         layer_nums[i],
                                         in_channel=in_channels[i],
                                         out_channel=out_channels[i],
                                         stride=2,
                                         training=bn_training,
                                         weights_update=weights_update)
        layers.append(layer)
    return ResNetFea(stem, layers, out_features)
