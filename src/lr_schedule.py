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
"""lr generator for defrcn"""
from bisect import bisect_right

def _get_warmup_factor_at_iter(
    method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See :paper:`ImageNet in 1h` for more details.
    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).
    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))


def multistep_lr(config, steps_per_epoch):
    base_lr = float(config.base_lr)
    bias_lr = config.base_lr * config.bias_lr_factor
    milestones = config.milestones
    warmup_method = config.warmup_method
    warmup_iter = config.warmup_iter
    warmup_factor = config.warmup_factor
    gamma = config.warmup_gamma
    last_epoch = steps_per_epoch * config.epoch_size
    lr = []
    bias_lr_l = []
    for step in range(last_epoch):
        warmup_factor = _get_warmup_factor_at_iter(warmup_method, step, warmup_iter, warmup_factor)
        lr.append(base_lr * warmup_factor * gamma ** bisect_right(milestones, step))
        bias_lr_l.append(bias_lr * warmup_factor * gamma ** bisect_right(milestones, step))
    return lr, bias_lr
