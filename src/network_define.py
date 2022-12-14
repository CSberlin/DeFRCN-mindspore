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
"""Defrcn training network wrapper."""

import time
from eval import eval_defrcn
import mindspore.common.dtype as mstype
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import ParameterTuple, Tensor
from mindspore.train.callback import Callback
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore import Profiler
import os

time_stamp_init = False
time_stamp_first = 0

class LossCallBack(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF terminating training.

    Note:
        If per_print_times is 0 do not print loss.

    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, per_print_times=1, rank_id=0, lr=None):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.count = 0
        self.loss_sum = 0
        self.rank_id = rank_id
        self.lr = lr

        global time_stamp_init, time_stamp_first
        if not time_stamp_init:
            time_stamp_first = time.time()
            time_stamp_init = True

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs.asnumpy()
        cur_step_in_epoch = (cb_params.cur_step_num -
                             1) % cb_params.batch_num + 1

        self.count += 1
        self.loss_sum += float(loss)

        if self.count >= 1:
            global time_stamp_first
            time_stamp_current = time.time()
            total_loss = self.loss_sum / self.count
            loss_file = open("./loss_{}.log".format(self.rank_id), "a+")
            loss_file.write("%lu s | epoch: %s step: %s total_loss: %.5f  lr: %.6f" %
                            (time_stamp_current - time_stamp_first, cb_params.cur_epoch_num, cur_step_in_epoch,
                            total_loss, self.lr[cb_params.cur_step_num - 1]))
            loss_file.write("\n")
            loss_file.close()

            self.count = 0
            self.loss_sum = 0

class EvalCallBack(Callback):
    def __init__(self, train_epoch, step_per_epoch, eval_checkpoint_path) -> None:
        super(EvalCallBack, self).__init__()
        self.step_per_epoch = step_per_epoch
        self.epoch_num = train_epoch
        self.eval_checkpoint_path = eval_checkpoint_path

    # def on_train_epoch_end(self, run_context):
    #     cb_params = run_context.original_args()
    #     # print("epoch ", cb_params.cur_epoch_num, "end", "now start eval")
    #     # ckpt_file_name = "defrcn-{}_{}.ckpt".format(cb_params.cur_epoch_num, self.step_per_epoch)
    #     # checkpoint_path = os.path.join(self.eval_checkpoint_path, ckpt_file_name)
    #     # if (cb_params.cur_step_num >= self.step_per_epoch) and (cb_params.cur_epoch_num < self.epoch_num):
    #     #     eval_defrcn(checkpoint_path)
    #     if cb_params.cur_epoch_num == 300:
    #         print("epoch ", cb_params.cur_epoch_num, "end", "now start eval")
    #         ckpt_file_name = "defrcn-{}_{}.ckpt".format(cb_params.cur_epoch_num, self.step_per_epoch)
    #         checkpoint_path = os.path.join(self.eval_checkpoint_path, ckpt_file_name)
    #         if (cb_params.cur_step_num >= self.step_per_epoch) and (cb_params.cur_epoch_num < self.epoch_num):
    #             eval_defrcn(checkpoint_path)

    def on_train_end(self, run_context):
        print("train parse end ... now begin eval")
        cb_params = run_context.original_args()
        print("epoch ", cb_params.cur_epoch_num, "end", "now start eval")
        ckpt_file_name = "defrcn-{}_{}.ckpt".format(cb_params.cur_epoch_num, self.step_per_epoch)
        checkpoint_path = os.path.join(self.eval_checkpoint_path, ckpt_file_name)
        if (cb_params.cur_step_num >= self.step_per_epoch) and (cb_params.cur_epoch_num == self.epoch_num):
            eval_defrcn(checkpoint_path)

class LossNet(nn.Cell):
    """defrcn loss method"""
    def construct(self, x1, x2, x3, x4, x5, x6):
        return x1 + x2


class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to compute loss.

    Args:
        backbone (Cell): The target network to wrap.
        loss_fn (Cell): The loss function used to compute loss.
    """

    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
    def construct(self, x, image_id, img_shape, gt_bboxe, gt_label, gt_num):
        loss1, loss2, loss3, loss4, loss5, loss6 = self._backbone(
            x, image_id, img_shape, gt_bboxe, gt_label, gt_num)
        return self._loss_fn(loss1, loss2, loss3, loss4, loss5, loss6)

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, return backbone network.
        """
        return self._backbone


class TrainOneStepCell(nn.Cell):
    """
    Network training package class.

    Append an optimizer to the training network after that the construct function
    can be called to create the backward graph.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default value is 1.0.
        reduce_flag (bool): The reduce flag. Default value is False.
        mean (bool): Allreduce method. Default value is False.
        degree (int): Device number. Default value is None.
    """

    def __init__(self, network, optimizer, sens=1.0, reduce_flag=False, mean=False, degree=None):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True,
                                      sens_param=True)
        self.sens = Tensor([sens, ], mstype.float32)
        self.reduce_flag = reduce_flag
        if reduce_flag:
            self.grad_reducer = DistributedGradReducer(
                optimizer.parameters, mean, degree)

    def construct(self, x, image_id, img_shape, gt_bboxe, gt_label, gt_num):
        weights = self.weights
        loss = self.network(x, image_id, img_shape, gt_bboxe, gt_label, gt_num)
        grads = self.grad(self.network, weights)(
            x, image_id, img_shape, gt_bboxe, gt_label, gt_num, self.sens)
        if self.reduce_flag:
            grads = self.grad_reducer(grads)
        return ops.depend(loss, self.optimizer(grads))

class StopAtStep(ms.Callback):
    def __init__(self, start_step, stop_step):
        super(StopAtStep, self).__init__()
        self.start_step = start_step
        self.stop_step = stop_step
        self.profiler = Profiler(start_profile=False, output_path='./profiler_data')
    def step_begin(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.start_step:
            self.profiler.start()
    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.stop_step:
            self.profiler.stop()
    def end(self, run_context):
        self.profiler.analyse()

# grad_scale = ops.MultitypeFuncGraph("grad_scale")
# @grad_scale.register("Tensor", "Tensor")
# def tensor_grad_scale(scale, grad):
#     return grad * ops.Reciprocal()(scale)

# class TrainOneStepCell(nn.TrainOneStepWithLossScaleCell):
#     """
#     Append an optimizer to the training network after that the construct
#     function can be called to create the backward graph.

#     Args:
#         network (Cell): The training network. Note that loss function should have been added.
#         optimizer (Optimizer): Optimizer for updating the weights.
#         sens (Number): The adjust parameter. Default: 1.0.
#         grad_clip(bool): Whether apply global norm gradient clip before optimizer. Default: True
#     """
#     def __init__(self, network, optimizer, sens=1.0, grad_clip=True, clip_norm=1.0, overflow_check=True):
#         scaling_sens = sens
#         if isinstance(scaling_sens, (int, float)):
#             scaling_sens = ms.Tensor(scaling_sens, ms.float32)
#         super(TrainOneStepCell, self).__init__(network, optimizer, scaling_sens)
#         self.grad_clip = grad_clip
#         self.overflow_check = overflow_check
#         self.clip_norm = clip_norm

#     def construct(self, *inputs):
#         weights = self.weights
#         loss = self.network(*inputs)
#         scaling_sens = self.scale_sense

#         status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

#         scaling_sens_filled = ops.ones_like(loss) * ops.cast(scaling_sens, ops.dtype(loss))
#         grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
#         # for g in grads:
#         #    print("grad", g.max(), g.min(), g.shape)
#         grads = self.hyper_map(ops.partial(grad_scale, scaling_sens), grads)
#         # apply grad reducer on grads
#         grads = self.grad_reducer(grads)
#         if self.grad_clip:
#             grads = ops.clip_by_global_norm(grads, clip_norm=self.clip_norm)

#         # get the overflow buffer
#         cond = self.get_overflow_status(status, grads)
#         overflow = self.process_loss_scale(cond)
#         # if there is no overflow, do optimize
#         if not overflow:
#             loss = ops.depend(loss, self.optimizer(grads))
#         if overflow:
#             print(f"overflow, scale_sens: {scaling_sens}")
#         return loss
