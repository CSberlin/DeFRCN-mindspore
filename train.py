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

"""train Defrcn and get checkpoint files."""
import re
import os
import time
from pprint import pprint
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor, Parameter, ParameterTuple
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.common import set_seed
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.train.callback import SummaryCollector
from src.defrcn.de_frcn import De_Frcn
from src.network_define import LossCallBack, EvalCallBack, WithLossCell, TrainOneStepCell, LossNet, StopAtStep
from src.dataset import data_to_mindrecord_byte_image, create_defrcn_dataset
from src.lr_schedule import multistep_lr
from src.sgd_optimizer import build_optimizer
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id

def train_defrcn_():
    """ train_defrcn_ """
    print("Start create dataset!")

    # It will generate mindrecord file in config.mindrecord_dir,
    # and the file name is mindrecord0, 1, ... file_num.
    prefix = "mindrecord"
    mindrecord_dir = os.path.join(config.train_mindrecord_dir, config.dataset_train[0])
    if config.is_pretrain:
        mindrecord_file = os.path.join(config.train_mindrecord_dir, config.dataset_train[0], prefix + "0")
    if config.is_finetune:
        mindrecord_file = os.path.join(config.train_mindrecord_dir, config.dataset_train[0], prefix)
    print("CHECKING MINDRECORD FILES ...")

    if rank == 0 and not os.path.exists(mindrecord_file + ".db"):
        if not os.path.exists(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if os.path.isdir(config.coco_root):
            print("Create Mindrecord. It may take some time.")
            if config.is_pretrain:
                data_to_mindrecord_byte_image(config, config.dataset_train, prefix, True, file_num=8)
            if config.is_finetune:
                data_to_mindrecord_byte_image(config, config.dataset_train, prefix, True, file_num=1)
            print("Create Mindrecord Done, at {}".format(mindrecord_dir))
        else:
            print("coco_root not exits.")
            raise ValueError(config.coco_root)
    while not os.path.exists(mindrecord_file + ".db"):
        print("wait 5 seconds")
        time.sleep(5)

    print("CHECKING MINDRECORD FILES DONE!")

    # When create MindDataset, using the first mindrecord file, such as defrcn.mindrecord0.
    dataset = create_defrcn_dataset(config, mindrecord_file, batch_size=config.train_batch_size,
                                        device_num=device_num, rank_id=rank,
                                        num_parallel_workers=config.num_parallel_workers,
                                        python_multiprocessing=config.python_multiprocessing)

    dataset_size = dataset.get_dataset_size()
    print("Create dataset done!")

    return dataset_size, dataset


def modelarts_pre_process():
    config.save_checkpoint_path = config.output_path


def load_ckpt_to_network(net, is_pretrain, is_finetune):
    if is_pretrain == True and is_finetune == False:
        load_path = config.backbone_pretrained_path
    elif is_pretrain == False and is_finetune == True:
        load_path = config.model_pretrained_path
    if load_path != "":
        print(f"\n[{rank}]", "===> Loading from checkpoint:", load_path)
        param_dict = ms.load_checkpoint(load_path)
        try:
            ms.load_param_into_net(net, param_dict)
        except RuntimeError as ex:
            ex = str(ex)
            print("Traceback:\n", ex, flush=True)
            if "reg_scores.weight" in ex:
                exit("[ERROR] The loss calculation of faster_rcnn has been updated. "
                     "If the training is on an old version, please set `without_bg_loss` to False.")
    print(f"[{rank}]", "\tDone!\n")
    return net

@moxing_wrapper(pre_process=modelarts_pre_process)
def train_defrcn():
    """ train_defrcn """
    print(f"\n[{rank}] - rank id of process")
    dataset_size, dataset = train_defrcn_()

    print(f"\n[{rank}]", "===> Creating network...")
    net = De_Frcn(config=config)
    net = net.set_train()
    net = load_ckpt_to_network(net, config.is_pretrain, config.is_finetune)

    device_type = ms.get_context("device_target")
    print(f"\n[{rank}]", "===> Device type:", device_type, "\n")
    
    # single card, original base_lr is for 8 cards
    if not config.run_distribute:
        config.base_lr = config.base_lr / 8

    if device_type == "Ascend":
        net.to_float(ms.float32)

    print(f"\n[{rank}]", "===> Creating loss, lr and opt objects...")
    loss = LossNet()
    weight_lr, bias_lr = multistep_lr(config, dataset_size)
    weight_lr = Tensor(weight_lr, ms.float32)
    bias_lr = Tensor(bias_lr, ms.float32)
    
    opt = build_optimizer(config, net, weight_lr, bias_lr)

    print(f"[{rank}]", "\tDone!\n")

    if device_type == "Ascend":
        net_with_loss = WithLossCell(net.to_float(ms.float32), loss.to_float(ms.float32))
        if config.run_distribute:
            net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale, reduce_flag=True,
                                mean=True, degree=device_num)
        else:
            net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale)
    elif device_type == "GPU":
        net_with_loss = WithLossCell(net, loss)
        if config.run_distribute:
            net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale, reduce_flag=True,
                                mean=True, degree=device_num)
        else:
            net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale)
    else:
        net_with_loss = WithLossCell(net, loss)
        net = nn.TrainOneStepCell(net_with_loss, opt)

    print(f"\n[{rank}]", "===> Creating callbacks...")
    specified = {"collect_metric": True, "histogram_regular": "^layer1.*｜^layer2.*|^layer3.1*", "collect_graph": True,
                 "collect_dataset_graph": True}
    #summary_collector = SummaryCollector(summary_dir, collect_specified_data=specified, collect_freq=200)
    time_cb = TimeMonitor(data_size=dataset_size)
    eval_checkpoint_path = os.path.join(config.eval_checkpoint_path, "ckpt_" + str(rank) + "/")
    eval_cb = EvalCallBack(train_epoch=config.epoch_size, step_per_epoch=dataset_size, 
                            eval_checkpoint_path=eval_checkpoint_path)
    loss_cb = LossCallBack(per_print_times=dataset_size, rank_id=rank, lr=weight_lr.asnumpy())
    cb = [time_cb, loss_cb, eval_cb]#, summary_collector]
    print(f"[{rank}]", "\tDone!\n")

    print(f"\n[{rank}]", "===> Configurating checkpoint saving...")
    if config.save_checkpoint:
        ckptconfig = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * dataset_size,
                                      keep_checkpoint_max=config.keep_checkpoint_max)
        save_checkpoint_path = os.path.join(config.save_checkpoint_path, "ckpt_" + str(rank) + "/")
        ckpoint_cb = ModelCheckpoint(prefix='defrcn', directory=save_checkpoint_path, config=ckptconfig)
        cb += [ckpoint_cb]
    # profile_cb = StopAtStep(1, 5)
    # cb += [profile_cb]
    print(f"[{rank}]", "\tDone!\n")
    model = Model(net)
    model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)


if __name__ == '__main__':
    set_seed(config.seed)
    if "seed" in config.dataset_train[0]:
        config.dataset_train[0] = re.sub("seed[0-9]", "seed"+str(config.seed), config.dataset_train[0])
    local_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
    summary_dir = local_path + "/train/summary/"
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=config.device_target, device_id=get_device_id())
    if config.run_distribute:
        init()
        rank = get_rank()
        summary_dir += "thread_num_" + str(rank) + "/"     
        device_num = get_group_size()
        ms.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                     gradients_mean=True, parameter_broadcast=True)
    else:
        local_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
        summary_dir = local_path + "/train/summary/"
        rank = 0
        device_num = 1

    train_defrcn()
