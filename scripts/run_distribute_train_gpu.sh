#!/bin/bash
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
if [ $# -le 0 ]
then 
    echo "Usage: sh run_distribute_train_gpu.sh [EXPNAME]"
exit 1
fi

# EXPNAME一般写 defrcn_det_r101
EXPNAME=$1
SAVEDIR=checkpoints/coco/${EXPNAME}
IMAGENET_PRETRAIN=../../ImageNetPretrained/pkl_convert/backbone.ckpt                            # <-- change it to you path
IMAGENET_PRETRAIN_ASCEND=../../ImageNetPretrained/pth_convert/res101_ms.ckpt  # <-- change it to you path

if [ -d "./train" ];
then
    rm -rf ./train
fi
mkdir ./train

if [ -d "./checkpoints/coco/defrcn_det_r101/defrcn_det_r101_base" ];
then
    rm -rf ./checkpoints/coco/defrcn_det_r101/defrcn_det_r101_base
fi

cp ../*.py ./train
cp -r ../configs ./train
cp *.sh ./train
cp -r ../src ./train
cd ./train || exit

export RANK_SIZE=8
echo "start training on $RANK_SIZE devices"
env > env.log
mpirun -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout --allow-run-as-root \
    python train.py  \
    --config_path=./configs/coco/defrcn_det_r101_base.yaml \
    --run_distribute=True \
    --device_target="GPU" \
    --device_num=$RANK_SIZE \
    --backbone_pretrained_path=$IMAGENET_PRETRAIN > train.log 2>&1 &