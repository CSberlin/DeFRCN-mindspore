#!/bin/bash
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

if [ $# -le 0 ]
then 
    echo "Usage: sh run_distribute_train_ascend.sh [EXPNAME]"
exit 1
fi


# EXPNAME一般写 defrcn_det_r101
EXPNAME=$1
SAVEDIR=checkpoints/coco/${EXPNAME}
IMAGENET_PRETRAIN=../../ImageNetPretrained/pkl_convert/backbone.ckpt                            # <-- change it to you path
IMAGENET_PRETRAIN_ASCEND=../../ImageNetPretrained/pth_convert/res101_ms.ckpt  # <-- change it to you path

export DEVICE_NUM=1
export RANK_ID=0
export DEVICE_ID=0
export RANK_SIZE=1

if [ -d "./train" ];
then
    rm -rf ./train
fi

if [ -d "./checkpoints/coco/defrcn_det_r101/defrcn_det_r101_base" ];
then
    rm -rf ./checkpoints/coco/defrcn_det_r101/defrcn_det_r101_base
fi
mkdir ./train
cp ../*.py ./train
cp -r ../configs ./train
cp *.sh ./train
cp -r ../src ./train
cd ./train || exit
echo "start training for rank $RANK_ID, device $DEVICE_ID"
env > env.log
python train.py --config_path=./configs/coco/defrcn_det_r101_base.yaml --device_id=$DEVICE_ID --device_target="Ascend" \
--rank_id=$RANK_ID --run_distribute=False --device_num=$DEVICE_NUM --backbone_pretrained_path=$IMAGENET_PRETRAIN &> log &
cd ..