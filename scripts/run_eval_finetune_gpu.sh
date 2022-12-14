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

export DEVICE_NUM=1
export RANK_SIZE=$DEVICE_NUM
export DEVICE_ID=7
export RANK_ID=0

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp -r ../configs ./eval
cp *.sh ./eval
cp -r ../src ./eval
cd ./eval || exit
env > env.log
echo "start eval for device $DEVICE_ID"
# remember change seed and eval_checkpoint_path in your config according to exp
python eval.py --seed=0 --run_distribute=False --config_path=./configs/coco/defrcn_fsod_r101_novel_10shot.yaml --device_target="GPU" \
--device_id=$DEVICE_ID --eval_checkpoint_path=../checkpoints/coco/defrcn_det_r101/defrcn_fsod_r101_novel/fsrw-like/10shot_seed0/repeat0/ckpt_0/defrcn-200_8.ckpt &> log &
cd ..