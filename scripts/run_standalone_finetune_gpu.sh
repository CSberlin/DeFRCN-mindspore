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

# ------------------------------ Model Preparation -------------------------------- #
EXPNAME=$1

SRCPATH=./checkpoints/coco/${EXPNAME}
SAVEDIR=./checkpoints/coco/${EXPNAME}

IMAGENET_PRETRAIN_ASCEND=../../ImageNetPretrained/pth_convert/res101_ms.ckpt  # <-- change it to you path

python3 ../src/model_surgery.py --dataset coco --method remove --device-target GPU                   \
    --src-path ${SRCPATH}/defrcn_det_r101_base/ckpt_0/defrcn-18_6154.ckpt                     \
    --save-dir ${SAVEDIR}/defrcn_det_r101_base
BASE_WEIGHT=${SAVEDIR}/defrcn_det_r101_base/model_reset_remove.ckpt

export DEVICE_NUM=1
export RANK_ID=0
export DEVICE_ID=0
export RANK_SIZE=1

# ------------------------------ Novel Fine-tuning -------------------------------- #
# --> 1. FSRW-like, i.e. run seed0 10 times (the FSOD results on coco in most papers)
for repeat_id in 0 1 2 3 4 5 6 7 8 9
do
    for shot in 1 2 3 5 10 30
    do
        for seed in 0
        do
            read -r -p "FSRW-like fsod Are You Sure to continue this finetune? now repeat_id: ${repeat_id}, shot: ${shot}, seed: ${seed}[Y/n/q] " confirm
            case ${confirm} in 
                [yY][eE][sS]|[yY])
                    if [ -d "./checkpoints/coco/defrcn_det_r101/defrcn_fsod_r101_novel/fsrw-like/${shot}shot_seed${seed}/repeat${repeat_id}" ];
                    then
                        rm -rf ./checkpoints/coco/defrcn_det_r101/defrcn_fsod_r101_novel/fsrw-like/${shot}shot_seed${seed}/repeat${repeat_id}
                    fi
                    if [ -d "finetune" ];
                    then
                        rm -rf ./finetune
                    fi
                    mkdir ./finetune
                    
                    CONFIG_PATH=./configs/coco/defrcn_fsod_r101_novel_${shot}shot.yaml
                    OUTPUT_DIR=${SAVEDIR}/defrcn_fsod_r101_novel/fsrw-like/${shot}shot_seed${seed}/repeat${repeat_id}

                    cp ../*.py ./finetune
                    cp -r ../configs ./finetune
                    cp *.sh ./finetune
                    cp -r ../src ./finetune
                    cd ./finetune || exit
                    echo "start finetune for rank $RANK_ID, device $DEVICE_ID"
                    env > env.log
                    python train.py \
                    --config_path=${CONFIG_PATH} \
                    --device_id=${DEVICE_ID} \
                    --device_target="GPU" \
                    --save_checkpoint_path=../${OUTPUT_DIR}  \
                    --eval_checkpoint_path=../${OUTPUT_DIR} \
                    --rank_id=${RANK_ID} \
                    --run_distribute=False \
                    --device_num=${DEVICE_NUM} \
                    --pcb_modelpath=${IMAGENET_PRETRAIN_ASCEND} \
                    --model_pretrained_path=../${BASE_WEIGHT} &> log &
                    cd ..
                    ;;
                [nN][oO]|[nN])
                    echo "Jump repeat_id: ${repeat_id}, shot: ${shot}, seed: ${seed} finetune"
                        ;;
                [qQ][uU][iI][tT]|[qQ])
                    echo "quit..."
                    exit 0
                    ;;
                *)
                    echo "invalid input..."
                    exit 1
                    ;;
            esac
        done
    done
done


# ----------------------------- Model Preparation --------------------------------- #
python3 ../src/model_surgery.py --dataset coco --method randinit --device-target GPU                        \
    --src-path ${SRCPATH}/defrcn_det_r101_base/ckpt_0/defrcn-18_6154.ckpt                     \
    --save-dir ${SAVEDIR}/defrcn_det_r101_base
BASE_WEIGHT=${SAVEDIR}/defrcn_det_r101_base/model_reset_surgery.ckpt


# ------------------------------ Novel Fine-tuning ------------------------------- #
# --> 2. TFA-like, i.e. run seed0~9 for robust results (G-FSOD, 80 classes)
for seed in 0 1 2 3 4 5 6 7 8 9
do
    for shot in 1 2 3 5 10 30
    do
        read -r -p "TFA-like gfsod Are You Sure to continue this finetune? now shot: ${shot}, seed: ${seed}[Y/n/q] " confirm
        case ${confirm} in 
            [yY][eE][sS]|[yY])
                if [ -d "./checkpoints/coco/defrcn_det_r101/defrcn_gfsod_r101_novel/tfa-like/${shot}shot_seed${seed}" ];
                then
                    rm -rf ./checkpoints/coco/defrcn_det_r101/defrcn_gfsod_r101_novel/tfa-like/${shot}shot_seed${seed}
                fi
                if [ -d "./finetune" ];
                then
                    rm -rf ./finetune
                fi
                mkdir ./finetune
                CONFIG_PATH=./configs/coco/defrcn_gfsod_r101_novel_${shot}shot.yaml
                OUTPUT_DIR=${SAVEDIR}/defrcn_gfsod_r101_novel/tfa-like/${shot}shot_seed${seed}
                
                cp ../*.py ./finetune
                cp -r ../configs ./finetune
                cp *.sh ./finetune
                cp -r ../src ./finetune
                cd ./finetune || exit
                echo "start finetuning on $RANK_SIZE devices"
                env > env.log

                python3 train.py  \
                --seed=${seed}  \
                --config_path=${CONFIG_PATH}\
                --run_distribute=False \
                --device_target="GPU" \
                --device_num=$RANK_SIZE \
                --save_checkpoint_path=../${OUTPUT_DIR} \
                --eval_checkpoint_path=../${OUTPUT_DIR} \
                --pcb_modelpath=${IMAGENET_PRETRAIN_ASCEND} \
                --model_pretrained_path=../${BASE_WEIGHT} &> log &
                cd ..
                ;;
            [nN][oO]|[nN])
                echo "Jump shot: ${shot}, seed: ${seed} finetune"
                    ;;
            [qQ][uU][iI][tT]|[qQ])
                echo "quit..."
                exit 0
                ;;
            *)
                echo "invalid input..."
                exit 1
                ;;
        esac
    done
done


# ------------------------------ Novel Fine-tuning ------------------------------- #  not necessary, just for the completeness of defrcn
# --> 3. TFA-like, i.e. run seed0~9 for robust results
BASE_WEIGHT=${SAVEDIR}/defrcn_det_r101_base/model_reset_remove.ckpt
for seed in 0 1 2 3 4 5 6 7 8 9
do
    for shot in 1 2 3 5 10 30
    do
        read -r -p "TFA-like fsod Are You Sure to continue this finetune? now shot: ${shot}, seed: ${seed}[Y/n/q] " confirm
        case ${confirm} in 
            [yY][eE][sS]|[yY])
                if [ -d "./checkpoints/coco/defrcn_det_r101/defrcn_gfsod_r101_novel/tfa-like/${shot}shot_seed${seed}" ];
                then
                    rm -rf ./checkpoints/coco/defrcn_det_r101/defrcn_fsod_r101_novel/tfa-like/${shot}shot_seed${seed}
                fi
                if [ -d "./finetune" ];
                then
                    rm -rf ./finetune
                fi
                mkdir ./finetune
                CONFIG_PATH=configs/coco/defrcn_fsod_r101_novel_${shot}shot.yaml
                OUTPUT_DIR=${SAVEDIR}/defrcn_fsod_r101_novel/tfa-like/${shot}shot_seed${seed}
                cp ../*.py ./finetune
                cp -r ../configs ./finetune
                cp *.sh ./finetune
                cp -r ../src ./finetune
                cd ./finetune || exit
                echo "start finetuning on $RANK_SIZE devices"
                env > env.log
                python3 train.py  \
                --seed=${seed}  \
                --config_path=${CONFIG_PATH}\
                --run_distribute=False \
                --device_target="GPU" \
                --device_num=$RANK_SIZE \
                --save_checkpoint_path=../${OUTPUT_DIR} \
                --eval_checkpoint_path=../${OUTPUT_DIR} \
                --pcb_modelpath=${IMAGENET_PRETRAIN_ASCEND} \
                --model_pretrained_path=../${BASE_WEIGHT} > train.log 2>&1 &
                cd ..
                ;;
            [nN][oO]|[nN])
                    echo "Jump shot: ${shot}, seed: ${seed} finetune"
                        ;;
            [qQ][uU][iI][tT]|[qQ])
                echo "quit..."
                exit 0
                ;;
            *)
                echo "invalid input..."
                exit 1
                ;;
        esac
    done
done
