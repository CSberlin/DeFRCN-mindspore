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

if [[ $# -lt 4 || $# -gt 7 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANNO_PATH] [DEVICE_TARGET] [IMAGE_WIDTH](optional) [IMAGE_HEIGHT](optional) [DEVICE_ID](optional)
    DEVICE_TARGET can choose from [Ascend, GPU, CPU], 
    IMAGE_WIDTH, IMAGE_HEIGHT and DEVICE_ID is optional. 
    IMAGE_WIDTH and IMAGE_HEIGHT must be set at the same time
    or not at the same time. IMAGE_WIDTH default value is 1280, IMAGE_HEIGHT default value is 768,
    DEVICE_ID can be set by environment variable device_id, otherwise the value is zero"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}
model=$(get_real_path $1)
data_path=$(get_real_path $2)
anno_path=$(get_real_path $3)
device_target=$4
device_id=0
image_width=1280
image_height=768
# If restore_bbox is set to False during export.py, it should be also set to false here (only support true or false and case sensitive).
restore_bbox=true

if [ $# -eq 5 ]; then
    device_id=$5
fi

if [ $# -eq 6 ]; then
    image_width=$5
    image_height=$6
fi

if [ $# -eq 7 ]; then
    image_width=$5
    image_height=$6
    device_id=$7
fi

echo "mindir name: "$model
echo "dataset path: "$data_path
echo "anno_path: " $anno_path
echo "device id: "$device_id
echo "image_width: "$image_width
echo "image_height: "$image_height
echo "restore_bbox: "$restore_bbox
echo "device_target: "$device_target

export ASCEND_HOME=/usr/local/Ascend/
if [ -d ${ASCEND_HOME}/ascend-toolkit ]; then
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/ascend-toolkit/latest/atc/lib64:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export TBE_IMPL_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:${TBE_IMPL_PATH}:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp
else
    export ASCEND_HOME=/usr/local/Ascend/latest/
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/atc/ccec_compiler/bin:$ASCEND_HOME/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/atc/lib64:$ASCEND_HOME/acllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:$ASCEND_HOME/atc/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/opp
fi

if [ -d ./input_Files ]; then
    rm -rf ./input_Files
fi
mkdir ./input_Files

function preprocess_data()
{
    python ../preprocess.py \
    --input_path=$data_path \
    --config_path=$anno_path \
    --coco_root=../datasets/coco \
    --test_mindrecord_dir=../datasets/coco2014/MindRecord_COCO_TEST \
    --datasets_root=../datasets
}
preprocess_data
if [ $? -ne 0 ]; then
    echo "preprocess_data code failed"
    exit 1
fi

function compile_app()
{
    cd ../ascend310_infer || exit
    bash build.sh &> build.log
    cd - || exit
}

function infer()
{
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    ../ascend310_infer/out/main --mindir_path=$model \
    --dataset_path=$data_path \
    --device_id=$device_id \
    --IMAGEWIDTH=$image_width \
    --IMAGEHEIGHT=$image_height  \
    --device_type=$device_target \
    --RESTOREBBOX=$restore_bbox &> infer.log
}

function cal_acc()
{
    python ../postprocess.py \
    --input_path=$data_path \
    --config_path=$anno_path \
    --coco_root=../datasets/coco \
    --test_mindrecord_dir=../datasets/coco2014/MindRecord_COCO_TEST \
    --datasets_root=../datasets \
    --result_path=./result_Files &> acc.log &
}

compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi
infer
if [ $? -ne 0 ]; then
    echo "execute inference failed"
    exit 1
fi
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi