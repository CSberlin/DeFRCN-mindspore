# Contents

- [Contents](#contents)
- [DeFRCN Description](#defrcn-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
  - [Run on Ascend](#run-on-ascend)
  - [Run on GPU](#run-on-gpu)
- [Script Description](#script-description)
  - [Script and Sample Code](#script-and-sample-code)
  - [Training Process](#training-process)
    - [Usage](#usage)
      - [on Ascend](#on-ascend)
      - [on GPU](#on-gpu)
    - [Result](#result)
  - [Evaluation Process](#evaluation-process)
    - [Usage](#usage-1)
      - [on Ascend](#on-ascend-1)
      - [on GPU](#on-gpu-1)
    - [Result](#result-1)
  - [Model Export](#model-export)
  - [Inference Process](#inference-process)
    - [Usage](#usage-2)
    - [result](#result-2)
- [Model Description](#model-description)
  - [Performance](#performance)
    - [Trian Phrase Performance](#trian-phrase-performance)
      - [Pretrain Train Phrase Performance](#pretrain-train-phrase-performance)
      - [Finetune Train Phrase Performance](#finetune-train-phrase-performance)
        - [FSRW-like fsod Train Phrase Performance](#fsrw-like-fsod-train-phrase-performance)
          - [GPU](#gpu)
        - [TFA-like gfsod Train Phrase Performance](#tfa-like-gfsod-train-phrase-performance)
          - [GPU](#gpu-1)
    - [Evaluation Phrase Performance](#evaluation-phrase-performance)
      - [Pretrain Evaluation Phrase Performance](#pretrain-evaluation-phrase-performance)
      - [Finetune Evaluation Phrase Performance](#finetune-evaluation-phrase-performance)
        - [FSRW-like fsod Eval Phrase Performance](#fsrw-like-fsod-eval-phrase-performance)
          - [GPU](#gpu-2)
        - [TFA-like fsod Eval Phrase Performance](#tfa-like-fsod-eval-phrase-performance)
          - [GPU](#gpu-3)
    - [Inference Performance](#inference-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# DeFRCN Description

To be concrete, DeFRCN extend Faster R-CNN by introducing Gradient Decoupled Layer for multistage decoupling and Prototypical Calibration Block for multi-task decoupling. The former is a novel deep layer with redefining the feature-forward operation and gradientbackward operation for decoupling its subsequent layer and preceding layer, and the latter is an offline prototype-based classification model with taking the proposals from detector as input and boosting the original classification scores with additional pairwise scores for calibration.

[Paper](https://arxiv.org/abs/2108.09017)：Limeng Qiao, Yuxuan Zhao, Zhiyuan Li, Xi Qiu, Jianan Wu and Chi Zhang. “DeFRCN: Decoupled Faster R-CNN for Few-Shot Object Detection..” international conference on computer vision(2021): n. pag.

# Model Architecture

DeFRCN is a model structure composed of Faster-rcnn (modified after Model-zoo copy) +GDL+PCB module. The model is suitable for small sample tasks through two stages of large-scale pre-training and finetuning. GDL module is used to clip the gradient of RPN and RCNN return, which is equivalent to decoupling operation. PCB was only used in the eval stage of the fine-tuning stage. After establishing the feature database of each category by extracting features from Support Set samples, each Query Set sample was matched into the category of cosine distance. Then a weighted sum is made with the model output to increase the accuracy of classification.

# Dataset

Data set used:

|  dataset  | 大小  |                                             谷歌云                                             |                           百度云                            |                                      参考                                      |
| :-------: | :---: | :--------------------------------------------------------------------------------------------: | :---------------------------------------------------------: | :----------------------------------------------------------------------------: |
|  VOC2007  | 0.8G  | [download](https://drive.google.com/file/d/1BcuJ9j9Mtymp56qGSOfYxlXN4uEVyxFm/view?usp=sharing) | [download](https://pan.baidu.com/s/1kjAmHY5JKDoG0L65T3dK9g) |                                       -                                        |
|  VOC2012  | 3.5G  | [download](https://drive.google.com/file/d/1NjztPltqm-Z-pG94a6PiPVP4BgD8Sz1H/view?usp=sharing) | [download](https://pan.baidu.com/s/1DUJT85AG_fqP9NRPhnwU2Q) |                                       -                                        |
| vocsplit  |  <1M  | [download](https://drive.google.com/file/d/1BpDDqJ0p-fQAFN_pthn2gqiK5nWGJ-1a/view?usp=sharing) | [download](https://pan.baidu.com/s/1518_egXZoJNhqH4KRDQvfw) | refer from [TFA](https://github.com/ucbdrive/few-shot-object-detection#models) |
|   COCO    | ~19G  |                                               -                                                |                              -                              |           download from [offical](https://cocodataset.org/#download)           |
| cocosplit | 174M  | [download](https://drive.google.com/file/d/1T_cYLxNqYlbnFNJt8IVvT7ZkWb5c0esj/view?usp=sharing) | [download](https://pan.baidu.com/s/1NELvshrbkpRS8BiuBIr5gA) | refer from [TFA](https://github.com/ucbdrive/few-shot-object-detection#models) |

  - Unzip the downloaded data-source to datasets and put it into your project directory:
    ```angular2html
      ...
      datasets
        | -- coco (trainval2014/*.jpg, val2014/*.jpg, annotations/*.json)
        | -- cocosplit
        | -- VOC2007
        | -- VOC2012
        | -- vocsplit
      de_frcn
      ...
    ```
# Environment Requirements

- Hardware（Ascend/GPU）

    - Prepare hardware environment with Ascend910 processor.
    - GPU NVIDIA 3090

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset COCO2017.

- Download Coco-related datasets and put them in the datasets directory

    1. If coco dataset is used. **Select dataset to coco when run script.**
        Install Cython and pycocotool, and you can also install mmcv to process data.

        ```pip
        pip install Cython

        pip install pycocotools

        pip install mmcv==0.2.14
        ```

        Change COCO_ROOT and any other required Settings in the 'configs' directory accordingly, as required by the model run. The directory structure is as follows. After downloading 2014 data set, remember to copy val2014/*.jpg data to trainval2014 directory:

        ```angular2html
        ...
        configs
        datasets
          | -- coco (trainval2014/*.jpg, val2014/*.jpg, annotations/*.json)
          | -- cocosplit
        de_frcn
        ...
        ```

# Quick Start

After installing MindSpore through the official website, you can follow the following steps to train and evaluate:

Note:

1. the first run will generate the mindeocrd file, which will take a long time.
2. We use imagenet preprocessing weights to initialize our model. Download the same model from here: [GoogleDrive](https://drive.google.com/file/d/1rsE20_fSkYeIhFaNU04rBfEDkMENLibj/view?usp=sharing) [BaiduYun](https://pan.baidu.com/s/1IfxFq15LVUI3iIMGFT8slw)
  - The extract code for all BaiduYun link is **0000**
  - included backbone and pcb 's resnet101 param
1. The directory structure in the downloaded pre-training parameter file is `ImageNetPretrained`。
    ```angular2html
        ...
        configs
        ImageNetPretrained
          | -- MSRA
          | -- torchvision
          | -- pkl_convert（create by yourself）
          | -- pth_convert（create by yourself）
        de_frcn
        ...
    ```
4. Copy the 'pkl2ckpt.py' file from the 'src' directory to' pkl_convert 'and the' pth2ckpt.py 'file from the' src' directory to 'pth_convert'. Run the following commands in the respective directories to generate ckpt files
    ```angular2html
    python pkl2ckpt.py
    python pth2ckpt.py
    ```

## Run on Ascend

```shell
# pretrain standalone train phrase
bash run_standalone_train_ascend.sh [EXP_NAME]

# pretrain distribute train phrase
bash run_distribute_train_ascend.sh [EXP_NAME]

# pretrain standalone eval phrase
bash run_eval_pretrain_ascend.sh

# finetune standalone train phrase
bash run_standalone_finetune_ascend.sh [EXP_NAME]

# finetune distribute train phrase
bash run_distribute_finetune_ascend.sh [EXP_NAME]

# finetune standalone eval phrase
bash run_eval_finetune_ascend.sh

# inference phrase (IMAGE_WIDTH and IMAGE_HEIGHT must be use the default values at the same time.)
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [DEVICE_TARGET] [IMAGE_WIDTH](optional) [IMAGE_HEIGHT](optional) [DEVICE_ID](optional)
```

## Run on GPU

```shell

# pretrain standalone train phrase
bash run_standalone_train_gpu.sh [EXP_NAME]

# pretrain distribute train phrase
bash run_distribute_train_gpu.sh [EXP_NAME]

# pretrain standalone eval phrase
bash run_eval_pretrain_gpu.sh

# finetune standalone train phrase
bash run_standalone_finetune_gpu.sh [EXP_NAME]

# finetune distribute train phrase
bash run_distribute_finetune_gpu.sh [EXP_NAME]

# finetune standalone eval phrase
bash run_eval_finetune_gpu.sh

```

5. Inference

```shell
# inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [IMAGE_WIDTH](optional) [IMAGE_HEIGHT](optional) [DEVICE_ID](optional)
```

# Script Description

## Script and Sample Code
```shell
de-frcn
├─ README.md                                     // DeFRCN EN_markdown
├─ README_CN.md                                  // DeFRCN CN_markdown
├─ ascend310_infer                               // 310 inference code
├─ configs                                       // yaml config file
├─ eval.py                                       // model eval script
├─ requirements.txt                              // requirement file
├─ scripts
│  ├─ run_distribute_finetune_ascend.sh          // distribute Ascend finetune train script
│  ├─ run_distribute_finetune_gpu.sh             // distribute GPU finetune train script
│  ├─ run_distribute_train_ascend.sh             // distribute Ascend pretrain train script
│  ├─ run_distribute_train_gpu.sh                // distribute GPU pretrain train script
│  ├─ run_eval_finetune_ascend.sh                // standalone Ascend finetune eval script
│  ├─ run_eval_finetune_gpu.sh                   // standalone GPU eval finetune eval script
│  ├─ run_eval_pretrain_ascend.sh                // standalone Ascend pretrain eval script
│  ├─ run_eval_pretrain_gpu.sh                   // standalone GPU pretrain eval script
│  ├─ run_infer_310.sh                           // 310 Ascend infer script
│  ├─ run_standalone_finetune_ascend.sh          // standalone Ascend finetune train script
│  ├─ run_standalone_finetune_gpu.sh             // standalone GPU finetune train script
│  ├─ run_standalone_train_ascend.sh             // standalone Ascend pretrain train script
│  └─ run_standalone_train_gpu.sh                // standalone GPU pretrain train script
├─ src
│  ├─ dataset.py                                 // create and aug dataset
│  ├─ defrcn
│  │  ├─ __init__.py                             // init file
│  │  ├─ anchor_generator.py                     // anchor_generator file
│  │  ├─ bbox_assign_sample.py                   // box sampler for rpn
│  │  ├─ bbox_assign_sample_stage2.py            // box sampler for rcnn
│  │  ├─ built_datacategory.py                   // built dataset category
│  │  ├─ de_frcn.py                              // DeFRCN net
│  │  ├─ gdl.py                                  // gradient decouple layer
│  │  ├─ meta_coco.py                            // register coco dataset
│  │  ├─ pcb.py                                  // Prototypical Calibration Block
│  │  ├─ pcb_resnet.py                           // pcb feature extractor
│  │  ├─ proposal_generator.py                   // proposal bbox
│  │  ├─ rcnn.py                                 // R-CNN net
│  │  ├─ resnet.py                               // backbone
│  │  ├─ roi_align.py                            // ROI_align net
│  │  └─ rpn.py                                  // region proposal net
│  ├─ detecteval.py                              // detecteval uitls
│  ├─ lr_schedule.py                             // scheduler
│  ├─ model_surgery.py                           // pretrain ckpt to finetune ckpt
│  ├─ pkl2ckpt.py                                // pkl to ckpt
│  ├─ pth2ckpt.py                                // pth to ckpt
│  ├─ model_utils
│  │  ├─ __init__.py                             // init file
│  │  ├─ config.py                               // generate config var
│  │  ├─ device_adapter.py                       // get cloud id
│  │  ├─ local_adapter.py                        // get local id
│  │  └─ moxing_adapter.py                       // prepare cloud data
│  ├─ network_define.py                          // DeFRCN net define
│  ├─ sgd_optimizer.py                           // optimizer
│  └─ util.py                                    // utils func
└─ train.py                                      // train py
└─ preprocess.py                                 // infer preprocess
└─ postprocess.py                                // infer postprocess
```

## Training Process

### Usage

#### on Ascend

```shell
# standalone pretrain ascend train
bash run_standalone_train_ascend.sh [EXP_NAME]

# distribute pretrain ascend train
bash run_distribute_train_ascend.sh [EXP_NAME]

# standalone finetune ascend eval
bash run_standalone_finetune_ascend.sh [EXP_NAME]

# distribute pretrain ascend eval
bash run_distribute_finetune_ascend.sh [EXP_NAME]
```

#### on GPU

```shell
# standalone pretrain ascend train
bash run_standalone_train_gpu.sh [EXP_NAME]

# distribute pretrain ascend train
bash run_distribute_train_gpu.sh [EXP_NAME]

# standalone finetune ascend eval
bash run_standalone_finetune_gpu.sh [EXP_NAME]

# distribute finetune ascend eval
bash run_distribute_finetune_gpu.sh [EXP_NAME]
```

Notes:

1. Rank_table.json which is specified by RANK_TABLE_FILE is needed when you are running a distribute task. You can generate it by using the [hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).
2. Currently, it only runs on coco data sets.
3. Modify the seed parameter eval_checkpoint_path when using finetune phase eval.
4. In finetune stage, if device_id has problems, remember to manually change device_id of context in model_surgery.py.
5. If evalcallback, which is automatically invoked after the training phase, runs for a long time without error, you can manually end the process after the ckpt file is generated and then invoke the eval script to start eval.
6. During the training process, it will be found that the loss tends to be stable and no longer declines after dozens of steps, which is a normal phenomenon for the target detection task. Please continue to train more epoch to get better results.

### Result

The training results are saved in the sample path with a folder name beginning with "train" or "train_parallel". You can find the result in the corresponding loss.log, as shown below.
checkpoints are saved in the script directory of checkpoints.

```log
# distribute training result(8p)
3075 s | epoch: 1 step: 6154 total_loss: 0.35424  lr: 0.020000
6131 s | epoch: 2 step: 6154 total_loss: 0.08087  lr: 0.020000
9449 s | epoch: 3 step: 6154 total_loss: 0.43327  lr: 0.020000
12455 s | epoch: 4 step: 6154 total_loss: 0.85868  lr: 0.020000
...
48667 s | epoch: 16 step: 6154 total_loss: 0.09816  lr: 0.002000
51577 s | epoch: 17 step: 6154 total_loss: 0.17274  lr: 0.000200
54484 s | epoch: 18 step: 6154 total_loss: 0.22159  lr: 0.000200
```

## Evaluation Process

### Usage

#### on Ascend

```shell
# pretrain eval
bash run_eval_pretrain_ascend.sh

# finetune eval
bash run_eval_finetune_ascend.sh
```

#### on GPU

```shell
# pretrain eval
bash run_eval_pretrain_gpu.sh

# finetune eval
bash run_eval_finetune_gpu.sh
```

### Result

Eval result will be stored in the example path, whose folder name is "eval". Under this, you can find result like the following in log.

```log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.346
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.516
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.380
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.169
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.389
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.513
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.298
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.411
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.415
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.201
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.620
```

```log
Per-category bbox AP: 
| category       | AP     | category      | AP     | category     | AP     |
| :------------- | :----- | :------------ | :----- | :----------- | :----- |
| truck          | 32.593 | traffic light | 22.976 | fire hydrant | 65.070 |
| stop sign      | 67.771 | parking meter | 36.660 | bench        | 18.118 |
| elephant       | 65.753 | bear          | 59.979 | zebra        | 60.796 |
| giraffe        | 65.920 | backpack      | 15.580 | umbrella     | 31.583 |
| handbag        | 14.022 | tie           | 31.889 | suitcase     | 31.019 |
| frisbee        | 54.297 | skis          | 22.370 | snowboard    | 33.025 |
| sports ball    | 35.419 | kite          | 33.957 | baseball bat | 28.330 |
| baseball glove | 28.363 | skateboard    | 45.186 | surfboard    | 38.465 |
| tennis racket  | 48.659 | wine glass    | 29.142 | cup          | 32.225 |
| fork           | 30.701 | knife         | 14.685 | spoon        | 14.357 |
| bowl           | 33.161 | banana        | 20.671 | apple        | 14.205 |
| sandwich       | 35.420 | orange        | 22.179 | broccoli     | 19.296 |
| carrot         | 18.010 | hot dog       | 29.252 | pizza        | 50.440 |
| donut          | 46.093 | cake          | 35.906 | bed          | 39.946 |
| toilet         | 53.539 | laptop        | 53.899 | mouse        | 41.457 |
| remote         | 27.970 | keyboard      | 50.356 | cell phone   | 25.025 |
| microwave      | 48.486 | oven          | 32.957 | toaster      | 17.426 |
| sink           | 31.630 | refrigerator  | 49.336 | book         | 6.091  |
| clock          | 43.641 | vase          | 30.985 | scissors     | 26.901 |
| teddy bear     | 40.281 | hair drier    | 7.228  | toothbrush   | 14.662 |
```

## Model Export

```shell
python export.py --config_path [CONFIG_PATH] --export_ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --export_file_format[EXPORT_FORMAT]
```

`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]


## Inference Process

### Usage

Before performing inference, the model file must be exported by export script on the Ascend910 environment.
The following example only supports mindir inference with batch_size=1.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANNO_PATH] [DEVICE_TARGET] [IMAGE_WIDTH](optional) [IMAGE_HEIGHT](optional) [DEVICE_ID](optional)
```

### result

Inference result is saved in current path, you can find result like this in acc.log file.



# Model Description

## Performance
### Trian Phrase Performance
#### Pretrain Train Phrase Performance
| Parameters        | Ascend                                                     | GPU                                                        |
| ----------------- | ---------------------------------------------------------- | ---------------------------------------------------------- |
| Model Version     | V1                                                         | V1                                                         |
| Resource          | Ascend 910；CPU 2.60GHz，192 cores；32G                    | NVIDIA 3090 24G                                            |
| Uploaded Date     | 2022/12/12                                                 | 2022/12/12                                                 |
| MindSpore Version | 1.9.0                                                      | mindspore-gpu 1.8.1                                        |
| Dataset           | COCO 2014                                                  | COCO 2014                                                  |
| Train param       | epoch=18, batch_size=2                                     | epoch=18, batch_size=2                                     |
| Optimizer         | SGD                                                        | SGD                                                        |
| Loss              | Softmax cross entropy，Sigmoid cross entropy，SmoothL1Loss | Softmax cross entropy，Sigmoid cross entropy，SmoothL1Loss |
| Speed             |                                                            | 1pc：264ms/step；8pcs：460ms/step                          |
| Total Time        |                                                            | 1pc：65.03h；8pcs：14.13h                                  |
| Param(M)          | 658.3                                                      | 658.3                                                      |

#### Finetune Train Phrase Performance
##### FSRW-like fsod Train Phrase Performance
- base enviroment the same as Pretrain Phrase
###### GPU
| Parameters | 1shot                 | 2shot                 | 3shot                | 5shot                | 10shot           | 30shot           |
| ---------- | --------------------- | --------------------- | -------------------- | -------------------- | ---------------- | ---------------- |
| Speed      | 8pcs: 2478.433ms/step | 8pcs: 1312.433ms/step | 8pcs: 983.971ms/step | 8pcs: 690.911ms/step | 8pcs: 573ms/step | 8pcs: 420ms/step |
| Total Time | 0h33m2s               | 0h26m48s              | 0h26m42s             | 0h23m12s             | 0h15m21s         | 0h14m24s         |
| Param(M)   | 473                   | 473                   | 473                  | 473                  | 473              | 473              |

##### TFA-like gfsod Train Phrase Performance
- base enviroment the same as Pretrain Phrase
###### GPU
| Parameters | 1shot                | 2shot                | 3shot                | 5shot                | 10shot               | 30shot               |
| ---------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| Speed      | 8pcs: 715.817ms/step | 8pcs: 582.317ms/step | 8pcs: 512.302ms/step | 8pcs: 462.155ms/step | 8pcs: 416.027ms/step | 8pcs: 371.619ms/step |
| Total Time | 0h38m41s             | 0h31m3s              | 0h32m1s              | 0h29m36s             | 0h21m2s              | 0h36m39s             |
| Param(M)   | 481                  | 481                  | 481                  | 481                  | 481                  | 481                  |



### Evaluation Phrase Performance

#### Pretrain Evaluation Phrase Performance
| Param             | Ascend              | GPU                 |
| ----------------- | ------------------- | ------------------- |
| Model Version     | V1                  | V1                  |
| Resource          | Ascend 910          | NVIDIA 3090 24G     |
| Uploaded Date     | 2022/12/12          | 2022/12/12          |
| MindSpore Version | 1.9.0               | mindspore-gpu 1.8.1 |
| Dataset           | COCO2014            | COCO2014            |
| batch_size        | 2                   | 2                   |
| Output            | mAP                 | mAP                 |
| Acc               |                     | IoU=0.50：51.7%     |
| Model             | 658.3M（.ckpt文件） | 658.3M（.ckpt文件） |

#### Finetune Evaluation Phrase Performance

##### FSRW-like fsod Eval Phrase Performance
###### GPU
- base enviroment the same as Pretrain Phrase

| shot    | mAP<sup>novel</sup> | infer model     |
| ------- | ------------------- | --------------- |
| 1 shot  | 5.9                 | 472.6M（.ckpt)  |
| 2 shot  | 8.1                 | 472.7M（.ckpt)  |
| 3 shot  | 11.0                | 472.7M（.ckpt)  |
| 5 shot  | 11.0                | 472.96M（.ckpt) |
| 10 shot | 13.4                | 472.84M（.ckpt) |
| 30 shot | 15.9                | 472.98M（.ckpt) |
##### TFA-like fsod Eval Phrase Performance
###### GPU
- base enviroment the same as Pretrain Phrase

| shot    | mAP<sup>novel</sup> | infer model     |
| ------- | ------------------- | --------------- |
| 1 shot  | 5.0                 | 480.35M（.ckpt) |
| 2 shot  | 8.0                 | 480.4M（.ckpt)  |
| 3 shot  | 10.6                | 480.54M（.ckpt) |
| 5 shot  | 11.7                | 480.54M（.ckpt) |
| 10 shot | 13.5                | 480.29M（.ckpt) |
| 30 shot | 17.6                | 481.15M（.ckpt) |
### Inference Performance



# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).

