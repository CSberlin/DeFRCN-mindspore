# 目录

- [目录](#目录)
- [DeFRCN 描述](#defrcn-描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
  - [在Ascend上运行](#在ascend上运行)
  - [在GPU上运行](#在gpu上运行)
- [脚本说明](#脚本说明)
  - [脚本及样例代码](#脚本及样例代码)
  - [训练过程](#训练过程)
    - [用法](#用法)
      - [在Ascend上运行](#在ascend上运行-1)
      - [在GPU上运行](#在gpu上运行-1)
    - [结果](#结果)
  - [评估过程](#评估过程)
    - [用法](#用法-1)
      - [在Ascend上运行](#在ascend上运行-2)
      - [在GPU上运行](#在gpu上运行-2)
    - [结果](#结果-1)
  - [模型导出](#模型导出)
  - [推理过程](#推理过程)
    - [使用方法](#使用方法)
    - [结果](#结果-2)
- [模型描述](#模型描述)
  - [性能](#性能)
    - [训练性能](#训练性能)
      - [大规模训练阶段](#大规模训练阶段)
      - [微调训练阶段性能](#微调训练阶段性能)
        - [FSRW-like fsod 训练阶段性能](#fsrw-like-fsod-训练阶段性能)
          - [GPU](#gpu)
        - [TFA-like gfsod 训练阶段性能](#tfa-like-gfsod-训练阶段性能)
          - [GPU](#gpu-1)
    - [评估性能](#评估性能)
      - [大规模训练阶段评估性能](#大规模训练阶段评估性能)
      - [微调阶段评估性能](#微调阶段评估性能)
        - [FSRW-like fsod 微调阶段评估性能](#fsrw-like-fsod-微调阶段评估性能)
          - [GPU](#gpu-2)
        - [TFA-like fsod 微调阶段评估性能](#tfa-like-fsod-微调阶段评估性能)
          - [GPU](#gpu-3)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# DeFRCN 描述

DeFRCN提出，引入用于多阶段解耦的梯度解耦层和用于多任务解耦的原型校准块来扩展Faster R-CNN。前者是一种新的深层分类模型，通过重新定义特征前向操作和梯度后向操作来分离其后层和前层；后者是一种基于离线原型的分类模型，它以检测器的建议作为输入，并用额外的余弦分数来提高原始分类分数以进行校正。

[论文](https://arxiv.org/abs/2108.09017)：Limeng Qiao, Yuxuan Zhao, Zhiyuan Li, Xi Qiu, Jianan Wu and Chi Zhang. “DeFRCN: Decoupled Faster R-CNN for Few-Shot Object Detection..” international conference on computer vision(2021): n. pag.

# 模型架构

DeFRCN属于模型结构由Faster-rcnn（model-zoo照搬后修改）+GDL+PCB模块组成。通过大规模预训练和微调两阶段使模型适用于小样本任务。其中GDL用于对RPN和RCNN回传的梯度进行衰减，相当于解耦操作。PCB只在微调阶段的eval阶段使用，通过Support Set样本提取特征建立每个类别的特征库后，将每个Query Set样本进行cosine距离类别匹配。随后与模型输出做一个加权求和，以增加分类的准确率。

# 数据集

使用的数据集：


|  数据集   | 大小  |                                             谷歌云                                             |                           百度云                            |                                      参考                                      |
| :-------: | :---: | :--------------------------------------------------------------------------------------------: | :---------------------------------------------------------: | :----------------------------------------------------------------------------: |
|  VOC2007  | 0.8G  | [download](https://drive.google.com/file/d/1BcuJ9j9Mtymp56qGSOfYxlXN4uEVyxFm/view?usp=sharing) | [download](https://pan.baidu.com/s/1kjAmHY5JKDoG0L65T3dK9g) |                                       -                                        |
|  VOC2012  | 3.5G  | [download](https://drive.google.com/file/d/1NjztPltqm-Z-pG94a6PiPVP4BgD8Sz1H/view?usp=sharing) | [download](https://pan.baidu.com/s/1DUJT85AG_fqP9NRPhnwU2Q) |                                       -                                        |
| vocsplit  |  <1M  | [download](https://drive.google.com/file/d/1BpDDqJ0p-fQAFN_pthn2gqiK5nWGJ-1a/view?usp=sharing) | [download](https://pan.baidu.com/s/1518_egXZoJNhqH4KRDQvfw) | refer from [TFA](https://github.com/ucbdrive/few-shot-object-detection#models) |
|   COCO    | ~19G  |                                               -                                                |                              -                              |           download from [offical](https://cocodataset.org/#download)           |
| cocosplit | 174M  | [download](https://drive.google.com/file/d/1T_cYLxNqYlbnFNJt8IVvT7ZkWb5c0esj/view?usp=sharing) | [download](https://pan.baidu.com/s/1NELvshrbkpRS8BiuBIr5gA) | refer from [TFA](https://github.com/ucbdrive/few-shot-object-detection#models) |
  - 解压到 `datasets` 目录中加入工程，目录如下，记得将val2014目录内容复制到根据train2014重命名好的trainval2014内:
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

# 环境要求

- 硬件（Ascend/GPU）

    - 使用Ascend处理器来搭建硬件环境。
    - GPU NVIDIA 3090

- 安装[MindSpore](https://www.mindspore.cn/install)。

- 下载COCO相关的数据集并放入datasets目录下

    1. 使用COCO数据集，**执行脚本时选择数据集COCO。**
        安装Cython和pycocotool，也可以安装mmcv进行数据处理。

        ```python
        pip install Cython

        pip install pycocotools

        pip install mmcv==0.2.14
        ```

        根据模型运行需要，对应地在`configs`目录中更改COCO_ROOT和其他需要的设置。目录结构如下，下载好2014数据集后，记得将val2014/*.jpg的数据复制一份到trainval2014目录下：
        ```angular2html
        ...
        configs
        datasets
          | -- coco (trainval2014/*.jpg, val2014/*.jpg, annotations/*.json)
          | -- cocosplit
        de_frcn
        ...
        ```

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

注意：

1. 第一次运行生成MindRecord文件，耗时较长。
2. 预训练参数准备
  - 我们使用imagenet预处理权值来初始化我们的模型。从这里下载相同的模型: [GoogleDrive](https://drive.google.com/file/d/1rsE20_fSkYeIhFaNU04rBfEDkMENLibj/view?usp=sharing) [BaiduYun](https://pan.baidu.com/s/1IfxFq15LVUI3iIMGFT8slw)
  - The extract code for all BaiduYun link is **0000**
  - 其中里面包括了backbone与pcb中的resnet101参数
3. 下载好的预训练参数文件内目录结构为`ImageNetPretrained`。
    ```angular2html
    ...
    configs
    ImageNetPretrained
      | -- MSRA
      | -- torchvision
      | -- pkl_convert（自己创建）
      | -- pth_convert（自己创建）
    de_frcn
    ...
    ```
4. 将`model_utils`目录下的`pkl2ckpt.py`文件拷入`pkl_convert`， 将`model_utils`目录下的`pth2ckpt.py`文件拷入`pth_convert`。并进入各自目录下分别运行如下命令，生成对应的ckpt文件
    ```angular2html
    python pkl2ckpt.py
    python pth2ckpt.py
    ```

## 在Ascend上运行

```shell
# 大规模预训练阶段单机训练
bash run_standalone_train_ascend.sh [EXP_NAME]

# 大规模预训练阶段分布式训练
bash run_distribute_train_ascend.sh [EXP_NAME]

# 大规模预训练阶段评估
bash run_eval_pretrain_ascend.sh

# 微调阶段单机训练
bash run_standalone_finetune_ascend.sh [EXP_NAME]

# 微调阶段分布式训练
bash run_distribute_finetune_ascend.sh [EXP_NAME]

# 微调阶段评估
bash run_eval_finetune_ascend.sh

#推理(IMAGE_WIDTH和IMAGE_HEIGHT必须同时设置或者同时使用默认值。)
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [CONFIG_PATH] [DEVICE_TARGET] [IMAGE_WIDTH](optional) [IMAGE_HEIGHT](optional) [DEVICE_ID](optional)
```

## 在GPU上运行

```shell

# 大规模预训练阶段单机训练
bash run_standalone_train_gpu.sh [EXP_NAME]

# 大规模预训练阶段分布式训练
bash run_distribute_train_gpu.sh [EXP_NAME]

# 大规模预训练阶段评估
bash run_eval_pretrain_gpu.sh

# 微调阶段单机训练
bash run_standalone_finetune_gpu.sh [EXP_NAME]

# 微调阶段分布式训练
bash run_distribute_finetune_gpu.sh [EXP_NAME]

# 微调阶段评估
bash run_eval_finetune_gpu.sh

```

# 脚本说明

## 脚本及样例代码
```shell
de-frcn
├─ README.md                                     // DeFRCN英文相关说明
├─ README_CN.md                                  // DeFRCN中文相关说明
├─ ascend310_infer                               // 实现310推理源代码
├─ configs                                       // yaml配置文件
├─ eval.py                                       // 评估脚本
├─ requirements.txt                              // 环境依赖文件
├─ scripts
│  ├─ run_distribute_finetune_ascend.sh          // 多卡Ascend微调阶段脚本
│  ├─ run_distribute_finetune_gpu.sh             // 多卡GPU微调阶段脚本
│  ├─ run_distribute_train_ascend.sh             // 多卡Ascend大规模预训练阶段脚本
│  ├─ run_distribute_train_gpu.sh                // 多卡GPU大规模预训练阶段脚本
│  ├─ run_eval_finetune_ascend.sh                // Ascend微调阶段脚本
│  ├─ run_eval_finetune_gpu.sh                   // GPU微调阶段脚本
│  ├─ run_eval_pretrain_ascend.sh                // Ascend大规模预训练阶段脚本
│  ├─ run_eval_pretrain_gpu.sh                   // GPU大规模预训练阶段脚本
│  ├─ run_infer_310.sh                           // Ascend推理shell脚本
│  ├─ run_standalone_finetune_ascend.sh          // 单卡Ascend微调阶段脚本
│  ├─ run_standalone_finetune_gpu.sh             // 单卡GPU微调阶段脚本
│  ├─ run_standalone_train_ascend.sh             // 单卡Ascend大规模预训练阶段脚本
│  └─ run_standalone_train_gpu.sh                // 单卡GPU大规模预训练阶段脚本
├─ src
│  ├─ dataset.py                                 // 创建并增强数据集
│  ├─ defrcn
│  │  ├─ __init__.py                             // init文件
│  │  ├─ anchor_generator.py                     // 锚点生成器
│  │  ├─ bbox_assign_sample.py                   // 第一阶段采样器
│  │  ├─ bbox_assign_sample_stage2.py            // 第二阶段采样器
│  │  ├─ built_datacategory.py                   // 建立数据集目录
│  │  ├─ de_frcn.py                              // DeFRCN网络
│  │  ├─ gdl.py                                  // 多阶段解耦模块
│  │  ├─ meta_coco.py                            // 注册coco数据集
│  │  ├─ pcb.py                                  // 多任务解耦的原型校准块
│  │  ├─ pcb_resnet.py                           // pcb特征提取器
│  │  ├─ proposal_generator.py                   // 候选生成器
│  │  ├─ rcnn.py                                 // R-CNN网络
│  │  ├─ resnet.py                               // backbone
│  │  ├─ roi_align.py                            // ROI对齐网络
│  │  └─ rpn.py                                  // 区域候选网络
│  ├─ detecteval.py                              // detecteval工具函数
│  ├─ lr_schedule.py                             // lr调度器
│  ├─ model_surgery.py                           // 大规模预训练参数转化为微调参数
│  ├─ pkl2ckpt.py                                // 将pkl参数转化成ckpt用作backbone
│  ├─ pth2ckpt.py                                // 将pth参数转化成ckpt用作pcb特征提取
│  ├─ model_utils
│  │  ├─ __init__.py                             // init文件
│  │  ├─ config.py                               // 获取yaml和命令行配置
│  │  ├─ device_adapter.py                       // 获取云上id
│  │  ├─ local_adapter.py                        // 获取本地id
│  │  └─ moxing_adapter.py                       // 云上数据准备
│  ├─ network_define.py                          // DeFRCN网络定义
│  ├─ sgd_optimizer.py                           // 优化器实现
│  └─ util.py                                    // 工具函数
└─ train.py                                      // 训练脚本
└─ preprocess.py                                 // 推理前处理
└─ postprocess.py                                // 推理后处理
```

## 训练过程

### 用法

#### 在Ascend上运行

```shell
# 大规模预训练阶段单机训练
bash run_standalone_train_ascend.sh [EXP_NAME]

# 大规模预训练阶段分布式训练
bash run_distribute_train_ascend.sh [EXP_NAME]

# 微调阶段单机训练
bash run_standalone_finetune_ascend.sh [EXP_NAME]

# 微调阶段分布式训练
bash run_distribute_finetune_ascend.sh [EXP_NAME]
```

#### 在GPU上运行

```shell
# 大规模预训练阶段单机训练
bash run_standalone_train_gpu.sh [EXP_NAME]

# 大规模预训练阶段分布式训练
bash run_distribute_train_gpu.sh [EXP_NAME]

# 微调阶段单机训练
bash run_standalone_finetune_gpu.sh [EXP_NAME]

# 微调阶段分布式训练
bash run_distribute_finetune_gpu.sh [EXP_NAME]
```

Notes:

1. 运行分布式任务时需要用到RANK_TABLE_FILE指定的rank_table.json。您可以使用[hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)生成该文件。
2. 目前只在coco数据集上实现运行。
3. 在使用finetune阶段的eval时，记得修改对应`seed`参数`eval_checkpoint_path`。
4. finetune阶段若device_id出现问题，记得手动更改model_surgery.py的context的device_id。
5. 若训练阶段结束后自动调用的evalcallback长时间运行且不报错，可以考虑在正常生成ckpt文件后，手动结束进程，随后调用相应的eval脚本开始eval。
6. 训练过程会发现几十个step后loss趋于稳定并且不再下降的情况，此时对于目标检测任务来说是正常现象，请继续让其训练更多的epoch，会得到更好的效果。

### 结果

训练结果保存在示例路径中，文件夹名称以“train”或“train_parallel”开头。您可以在对应的loss.log中找到结果，如下所示。
检查点保存在script同级目录下的checkpoints内。

```log
# 分布式训练结果（8P）
3075 s | epoch: 1 step: 6154 total_loss: 0.35424  lr: 0.020000
6131 s | epoch: 2 step: 6154 total_loss: 0.08087  lr: 0.020000
9449 s | epoch: 3 step: 6154 total_loss: 0.43327  lr: 0.020000
12455 s | epoch: 4 step: 6154 total_loss: 0.85868  lr: 0.020000
...
48667 s | epoch: 16 step: 6154 total_loss: 0.09816  lr: 0.002000
51577 s | epoch: 17 step: 6154 total_loss: 0.17274  lr: 0.000200
54484 s | epoch: 18 step: 6154 total_loss: 0.22159  lr: 0.000200
```

## 评估过程

### 用法

#### 在Ascend上运行

```shell
# 大规模预训练阶段评估
bash run_eval_pretrain_ascend.sh

# 微调阶段评估
bash run_eval_finetune_ascend.sh
```

#### 在GPU上运行

```shell
# 大规模预训练阶段评估
bash run_eval_pretrain_gpu.sh

# 微调阶段评估
bash run_eval_finetune_gpu.sh
```


### 结果

评估结果将保存在示例路径中，文件夹名为“eval”。在此文件夹下，您可以在日志中找到类似以下的结果。

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

## 模型导出

```shell
python export.py --config_path [CONFIG_PATH] --export_ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --export_file_format[EXPORT_FORMAT]
```

`EXPORT_FORMAT` 可选 ["AIR", "MINDIR"]


## 推理过程

### 使用方法

在推理之前需要在昇腾910环境上完成模型的导出。以下示例仅支持batch_size=1的mindir推理。

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANNO_PATH] [DEVICE_TARGET] [IMAGE_WIDTH](optional) [IMAGE_HEIGHT](optional) [DEVICE_ID](optional)
```

### 结果

推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

```log

```

# 模型描述

## 性能

### 训练性能

#### 大规模训练阶段
| 参数          | Ascend                                     | GPU                                        |
| ------------- | ------------------------------------------ | ------------------------------------------ |
| 模型版本      | V1                                         | V1                                         |
| 资源          | Ascend 910；CPU 2.60GHz，192核；显存：32G  | NVIDIA 3090 24G                            |
| 上传日期      | 2022/12/05                                 | 2022/12/05                                 |
| MindSpore版本 | 1.9.0                                      | mindspore-gpu 1.8.1                        |
| 数据集        | COCO 2014                                  | COCO 2014                                  |
| 训练参数      | epoch=18, batch_size=2                     | epoch=18, batch_size=2                     |
| 优化器        | SGD                                        | SGD                                        |
| 损失函数      | Softmax交叉熵，Sigmoid交叉熵，SmoothL1Loss | Softmax交叉熵，Sigmoid交叉熵，SmoothL1Loss |
| 速度          |                                            | 1卡：264毫秒/步；8卡：460毫秒/步           |
| 总时间        |                                            | 1卡：65.03小时；8卡：14.13小时             |
| 参数(M)       | 658.3                                      | 658.3                                      |

#### 微调训练阶段性能
- 基本配置与大规模训练阶段相同
##### FSRW-like fsod 训练阶段性能
###### GPU
| 参数     | 1shot                 | 2shot                 | 3shot                | 5shot                | 10shot           | 30shot           |
| -------- | --------------------- | --------------------- | -------------------- | -------------------- | ---------------- | ---------------- |
| 速度     | 8pcs: 2478.433ms/step | 8pcs: 1312.433ms/step | 8pcs: 983.971ms/step | 8pcs: 690.911ms/step | 8pcs: 573ms/step | 8pcs: 420ms/step |
| 整体时间 | 0h33m2s               | 0h26m48s              | 0h26m42s             | 0h23m12s             | 0h15m21s         | 0h14m24s         |
| 参数(M)  | 473                   | 473                   | 473                  | 473                  | 473              | 473              |

##### TFA-like gfsod 训练阶段性能
- 基本配置与大规模训练阶段相同
###### GPU
| 参数 | 1shot                | 2shot                | 3shot                | 5shot                | 10shot               | 30shot               |
| ---------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| 速度      | 8pcs: 715.817ms/step | 8pcs: 582.317ms/step | 8pcs: 512.302ms/step | 8pcs: 462.155ms/step | 8pcs: 416.027ms/step | 8pcs: 371.619ms/step |
| 整体时间 | 0h38m41s             | 0h31m3s              | 0h32m1s              | 0h29m36s             | 0h21m2s              | 0h36m39s             |
| 参数(M)   | 481                  | 481                  | 481                  | 481                  | 481                  | 481                  |

### 评估性能

#### 大规模训练阶段评估性能

| 参数          | Ascend              | GPU                 |
| ------------- | ------------------- | ------------------- |
| 模型版本      | V1                  | V1                  |
| 资源          | Ascend 910          | NVIDIA 3090 24G     |
| 上传日期      | 2022/12/12          | 2022/12/12          |
| MindSpore版本 | 1.9.0               | mindspore-gpu 1.8.1 |
| 数据集        | COCO2014            | COCO2014            |
| batch_size    | 2                   | 2                   |
| 输出          | mAP                 | mAP                 |
| 准确率        |                     | IoU=0.50：51.7%     |
| 推理模型      | 658.3M（.ckpt文件） | 658.3M（.ckpt文件） |

#### 微调阶段评估性能
##### FSRW-like fsod 微调阶段评估性能
- 基本配置与大规模训练微调阶段一致
###### GPU

| shot    | mAP<sup>novel</sup> | 推理模型            |
| ------- | ------------------- | ------------------- |
| 1 shot  | 5.9                 | 472.6M（.ckpt文件)  |
| 2 shot  | 8.1                 | 472.7M（.ckpt文件)  |
| 3 shot  | 11.0                | 472.7M（.ckpt文件)  |
| 5 shot  | 11.0                | 472.96M（.ckpt文件) |
| 10 shot | 13.4                | 472.84M（.ckpt文件) |
| 30 shot | 15.9                | 472.98M（.ckpt文件) |


##### TFA-like fsod 微调阶段评估性能
- 基本配置与大规模训练微调阶段一致
###### GPU

| shot    | mAP<sup>novel</sup> | 推理模型            |
| ------- | ------------------- | ------------------- |
| 1 shot  | 5.0                 | 480.35M（.ckpt文件) |
| 2 shot  | 8.0                 | 480.4M（.ckpt文件)  |
| 3 shot  | 10.6                | 480.54M（.ckpt文件) |
| 5 shot  | 11.7                | 480.54M（.ckpt文件) |
| 10 shot | 13.5                | 480.29M（.ckpt文件) |
| 30 shot | 17.6                | 481.15M（.ckpt文件) |
# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
