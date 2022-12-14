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

from mindspore import Tensor, Parameter
import os
import mindspore as ms
import mindspore.ops as ops
from mindspore.common.initializer import Normal, initializer
import argparse
from model_utils.local_adapter import get_device_id

def surgery_loop(args, surgery):

    save_name = args.tar_name + '_' + ('remove' if args.method == 'remove' else 'surgery') + '.ckpt'
    save_path = os.path.join(args.save_dir, save_name)
    try:
        os.makedirs(args.save_dir, exist_ok=True)
        print("Directory '%s' created successfully" %args.save_dir)
    except OSError as error:
        print("Directory '%s' can not be created")

    if args.method == 'remove':
        ckpt = ms.load_checkpoint(args.src_path, filter_prefix=args.remove_param_name)
    elif args.method == 'randinit': # 对应80类
        ckpt = ms.load_checkpoint(args.src_path, filter_prefix="learning_rate")
        tar_sizes = [TAR_SIZE + 1, TAR_SIZE * 4]
        for idx, (param_name, tar_size) in enumerate(zip(args.rand_init_param_name, tar_sizes)):
            surgery(param_name, True, tar_size, ckpt)
            surgery(param_name, False, tar_size, ckpt)
    else:
        raise NotImplementedError
    if 'scheduler' in ckpt:
        del ckpt['scheduler']
    if 'optimizer' in ckpt:
        del ckpt['optimizer']
    if 'iteration' in ckpt:
        ckpt['iteration'] = 0

    ckpt_save_list = []
    for name, value in ckpt.items():
        ckpt_save = {}
        ckpt_save["name"] = name
        ckpt_save["data"] = value
        ckpt_save_list.append(ckpt_save)
    ms.save_checkpoint(ckpt_save_list, save_path)
    print('save changed ckpt to {}'.format(save_path))


def main(args):
    """
    Either remove the final layer weights for fine-tuning on novel dataset or
    append randomly initialized weights for the novel classes.
    """
    def surgery(param_name, is_weight, tar_size, ckpt):
        weight_name = param_name + ('.weight' if is_weight else '.bias')
        pretrained_weight = ckpt[weight_name]
        if "cls_score" in param_name:
            pretrained_weight_tensor = pretrained_weight.transpose()
        elif "reg_scores" in param_name:
            pretrained_weight_tensor = pretrained_weight.transpose()

        pretrained_weight_tensor = ops.Squeeze()(pretrained_weight_tensor)

        prev_cls = pretrained_weight_tensor.shape[0]
        if 'cls_score' in param_name:
            prev_cls -= 1
        if is_weight:
            feat_size = pretrained_weight_tensor.shape[1]
            new_weight = initializer(Normal(sigma=0.01, mean=0.0), shape=[tar_size, feat_size], dtype=ms.float32) #(2048, 81)
        else:
            new_weight = initializer("zeros", [tar_size], ms.float32) # (81,)

        new_weight_copy = new_weight.copy()

        if args.dataset == 'coco':
            for idx, c in enumerate(BASE_CLASSES):
                _idx = idx + 1
                if 'cls_score' in param_name:
                    # new_weight_copy[IDMAP[c]] map continueous 80
                    new_weight_copy[IDMAP[c]] = pretrained_weight_tensor[_idx] # 1->60
                else:
                    new_weight_copy[IDMAP[c]*4:(IDMAP[c]+1)*4] = \
                        pretrained_weight_tensor[idx*4:(idx+1)*4]

            new_weight = new_weight_copy

        if 'cls_score' in param_name:
            new_weight[0] = pretrained_weight_tensor[0]  # bg class

        if "cls_score" in param_name:
            save_weight = new_weight.transpose()
        elif "reg_scores" in param_name:
            save_weight = new_weight.transpose()
        save_weight = ops.Squeeze()(save_weight)
        save_weight = Parameter(save_weight, requires_grad=True)
        
        ckpt[weight_name] = save_weight

    surgery_loop(args, surgery)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device-target', type=str, default='Ascend', choices=['Ascend', 'GPU', 'CPU'])
    parser.add_argument('--dataset', type=str, default='coco', choices=['voc', 'coco'])
    parser.add_argument('--src-path', type=str, default='', help='Path to the main checkpoint')
    parser.add_argument('--save-dir', type=str, default='', required=True, help='Save directory')
    parser.add_argument('--method', choices=['remove', 'randinit'], required=True,
                        help='remove = remove the final layer of the base detector. '
                             'randinit = randomly initialize novel weights.')
    parser.add_argument('--remove-param-name', type=str, nargs='+', help='Target parameter names',
                        default=["learning_rate", "stat.rcnn.cls_scores.weight", "stat.rcnn.cls_scores.bias",
                                "stat.rcnn.reg_scores.weight", "stat.rcnn.reg_scores.bias",
                                "rcnn.cls_scores.weight", "rcnn.cls_scores.bias", "rcnn.reg_scores.weight",
                                "rcnn.reg_scores.bias", "accum.rcnn.cls_scores.weight", "accum.rcnn.cls_scores.bias",
                                "accum.rcnn.reg_scores.weight", "accum.rcnn.reg_scores.bias"
                              ])
    parser.add_argument('--rand-init-param-name', type=str, nargs='+', help='Target parameter names',
                        default=["rcnn.cls_scores", "rcnn.reg_scores"])
    parser.add_argument('--tar-name', type=str, default='model_reset', help='Name of the new ckpt')
    args = parser.parse_args()
    ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device_target, device_id=get_device_id())
    if args.dataset == 'coco':
        NOVEL_CLASSES = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
        BASE_CLASSES = [8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
                        39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                        61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        ALL_CLASSES = sorted(BASE_CLASSES + NOVEL_CLASSES)
        IDMAP = {v: i+1 for i, v in enumerate(ALL_CLASSES)}
        # print("IDMAP:   ", IDMAP)
        TAR_SIZE = 80
    elif args.dataset == 'voc':
        TAR_SIZE = 20
    else:
        raise NotImplementedError

    main(args)
