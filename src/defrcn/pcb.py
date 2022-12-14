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

import os
import logging
from mindspore import load_checkpoint, load_param_into_net, context
from src.defrcn.pcb_resnet import resnet101
from mindspore import Tensor
import mindspore.ops as ops
import mindspore as ms
from src.dataset import create_defrcn_dataset, data_to_mindrecord_byte_image
from mindspore.communication.management import init, get_rank
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
class PrototypicalCalibrationBlock:

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.alpha = self.cfg.pcb_alpha
        self.test_per_size = 1
        self.dtype = np.float32
        self.ms_type = ms.float32
        self.cast = ops.Cast()
        roi_align_index_test = [np.array(np.ones((cfg.num_gts, 1)) * i, dtype=self.dtype) \
                                for i in range(self.test_per_size)]
        self.roi_align_index_test_tensor = Tensor(np.concatenate(roi_align_index_test), self.ms_type)
        self.pcb_upper = Tensor([self.cfg.pcb_upper], dtype=ms.float32)
        self.pcb_lower = Tensor([self.cfg.pcb_lower], dtype=ms.float32)
        self.concat_0 = ops.Concat(axis=0)
        self.concat_1 = ops.Concat(axis=1)
        self.greater = ops.Greater()
        self.imagenet_model = self.build_model()

        
        self.prefix = "mindrecord_pcb"
        # use train data to build mindrecord，save in test dir，cause pcb use in test phrase
        self.mindrecord_file = os.path.join(self.cfg.test_mindrecord_dir, self.cfg.dataset_train[0], self.prefix)
        self.mindrecord_dir = os.path.join(self.cfg.test_mindrecord_dir, self.cfg.dataset_train[0])
        print("CHECKING MINDRECORD FILES ...")
        rank = 0
        if self.cfg.run_distribute==True:
            init()
            rank=get_rank()
        if rank==0 and not os.path.exists(self.mindrecord_file + ".db"):
            if not os.path.exists(self.mindrecord_dir):
                os.makedirs(self.mindrecord_dir)
            if os.path.isdir(self.cfg.coco_root):
                print("Create PCB Mindrecord. It may take some time.")
                data_to_mindrecord_byte_image(self.cfg, self.cfg.dataset_train, self.prefix, False, file_num=1)
                print("Create PCB Mindrecord Done, at {}".format(self.mindrecord_dir))
            else:
                print("coco_root not exits.")
                raise ValueError(self.config.coco_root)
        while not os.path.exists(self.mindrecord_file + ".db"):
            print("wait 5 seconds")
            time.sleep(5)
        self.dataset = create_defrcn_dataset(self.cfg, mindrecord_file=self.mindrecord_file, is_training=False, batch_size=1)
        self.roi_pooler = ops.ROIAlign(pooled_height=1, pooled_width=1, spatial_scale=1/32, sample_num=2)
        self.prototypes = self.build_prototypes()
        self.exclude_cls = self.clsid_filter()
        
        self.reducesum = ops.ReduceSum(keep_dims=False)
        

    def build_model(self):
        if self.cfg.pcb_modeltype == 'resnet':
            imagenet_model = resnet101()
        else:
            raise NotImplementedError
        state_dict = load_checkpoint(self.cfg.pcb_modelpath)
        load_param_into_net(imagenet_model, state_dict)
        imagenet_model.set_train(mode = False)
        return imagenet_model

    def build_prototypes(self):
        
        all_features, all_labels = [], []
        for item in self.dataset.create_dict_iterator(num_epochs=1):
            masks = item["valid_num"]
            valid = False
            index = -1
            # TODO: this is ugly find first valid
            for mask in masks: # masks shape (1, 128)
                for i, value in enumerate(mask):
                    if value==False:
                       index = i
                       if index != 0:
                           valid = True
                       break
            if not valid:
                continue

            # load support images and gt-boxes
            img = item["image"] # RGB
            img_h, img_w = img.shape[-2], img.shape[-1] # 768 1280
            ratio = item["image_shape"][0][-1]
            boxes = item["box"][0] * ratio
            boxes = boxes.asnumpy()
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, img_w - 1)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, img_h - 1)
            boxes = Tensor(boxes, dtype=self.ms_type)

            rois = self.concat_1((self.roi_align_index_test_tensor, boxes))
            rois = rois[:index,:]

            # extract roi features
            features = self.extract_roi_features(img, rois)
            all_features.append(features)
            gt_classes = item["label"][0][:index]
            all_labels.append(gt_classes)

        all_features = self.concat_0(all_features)
        all_labels = self.concat_0(all_labels)
        all_features = Tensor(all_features, dtype=self.ms_type)
        all_labels = Tensor(all_labels, dtype=ms.int32)
        assert all_features.shape[0] == all_labels.shape[0]

        # calculate prototype
        features_dict = {}
        expand_dims = ops.ExpandDims()
        for i, label in enumerate(all_labels):
            label = int(label)
            if label not in features_dict:
                features_dict[label] = []
            features_dict[label].append(expand_dims(all_features[i], 0))

        prototypes_dict = {}
        for label in features_dict:
            features = self.concat_0(features_dict[label])
            prototypes_dict[label] = features.mean(axis=0, keep_dims=True)
        print("Build prototypes success! the size is {}".format(len(prototypes_dict)))
        return prototypes_dict

    def extract_roi_features(self, img, boxes):
        """
        :param img:
        :param boxes:
        :return:
        """

        mean = Tensor([0.406, 0.456, 0.485]).reshape((3, 1, 1))
        std = Tensor([[0.225, 0.224, 0.229]]).reshape((3, 1, 1))

        images = (img / 255. - mean) / std

        # padding 768 1280
        conv_feature = self.imagenet_model(images[:, [2,1,0]])[1]  # size: BxCxHxW
        squeeze = ops.Squeeze(axis=2)
        box_features = self.roi_pooler(conv_feature, boxes)
        box_features = squeeze(squeeze(box_features))

        activation_vectors = self.imagenet_model.fc(box_features)

        return activation_vectors

    def execute_calibration(self, inputs, dts):
        # inputs["image"] shape: (bs, c, h, w)
        all_bbox = dts[0][:,:-1]
        all_scores = dts[0][:,-1]
        all_label = dts[1]

        if all_label.shape[0] == 0:
            return dts

        ileft, iright = 0, 0
        for score in all_scores:
            if score > self.pcb_upper:
                ileft += 1
        for score in all_scores:
            if score > self.pcb_lower:
                iright += 1

        assert ileft <= iright
        
        boxes = all_bbox[ileft:iright]
        roi_align_index_test = [np.array(np.ones((iright-ileft, 1)) * i, dtype=self.dtype) \
                        for i in range(self.test_per_size)]
        roi_align_index_test_tensor = Tensor(np.concatenate(roi_align_index_test), ms.float32)

        boxes = Tensor(boxes, ms.float32)
        rois = self.concat_1((roi_align_index_test_tensor, boxes))
        
        features = self.extract_roi_features(inputs["image"], rois)

        for i in range(ileft, iright):
            tmp_class = all_label[i] + 1
            if tmp_class in self.exclude_cls:
                continue
            tmp_cos = cosine_similarity(features[i - ileft].asnumpy().reshape((1, -1)),
                                        self.prototypes[tmp_class].asnumpy())[0][0]
            dts[0][i,-1] = all_scores[i] * self.alpha + tmp_cos * (1 - self.alpha)
        return dts

    def clsid_filter(self):
        dsname = self.cfg.dataset_test[0]
        exclude_ids = []
        if 'test_all' in dsname:
            if 'coco' in dsname:
                exclude_ids = [7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                               30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45,
                               46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 61, 63, 64, 65,
                               66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
            elif 'voc' in dsname:
                exclude_ids = list(range(0, 15))
            else:
                raise NotImplementedError
        return exclude_ids
