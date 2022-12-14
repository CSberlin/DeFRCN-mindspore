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

import io
import os
import contextlib
import numpy as np
from pycocotools.coco import COCO
from src.model_utils.config import config

def load_coco_json(json_file, image_root, metadata, dataset_name):
    """Use Generate coco image path list and annotation list

    Args:
        json_file (string): annotation file path
        image_root (string): images file path
        metadata (dict): coco base train or novel finetuning metadata
        dataset_name (string)
    
    """
    is_shots = "shot" in dataset_name  # few-shot
    if is_shots:
        imgid2info = {}
        shot = dataset_name.split('_')[-2].split('shot')[0]
        seed = int(dataset_name.split('_seed')[-1])
        split_dir = os.path.join(config.datasets_root, 'cocosplit', 'seed{}'.format(seed))

        for idx, cls in enumerate(metadata["thing_classes"][1:]):
            json_file = os.path.join(split_dir, "full_box_{}shot_{}_trainval.json".format(shot, cls))
            with contextlib.redirect_stdout(io.StringIO()):
                coco_api = COCO(json_file)
            img_ids = sorted(list(coco_api.imgs.keys()))
            for img_id in img_ids:
                if img_id not in imgid2info:
                    imgid2info[img_id] = [coco_api.loadImgs([img_id])[0], coco_api.imgToAnns[img_id]]
                else:
                    for item in coco_api.imgToAnns[img_id]:
                        imgid2info[img_id][1].append(item)
            
        imgs, anns = [], []
        for img_id in imgid2info:
            imgs.append(imgid2info[img_id][0])
            anns.append(imgid2info[img_id][1])
        
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(json_file)
        # sort indices for reproducible results
        img_ids = sorted(list(coco_api.imgs.keys()))
        imgs = coco_api.loadImgs(img_ids)
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    
    imgs_anns = list(zip(imgs, anns))
    id_map = metadata["thing_dataset_id_to_contiguous_id"]
    dataset_dicts = []
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(
            image_root, img_dict["file_name"]
        )
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        # annotations different from origin code process
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id
            assert anno.get("ignore", 0) == 0

            obj = {}
            bbox = anno["bbox"]
            if anno["category_id"] in id_map:
                x1, x2 = bbox[0], bbox[0] + bbox[2]
                y1, y2 = bbox[1], bbox[1] + bbox[3]
                obj["category_id"] = [id_map[anno["category_id"]]]
                obj["bbox"] = [x1, y1, x2, y2]
                obj["iscrowd"] = [int(anno["iscrowd"])]
                objs.append(obj)                   
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts