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

from builtins import float, int, len, min, max
from typing import Tuple
from src.defrcn.meta_coco import load_coco_json
from src.model_utils.config import config
from src.defrcn.built_datacategory import register_all_coco, DATACATALOG, METADATACATALOG
from typing import Any, Dict, List, Set
import itertools
import logging
import numpy as np
from numpy import random
import os
import cv2
import mindspore as ms
import mindspore.dataset as de
from mindspore.mindrecord import FileWriter
from tabulate import tabulate
from termcolor import colored

def filter_images_with_only_crowd_annotations(dataset_dicts):
    """
    Filter out images with none annotations or only crowd annotations
    (i.e., images without non-crowd annotations).
    A common training-time preprocessing on COCO dataset.
    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.
    Returns:
        list[dict]: the same format, but filtered.
    """

    def valid(anns):
        for ann in anns:
            if ann.get("iscrowd", [0]) == [0]:
                return True
        return False

    dataset_dicts = [x for x in dataset_dicts if valid(x["annotations"])]
    return dataset_dicts

def filter_empty_boxes(dataset_dicts, box_threshold=1e-5):
    def delete_empty_boxes(annos, threshold):
        for i, anno in enumerate(annos):
            if anno.get("bbox",0) == 0:
                continue
            else:
                box = np.array(anno["bbox"],dtype=np.float32).reshape(-1,4)
                widths = box[:, 2] - box[:, 0]
                heights = box[:, 3] - box[:, 1]
                keep = (widths > threshold) & (heights > threshold)
                if not keep:
                    del annos[i]
            
    for dataset_dict in dataset_dicts:
        delete_empty_boxes(dataset_dict["annotations"], box_threshold)
    return dataset_dicts

def print_instances_class_histogram(dataset_dicts, class_names):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)

    for entry in dataset_dicts:
        annos = entry["annotations"]
        classes = [x["category_id"][0] for x in annos if x.get("iscrowd", [0])==[0]]
        histogram += np.histogram(classes, bins=hist_bins)[0]
    
    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    print("Distribution of instances among all {} categories:\n".format(num_classes) + colored(table, "cyan"))

def get_detection_dataset_dicts(config, dataset_names, filter_empty=True):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.
    Args:
        dataset_names (list[str]): a list of dataset names (e.g coco14_trainval_base)
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
             `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
             that match each dataset in `dataset_names`.
    Returns:
         list[dict]: a list of dicts following the standard dataset dict format.
     """
    assert len(dataset_names)
     # List[tuple] get all coco experiment set metadata (name, imgdir, annofile)
    register_all_coco(config.datasets_root)
    dataset_dicts : List[Dict] = []
    datasets_catalog = [DATACATALOG[dataset_name] for dataset_name in dataset_names]
    for dataset_meta_info in datasets_catalog:
        dataset_dict = load_coco_json(dataset_meta_info["annofile"] ,dataset_meta_info["imgdir"],
                                        dataset_meta_info["metadata"],dataset_meta_info["name"])
        dataset_dicts.append(dataset_dict)
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)
    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))
    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    filter_empty_boxes(dataset_dicts)
    
    if has_instances:
        try:
            class_names = METADATACATALOG[dataset_names[0]]["metadata"]["thing_classes"]
            print_instances_class_histogram(dataset_dicts, class_names)
            for dataset_name in dataset_names:
                if not METADATACATALOG[dataset_name]["metadata"].has_key('thing_classes'):
                    raise ValueError("Datasets have different metadata '{}'!".format('thing_classes'))
        except AttributeError:  # class names are not available for this dataset
            pass
    print("image sample num:   ", len(dataset_dicts)) # 98459
    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(dataset_names))
    return dataset_dicts

def flip_column(img, image_id, img_shape, gt_bboxes, gt_label, gt_num):
    """flip operation for image"""
    img_data = img
    img_data = np.flip(img_data, axis=1)
    flipped = gt_bboxes.copy()
    _, w, _ = img_data.shape

    flipped[..., 0::4] = w - gt_bboxes[..., 2::4] - 1
    flipped[..., 2::4] = w - gt_bboxes[..., 0::4] - 1
    return (img_data, image_id, img_shape, flipped, gt_label, gt_num)
    
def rescale_with_tuple(img, scale):
    h, w = img.shape[:2]
    scale_factor = min(max(scale) / max(h, w), min(scale) / min(h, w))
    new_size = int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5)
    rescaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    return rescaled_img, scale_factor

def rescale_column(img, image_id, img_shape, gt_bboxes, gt_label, gt_num, config):
    """rescale operation for image"""
    img_data, scale_factor = rescale_with_tuple(img, (config.img_width, config.img_height))
    if img_data.shape[0] > config.img_height:
        img_data, scale_factor2 = rescale_with_tuple(img_data, (config.img_height, config.img_height))
        scale_factor = scale_factor * scale_factor2

    gt_bboxes = gt_bboxes * scale_factor
    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_data.shape[1] - 1)
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_data.shape[0] - 1)

    pad_h = config.img_height - img_data.shape[0]
    pad_w = config.img_width - img_data.shape[1]
    assert ((pad_h >= 0) and (pad_w >= 0))

    pad_img_data = np.zeros((config.img_height, config.img_width, 3)).astype(img_data.dtype)
    pad_img_data[0:img_data.shape[0], 0:img_data.shape[1], :] = img_data

    img_shape = (config.img_height, config.img_width, 1.0)
    img_shape = np.asarray(img_shape, dtype=np.float32)

    return (pad_img_data, image_id, img_shape, gt_bboxes, gt_label, gt_num)

def rescale_column_test(img, image_id, img_shape, gt_bboxes, gt_label, gt_num, config):
    """rescale operation for image of eval"""
    img_data, scale_factor = rescale_with_tuple(img, (config.img_width, config.img_height))
    if img_data.shape[0] > config.img_height:
        img_data, scale_factor2 = rescale_with_tuple(img_data, (config.img_height, config.img_height))
        scale_factor = scale_factor * scale_factor2

    pad_h = config.img_height - img_data.shape[0]
    pad_w = config.img_width - img_data.shape[1]
    assert ((pad_h >= 0) and (pad_w >= 0))

    pad_img_data = np.zeros((config.img_height, config.img_width, 3)).astype(img_data.dtype)
    pad_img_data[0:img_data.shape[0], 0:img_data.shape[1], :] = img_data

    img_shape = np.append(img_shape, (scale_factor, scale_factor))
    img_shape = np.asarray(img_shape, dtype=np.float32)

    return (pad_img_data, image_id, img_shape, gt_bboxes, gt_label, gt_num)

def transpose_column(img, image_id, img_shape, gt_bboxes, gt_label, gt_num):
    """transpose operation for image"""
    img_data = img.transpose(2, 0, 1).copy()
    img_data = img_data.astype(np.float32)
    img_shape = img_shape.astype(np.float32)
    gt_bboxes = gt_bboxes.astype(np.float32)
    gt_label = gt_label.astype(np.int32)
    gt_num = gt_num.astype(np.bool)

    return (img_data, image_id, img_shape, gt_bboxes, gt_label, gt_num)

class Expand:
    """expand image"""

    def __init__(self, mean=(0, 0, 0), to_rgb=False, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, boxes, labels):
        if random.randint(2):
            return img, boxes, labels

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img
        boxes += np.tile((left, top), 2)
        return img, boxes, labels

def expand_column(img, image_id, img_shape, gt_bboxes, gt_label, gt_num):
    """expand operation for image"""
    expand = Expand()
    img, gt_bboxes, gt_label = expand(img, gt_bboxes, gt_label)

    return (img, image_id, img_shape, gt_bboxes, gt_label, gt_num)

def preprocess_fn(image, image_id, box, is_training, config):
    """Preprocess function for dataset."""

    def _infer_data(image_bgr, image_id, image_shape, gt_box_new, gt_label_new, gt_iscrowd_new_revert):
        image_shape = image_shape[:2]
        input_data = image_bgr, image_id, image_shape, gt_box_new, gt_label_new, gt_iscrowd_new_revert

        if config.keep_ratio:
            input_data = rescale_column_test(*input_data, config=config)

        output_data = transpose_column(*input_data)
        return output_data

    def _data_aug(image, image_id, box, is_training):
        """Data augmentation function."""
        pad_max_number = config.num_gts
        # fill empty box
        if box.shape == ():
            box = np.array([[0,0,0,0,-1,1]])
        if pad_max_number < box.shape[0]:
            box = box[:pad_max_number, :]
        image_bgr = image.copy()
        # H W C
        image_bgr[:, :, 0] = image[:, :, 2]
        image_bgr[:, :, 1] = image[:, :, 1]
        image_bgr[:, :, 2] = image[:, :, 0]
        image_shape = image_bgr.shape[:2]
        gt_box = box[:, :4]
        gt_label = box[:, 4]
        gt_iscrowd = box[:, 5]

        gt_box_new = np.pad(gt_box, ((0, pad_max_number - box.shape[0]), (0, 0)), mode="constant", constant_values=0)
        gt_label_new = np.pad(gt_label, ((0, pad_max_number - box.shape[0])), mode="constant", constant_values=-1)
        gt_iscrowd_new = np.pad(gt_iscrowd, ((0, pad_max_number - box.shape[0])), mode="constant", constant_values=1)
        gt_iscrowd_new_revert = (~(gt_iscrowd_new.astype(np.bool))).astype(np.int32)

        if not is_training:
            return _infer_data(image_bgr, image_id, image_shape, gt_box_new, gt_label_new, gt_iscrowd_new_revert)

        flip = np.random.uniform(0, 1, []) < config.flip_ratio
        expand = (np.random.rand() < config.expand_ratio)
        input_data = image_bgr, image_id, image_shape, gt_box_new, gt_label_new, gt_iscrowd_new_revert

        if expand:
            input_data = expand_column(*input_data)
        if config.keep_ratio:
            input_data = rescale_column(*input_data, config=config)
        if flip:
            input_data = flip_column(*input_data)

        output_data = transpose_column(*input_data)
        return output_data

    return _data_aug(image, image_id, box, is_training)

def data_to_mindrecord_byte_image(config, dataset_names, prefix="mindrecord", is_training=True, file_num=8):
    if is_training:
        dataset_dicts = get_detection_dataset_dicts(config, dataset_names, filter_empty=True)
    else:
        dataset_dicts = get_detection_dataset_dicts(config, dataset_names, filter_empty=False)
    if is_training:
        mindrecord_dir = config.train_mindrecord_dir
    else:
        mindrecord_dir = config.test_mindrecord_dir
    mindrecord_path = os.path.join(mindrecord_dir, dataset_names[0], prefix)
    # over_write=False
    # if file_num==1:
    #     over_write=True
    writer = FileWriter(mindrecord_path, file_num)
    defrcn_json = {
        "image": {"type": "bytes"},
        "image_id": {"type": "int32"},
        "annotation": {"type": "int32", "shape": [-1, 6]},
    }
    writer.add_schema(defrcn_json, "defrcn_json")

    for dataset_dict in dataset_dicts:
        with open(dataset_dict['file_name'], 'rb') as f:
            img = f.read()
        img_id = dataset_dict['image_id']
        annos = []
        for anno in dataset_dict['annotations']:
            annos.append(anno["bbox"] + anno["category_id"] + anno["iscrowd"])
        annos = np.array(annos, np.int32)
        row = {"image": img, "image_id": img_id, "annotation": annos}
        writer.write_raw_data([row])
    writer.commit()

def create_defrcn_dataset(config, mindrecord_file, batch_size=2, device_num=1, rank_id=0, is_training=True,
                              num_parallel_workers=8, python_multiprocessing=False):
    """Create defrcn dataset with MindDataset."""
    cv2.setNumThreads(0)
    de.config.set_prefetch_size(8)
    if is_training:
        sampler = de.DistributedSampler(num_shards=device_num, shard_id=rank_id, shuffle=True)
    else:
        sampler = de.DistributedSampler(num_shards=device_num, shard_id=rank_id, shuffle=False)
    ds = de.MindDataset(mindrecord_file, columns_list=["image", "image_id", "annotation"], 
                        num_parallel_workers=num_parallel_workers, sampler=sampler)
    decode = ms.dataset.vision.Decode()
    ds = ds.map(input_columns=["image"], operations=decode)
    compose_map_func = (lambda image, image_id, annotation: preprocess_fn(image, image_id ,annotation, is_training, config=config))

    if is_training:
        ds = ds.map(input_columns=["image", "image_id", "annotation"],
                    output_columns=["image", "image_id", "image_shape", "box", "label", "valid_num"],
                    column_order=["image", "image_id", "image_shape", "box", "label", "valid_num"],
                    operations=compose_map_func, python_multiprocessing=python_multiprocessing,
                    num_parallel_workers=num_parallel_workers)
        # ds = ds.project(["image", "image_id", "image_shape", "box", "label", "valid_num"])
        ds = ds.batch(batch_size, drop_remainder=True)
    else:
        ds = ds.map(input_columns=["image", "image_id", "annotation"],
                    output_columns=["image", "image_id", "image_shape", "box", "label", "valid_num"],
                    column_order=["image", "image_id", "image_shape", "box", "label", "valid_num"],
                    operations=compose_map_func,
                    num_parallel_workers=num_parallel_workers)
        # ds = ds.project(["image", "image_id", "image_shape", "box", "label", "valid_num"])
        ds = ds.batch(batch_size, drop_remainder=False)
    return ds