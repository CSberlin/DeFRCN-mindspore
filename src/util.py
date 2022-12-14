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
"""coco eval for defrcn"""

import json
import os
import csv
import shutil
import numpy as np
import mmcv
from pycocotools.coco import COCO
from src.detecteval import DetectEval
from pycocotools.cocoeval import COCOeval

_init_value = np.array(0.0)
summary_init = {
    'Precision/mAP': _init_value,
    'Precision/mAP@.50IOU': _init_value,
    'Precision/mAP@.75IOU': _init_value,
    'Precision/mAP (small)': _init_value,
    'Precision/mAP (medium)': _init_value,
    'Precision/mAP (large)': _init_value,
    'Recall/AR@1': _init_value,
    'Recall/AR@10': _init_value,
    'Recall/AR@100': _init_value,
    'Recall/AR@100 (small)': _init_value,
    'Recall/AR@100 (medium)': _init_value,
    'Recall/AR@100 (large)': _init_value,
}

def write_list_to_csv(file_path, data_to_write, append=False):
    # print('Saving data into file [{}]...'.format(file_path))
    if append:
        open_mode = 'a'
    else:
        open_mode = 'w'
    with open(file_path, open_mode) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data_to_write)


def coco_eval(config, split, result_files, result_types, coco, catIds=None, max_dets=(100, 300, 1000), single_result=False,
              plot_detect_result=False):
    """coco eval for defrcn"""
    anns = json.load(open(result_files['bbox']))

    for res_type in result_types:
        result_file = result_files[res_type]
        assert result_file.endswith('.json')

        coco_dets = coco.loadRes(result_file)
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        # cocoEval = DetectEval(coco, coco_dets, iou_type)
        cocoEval = DetectEval(coco, coco_dets, iou_type, catIds)
        if catIds is not None:
            cocoEval.params.catIds = catIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        if plot_detect_result:
            calcuate_pr_rc_f1(config, split, coco, coco_dets, iou_type, catIds)

    return cocoEval


def calcuate_pr_rc_f1(config, split, coco, coco_dets, iou_type, catIds):
    cocoEval = DetectEval(coco, coco_dets, iou_type, catIds)
    if catIds is not None:
        cocoEval.params.catIds = catIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    stats_all = cocoEval.stats

    eval_result_path = os.path.abspath("./eval_result_"+split)
    if os.path.exists(eval_result_path):
        shutil.rmtree(eval_result_path)
    os.mkdir(eval_result_path)

    result_csv = os.path.join(eval_result_path, "statistics.csv")
    eval_item = ["ap@0.5:0.95", "ap@0.5", "ap@0.75", "ar@0.5:0.95", "ar@0.5", "ar@0.75"]
    write_list_to_csv(result_csv, eval_item, append=False)
    eval_result = [round(stats_all[0], 3), round(stats_all[1], 3), round(stats_all[2], 3), round(stats_all[6], 3),
                   round(stats_all[7], 3), round(stats_all[8], 3)]
    write_list_to_csv(result_csv, eval_result, append=True)
    write_list_to_csv(result_csv, [], append=True)
    # 1.2 plot_pr_curve
    cocoEval.plot_pr_curve(eval_result_path)

    # 2
    E = DetectEval(coco, coco_dets, iou_type, catIds)
    if catIds is not None:
        E.params.catIds = catIds
    E.params.iouThrs = [0.5]
    E.params.maxDets = [100]
    E.params.areaRng = [[0 ** 2, 1e5 ** 2]]
    E.evaluate()
    # 2.1 plot hist_curve of every class's tp's confidence and fp's confidence
    confidence_dict = E.compute_tp_fp_confidence()
    E.plot_hist_curve(confidence_dict, eval_result_path)

    # 2.2 write best_threshold and p r to csv and plot
    cat_pr_dict, cat_pr_dict_origin = E.compute_precison_recall_f1()
    # E.write_best_confidence_threshold(cat_pr_dict, cat_pr_dict_origin, eval_result_path)
    best_confidence_thres = E.write_best_confidence_threshold(cat_pr_dict, cat_pr_dict_origin, eval_result_path)
    print("best_confidence_thres: ", best_confidence_thres)
    E.plot_mc_curve(cat_pr_dict, eval_result_path)

    # 3
    # 3.1 compute every class's p r and save every class's p and r at iou = 0.5
    E = DetectEval(coco, coco_dets, iou_type, catIds)
    if catIds is not None:
        E.params.catIds = catIds
    E.params.iouThrs = [0.5]
    E.params.maxDets = [100]
    E.params.areaRng = [[0 ** 2, 1e5 ** 2]]
    E.evaluate()
    E.accumulate()
    result = E.evaluate_every_class()
    print_info = ["class_name", "tp_num", "gt_num", "dt_num", "precision", "recall"]
    write_list_to_csv(result_csv, print_info, append=True)
    print("class_name", "tp_num", "gt_num", "dt_num", "precision", "recall")
    for class_result in result:
        print(class_result)
        write_list_to_csv(result_csv, class_result, append=True)

    # 3.2 save ng / ok images
    E.save_images(config, eval_result_path, 0.5)

    return stats_all[0]


def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]


def bbox2result_1image(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        result = [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)]
    else:
        result = [bboxes[labels == i, :] for i in range(num_classes - 1)]
    return result


def proposal2json(dataset, results):
    """convert proposal to json mode"""
    img_ids = dataset.getImgIds()
    json_results = []
    dataset_len = dataset.get_dataset_size() * 2
    for idx in range(dataset_len):
        img_id = img_ids[idx]
        bboxes = results[idx]
        for i in range(bboxes.shape[0]):
            data = dict()
            data['image_id'] = img_id
            data['bbox'] = xyxy2xywh(bboxes[i])
            data['score'] = float(bboxes[i][4])
            data['category_id'] = 1
            json_results.append(data)
    return json_results

def det2json(dataset, img_ids_list, results, metadata):
    """convert det to json mode"""
    metadata_dict = metadata['metadata']
    # print("metadata_dict:   ", metadata_dict.keys())

    if "thing_dataset_id_to_contiguous_id" in metadata_dict:
        reverse_id_mapping = {
            v: k for k, v in metadata_dict["thing_dataset_id_to_contiguous_id"].items()
        }
    json_results = []
    dataset_len = len(img_ids_list)
    for idx in range(dataset_len):
        # align img_id
        img_id = img_ids_list[idx]
        if idx == len(results): break
        result = results[idx]
        for label, result_label in enumerate(result):
            # result list len:60  element: array shape(n, 5) n:0-N
            bboxes = result_label
            for i in range(bboxes.shape[0]):
                data = dict()
                # label : 0 - 59
                data['image_id'] = img_id.asnumpy()[0]
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                # 转label然后再cat_ids
                if "thing_dataset_id_to_contiguous_id" in metadata_dict:
                    data['category_id'] = reverse_id_mapping[label+1]
                else:
                    # TODO
                    pass
                    # data['category_id'] = cat_ids[label]
                json_results.append(data)
    return json_results


def results2json(dataset, img_ids, results, metadata, out_file):
    """convert result convert to json mode"""
    result_files = dict()
    if isinstance(results[0], list):
        json_results = det2json(dataset, img_ids, results, metadata)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        mmcv.dump(json_results, result_files['bbox'])
    elif isinstance(results[0], np.ndarray):
        json_results = proposal2json(dataset, results)
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'proposal')
        mmcv.dump(json_results, result_files['proposal'])
    else:
        raise TypeError('invalid type of results')
    return result_files
