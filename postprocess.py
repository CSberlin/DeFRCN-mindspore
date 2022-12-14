# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""post process for 310 inference"""
import os
import numpy as np
from typing import Any, Dict, List, Set

from pycocotools.coco import COCO

from src.util import coco_eval, bbox2result_1image, results2json
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.defrcn.built_datacategory import register_all_coco, DATACATALOG, METADATACATALOG
from src.defrcn.meta_coco import load_coco_json
from src.defrcn.pcb import PrototypicalCalibrationBlock
from src.defrcn.built_datacategory import register_all_coco, METADATACATALOG

import mindspore as ms
from mindspore.common.tensor import Tensor

dst_width = config.img_width
dst_height = config.img_height


def modelarts_pre_process():
    pass

def _derive_coco_results(coco_eval, iou_type, class_names=None):
    """
    Derive the desired score numbers from summarized COCOeval.

    Args:
        coco_eval (None or COCOEval): None represents no predictions from model.
        iou_type (str):
        class_names (None or list[str]): if provided, will use it to predict
            per-category AP.

    Returns:
        a dict of {metric name: score}
    """

    metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]

    # the standard metrics
    results = {
        metric: float(coco_eval.stats[idx] * 100) \
            for idx, metric in enumerate(metrics)
    }

    if class_names is None or len(class_names) <= 1:
        return results
    # Compute per-category AP
    precisions = coco_eval.eval["precision"]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    results_per_category = []
    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        results_per_category.append(("{}".format(name), float(ap * 100)))

    # tabulate it
    N_COLS = min(6, len(results_per_category) * 2)
    results_flatten = list(itertools.chain(*results_per_category))
    results_2d = itertools.zip_longest(
        *[results_flatten[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        results_2d,
        tablefmt="pipe",
        floatfmt=".3f",
        headers=["category", "AP"] * (N_COLS // 2),
        numalign="left",
    )
    print("Per-category {} AP: \n".format(iou_type) + table)

    results.update({"AP-" + name: ap for name, ap in results_per_category})
    return results

@moxing_wrapper(pre_process=modelarts_pre_process)
def get_eval_result(input_path, anno_path, result_path):
    """ get evaluation result of defrcn"""
    input_path = config.input_path
    max_num = config.num_gts
    result_path = result_path
    outputs = []
    
    dataset_root = './datasets'
    register_all_coco(dataset_root)
    dataset_dicts : List[Dict] = []
    datasets_catalog = [DATACATALOG[dataset_name] for dataset_name in config.dataset_test]
    
    for dataset_meta_info in datasets_catalog:
        dataset_coco = COCO(dataset_meta_info["annofile"])

    img_ids = dataset_coco.getImgIds()

    if config.pcb_enable:
        print("Start initializing PCB module, it will take some time, please wait...")
        pcb = PrototypicalCalibrationBlock(config)
    
    # use test mindrecord save
    img_ids=[]
    for ind, img_id in enumerate(img_ids):
        image_info = dataset_coco.loadImgs(img_id)
        img_file = os.path.join(input_path, "eval_img_bin", "eval_input_" + str(ind+1) + ".bin")
        img_ids.append(img_id)
        bbox_result_file = os.path.join(result_path, file_id + "_0.bin")
        label_result_file = os.path.join(result_path, file_id + "_1.bin")
        mask_result_file = os.path.join(result_path, file_id + "_2.bin")
        
        data = Tensor(np.fromfile(img_file, dtype=np.float32).reshape(1, 3, 768, 1280), ms.float32)
        
        all_bbox = np.fromfile(bbox_result_file, dtype=np.float16).reshape(-1, 5)
        all_label = np.fromfile(label_result_file, dtype=np.int32).reshape(-1, 1)
        all_mask = np.fromfile(mask_result_file, dtype=np.bool_).reshape(-1, 1)

        all_bbox_squee = np.squeeze(all_bbox)
        all_label_squee = np.squeeze(all_label)
        all_mask_squee = np.squeeze(all_mask)

        all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
        all_labels_tmp_mask = all_label_squee[all_mask_squee]

        inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
        if all_bboxes_tmp_mask.shape[0] > max_num:    
            inds = inds[:max_num]
        all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
        all_labels_tmp_mask = all_labels_tmp_mask[inds]

        if config.pcb_enable:
            all_bboxes_tmp_mask, all_labels_tmp_mask = pcb.execute_calibration(data, [all_bboxes_tmp_mask, all_labels_tmp_mask])
        
        outputs_tmp = bbox2result_1image(all_bboxes_tmp_mask, all_labels_tmp_mask, config.num_classes)
        outputs.append(outputs_tmp)

    metadata = METADATACATALOG.get(config.dataset_test[0])
    eval_types = ["bbox"]
    result_files = results2json(dataset_coco, img_ids, outputs, metadata, "./results.pkl")
    coco_eval(config, "", result_files, eval_types, dataset_coco, single_result=False)

    is_splits = "all" in config.dataset_test[0] or "base" in config.dataset_test[0] \
                or "novel" in config.dataset_test[0]
        base_classes = [
                8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80,
                81, 82, 84, 85, 86, 87, 88, 89, 90,
            ]
        novel_classes = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
        if is_splits:
            _results = OrderedDict()
            _results["bbox"] = {}
            for split, classes, names in [
                        ("all", base_classes+novel_classes, metadata['metadata'].get("thing_classes")[1:]), # [1:] 为了去除background
                        ("base", base_classes, metadata['metadata'].get("base_classes")[1:]),
                        ("novel", novel_classes, metadata['metadata'].get("novel_classes")[1:])]:
                if "all" not in config.dataset_test[0] and \
                            split not in config.dataset_test[0]:
                        continue
                coco_evaluation = coco_eval(config, split, result_files, eval_types, dataset_coco, classes,
                                    single_result=False, plot_detect_result=True)
                res_ = _derive_coco_results(coco_evaluation, "bbox", class_names=names)
                
                res = {}
                for metric in res_.keys():
                    if len(metric) <= 4:
                        if split == "all":
                            res[metric] = res_[metric]
                        elif split == "base":
                            res["b"+metric] = res_[metric]
                        elif split == "novel":
                            res["n"+metric] = res_[metric]

                _results["bbox"].update(res)
                # add "AP" if not already in
                if "AP" not in _results["bbox"]:
                    if "nAP" in _results["bbox"]:
                        _results["bbox"]["AP"] = _results["bbox"]["nAP"]
                    else:
                        _results["bbox"]["AP"] = _results["bbox"]["bAP"]

        else:
            coco_evaluation = coco_eval(config, "", result_files, eval_types, dataset_coco,
                                    single_result=False, plot_detect_result=True)
            res_ = _derive_coco_results(coco_evaluation, "bbox", class_names=names)
            _results["bbox"] = res
        print("\nEvaluation done!")

if __name__ == '__main__':
    get_eval_result(config.input_path, config.anno_path, config.result_path)
