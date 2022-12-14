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

"""Evaluation for Defrcn"""
import os
import time
import numpy as np
from tabulate import tabulate
from pycocotools.coco import COCO
import mindspore as ms
from mindspore.common import set_seed, Parameter
import itertools
from src.defrcn.built_datacategory import register_all_coco, METADATACATALOG
from src.dataset import data_to_mindrecord_byte_image, create_defrcn_dataset
from src.util import coco_eval, bbox2result_1image, results2json
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id
from src.defrcn.de_frcn import De_Frcn
from src.defrcn.pcb import PrototypicalCalibrationBlock
from collections import OrderedDict
from mindspore.communication.management import init, get_rank

def defrcn_eval(dataset_path, ckpt_path, anno_path):
    """defrcn evaluation."""
    rank=0
    if config.run_distribute==True:
        init()
        rank = get_rank()
    if rank == 0:
        print("Eval Loading from checkpoint:", ckpt_path)
        if not os.path.isfile(ckpt_path):
            raise RuntimeError("CheckPoint file {} is not valid.".format(ckpt_path))
        ds = create_defrcn_dataset(config, dataset_path, batch_size=config.test_batch_size, is_training=False, num_parallel_workers=1)
        net = De_Frcn(config)
        param_dict = ms.load_checkpoint(ckpt_path)
        if config.device_target == "GPU":
            for key, value in param_dict.items():
                tensor = value.asnumpy().astype(np.float32)
                param_dict[key] = Parameter(tensor, key)
        ms.load_param_into_net(net, param_dict)
        net.set_train(False)
        device_type = "Ascend" if ms.get_context("device_target") == "Ascend" else "Others"
        if device_type == "Ascend":
            net.to_float(ms.float32)

        eval_iter = 0
        total = ds.get_dataset_size()
        outputs = []
        
        metadata = METADATACATALOG.get(config.dataset_test[0])
        dataset_coco = COCO(metadata["json_file"])
        print("\n========================================\n")
        print("total images num: ", total)
        print("Processing, please wait a moment.")
        
        if config.pcb_enable:
            print("Start initializing PCB module, it will take some time, please wait...")
            pcb = PrototypicalCalibrationBlock(config)

        img_ids = []
        max_num = config.num_gts
        for data in ds.create_dict_iterator(num_epochs=1):
            eval_iter = eval_iter + 1
            img_data = data['image']
            img_id = data['image_id']
            img_metas = data['image_shape']
            gt_bboxes = data['box']
            gt_labels = data['label']
            gt_num = data['valid_num']
            img_ids.append(img_id)
            start = time.time()
            # run net
            output = net(img_data, img_id, img_metas, gt_bboxes, gt_labels, gt_num)
            end = time.time()
            print("Iter {} cost time {}".format(eval_iter, end - start))
            
            
            # output
            all_bbox = output[0]
            all_label = output[1]
            all_mask = output[2]
            for j in range(config.test_batch_size):
                all_bbox_squee = np.squeeze(all_bbox.asnumpy()[j, :, :])
                all_label_squee = np.squeeze(all_label.asnumpy()[j, :, :])
                all_mask_squee = np.squeeze(all_mask.asnumpy()[j, :, :])
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


        eval_types = ["bbox"]
        result_files = results2json(dataset_coco, img_ids, outputs, metadata, "./results.pkl")
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
    
def modelarts_pre_process():
    pass


@moxing_wrapper(pre_process=modelarts_pre_process)
def eval_defrcn(eval_checkpoint_path):
    """ eval_defrcn """
    prefix = "mindrecord"
    mindrecord_dir = os.path.join(config.test_mindrecord_dir, config.dataset_test[0])
    mindrecord_file = os.path.join(mindrecord_dir, prefix)
    print("CHECKING MINDRECORD FILES ...")
    rank = 0
    if config.run_distribute==True:
        init()
        rank=get_rank()
    if rank == 0 and not os.path.exists(mindrecord_file + ".db"):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if config.dataset == "coco":
            if os.path.isdir(config.coco_root):
                print("Create Mindrecord. It may take some time.")
                data_to_mindrecord_byte_image(config, config.dataset_test, prefix, False, file_num=1)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("coco_root not exits.")

    while not os.path.exists(mindrecord_file + ".db"):
        print("wait 5 seconds")
        time.sleep(5)

    print("CHECKING MINDRECORD FILES DONE!")
    print("register_all_coco_dataset... please wait...")
    register_all_coco(config.datasets_root)
    print("register_all_coco done!")
    print("Start Eval!")
    start_time = time.time()

    eval_anno_path = METADATACATALOG.get(config.dataset_test[0])["json_file"]

    defrcn_eval(mindrecord_file, eval_checkpoint_path, eval_anno_path)
    
    end_time = time.time()
    total_time = end_time - start_time
    print("\nDone!\nTime taken: {:.2f} seconds".format(total_time))

    flags = [0] * 3
    config.eval_result_path = os.path.abspath("./eval_result")
    if os.path.exists(config.eval_result_path):
        result_files = os.listdir(config.eval_result_path)
        for file in result_files:
            if file == "statistics.csv":
                with open(os.path.join(config.eval_result_path, "statistics.csv"), "r") as f:
                    res = f.readlines()
                if len(res) > 1:
                    if "class_name" in res[3] and "tp_num" in res[3] and len(res[4].strip().split(",")) > 1:
                        flags[0] = 1
            elif file in ("precision_ng_images", "recall_ng_images", "ok_images"):
                imgs = os.listdir(os.path.join(config.eval_result_path, file))
                if imgs:
                    flags[1] = 1

            elif file == "pr_curve_image":
                imgs = os.listdir(os.path.join(config.eval_result_path, "pr_curve_image"))
                if imgs:
                    flags[2] = 1
            else:
                pass

    if sum(flags) == 3:
        print("Successfully created 'eval_results' visualizations")
        exit(0)
    else:
        print("Failed to create 'eval_results' visualizations")
        exit(-1)


if __name__ == '__main__':
    set_seed(config.seed)
    ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id())
    eval_defrcn(config.eval_checkpoint_path)
    