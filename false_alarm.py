import argparse
import json
import pickle as pkl
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from mmrotate.evaluation.functional.mean_ap import tpfp_default
from mmrotate.structures import hbox2qbox, rbox2qbox
from terminaltables import AsciiTable
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('res_dir', type=str, default=None)
    parser.add_argument('save_dir', type=str, default=None)
    parser.add_argument('classes_file', type=str, default=None)

    parser.add_argument('--gt-dir', type=str, default=None)
    parser.add_argument('--img-dir', type=str, default=None)

    parser.add_argument('--load-type', choices=['dota', 'coco', 'pkl'], type=str, default="pkl")

    parser.add_argument('--keep-metric',
                        choices=[
                            'f1_score', 'precision', 'recall', 'tp', 'fp',
                            'cls_tp', 'cls_fp'
                        ],
                        default='f1_score')
    parser.add_argument('--metric-thre', type=float, default=1)
    parser.add_argument('--score-thre', type=float, default=0.05)
    parser.add_argument('--img-suffix', type=str, default='.png')
    parser.add_argument('--visualizer', action="store_true", default=True)
    return parser.parse_args()


def resp_mkdir(dirname: Path, msg: str = None):
    if dirname.exists():
        ans = ""
        while ans not in ["yes", "no"]:
            msg = msg if msg is not None else \
                f"find exists {dirname}, do you want to" \
                " delete it (yes or no): "
            ans = input(msg).lower()
            if ans == "yes":
                shutil.rmtree(dirname)
                dirname.mkdir()
            elif ans == "no":
                exit(0)
    else:
        dirname.mkdir()


def load_dota_gt(data_path: Path, 
                 classes_map: dict, 
                 mode="dota", 
                 ignore_classes: list = None) -> dict:

    logger.info("load dota ground truth info ...")
    if mode == "coco" or mode == "dota":
        if not isinstance(data_path, Path):
            data_path = Path(data_path)
        glob_label_files = list(data_path.glob("*.txt"))

        infos = {}
        for label_file_path in tqdm(glob_label_files, desc="loading", ncols=100):
            file_stem = label_file_path.stem
            with open(label_file_path, "r") as f:
                lines = f.readlines()
            single_info = {"bboxes": [], "labels": []}
            for line in lines:
                if line.startswith('imagesource') or \
                line.startswith('gsd') or \
                line.startswith('NaN'):
                    continue
                line = line.strip()
                line = line.split(' ')
                class_name = line[-2]
                if ignore_classes is not None and class_name in ignore_classes:
                    continue
                box = list(map(float, line[:8]))
                single_info["bboxes"].append(box)
                single_info["labels"].append(class_name)
            infos[file_stem] = single_info
        logger.info("finish loading dota ann!")

    elif mode == "pkl":
        reverse_classes_map = {v:k for k, v in classes_map.items()}
        with open(data_path, "rb") as f:
            data = pkl.load(f)

        infos = {}
        for data_item in tqdm(data, desc="loading...", ncols=100):
            img_path = Path(data_item["img_path"])
            img_name = img_path.stem
            gt_instances = data_item["gt_instances"]
            bboxes = gt_instances["bboxes"]
            labels = gt_instances["labels"]

            single_info = {"bboxes": [], "labels": []}
            for box_torch, label_torch in zip(bboxes, labels):
                box = rbox2qbox(box_torch[None])[0]
                box = box.tolist()
                label = label_torch.tolist()
                class_name = reverse_classes_map[label]
                if ignore_classes is not None and class_name in ignore_classes:
                    continue
                single_info["bboxes"].append(box)
                single_info["labels"].append(class_name)
            infos[img_name] = single_info
    else:
        raise NotImplementedError
        

    return infos


def load_dota_res(data_path: Path, 
                  classes_map: dict, 
                  mode="coco", 
                  ignore_classes: list = None, 
                  score_thre=0.05) -> dict:
    if not isinstance(data_path, Path):
        data_path = Path(data_path)

    if mode == "dota":
        glob_label_files = list(data_path.glob("*.txt"))

        infos = {}
        logger.info("load dota results info ...")
        for label_file_path in tqdm(glob_label_files, desc="loading", ncols=100):
            file_stem = label_file_path.stem
            if file_stem.startswith("Task1") or file_stem.startswith("Task2"):
                class_name = file_stem[file_stem.find("_") + 1:]
            else:
                class_name = file_stem
            if ignore_classes is not None and class_name in ignore_classes:
                continue

            with open(label_file_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = line.split(' ')
                img_name = line[0]
                score = float(line[1])
                if score < score_thre: continue
                box = list(map(float, line[2:]))
                if img_name in infos:
                    infos[img_name]["bboxes"].append(box)
                    infos[img_name]["scores"].append(score)
                    infos[img_name]["labels"].append(class_name)
                else:
                    infos[img_name] = {
                        "bboxes": [box],
                        "scores": [score],
                        "labels": [class_name]
                    }

    elif mode == "coco":
        reverse_classes_map = {v:k for k, v in classes_map.items()}
        with open(data_path, "rb") as f:
            data = json.load(f)
        infos = {}
        for data_item in tqdm(data, desc="loading...", ncols=100):
            img_name = data_item['image_id']
            box = data_item['bbox']
            box_torch = torch.tensor(box, dtype=torch.float32)
            if len(box_torch) == 5:
                box = rbox2qbox(box_torch[None])[0]
            elif len(box_torch) == 4:               
                box = hbox2qbox(box_torch[None])[0]
            box = box_torch.tolist()
            score = data_item["score"]
            if score < score_thre: continue
            class_name = reverse_classes_map[data_item['category_id']]
            if ignore_classes is not None and class_name in ignore_classes:
                continue

            if img_name in infos:
                infos[img_name]["bboxes"].append(box)
                infos[img_name]["scores"].append(score)
                infos[img_name]["labels"].append(class_name)
            else:
                infos[img_name] = {
                    "bboxes": [box],
                    "scores": [score],
                    "labels": [class_name]
                }
    
    elif mode == "pkl":
        reverse_classes_map = {v:k for k, v in classes_map.items()}
        with open(data_path, "rb") as f:
            data = pkl.load(f)
        infos = {}
        for data_item in tqdm(data, desc="loading...", ncols=100):
            img_path = Path(data_item["img_path"])
            img_name = img_path.stem
            pred_instances = data_item["pred_instances"]
            bboxes = pred_instances["bboxes"]
            labels = pred_instances["labels"]
            scores = pred_instances["scores"]
            if len(labels) == 0: continue

            single_info = dict(bboxes = [], scores = [], labels = [])
            for box_torch, label_torch, score_torch in zip(bboxes, labels, scores):
                if len(box_torch) == 5:
                    box = rbox2qbox(box_torch[None])[0]
                elif len(box_torch) == 4:               
                    box = hbox2qbox(box_torch[None])[0]

                box = box.tolist()
                label = label_torch.tolist()
                score = score_torch.tolist()
                if score < score_thre: continue
                class_name = reverse_classes_map[label]
                if ignore_classes is not None and class_name in ignore_classes:
                    continue
                single_info["bboxes"].append(box)
                single_info["labels"].append(class_name)
                single_info["scores"].append(score)
            infos[img_name] = single_info
            

    logger.info("finish load dota results!")
    return infos


def load_class(file_path: Path, ignore_classes: list = None) -> dict:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    
    with open(file_path, "r") as f: 
        lines = f.readlines()
    infos = {}
    for i, line in enumerate(lines):
        line = line.strip()
        if len(line) == 0:
            continue
        class_name = line
        if ignore_classes is not None and class_name in ignore_classes:
            continue
        infos[class_name] = i
    return infos


def main():
    args = get_args()
    assert args.res_dir is not None
    assert args.save_dir is not None

    res_dir = Path(args.res_dir)
    save_dir = Path(args.save_dir)
    classes_file = Path(args.classes_file)
    load_type = args.load_type
    score_thre = args.score_thre

    gt_dir = args.gt_dir
    img_dir = args.img_dir

    assert gt_dir is not None, "gt_dir is None"

    if args.visualizer:
        assert img_dir is not None, "img_dir is None"
        img_dir = Path(img_dir)

    gt_dir = Path(gt_dir)

    img_suffix = args.img_suffix
    save_img_dir = save_dir / "images"
    save_label_dir = save_dir / "labels"

    keep_metric = args.keep_metric
    metric_thre = args.metric_thre

    logger.info(f'apply metric {keep_metric} and thre {metric_thre}')
    if keep_metric in ['precision', 'recall', 'f1_score']:
        logger.info(
            f'if metric val of the image less than {metric_thre}, it will be selected'
        )
        assert 0 <= metric_thre <= 1, f"{keep_metric} must be 0~1"
    elif keep_metric in ['fp']:
        logger.info(
            f'if metric val of the image more than {metric_thre}, it will be selected'
        )
        assert metric_thre >= 0, f"{keep_metric} must >= 0"
    elif keep_metric in ['tp']:
        logger.info(
            f'if metric val of the image less than {metric_thre}, it will be selected'
        )
        assert metric_thre >= 0, f"{keep_metric} must >= 0"
    elif keep_metric in ['cls_fp']:
        logger.info(
            f'if any val of the metric of the image more than {metric_thre}, it will be selected'
        )
        assert metric_thre >= 0, f"{keep_metric} must >= 0"
    elif keep_metric in ['cls_tp']:
        logger.info(
            f'if any val of the metric of the image less than {metric_thre}, it will be selected'
        )
        assert metric_thre >= 0, f"{keep_metric} must >= 0"

    assert res_dir.exists(), f"{res_dir} don't exist"

    resp_mkdir(save_img_dir)
    resp_mkdir(save_label_dir)

    class_map = load_class(classes_file)
    res_infos = load_dota_res(res_dir, class_map, load_type, score_thre = score_thre)
    gt_infos = load_dota_gt(gt_dir, class_map)

    logger.info(f"load {len(class_map)} classes")
    logger.info(f"load {len(res_infos)} results")
    logger.info(f"load {len(gt_infos)} ground truth")

    classes_tp = [0 for _ in range(len(class_map))]
    classes_fp = [0 for _ in range(len(class_map))]
    classes_gt = [0 for _ in range(len(class_map))]
    img_eval_info = dict()
    eps = np.finfo(np.float32).eps
    keep_img = []
    for img_name, gt_info in tqdm(gt_infos.items(), desc="checking",
                                  ncols=100):
        single_img_eval_info = {}
        bboxes = gt_info["bboxes"]
        labels = gt_info["labels"]

        def get_labels_map(labels):
            new_labels = []
            for label in labels:
                new_labels.append(class_map[label])
            return new_labels

        labels = get_labels_map(labels)

        labels = np.asarray(labels, dtype=np.int64)
        bboxes = np.asarray(bboxes, dtype=np.float32) if len(bboxes) else \
                    np.empty((0, 8), dtype=np.float32)

        def split_to_list(bboxes, labels, ndims=8):
            bboxes_list = [
                np.empty((0, ndims), dtype=np.float32)
                for _ in range(len(class_map))
            ]
            if len(bboxes) == 0:
                return bboxes_list

            for class_id in range(len(class_map)):
                class_mask = class_id == labels
                bboxes_list[class_id] = bboxes[class_mask]
            return bboxes_list

        gt_bboxes_list = split_to_list(bboxes, labels)

        if img_name not in res_infos:
            res_bboxes_list = [
                np.empty((0, 9), dtype=np.float32)
                for _ in range(len(class_map))
            ]
        else:
            res_info = res_infos[img_name]
            bboxes = res_info["bboxes"]
            labels = res_info["labels"]
            scores = res_info["scores"]

            def merge_scores_bboxes(bboxes, scores):
                new_bboxes = []
                for bbox, score in zip(bboxes, scores):
                    bbox.append(score)
                    new_bboxes.append(bbox)
                return new_bboxes

            bboxes = merge_scores_bboxes(bboxes, scores)
            labels = get_labels_map(labels)
            bboxes = np.asarray(bboxes, dtype=np.float32) if len(bboxes) else \
                        np.empty((0, 9), dtype=np.float32)
            labels = np.asarray(labels, dtype=np.int64)
            res_bboxes_list = split_to_list(bboxes, labels, ndims=9)
        total_fp = 0
        total_tp = 0
        cls_fp = []
        cls_tp = []
        num_gts = 0
        for i, (res_bboxes,
                gt_bboxes) in enumerate(zip(res_bboxes_list, gt_bboxes_list)):
            num_gts += len(gt_bboxes)
            tp, fp = tpfp_default(res_bboxes,
                                  gt_bboxes,
                                  np.empty((0, 8), dtype=np.float32),
                                  iou_thr=0.1,
                                  box_type='qbox')
            if len(classes_gt) == 0:
                continue

            total_tp += tp.sum()
            total_fp += fp.sum()
            classes_tp[i] += tp.sum()
            classes_fp[i] += fp.sum()
            classes_gt[i] += len(gt_bboxes)
            cls_tp.append(tp.sum())
            cls_fp.append(fp.sum())

        single_img_eval_info["precision"] = precision = total_tp / \
            max(total_tp + total_fp, eps)
        single_img_eval_info["recall"] = recall = total_tp / \
            max(len(gt_info["bboxes"]), eps)

        single_img_eval_info["f1_score"] = precision * recall / \
            max(precision + recall, eps) if num_gts != 0 else 0
        single_img_eval_info["tp"] = tp
        single_img_eval_info["fp"] = fp
        single_img_eval_info["cls_tp"] = cls_tp
        single_img_eval_info["cls_fp"] = cls_fp
        img_eval_info[img_name] = single_img_eval_info

        eval_val = single_img_eval_info[keep_metric]
        if keep_metric in ['precision', 'recall', 'f1_score', 'tp']:
            if eval_val < metric_thre:
                keep_img.append(img_name)
        elif keep_metric in ['fp']:
            if eval_val > metric_thre:
                keep_img.append(img_name)
        elif keep_metric in ['cls_tp']:
            if any(eval_val < m for m in eval_val):
                keep_img.append(img_name)
        elif keep_metric in ['cls_fp']:
            if any(eval_val > m for m in eval_val):
                keep_img.append(img_name)
        else:
            raise KeyError

    header = ['class', 'recall', 'precision', 'f1_score']
    reverse_class_map = {v: k for k, v in class_map.items()}
    table_data = [header]
    mean_f1_score = []
    for i in range(len(class_map)):
        if classes_gt[i] > 0:
            recall = classes_tp[i] / classes_gt[i]
            precision = classes_tp[i] / (classes_fp[i] + classes_tp[i])
            f1_score = (2 * precision * recall) / (precision + recall)
        else:
            recall = 0
            precision = 0
            f1_score = 0
        row_data = [
            reverse_class_map[i], f'{recall:.3f}', f'{precision:.3f}',
            f'{f1_score:.3f}'
        ]
        table_data.append(row_data)
        if classes_gt[i] > 0:
            mean_f1_score.append(f1_score)
    table_data.append([
        'mean_f1_score', '', '',
        f'{sum(mean_f1_score) / len(mean_f1_score):.3f}'
    ])
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True

    logger.info('\n********eval results********\n' + table.table)

    if len(keep_img) > 0:
        logger.info(f'{len(keep_img)} images meet the metric')
        logger.info('save img ...')
        for img_name in tqdm(keep_img, ncols=100, desc="saving..."):
            ori_img_file = img_dir / (img_name + img_suffix)
            ori_label_file = gt_dir / (img_name + ".txt")
            save_img_file = save_img_dir / (img_name + img_suffix)
            save_label_file = save_label_dir / (img_name + ".txt")
            shutil.copyfile(ori_img_file, save_img_file)
            shutil.copyfile(ori_label_file, save_label_file)

    if args.visualizer:
        logger.info(f"visualizer images...")
        vis_dir = save_dir / "vis_dir"
        resp_mkdir(vis_dir)

        def visualizer(img_name: str):
            ann = res_infos[img_name] if img_name in res_infos else None
            gt = gt_infos[img_name]
            img_file = img_dir / (img_name + img_suffix)
            img = cv2.imread(str(img_file))
            assert img is not None
            if ann is not None:
                ann_bboxes = ann["bboxes"]
                ann_labels = ann["labels"]
                ann_scores = ann["scores"]

            gt_bboxes = gt["bboxes"]
            gt_labels = gt["labels"]
            vis_path = vis_dir / (img_name + img_suffix)

            if ann is not None:
                for ann_bbox, ann_label, ann_score in zip(ann_bboxes, ann_labels, ann_scores):
                    info = f"{ann_label}:{ann_score:.3f}" 
                    ann_bbox_np = np.asarray(ann_bbox[:8], dtype=np.int32).reshape(-1, 1, 2)
                    ctr_x = ann_bbox_np[:, 0, 0].mean()
                    ctr_y = ann_bbox_np[:, 0, 1].mean()

                    cv2.putText(img, info, (int(ctr_x), int(ctr_y)), 
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                                color=(0, 0, 255), thickness=2)

                    cv2.polylines(img, [ann_bbox_np], True, (255, 0, 0), thickness=3)
            
            for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
                info = f"{gt_label}" 
                gt_bbox_np = np.asarray(gt_bbox, dtype=np.int32).reshape(-1, 1, 2)
                ctr_x = gt_bbox_np[:, 0, 0].mean()
                ctr_y = gt_bbox_np[:, 0, 1].mean()

                cv2.putText(img, info, (int(ctr_x), int(ctr_y)), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                            color=(0, 0, 255), thickness=2)

                cv2.polylines(img, [gt_bbox_np], True, (0, 255, 0), thickness=3)
            cv2.imwrite(str(vis_path), img)
            
        
        res = list(map(visualizer, keep_img))




if __name__ == "__main__":
    main()
