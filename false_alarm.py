import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from loguru import logger
from mmrotate.evaluation.functional.mean_ap import tpfp_default
from terminaltables import AsciiTable
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('res_dir', type=str, default=None)
    parser.add_argument('gt_dir', type=str, default=None)
    parser.add_argument('save_dir', type=str, default=None)
    parser.add_argument('--img-dir', type=str, default=None)
    parser.add_argument('--keep-metric',
                        choices=[
                            'f1_score', 'precision', 'recall', 'tp', 'fp',
                            'cls_tp', 'cls_fp'
                        ],
                        default='f1_score')
    parser.add_argument('--metric-thre', type=float, default=0.8)
    parser.add_argument('--img-suffix', type=str, default='.png')
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


def load_dota_gt(data_dir: Path, ignore_classes: list = None) -> dict:
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
    glob_label_files = list(data_dir.glob("*.txt"))

    infos = {}
    logger.info("load dota ground truth info ...")
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
    return infos


def load_dota_res(data_dir: Path, ignore_classes: list = None) -> dict:
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    glob_label_files = list(data_dir.glob("*.txt"))

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

    logger.info("finish load dota results!")
    return infos


def load_class(data_dir: Path, ignore_classes: list = None) -> dict:
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
    glob_label_files = list(data_dir.glob("*.txt"))

    infos = {}
    index = 0
    for label_file_path in glob_label_files:
        file_stem = label_file_path.stem
        if file_stem.startswith("Task1"):
            class_name = file_stem[file_stem.find("_") + 1:]
        else:
            class_name = file_stem
        if ignore_classes is not None and class_name in ignore_classes:
            continue
        if class_name not in infos:
            infos[class_name] = index
            index = index + 1

    return infos


def main():
    args = get_args()
    assert args.res_dir is not None
    assert args.gt_dir is not None
    assert args.save_dir is not None

    res_dir = Path(args.res_dir)
    gt_dir = Path(args.gt_dir)
    save_dir = Path(args.save_dir)
    img_dir = Path(args.img_dir)

    img_suffix = args.img_suffix
    save_img_dir = save_dir / "images"
    save_label_dir = save_dir / "labels"

    keep_metric = args.keep_metric
    metric_thre = args.metric_thre

    logger.info(f'apply metric {keep_metric} and thre {metric_thre}')
    if keep_metric in ['precision', 'recall', 'f1_score']:
        logger.info(
            f'if metric val of the image gather than {metric_thre}, it will keep'
        )
        assert 0 <= metric_thre <= 1, f"{keep_metric} must be 0~1"
    elif keep_metric in ['tp', 'fp']:
        logger.info(
            f'if metric val of the image less than {metric_thre}, it will keep'
        )
        assert metric_thre >= 0, f"{keep_metric} must >= 0"
    elif keep_metric in ['cls_tp', 'cls_fp']:
        logger.info(
            f'if all metric val of the list of the image less than {metric_thre}, it will keep'
        )
        assert metric_thre >= 0, f"{keep_metric} must >= 0"

    assert res_dir.exists(), f"{res_dir} don't exist"
    assert gt_dir.exists(), f"{gt_dir} don't exist"
    resp_mkdir(save_img_dir)
    resp_mkdir(save_label_dir)

    res_infos = load_dota_res(res_dir)
    gt_infos = load_dota_gt(gt_dir)
    class_map = load_class(res_dir)
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
        for i, (res_bboxes,
                gt_bboxes) in enumerate(zip(res_bboxes_list, gt_bboxes_list)):
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
        single_img_eval_info["precision"] = precision = (total_tp + eps) / (
            total_tp + total_fp + eps)
        single_img_eval_info["recall"] = recall = (total_tp + eps) / (
            len(gt_info["bboxes"]) + eps)
        single_img_eval_info["f1_score"] = (2 * precision * recall +
                                            eps) / (precision + recall + eps)
        single_img_eval_info["tp"] = tp
        single_img_eval_info["fp"] = fp
        single_img_eval_info["cls_tp"] = cls_tp
        single_img_eval_info["cls_fp"] = cls_fp
        img_eval_info[img_name] = single_img_eval_info

        if keep_metric in ['precision', 'recall', 'f1_score']:
            eval_val = single_img_eval_info[keep_metric]
            if eval_val > metric_thre:
                keep_img.append(img_name)
        elif keep_metric in ['tp', 'fp']:
            eval_val = single_img_eval_info[keep_metric]
            if eval_val < metric_thre:
                keep_img.append(img_name)
        elif keep_metric in ['cls_tp', 'cls_fp']:
            eval_val = single_img_eval_info[keep_metric]
            if all(eval_val < m for m in single_img_eval_info[keep_metric]):
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
            f1_score = (2 * precision * recall + eps) / (precision + recall +
                                                         eps)
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

    logger.info('\neval results:' + table.table)

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


if __name__ == "__main__":
    main()
