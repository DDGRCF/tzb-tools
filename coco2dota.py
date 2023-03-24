import argparse
import json
import shutil
from pathlib import Path

import torch
from loguru import logger
from mmrotate.structures import rbox2qbox
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-path', type=str, default='/path/to/coco/annotations')
    parser.add_argument('--save-dir', type=str, default='/dir/to/save/convert_results')
    parser.add_argument('--classes-file', type=str, default=None)
    parser.add_argument('--ignore-classes', nargs='+', type=str, default=None)
    parser.add_argument('--score-thre', type=float, default=0.05)
    opt = parser.parse_args()
    return opt

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

def load_infos(data_path: Path, 
               classes_map: dict, 
               ignore_classes: list = None, 
               score_thre=0.05) -> dict:
    if not isinstance(data_path, Path):
        data_path = Path(data_path)

    reverse_classes_map = {v:k for k, v in classes_map.items()}
    with open(data_path, "rb") as f:
        data = json.load(f)
    infos = {}
    for data_item in tqdm(data, desc="loading...", ncols=100):
        img_name = data_item['image_id']
        box = data_item['bbox']
        box_torch = torch.tensor(box, dtype=torch.float32)[None]
        box_torch = rbox2qbox(box_torch)
        box_torch = box_torch[0]
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
    
    logger.info("finish load dota results!")
    return infos


@logger.catch
def main(opt):
    json_path = Path(opt.json_path)
    save_dir = Path(opt.save_dir)
    class_file = Path(opt.classes_file)
    resp_mkdir(save_dir)

    class_map = load_class(class_file)
    json_infos = load_infos(json_path, class_map, opt.ignore_classes, opt.score_thre)
    for class_name in tqdm(class_map):
        class_dst_file = save_dir / ("Task1_" + class_name + ".txt")
        with open(class_dst_file, "w") as f:
            for img_name in json_infos:
                json_info = json_infos[img_name]
                labels = json_info["labels"]
                bboxes = json_info["bboxes"]
                scores = json_info["scores"]
                for label, bbox, score in zip(labels, bboxes, scores):
                    if label != class_name:
                        continue
                    line = [img_name, str(score)]
                    bbox_str = list(map(str, bbox))
                    line.extend(bbox_str)
                    line = ' '.join(line) + '\n'
                    f.write(line)

if __name__ == '__main__':
    opt = get_opt()
    main(opt)