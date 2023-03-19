import argparse
import json
import os
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path

import cv2
from loguru import logger
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser("dota 2 coco")
    parser.add_argument("ori_dir", type=str, default=None)
    parser.add_argument("save_path", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default="images")
    parser.add_argument("--label-dir", type=str, default="labels")
    parser.add_argument("--ignore_classes", nargs="+", default=["plane"])
    parser.add_argument("--image_suffix", type=str, default=".png")
    return parser.parse_args()


def read_dota_data(data_dir, 
                   image_dir_name="images", 
                   label_dir_name="labels", 
                   label_suffix=".txt", 
                   image_suffix=".tif",
                   ignore_classes=None):

    logger.info("Begin read dota infos...")
    per_image_anns = {}
    per_class_anns = {}

    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    image_dir = data_dir / image_dir_name
    label_dir = data_dir / label_dir_name
    glob_label_files = list(label_dir.glob("*" + label_suffix))
    for label_file_path in tqdm(glob_label_files):
        file_stem = label_file_path.stem 
        label_file_name = label_file_path.name
        image_file_name = file_stem + image_suffix
        image_file_path = image_dir / image_file_name
        with open(label_file_path, "r") as fr:
            lines = fr.readlines()

        for line in lines:
            line = line.strip("\n")
            line = line.lstrip(" ").strip(" ")
            if line.startswith("imagesource") or line.startswith("gsd"):
                continue
            items = line.split(" ")
            if len(items) < 9:
                raise ValueError
            class_name = items[8]
            if ignore_classes is not None and class_name in ignore_classes:
                continue

            box = list(map(float, items[:8]))
            info = dict(box=box, filepath=image_file_path)

            if class_name in per_class_anns:
                per_class_anns[class_name].append(info)
            else:
                per_class_anns[class_name] = [info]

            info = dict(box=box, class_name=class_name)
            if file_stem in per_image_anns:
                per_image_anns[image_file_path].append(info)
            else:
                per_image_anns[image_file_path] = [info]

    return per_class_anns, per_image_anns

def main():
    args = get_args()
    ori_dir = Path(args.ori_dir)
    save_path = Path(args.save_path)
    image_suffix = args.image_suffix

    ignore_classes = args.ignore_classes

    per_classs_anns, per_image_anns = read_dota_data(ori_dir, args.image_dir, args.label_dir, 
                                                    image_suffix=image_suffix, 
                                                    ignore_classes=ignore_classes)
    data_dict = {}
    info = {'contributor': 'ddgrcf',
            'data_created': '2023',
            'description': 'It is just from Tianzhi Cup.',
            'url': 'https://github.com/DDGRCF',
            'version': '1.0',
            'year': 2023}
    data_dict['info'] = info
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []

    class_map = dict()
    for idx, name in enumerate(per_classs_anns):
        single_cat = {'id': idx + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)
        class_map[name] = idx + 1

    inst_count = 1
    image_id = 1
    with open(save_path, 'w') as fw:
        for image_path, image_anns in tqdm(per_image_anns.items()):
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            height, width, _ = img.shape

            single_image = {}
            single_image['file_name'] = image_path.name
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            for obj in image_anns:
                box = obj["box"]
                class_name = obj["class_name"]

                single_obj = {}
                single_obj['category_id'] = class_map[class_name]

                single_obj['segmentation'] = []
                single_obj['iscrowd'] = 0

                xmin, ymin, xmax, ymax = min(box[0::2]), min(box[1::2]), \
                                    max(box[0::2]), max(box[1::2])

                width, height = xmax - xmin, ymax - ymin
                single_obj['area'] = width * height
                single_obj['bbox'] = xmin, ymin, width, height
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
        json.dump(data_dict, fw, indent=4) 


if __name__ == "__main__":
    main()