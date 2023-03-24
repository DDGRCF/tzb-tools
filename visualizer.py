import argparse
import os
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from tqdm.contrib.concurrent import process_map


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ori_dir", type=str, default=None)
    parser.add_argument("--img-dir", type=str, default="images")
    parser.add_argument("--ann-dir", type=str, default="labels")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--img-suffix", choices=[".png", ".tif", ".jpg"], 
            type=str, default=".png")
    parser.add_argument("--classes", nargs="+", type=str, default=None)
    parser.add_argument("--ignore_empty", action="store_true")
    return parser.parse_args()


def single_visualizer(ann_file, img_dir, save_img_dir,
                      img_suffix=".png", classes=None, ignore_empty=False):
    with open(ann_file, "r") as f:
        lines = f.readlines()

    if ignore_empty and len(lines) == 0:
        return

    img_file = img_dir / (ann_file.stem +  img_suffix)
    save_img_file = save_img_dir / (ann_file.stem + img_suffix)
    assert img_file.exists(), f"{img_file} dont exist"

    img = cv2.imread(str(img_file))
    for line in lines:
        if line.startswith("imagesource") or \
           line.startswith("gsd"):
           continue

        line = line.strip()
        line = line.split(" ")
        class_name = line[-2]
        if classes is not None and class_name not in class_name:
            continue
        box = list(map(float, line[0:8]))
        box = np.asarray(box, dtype=np.int32).reshape(-1, 1, 2)
        ctr_x = box[:, 0, 0].mean()
        ctr_y = box[:, 0, 1].mean()
        cv2.putText(img, class_name, (int(ctr_x), int(ctr_y)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 255), thickness=2)
        cv2.polylines(img, [box], True, [255, 255, 0], thickness=3)
    cv2.imwrite(str(save_img_file), img)
    return 0

def main():
    args = get_args()
    ori_dir = Path(args.ori_dir)
    img_dir = args.img_dir
    ann_dir = args.ann_dir
    save_dir = Path(args.save_dir)
    img_suffix = args.img_suffix
    ignore_empty = args.ignore_empty
    classes = args.classes

    img_dir = ori_dir / img_dir
    ann_dir = ori_dir / ann_dir
    save_img_dir = save_dir / img_dir.stem
    
    if not save_img_dir.exists():
        os.makedirs(save_img_dir)
    else:
        print("是否覆盖源文件:")
        if input().lower() == "yes":
            os.makedirs(save_img_dir, exist_ok=True)
        else:
            exit(0)

    ann_set = list(ann_dir.glob("**/*.txt"))
    deal_func = partial(single_visualizer, 
                        img_dir=img_dir, 
                        save_img_dir=save_img_dir, 
                        img_suffix=img_suffix,
                        classes=classes,
                        ignore_empty=ignore_empty)

    print("Begin visualize the images...")

    res = process_map(deal_func, ann_set, chunksize=max(int(len(ann_set) / 10), 1))
    list(res)


if __name__ == "__main__":
    main()