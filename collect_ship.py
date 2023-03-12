import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


def main():
    type="ship"
    root = "/disk0/dataset/TianzhiBk/ship/train"
    dst = "/disk0/dataset/Tianzhi/ship/tmp"
    img_dir = os.path.join(root, "images")        
    label_dir = os.path.join(root, "labelTxt")

    dst_img_dir = os.path.join(dst, "images")
    dst_label_dir = os.path.join(dst, "labels")
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)

    prefix = "tzb"
    for ann_file in tqdm(glob(os.path.join(label_dir, "*.txt"))):
        filename = os.path.splitext(os.path.basename(ann_file))[0]
        img_file = os.path.join(img_dir, filename + ".png")
        assert os.path.exists(img_file), "img file don't exist"

        dst_ann_file = os.path.join(dst_label_dir, prefix + "-" + filename + ".txt")
        dst_img_file = os.path.join(dst_img_dir, prefix + "-" + filename + ".tif")

        # 保存图片
        img = cv2.imread(img_file)
        cv2.imwrite(dst_img_file, img)

        with open(ann_file, "r") as fr:
            lines = fr.readlines()
        
        with open(dst_ann_file, "w") as fw:
            fw.write("imagesource:GoogleEarth\n")
            fw.write("gsd:null\n")
            for line in lines:
                line = line.strip()
                line = line.split(",")
                line = line[1:]
                box = np.asarray(list(map(float, line)), dtype=np.int32)
                rect = cv2.minAreaRect(box.reshape(-1, 2))
                box = cv2.boxPoints(rect).reshape(-1).tolist()
                line = list(map(str, box))
                line.append(type)
                line.append('0')
                line = " ".join(line)  + "\n"
                fw.write(line)


if __name__ == "__main__":
    main()