import argparse
import os
import xml.etree.ElementTree as ET
from glob import glob

import cv2
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser("各种数据格式转dota")
    parser.add_argument("type", type=str, 
        choices=type_choices, default="dior-r")
    parser.add_argument("ori_dir", type=str, default=None)
    parser.add_argument("save_dir", type=str, default=None)
    parser.add_argument("--classes", nargs="+", default=["airplane"])
    parser.add_argument("--with-prefix", action="store_true")
    return parser.parse_args()

def get_tree_val(doms, root, deal_func):
    for child in root:
        if len(doms):  
            if child.tag == doms[0]:
                get_tree_val(doms[1:] if len(doms) > 0 else [], child, deal_func)
        else:
            deal_func(child)

def dior_r_func(ori_dir: str, save_dir: str, classes: list, type: str):
    assert os.path.exists(ori_dir), f"{ori_dir} don't exist"
    # assert not os.path.exists(save_dir), f"{save_dir} already exist" 
    save_img_dir = os.path.join(save_dir, "images")
    save_label_dir = os.path.join(save_dir, "labels")
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_label_dir, exist_ok=True)

    ann_dir = os.path.join(ori_dir, "Annotations/obb")
    img_dir = os.path.join(ori_dir, "JPEGImages")
    
    for img_file in tqdm(glob(os.path.join(img_dir, "*.jpg"))):
        file_name = os.path.splitext(os.path.basename(img_file))[0]
        ann_file = os.path.join(ann_dir, file_name + ".xml")
        save_ann_file = os.path.join(save_label_dir, type + "-" + file_name + ".txt")
        save_img_file = os.path.join(save_img_dir, type + "-" + file_name + ".tif")
        assert os.path.exists(ann_file), "ann file exist"
        tree = ET.parse(ann_file)
        object = dict()

        def deal_func(root):
            class_name = ""
            if root.tag == "object":
                for child in root:
                    if child.tag == "name":
                        class_name = child.text
                        if class_name not in classes:
                            continue
                        if class_name not in object:
                            object[class_name] = []
                    if class_name not in classes:
                        continue
                    if child.tag == "robndbox":
                        box = []
                        for value in child:
                            box.append(float(value.text))
                        object[class_name].append(box)

        get_tree_val([], tree.getroot(), deal_func)
        if len(object):
            with open(save_ann_file, "w") as f:
                f.write("imagesource:null\n")
                f.write("gsd:null\n")
                for class_name, boxes in object.items():
                    for box in boxes:
                        box = list(map(str, box))
                        if class_name == "airplane":
                            class_name = "plane"
                        box.append(class_name.lower())
                        box.append("0")
                        box = " ".join(box)  + "\n"
                        f.write(box)
            img = cv2.imread(img_file)
            cv2.imwrite(save_img_file, img)

def hrsc_func(ori_dir: str, save_dir: str, classes: list, type: str):
    pass

def dota_func(ori_dir: str, save_dir: str, classes: list, type: str):
    assert os.path.exists(ori_dir), f"{ori_dir} don't exist"
    save_img_dir = os.path.join(save_dir, "images")
    save_label_dir = os.path.join(save_dir, "labels")
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_label_dir, exist_ok=True)

    ann_dir = os.path.join(ori_dir, "labelTxt-v2.0")
    img_dir = os.path.join(ori_dir, "images")
    
    for img_file in tqdm(glob(os.path.join(img_dir, "*.png"))):
        file_name = os.path.splitext(os.path.basename(img_file))[0]
        ann_file = os.path.join(ann_dir, file_name + ".txt")
        save_ann_file = os.path.join(save_label_dir, type + "-" + file_name + ".txt")
        save_img_file = os.path.join(save_img_dir, type + "-" + file_name + ".tif")
        assert os.path.exists(ann_file), "ann file exist"

        new_lines = []
        with open(ann_file, "r") as fr:
            lines = fr.readlines()
            for line in lines:
                if line.startswith("imagesource") or line.startswith("gsd"):
                    continue
                line = line.strip()
                line = line.split(" ")
                class_name = line[-2]
                if class_name not in classes:
                    continue
                new_lines.append(" ".join(line) + "\n")

        if len(new_lines):
            with open(save_ann_file, "w") as fw:
                fw.write("imagesource:GoogleEarth\n")
                fw.write("gsd:null\n")
                for line in new_lines:
                    fw.write(line)
            
            img = cv2.imread(img_file)
            cv2.imwrite(save_img_file, img)
                    
def fair1m_func(ori_dir: str, save_dir: str, classes:list, type: str):
    pass
   
type_choices = ["dior-r", "hrsc", "dota", "fair1m"]
type_funcs = [dior_r_func, hrsc_func, dota_func, fair1m_func]
assert len(type_choices) == len(type_funcs)
type_func_map = {choice: func for choice, func in 
                    zip(type_choices, type_funcs)}


def main():
    args = get_args()
    type = args.type
    ori_dir = args.ori_dir
    save_dir = args.save_dir
    with_prefix = args.with_prefix
    classes = args.classes
    assert ori_dir, "ori dir can't be empty"
    assert save_dir, "save dir can't be empty"
    type_func = type_func_map[type]
    type_func(ori_dir, save_dir, classes, type if with_prefix else "")

if __name__ == "__main__":
    main()