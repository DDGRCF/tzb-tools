import argparse
import os
import os.path as osp
from tqdm import tqdm
from functools import partial
from multiprocessing.pool import Pool

def parse_args():
    parser = argparse.ArgumentParser(description='convert dota format to tzb format')
    parser.add_argument(
        "res_dir",
        type=str,
        default=None,
        help='the dota results')
    parser.add_argument(
        "img_dir",
        type=str,
        default=None,
        help='tzb images dir')
    parser.add_argument(
        "dst_dir",
        type=str,
        default=None,
        help='wishing to write fair dir')
    parser.add_argument('--save_filter', action='store_true')
    args = parser.parse_args()
    return args

def load_dota(src_dir):
    src = os.listdir(src_dir)
    zip_file = osp.basename(src_dir) + '.zip'
    if zip_file in src:
        src.remove(zip_file)
    dict_content = {}
    for cls_txt in tqdm(src, desc="Reading dota results:"):
        res_dir = osp.join(src_dir, cls_txt)
        cls = cls_txt.replace('.txt', '')
        cls = cls.split('1_')[1]
        with open(res_dir, 'r') as f:
            for line in f.readlines():
                l = line.split(' ')
                if len(l) == 0:
                    continue
                img_name = l[0]
                ann = cls + " "
                for i in range(8):
                    ann += l[i+1] + " "
                ann += l[9]
                if img_name not in dict_content.keys():
                    dict_content[img_name] = []

                    dict_content[img_name].append(ann)
                else:
                    dict_content[img_name].append(ann)
    print("the number of images can be detected is {}".format(len(dict_content.keys())))
    return dict_content

def write_tzb(dst_dir, imgs_dir, dict_content, save_filter):
    for img_name in tqdm(dict_content.keys(), desc='Writing for tzb results:'):
            dst_file = osp.join(dst_dir, img_name+'.txt')
            with open(dst_file, 'a+') as f:
                for ann in dict_content[img_name]:
                    f.write(ann)
    if save_filter:
        img_files = os.listdir(imgs_dir)
        for img in img_files:
            img_name = img.split('.')[0]
            if img not in dict_content.keys():     
                dst_file = osp.join(dst_dir, img_name+'.txt')
                with open(dst_file, 'a+') as f:
                    f.write(" ")
        



def main():
    args = parse_args()
    res_dir = args.res_dir
    img_dir =args.img_dir
    dst_dir = args.dst_dir
    save_filter = args.save_filter
    if not osp.exists(dst_dir):
        os.makedirs(dst_dir)
    dict_content = load_dota(res_dir)
    write_tzb(dst_dir, img_dir, dict_content, save_filter)


if __name__ == '__main__':
    main()



