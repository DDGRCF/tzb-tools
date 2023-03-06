import argparse
import cv2
import os 
import os.path as osp
import numpy as np
import random 
from tqdm import tqdm
import torch
from multiprocessing.pool import Pool
from functools import partial
from mmrotate.structures.bbox import rbbox_overlaps, qbox2rbox

def make_folder(dir):
    """ create a folder with dir 
    
    Args:
        dir (str): want to be created   
    """
    if osp.exists(dir):
        print("文件夹已存在，是否覆盖：[y/n]")
        if input.lower() == 'y':
            os.makedirs(dir, exist_ok=True)
        else:
            exit(0)
    else:
        os.makedirs(dir)

def get_args():
    parser = argparse.ArgumentParser('copy some classes to another dataset')
    parser.add_argument('src_dir', type=str, help='ori_dirs')
    parser.add_argument('dst_dir', type=str, help='dst_dirs')
    parser.add_argument('--src_imgs_fname', type=str, help='folder name for imgs', default='images')
    parser.add_argument('--src_anns_fname', type=str, help='folder name for anns', default='labels')
    parser.add_argument('--dst_imgs_fname', type=str, help='folder name for imgs', default='images')
    parser.add_argument('--dst_anns_fname', type=str, help='folder name for anns', default='annfiles')
    parser.add_argument('--classes', nargs='+', type=str, default='ship')
    return parser.parse_args()


def _single_collect(file):
    """get object infos of single annotation file

    Args:
        file(str): dir of file 
    """
    rbboxs = []
    rects = []
    with open(file, 'r') as f:
        for line in f.readlines():
            l = line.split(' ')
            if len(l) < 8:
                continue
            else:
                poly = np.array(list(map(float,l[:8])),dtype=int).reshape((4,2))
                # obb and rbbox is the same box with different format
                obb = cv2.minAreaRect(poly)    # x,y,w,h,theta
                w_h = [i*1.2 for i in list(obb[1])]
                obb = list(obb)
                obb[1] = w_h
                obb = tuple(obb)
                rbbox = cv2.boxPoints(obb)   # poly (x1,y1,x2,y2,x3,y3,x4,y4)
                rect = cv2.boundingRect(rbbox) # (x_lt,y_lt,w,h)
                rbboxs.append(rbbox)
                rects.append(rect)
    return rbboxs, rects


def collect_objects(file, src_dir, src_imgs_fname, src_anns_fname):
    """collect all objects in this file, get a h x w list from img

    Args:
        file(str): file_name of annotation
    return:
        obj_collect(list): composed with (obj_img, obj_poly)
    """
    i = 0
    obj_collect = []
    src_imgs_dir = osp.join(src_dir, src_imgs_fname)
    src_anns_dir = osp.join(src_dir, src_anns_fname)
    
    img_name = file.split(".")[0] + ".tif"
    file_dir = osp.join(src_anns_dir, file)
    img = cv2.imread(osp.join(src_imgs_dir, img_name))
    
    rbboxs, rects = _single_collect(file_dir)
    assert len(rbboxs) == len(rects)

    for index in range(len(rbboxs)):  
        i += 1
        rbbox = rbboxs[index]
        rect = rects[index]
        x_lt, y_lt, x_rb, y_rb = rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3] 
        if x_lt < 0 or x_rb < 0 or y_lt < 0 or y_rb < 0:
            continue
        mask = np.zeros(img.shape[:2])
        cv2.fillPoly(mask, [rbbox.astype(np.int32)], 1)
        
        obj_rect = img[y_lt:y_rb+1, x_lt:x_rb+1, :]
        obj_mask = mask[y_lt:y_rb+1, x_lt:x_rb+1] 
        
        obj_img = np.expand_dims(obj_mask, 2).repeat(3, axis=2)  * obj_rect
        h, w = obj_img.shape[:2]
        ksize_w = (w//15) if (w//15) % 2 == 1 else (w//15) + 1
        ksize_h = (h//15) if (h//15) % 2 == 1 else (h//15) + 1
        obj_blur = cv2.GaussianBlur(obj_img, (ksize_h, ksize_w), 0)
        relative_coord = rbbox
        relative_coord[:, 0] = relative_coord[:, 0] - x_lt
        relative_coord[:, 1] = relative_coord[:, 1] - y_lt
        obj_info = (obj_blur, relative_coord)
        obj_collect.append(obj_info)
        
        # cv2.imwrite(osp.join(src_dir, "obj_vis", "{}_{}.png".format(file.split(".")[0], i)), obj_blur)
    
    return obj_collect

def _write_ann(poly, ann_dir, h, w):
    """write a poly annotation to ann file

    Args:
        poly(np.array): shape(4,2)
        ann_dir(str): dir of proposed annotation file 
    """
    poly[:, 0] = poly[:, 0] + w
    poly[:, 1] = poly[:, 1] + h
    poly = poly.reshape(1,-1).squeeze()
    line = ""
    for i in poly:
        line += str(i)+" " 
    line += "ship 2\n"
    with open(ann_dir, 'a+') as f:
        f.write(line)
    return poly
    

def copy_paste(obj_collect, dst_dir, dst_imgs_fname, dst_anns_fname):
    """select some of obj_collect to paste dst imgs and anns

    Args:
        obj_collect(list): composed with some list, this list is composed (obj_img, obj_poly)
    """
    dst_imgs_dir = osp.join(dst_dir, dst_imgs_fname)
    dst_anns_dir = osp.join(dst_dir, dst_anns_fname)
    all_obj = sum(obj_collect, [])
    for file in tqdm(os.listdir(dst_imgs_dir)):
        img = cv2.imread(osp.join(dst_imgs_dir, file))
        h_img, w_img = img.shape[:2]
        select_num_obj = np.random.randint(4,11)
        select_obj = random.choices(population=all_obj, k=select_num_obj)
        ann = file.split('.')[0] + '.txt'
        ann_dir = osp.join(dst_anns_dir, ann)
        rbboxs, _ = _single_collect(ann_dir)
        rbboxs = np.array(rbboxs).reshape(-1, 8)
        for obj in select_obj:
            h_obj, w_obj = obj[0].shape[:2]
            proposed_h = random.randint(0, h_img-h_obj) 
            proposed_w = random.randint(0, w_img-w_obj)
            if len(rbboxs) == 0:
                img[proposed_h:proposed_h+h_obj, proposed_w:proposed_w+w_obj, :] = obj[0]
                poly = _write_ann(obj[1], ann_dir, proposed_h, proposed_w)
                rbboxs = np.vstack([rbboxs, poly])
            else:    
                proposed_obj = torch.tensor([proposed_w + w_obj/2, proposed_h + h_obj/2, w_obj, h_obj, 0]).reshape(-1, 5)
                rotated_bboxs = qbox2rbox(torch.tensor(rbboxs, dtype=int))   # rotated_bboxs shape is (..., 5) (xc, yc, w, h, t)
                iof = rbbox_overlaps(proposed_obj, rotated_bboxs.float(), mode='iof').squeeze()

                while len(torch.nonzero(iof > 0.2)) != 0: 
                    proposed_h = random.randint(0, h_img-h_obj) 
                    proposed_w = random.randint(0, w_img-w_obj)
                    proposed_obj = torch.tensor([proposed_w + w_obj/2, proposed_h + h_obj/2, w_obj, h_obj, 0]).reshape(-1, 5)
                    iof = rbbox_overlaps(proposed_obj, rotated_bboxs.float(), mode='iof').squeeze()
                img[proposed_h:proposed_h+h_obj, proposed_w:proposed_w+w_obj, :] = obj[0]
                poly = _write_ann(obj[1], ann_dir, proposed_h, proposed_w)
                rbboxs = np.vstack([rbboxs, poly])





def main():
    args = get_args()
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    src_imgs_fname = args.src_imgs_fname
    src_anns_fname = args.src_anns_fname
    dst_imgs_fname = args.dst_imgs_fname
    dst_anns_fname = args.dst_anns_fname
    
    # make_folder(osp.join(src_dir, "obj_vis"))
    # collect all object and return mask img with obj
    src_anns_dir = osp.join(src_dir, src_anns_fname)
    fnames = os.listdir(src_anns_dir)
    func = partial(collect_objects, src_dir=src_dir, src_imgs_fname=src_imgs_fname, src_anns_fname=src_anns_fname)
    pool = Pool(10)
    obj_collect = pool.map(func, fnames)
    
    # random choose obj to paste to dst img and update dst_anns
    copy_paste(obj_collect, dst_dir, dst_imgs_fname, dst_anns_fname)

if __name__ == '__main__':
    main()
