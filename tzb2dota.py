import os
import argparse
import os.path as osp
from functools import partial
from multiprocessing.pool import Pool


def get_args():
    parser = argparse.ArgumentParser('change tzb format to dota format')
    parser.add_argument('src_dir', type=str, help='ori_dirs')
    parser.add_argument('dst_dir', type=str, help='dst_dirs')
    parser.add_argument('--src_anns_fname', type=str, help='folder name for anns', default='labels')
    parser.add_argument('--dst_anns_fname', type=str, help='folder name for anns', default='labelTxt')
    return parser.parse_args()

def tzb2dota(fname, src_dir, src_anns_fname, dst_dir, dst_anns_fname): 
    src_anns_dir = osp.join(src_dir, src_anns_fname)
    dst_anns_dir = osp.join(dst_dir, dst_anns_fname)
    src_file_dir = osp.join(src_anns_dir, fname)
    dst_file_dir = osp.join(dst_anns_dir, fname)
    with open(src_file_dir, 'r') as f:
        for line in f.readlines():
            l = line.split(' ')
            if len(l) < 8:
                continue
            else:
                cls = l[0]
                poly = l[1:9]
                res = ""
                for i in poly:
                    res += i+" " 
                res += cls+" 2\n"
                with open(dst_file_dir, 'a+') as fw:
                    fw.write(res)         

def main():
    args = get_args()
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    src_anns_fname = args.src_anns_fname
    dst_anns_fname = args.dst_anns_fname
    
    # make_folder(osp.join(src_dir, "obj_vis"))
    # collect all object and return mask img with obj
    src_anns_dir = osp.join(src_dir, src_anns_fname)
    fnames = os.listdir(src_anns_dir)
    func = partial(tzb2dota, src_dir=src_dir, src_anns_fname=src_anns_fname, 
                            dst_dir=dst_dir, dst_anns_fname=dst_anns_fname)
    pool = Pool(10)
    pool.map(func, fnames)
    

if __name__ == '__main__':
    main()