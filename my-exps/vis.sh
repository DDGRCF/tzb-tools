#! /bin/bash
env=tzb-mmrotate
source /disk0/r/anaconda3/etc/profile.d/conda.sh
conda activate ${env}

ori_dir=$1
ann_dir=$2
save_dir=$3

python visualizer.py ${ori_dir} --ann-dir ${ann_dir} --save-dir ${save_dir}