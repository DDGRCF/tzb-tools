#! /bin/bash
env=tzb-mmrotate
source /disk0/r/anaconda3/etc/profile.d/conda.sh
conda activate ${env}
src_dir=$1
dst_dir=$2

python copy_paste.py ${src_dir} ${dst_dir}