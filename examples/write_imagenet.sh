#!/bin/bash

write_dataset () {
	write_path=/mnt/cfs/home/engstrom/store/ffcv/${1}_${2}_${3}_${4}.ffcv
	echo $write_path
	python ../scripts/write_image_datasets.py \
		--cfg.dataset=imagenet \
		--cfg.split=${1} \
		--cfg.data_dir=/mnt/cfs/datasets/pytorch_imagenet/${1} \
		--cfg.write_path=$write_path \
		--cfg.max_resolution=${2} \
		--cfg.write_mode=proportion \
		--cfg.compress_probability=${3} \
		--cfg.jpeg_quality=$4
}

# Threshold: 600 * 600 * 3 bytes
# image_size frac_jpeg 100

# write_dataset train $1 $2 $3
write_dataset val $1 $2 $3
