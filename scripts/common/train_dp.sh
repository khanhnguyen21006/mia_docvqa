#!/bin/bash

model=donut
ckpt=naver-clova-ix/donut-base

dataset=docvqav0
data_root=./data  # change to DATA_ROOT
data_dir="${data_root}/${dataset}"

learning_rate=3e-5
num_epoch=10

# DP-SGD: epsilon = 8
batch_size=64
mini_batch_size=2
sensitivity=8
noise_multiplier=0.5767822266

expt_name=f"dp_${model}_${dataset}_eps=8_bzs=${batch_size}_c=${noise_multiplier}_S=${S}"

python train.py \
	--dp \
	--model $model \
	--ckpt $ckpt \
	--data_dir $data_dir \
	--learning_rate $learning_rate \
	--num_epoch $num_epoch \
	--batch_size $batch_size \
    --mini_batch_size $mini_batch_size \
	--sensitivity $sensitivity \
    --noise_multiplier $noise_multiplier \
	--expt $expt_name
