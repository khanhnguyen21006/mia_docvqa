#!/bin/bash

model=udop
ckpt=microsoft/udop-large

dataset=docvqav0
data_root=./data  # change to DATA_ROOT
data_dir="${data_root}/${dataset}"

batch_size=8
learning_rate=5e-5
weight_decay=0.01
num_epoch=10
warmup_steps=1000
max_steps=20000

expt_name=udop_docvqa_200k

# model=donut
# ckpt=naver-clova-ix/donut-base

# dataset=pfl
# data_root=./data  # change to DATA_ROOT
# data_dir="${data_root}/${dataset}"

# batch_size=6
# learning_rate=5e-5
# num_epoch=20
# warmup_steps=10000
# max_steps=800000

# expt_name=donut_pfl_800k

python train.py \
	--model $model \
	--ckpt $ckpt \
	--data_dir $data_dir \
	--learning_rate $learning_rate \
	--batch_size $batch_size \
	--num_epoch $num_epoch \
	--expt $expt_name
