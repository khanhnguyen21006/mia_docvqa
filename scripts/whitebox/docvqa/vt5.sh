#!/bin/bash

# VT5 as whitebox
declmh=language_backbone.lm_head
lastdecblkfc1=language_backbone.decoder.block.11.layer.2.DenseReluDense.wi
lastdecblkfc2=language_backbone.decoder.block.11.layer.2.DenseReluDense.wo

model=vt5
ckpt=rubentito/vt5-base-spdocvqa

dataset=docvqav0
data_root=./data  # /path/to/DATA_ROOT
data_dir="${data_root}/${dataset}"
pilot=300  # 0 if use all data

bl=vt5_docvqa_bl
fl=vt5_docvqa_fl
fl_lora=vt5_docvqav0_fl_lora
ig=vt5_docvqa_ig

fl_alpha=0.001
fl_tau=(1e-4 1e-5 1e-6)
fl_lora_alpha=0.001
fl_lora_tau=(1e-4 1e-5 1e-6)
ig_alpha=(1.0 0.001)
ig_tau=(1e-5 5e-6)

rand_seed=($((1 + RANDOM % 2000)))
echo "Random seed: $rand_seed"

for seed in ${rand_seed[@]}
do
      if [ "$pilot" -gt 0 ]; then
            python -m utils.create_pilot \
                  --data_dir $data_root \
                  --dataset $dataset \
                  --pilot $pilot \
                  --seed $seed
      fi

      echo "============ Baseline:"
      python run_white_box.py \
                  --attack bl \
                  --model $model \
                  --ckpt $ckpt \
                  --data_dir $data_dir \
                  --pilot $pilot \
                  --expt $bl \
                  --seed $seed

      echo "============ Method: FL"
      for tau in ${fl_tau[@]}
      do
            python run_white_box.py \
                  --attack fl \
                  --layer $lastdecblkfc2 \
                  --model $model \
                  --ckpt $ckpt \
                  --data_dir $data_dir \
                  --pilot $pilot \
                  --step_size $fl_alpha \
                  --threshold $tau  \
                  --max_step 200 \
                  --expt $fl \
                  --seed $seed
      done

      echo "============ Method: FL_lora"
      for tau in ${fl_lora_tau[@]}
      do
            python run_white_box.py \
                  --attack fl \
                  --layer $lastdecblkfc2 \
                  --model $model \
                  --ckpt $ckpt \
                  --data_dir $data_dir \
                  --pilot $pilot \
                  --step_size $fl_lora_alpha \
                  --threshold $tau \
                  --max_step 200 \
                  --lora \
                  --expt $fl_lora \
                  --seed $seed
      done

      echo "============ Method: IG"
      for alpha in ${ig_alpha[@]}
      do
            for tau in ${ig_tau[@]}
            do
                  python run_white_box.py \
                        --attack ig \
                        --model $model \
                        --ckpt $ckpt \
                        --data_dir $data_dir \
                        --pilot $pilot \
                        --step_size $alpha \
                        --threshold $tau \
                        --max_step 200 \
                        --expt $ig \
                        --seed $seed
            done
      done
done
