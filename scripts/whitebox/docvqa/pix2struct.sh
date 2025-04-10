#!/bin/bash

# pix2struct as whitebox
declmh=decoder.embed_tokens
lastdecblkfc1=decoder.layer.11.mlp.DenseReluDense.wi_0
lastdecblkfc2=decoder.layer.11.mlp.DenseReluDense.wo

model=pix2struct
ckpt=google/pix2struct-docvqa-base

dataset=docvqa
data_root=./data  # change to DATA_ROOT
data_dir="${data_root}/${dataset}"
pilot=300  # 0 if use all data

bl=pix2struct_docvqa_bl
fl=pix2struct_docvqa_fl
fl_lora=pix2struct_docvqa_fl_lora
ig=pix2struct_docvqa_ig

fl_alpha=0.001
fl_tau=(8e-4 5e-4 1e-4)
fl_lora_alpha=0.001
fl_lora_tau=(9e-5 8e-5 5e-5)
ig_alpha=(0.001)
ig_tau=(0.001 5e-4)

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
