#!/bin/bash

# VT5 as blackbox
model=vt5
ckpt=/path/to/pfl/checkpoint

# VT5 as proxy
proxy=vt5
proxy_ckpt=/path/to/checkpoint
proxy_save_name=proxy_vt5

dataset=pfl
data_root=./data  # change to DATA_ROOT
data_dir="${data_root}/${dataset}"
pilot=300  # 0 if use all data

declmh=language_backbone.lm_head
lastdecblkfc1=language_backbone.decoder.block.11.layer.2.DenseReluDense.wi
lastdecblkfc2=language_backbone.decoder.block.11.layer.2.DenseReluDense.wo

ig=blackbox_vt5_proxy_vt5_ig
fl=blackbox_vt5_proxy_vt5_fl
fl_lora=blackbox_vt5_proxy_vt5_fl_lora

fl_alpha=0.001
fl_tau=(1e-4 1e-5 1e-6)
fl_lora_alpha=0.001
fl_lora_tau=(1e-4 1e-5 1e-6)
ig_alpha=(1.0 0.001)
ig_tau=(1e-5 5e-6)

rand_seed=($((1 + RANDOM % 1000)))
echo "Random seed: $rand_seed"

for seed in ${rand_seed[@]}
do
      if [ "$pilot" -gt 0 ]; then
            python -m utils.create_pilot \
                  --data_dir $data_root \
                  --dataset $dataset \
                  --pilot $pilot \
                  --seed $seed
            proxy_save_dir="./save/blackbox/${model}/${dataset}/pilot/seed${seed}/${proxy_save_name}_checkpoints"
      else
            proxy_save_dir="./save/blackbox/${model}/${dataset}/${proxy_save_name}_checkpoints"
      fi

      echo "============ Blackbox VT5: Train Proxy VT5 ($proxy_save_dir)"
      python run_black_box.py \
            --model $model \
            --ckpt $ckpt \
            --proxy $proxy \
            --proxy_ckpt $proxy_ckpt \
            --data_dir $data_dir \
            --pilot $pilot \
            --num_epoch 512 \
            --batch_size 16 \
            --lr 5e-4 \
            --save_dir $proxy_save_dir \
            --save_name $proxy_save_name \
            --seed $seed

      echo "============ Proxy MIA: FL"
      for tau in ${fl_tau[@]}
      do
            python run_white_box.py \
                  --attack fl \
                  --layer $lastdecblkfc2 \
                  --model $proxy \
                  --ckpt "${proxy_save_dir}/last.ckpt/" \
                  --data_dir $data_dir \
                  --pilot $pilot \
                  --step_size $fl_alpha \
                  --threshold $tau  \
                  --max_step 200 \
                  --expt $fl \
                  --seed $seed
      done

      echo "============ Proxy MIA: FL_lora"
      for tau in ${fl_lora_tau[@]}
      do
            python run_white_box.py \
                  --attack fl \
                  --layer $lastdecblkfc2 \
                  --model $proxy \
                  --ckpt "${proxy_save_dir}/last.ckpt/" \
                  --data_dir $data_dir \
                  --pilot $pilot \
                  --step_size $fl_lora_alpha \
                  --threshold $tau \
                  --max_step 200 \
                  --lora \
                  --expt $fl_lora \
                  --seed $seed
      done

      echo "============ Proxy MIA: IG"
      for alpha in ${ig_alpha[@]}
      do
            for tau in ${ig_tau[@]}
            do
                  python run_white_box.py \
                        --attack ig \
                        --model $proxy \
                        --ckpt "${proxy_save_dir}/last.ckpt/" \
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
