#!/bin/bash

# VT5 as blackbox
model=vt5
ckpt=rubentito/vt5-base-spdocvqa

# Donut as proxy
proxy=donut
proxy_ckpt=naver-clova-ix/donut-base
proxy_save_name=proxy_donut

dataset=docvqav0
data_root=./data  # change to DATA_ROOT
data_dir="${data_root}/${dataset}"
pilot=300  # 0 if use all data

declmh=decoder.model.decoder.embed_tokens
lastdecblkfc1=decoder.model.decoder.layers.3.fc1
lastdecblkfc2=decoder.model.decoder.layers.3.fc2

ig=blackbox_donut_proxy_donut_ig
fl=blackbox_donut_proxy_donut_fl
fl_lora=blackbox_donut_proxy_donut_fl_lora

fl_alpha=0.001
fl_tau=(12.0 8.0 1.0)
fl_lora_alpha=0.001
fl_lora_tau=(6.0 5.0 4.0)
ig_alpha=(0.001)
ig_tau=(5.0 4.0 3.0 2.0)

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

      echo "============ Blackbox VT5: Train Proxy Donut ($proxy_save_dir)"
      python run_black_box.py \
            --model $model \
            --ckpt $ckpt \
            --proxy $proxy \
            --proxy_ckpt $proxy_ckpt \
            --data_dir $data_dir \
            --pilot $pilot \
            --num_epoch 128 \
            --batch_size 4 \
            --lr 1e-5 \
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
