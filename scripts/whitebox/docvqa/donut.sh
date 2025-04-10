#!/bin/bash

# donut as whitebox
declmh=decoder.model.decoder.embed_tokens
lastdecblkfc1=decoder.model.decoder.layers.3.fc1
lastdecblkfc2=decoder.model.decoder.layers.3.fc2

model=donut
ckpt=naver-clova-ix/donut-base-finetuned-docvqa

dataset=docvqa
data_root=/data/users/vkhanh/mia_docvqa/data
data_dir="${data_root}/${dataset}"
pilot=300  # 0 if use all data

bl=donut_docvqa_bl
fl=donut_docvqa_fl
fl_lora=donut_docvqa_fl_lora
ig=donut_docvqa_ig

fl_alpha=0.001
fl_tau=(12.0 8.0 1.0)
fl_lora_alpha=0.001
fl_lora_tau=(6.0 5.0 4.0)
ig_alpha=(0.001)
ig_tau=(5.0 4.0 3.0 2.0)

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
