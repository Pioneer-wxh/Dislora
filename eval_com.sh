#!/bin/bash

# 定义要测试的数据集列表
datasets=("openbookqa")

# 循环遍历数据集列表
for dataset_name in "${datasets[@]}"
do
  echo "======================================="
  echo "Running evaluation for dataset: $dataset_name"
  echo "======================================="

  CUDA_VISIBLE_DEVICES=$2 python commonsense_evaluate.py \
      --model Qwen2.5-7B-Instruct \
      --adapter mylora \
      --dataset $dataset_name \
      --base_model 'Qwen/Qwen2.5-7B-Instruct' \
      --batch_size 10 \
      --lora_weights $1|tee -a $1/${dataset_name}.txt

  echo "Finished evaluation for dataset: $dataset_name"
  echo ""
done

echo "All evaluations finished."