#!/bin/bash

# 定义要测试的数据集列表
datasets=("boolq" "piqa" "social_i_qa" "hellaswag" "winogrande" "ARC-Challenge" "ARC-Easy" "openbookqa")

# 循环遍历数据集列表
for dataset_name in "${datasets[@]}"
do
  echo "======================================="
  echo "Running evaluation for dataset: $dataset_name"
  echo "======================================="

  python commonsense_evaluate.py \
    --dataset "$dataset_name" \
    --model Qwen/Qwen2.5-7B-Instruct \
    --adapter mylora \
    --base_model "Qwen/Qwen2.5-7B-Instruct" \
    --lora_weights "./trained_models/qwen-lora-commonsense_15k-lr3e-4-alpha24" \
    --batch_size 16

  echo "Finished evaluation for dataset: $dataset_name"
  echo ""
done

echo "All evaluations finished."