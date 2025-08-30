#!/bin/bash

# 定义要测试的数据集列表
datasets=('AddSub' 'MultiArith' 'SingleEq' 'gsm8k' 'AQuA' 'SVAMP')

# 循环遍历数据集列表
for dataset_name in "${datasets[@]}"
do
  echo "======================================="
  echo "Running evaluation for dataset: $dataset_name"
  echo "======================================="

  python evaluate.py \
    --dataset "$dataset_name" \
    --model meta-llama/Meta-Llama-3-8B \
    --adapter LoRA \
    --base_model meta-llama/Meta-Llama-3-8B \
    --lora_weights "./trained_models/qwen-lora-math_50k-lr3e-4-alpha24" \

  echo "Finished evaluation for dataset: $dataset_name"
  echo ""
done

echo "All evaluations finished."