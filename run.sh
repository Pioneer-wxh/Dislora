#!/bin/bash

# 定义要尝试的学习率、lora_alpha 和 lora_r 值
learning_rates=(3e-4)
lora_alphas=(24 )
lora_ranks=(16)  # 添加不同的 LoRA rank 值

# 基础模型和数据路径等固定参数
BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
DATA_PATH='/root/autodl-tmp/Dislora/commonsense_170k.json'
# 从 DATA_PATH 提取数据集名称
DATASET_NAME=$(basename "${DATA_PATH}" .json)

BATCH_SIZE=12
MICRO_BATCH_SIZE=4
NUM_EPOCHS=1
CUTOFF_LEN=256
VAL_SET_SIZE=120
USE_GRADIENT_CHECKPOINTING=False
ADAPTER_NAME=mylora
TARGET_MODULES='[q_proj,v_proj,k_proj,o_proj]'
WANDB_PROJECT='llm-adapters-math'
WANDB_WATCH='gradients'
WANDB_LOG_MODEL='false'

# 遍历所有学习率、lora_alpha 和 lora_r 的组合
for lr in "${learning_rates[@]}"; do
  for alpha in "${lora_alphas[@]}"; do
    for rank in "${lora_ranks[@]}"; do
      # 为每个组合生成独特的输出目录和 W&B 运行名称，包含数据集名称
      output_dir="./trained_models/qwen-lora-${DATASET_NAME}-lr${lr}-alpha${alpha}-rank${rank}-epoch3"
      wandb_run_name="qwen-lora-${DATASET_NAME}-lr${lr}-alpha${alpha}-rank${rank}"
      if [ -d "$output_dir" ]; then
        echo "跳过已存在的训练组合: lr=${lr}, alpha=${alpha}, rank=${rank}"
        continue
      fi
      echo "Running with dataset=${DATASET_NAME}, lr=${lr}, lora_alpha=${alpha}, lora_r=${rank}"
      echo "Output directory: ${output_dir}"
      echo "Wandb run name: ${wandb_run_name}"

      deepspeed --num_gpus=3 finetune_MyLoRA.py \
        --base_model "${BASE_MODEL}" \
        --data_path "${DATA_PATH}" \
        --output_dir "${output_dir}" \
        --batch_size ${BATCH_SIZE} \
        --micro_batch_size ${MICRO_BATCH_SIZE} \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate ${lr} \
        --cutoff_len ${CUTOFF_LEN} \
        --val_set_size ${VAL_SET_SIZE} \
        --use_gradient_checkpointing ${USE_GRADIENT_CHECKPOINTING} \
        --adapter_name ${ADAPTER_NAME} \
        --lora_target_modules "${TARGET_MODULES}" \
        --lora_r ${rank} \
        --lora_alpha ${alpha} \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_run_name "${wandb_run_name}" \
        --wandb_watch "${WANDB_WATCH}" \
        --wandb_log_model "${WANDB_LOG_MODEL}"

      echo "Finished run with dataset=${DATASET_NAME}, lr=${lr}, lora_alpha=${alpha}, lora_r=${rank}"
      echo "-------------------------------------------------"
    done
  done
done

echo "All runs completed."