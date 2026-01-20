#!/bin/bash

setting='qwen2.5_7b_vizwiz_icl_eval'
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT=$setting

# Load config variables
source scripts/train_base_config_clarify.sh

# --config_file accelerate_configs/deepspeed_zero2.yaml \

# Run the training script with DeepSpeed
python -m accelerate.commands.launch \
    --main_process_port 20092 \
    --config_file accelerate_configs/deepspeed_zero2.yaml \
    grpo-gr/GRPO_GR.py \
    --train_data_path ./GRIT_data/clarification/jsonl/clarification_train.jsonl \
    --train_image_folder_path ./GRIT_data/clarification \
    --eval_data_path ./GRIT_data/vizwiz/jsonl/test_public.jsonl \
    --eval_image_folder_path ./GRIT_data/vizwiz \
    --setting $setting \
    --max_turns 1 \
    --output_dir output/$setting \
    --hub_model_id $setting \
    $COMMON_ARGS \
    --eval_steps 50 \
    --save_steps 20 \
    --num_train_epochs 0 \
    --lr_scheduler_type cosine \
    --per_device_eval_batch_size 2 \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --eval_on_start True \
    --max_prompt_length 1000