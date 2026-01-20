# config.sh

COMMON_ARGS="
    --project_root_path $(pwd) \
    --python_path_for_dino $(which python) \
    --dataset_name rr \
    --learning_rate 2e-6 \
    --num_generations 2 \
    --per_device_eval_batch_size 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --torch_empty_cache_steps 100 \
    --max_completion_length 768 \
    --max_prompt_length 1000 \
    --eval_strategy steps \
    --save_strategy steps \
    --num_train_epochs 1 \
    --save_total_limit 5 \
    --report_to wandb \
    --logging_steps 1 \
    --beta 0.01 \
    --bf16 True \
    --bf16_full_eval True \
    --log_completions True\
    --push_to_hub True 
"