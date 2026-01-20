CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
swift infer \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --adapters  /home/clarify-ambiguity/output/qwen2.5_3b_clarification_sft/v0-20251225-230221/checkpoint-33 \
  --val_dataset dataset/swift_vizwiz_test_dataset.jsonl \
  --merge_lora true