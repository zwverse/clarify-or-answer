CUDA_VISIBLE_DEVICES=1 \
MAX_PIXELS=1003520 \
swift infer \
  --model OpenGVLab/InternVL3-2B-Instruct \
  --adapters /home/clarify-ambiguity/output/internvl3_2b_clarification_sft/v4-20251224-184807/checkpoint-33 \
  --val_dataset dataset/swift_clearvqa_test_dataset.jsonl \
  --merge_lora true