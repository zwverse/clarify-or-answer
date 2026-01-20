CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
swift infer \
  --model OpenGVLab/InternVL3-2B-Instruct \
  --adapters /home/clarify-ambiguity/output/internvl3_2b_router_sft/v0-20251228-094231/checkpoint-46 \
  --val_dataset dataset/swift_router_clearvqa_test_dataset.jsonl \
  --merge_lora true