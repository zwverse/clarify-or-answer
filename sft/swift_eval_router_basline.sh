CUDA_VISIBLE_DEVICES=1 \
MAX_PIXELS=1003520 \
swift infer \
  --model OpenGVLab/InternVL3-8B-Instruct \
  --val_dataset dataset/swift_router_clearvqa_test_dataset.jsonl