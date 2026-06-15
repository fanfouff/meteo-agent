#!/usr/bin/env bash
set -euo pipefail

python -m meteo_agent_da.post_training.train_sft_lora \
  --train-file runs/pasbench_50/post_training_sft.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir runs/post_training/sft_lora \
  --max-steps 10
