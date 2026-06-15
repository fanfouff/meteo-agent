#!/usr/bin/env bash
set -euo pipefail

python -m meteo_agent_da.post_training.train_dpo_lora \
  --preference-file runs/pasbench_50/post_training_preferences.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir runs/post_training/dpo_lora \
  --max-steps 10
