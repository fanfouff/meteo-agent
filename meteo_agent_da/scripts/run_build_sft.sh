#!/usr/bin/env bash
set -euo pipefail

python -m meteo_agent_da.post_training.build_sft_data \
  --successful-reports-file runs/pasbench_50/filtered_reports.txt \
  --output runs/pasbench_50/post_training_sft.jsonl

python -m meteo_agent_da.post_training.verify_training_data \
  --input runs/pasbench_50/post_training_sft.jsonl \
  --format sft
