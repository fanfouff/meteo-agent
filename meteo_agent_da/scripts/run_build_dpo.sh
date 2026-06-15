#!/usr/bin/env bash
set -euo pipefail

python -m meteo_agent_da.post_training.build_preference_data \
  --scores-json runs/pasbench_50/scores.json \
  --chosen-method heuristic_tool \
  --rejected-method text_only \
  --preference-type tool_grounded_vs_text_only \
  --output runs/pasbench_50/post_training_preferences.jsonl

python -m meteo_agent_da.post_training.verify_training_data \
  --input runs/pasbench_50/post_training_preferences.jsonl \
  --format preference
