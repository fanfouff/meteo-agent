#!/usr/bin/env bash
set -euo pipefail

python -m meteo_agent_da.post_training.build_rollout_rewards \
  --scores-json runs/pasbench_50/scores.json \
  --output runs/pasbench_50/rollout_reward.jsonl

python -m meteo_agent_da.post_training.verify_training_data \
  --input runs/pasbench_50/rollout_reward.jsonl \
  --format rollout_reward
