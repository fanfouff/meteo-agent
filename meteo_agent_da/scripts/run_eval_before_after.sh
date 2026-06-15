#!/usr/bin/env bash
set -euo pipefail

python -m meteo_agent_da.bench.run_pasbench \
  --tasks examples/pasbench_da_50.jsonl \
  --method text_only heuristic_tool sft_qwen \
  --attempts 1 \
  --output-dir runs/pasbench_before_after
