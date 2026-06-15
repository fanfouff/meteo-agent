#!/usr/bin/env bash
set -euo pipefail

python -m meteo_agent_da.post_training.filter_traces \
  --successful-reports-file runs/pasbench_50/successful_reports.txt \
  --allow-no-artifacts \
  --dry-run-expected \
  --output runs/pasbench_50/filtered_reports.txt
