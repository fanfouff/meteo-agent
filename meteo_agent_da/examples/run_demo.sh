#!/usr/bin/env bash
set -euo pipefail

python -m meteo_agent_da.cli \
  --task "Compare PASNet and Swin-UNet on the 50pct split and generate a paper table." \
  --dry-run
