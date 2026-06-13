# MeteoAgent-DA

MeteoAgent-DA is a vertical-domain research agent for satellite data assimilation workflows.
It focuses on the project line you already own: **FY-3F MWTS-III + ERA5 + PASNet-DA temperature-profile correction**.

The project is intentionally narrower than a general weather QA agent. Its job is to help a researcher:

- inspect FY-3F/ERA5 `.npz` datasets and split files;
- generate PASNet/Swin-UNet/FuXi-DA training or inference commands;
- run or dry-run controlled experiments;
- summarize RMSE/MAE/vertical-profile diagnostics;
- create paper-ready tables, captions, and result paragraphs;
- collect successful tool trajectories for SFT/DPO/GRPO-style post-training.

## Design

```text
planner -> tool caller -> executor -> reflector -> reporter
```

The first version is a lightweight local harness rather than a heavy multi-agent platform. It borrows the useful engineering ideas from pico-style coding agents:

- explicit tool registry;
- structured tool inputs and outputs;
- bounded command execution;
- run traces and artifacts;
- short working memory;
- benchmark tasks with verifiable outputs.

## Repository Layout

```text
meteo_agent_da/
  agent/              # reflective runtime, schemas, memory
  tools/              # PASNet-DA domain tools
  sandbox/            # controlled command executor
  bench/              # PASBench-DA samples and evaluator helpers
  post_training/      # trace-to-SFT/preference conversion utilities
configs/
docs/
examples/
tests/
```

## Quick Start

From this directory:

```bash
python -m meteo_agent_da.cli \
  --task "Compare PASNet and Swin-UNet on the 50pct split and generate a paper table" \
  --dry-run
```

Run the smoke test:

```bash
python -m unittest discover -s tests
```

## Current Scope

The scaffold defaults to **dry-run** for expensive training/evaluation commands. Turning dry-run off should be done only after checking paths, GPU allocation, and output directories.

The default PASNet project root is:

```text
/home/lrx/Unet/satellite_assimilation_v2
```

## Paper Direction

Working title:

> MeteoAgent-DA: A Tool-Augmented and Post-Trained LLM Agent for Satellite Data Assimilation Research

Core contributions:

1. an agentic environment for satellite data-assimilation research;
2. PASBench-DA for tool-use, experiment-planning, result-reasoning, and paper-writing evaluation;
3. executable trajectory post-training for domain research agents;
4. validation on FY-3F MWTS-III / ERA5 / PASNet-DA workflows.
