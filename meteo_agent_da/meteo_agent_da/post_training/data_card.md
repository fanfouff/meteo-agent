# MeteoAgent-DA Post-training Data Card

## Data Sources

The post-training data is derived from MeteoAgent-DA runs over PASBench-DA
tasks. Each accepted sample should have:

- a user task from PASBench-DA or an equivalent satellite data-assimilation
  workflow;
- a `trace.jsonl` file containing plan, tool call, reflection, and report
  events;
- a `report.json` file containing status, tool results, artifacts, and next
  steps;
- verifier outputs showing the rollout is executable, grounded, and honest
  about dry-run versus real execution.

## Training Splits

- SFT: successful, tool-grounded trajectories after `filter_traces.py`.
- DPO: preference pairs over tool-grounded vs text-only, verified vs
  unverified, recovery vs failure, and concise reproducible vs hallucinated
  reports.
- GRPO-ready reward data: rollout records paired with an interpretable reward
  breakdown from PASBench or report verifiers.

## Exclusion Rules

Reject rollouts with failed tools, malformed traces, missing evidence,
hallucinated artifact paths, unsafe commands, ungrounded metric claims, false
training claims under dry-run, duplicate loops, or traces too long to audit.

## Intended Use

This data is for post-training a tool-augmented research agent, not for
claiming new meteorological skill by itself. Any scientific result must still
be validated against the original PASNet-DA experiments and held-out
evaluation artifacts.
