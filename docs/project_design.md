# MeteoAgent-DA Project Design

## Positioning

MeteoAgent-DA is not a broad weather-science assistant. It is a research workflow agent for offline satellite data assimilation and temperature-profile correction. The design deliberately narrows the task space to the PASNet-DA line:

- FY-3F MWTS-III microwave brightness temperatures;
- ERA5/NWP background temperature fields;
- sparse observation masks;
- PASNet-DA and baseline model experiments;
- RMSE/MAE/vertical-profile/spatial-error diagnostics;
- LaTeX-ready paper artifacts.

This avoids competing directly with general weather-agent frameworks and gives the project a clearer technical identity.

## Four Layers

### 1. Agent Runtime

The runtime follows a reflective loop:

```text
plan -> call tools -> observe -> reflect -> report
```

The first implementation is deterministic and local. A later LLM planner can be plugged in without changing the tool contracts.

### 2. Domain Tools

The domain tool layer wraps the existing PASNet-DA project:

- `data_indexer`: inspect `.npz` counts, split files, stats files, and dataset roots;
- `pasnet_runner`: build controlled training/evaluation commands;
- `evaluator`: summarize metric JSON/CSV/NPY artifacts;
- `plotter`: create or register paper figure commands;
- `paper_writer`: generate LaTeX tables and result paragraphs;
- `sanity_checker`: catch leakage, missing files, invalid split ratios, and unsafe run options.

### 3. Benchmark

`PASBench-DA` should evaluate executable research workflow competence rather than generic knowledge:

- data query;
- experiment planning;
- command generation;
- error repair;
- result interpretation;
- paper writing.

### 4. Post-Training

Successful traces are transformed into:

- SFT samples: user task -> tool trajectory -> final report;
- preference pairs: valid vs invalid experiment plans or reports;
- RL-style rewards: tool success, metric correctness, reproducibility, and scientific validity.

## Initial Milestones

1. Toolize existing PASNet workflows.
2. Implement reflective agent loop with trace logging.
3. Build 100-300 PASBench-DA tasks.
4. Collect successful trajectories.
5. Run SFT and then preference/RL experiments.

## Interview Story

The strongest interview framing is:

> I did not build a broad chatbot. I built a controlled research harness that lets an LLM interact with satellite-assimilation data, models, evaluation scripts, and paper artifacts. Then I used executable workflow traces as post-training data, so the model learns not just meteorology text, but the operational procedure of doing PASNet-style experiments.
