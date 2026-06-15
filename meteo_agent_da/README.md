# MeteoAgent-DA

**Post-training Data Pipeline for Tool-augmented Data Assimilation Agents**

面向卫星资料同化科研 Agent 的后训练数据、偏好优化与可执行评测框架。

MeteoAgent-DA 是一个面向**卫星资料同化科研工作流**的气象垂域 Agent。
它聚焦你已有的主线资产：**FY-3F MWTS-III + ERA5 + PASNet-DA 温度廓线订正**。

这个项目刻意不做泛化的天气问答助手，而是服务于科研工作流。它的目标是帮助研究者：

- 检查 FY-3F/ERA5 `.npz` 数据集和数据划分文件；
- 生成 PASNet、Swin-UNet、FuXi-DA 等模型的训练或推理命令；
- 以 dry-run 或受控执行方式运行实验；
- 汇总 RMSE、MAE、垂直廓线误差和空间误差诊断；
- 生成可直接进入论文的表格、图注和结果分析段落；
- 收集、过滤、验证成功工具调用轨迹，为 SFT/DPO/GRPO-ready reward 等后训练流程提供数据。

## 设计思路

```text
规划器 -> 工具调用器 -> 执行器 -> 反思器 -> 报告器
```

当前版本是一个轻量级本地 Agent Harness，而不是复杂的多智能体平台。它吸收了 pico 类代码 Agent 中比较有价值的工程设计：

- 显式工具注册表；
- 结构化工具输入和输出；
- 有边界的命令执行；
- 运行 trace 和 artifacts 落盘；
- 短期工作记忆；
- 带可验证输出的 benchmark 任务。

## 仓库结构

```text
meteo_agent_da/
  agent/              # 反思式运行时、数据结构、工作记忆
  tools/              # PASNet-DA 领域工具
  verifiers/          # artifact/command/metric/report/scientific verifier
  sandbox/            # 受控命令执行器
  bench/              # PASBench-DA 样例和评测辅助工具
  post_training/      # trace 到 SFT/preference 数据的转换工具
configs/
docs/
examples/
scripts/
tests/
```

## 快速开始

在当前目录执行：

```bash
python -m meteo_agent_da.cli \
  --task "Compare PASNet and Swin-UNet on the 50pct split and generate a paper table" \
  --dry-run
```

运行 smoke test：

```bash
python -m unittest discover -s tests
```

启动一个多轮对话式同化科研 Agent：

```bash
python -m meteo_agent_da.cli --chat --dry-run
```

使用 Qwen/OpenAI-compatible planner 启动 chat：

```bash
python -m meteo_agent_da.cli \
  --chat \
  --planner qwen \
  --llm-base-url http://localhost:8000/v1 \
  --llm-model <your-qwen-model> \
  --dry-run
```

交互式会话支持：

- `/memory`：查看当前会话记忆；
- `/session`：查看 session id 和存储路径；
- `/dry-run on|off`：切换 dry-run；
- `/exit`：退出。

生成 text-only baseline report：

```bash
python -m meteo_agent_da.baselines.text_only \
  --task "Compare PASNet and Swin-UNet on the 50pct split and generate a paper table." \
  --output runs/text_only_report.json
```

评估一个或多个 report：

```bash
python -m meteo_agent_da.bench.evaluate_trace \
  --tasks examples/pasbench_da_sample.jsonl \
  --report runs/demo/report.json runs/text_only_report.json
```

批量运行 PASBench-DA 50 条任务并生成核心结果表：

```bash
python -m meteo_agent_da.bench.run_pasbench \
  --tasks examples/pasbench_da_50.jsonl \
  --method text_only pico heuristic_tool \
  --attempts 1 \
  --output-dir runs/pasbench_50
```

`pico` baseline 只通过外部命令调用，不 vendoring Pico 源码。需要启用时传入：

```bash
python -m meteo_agent_da.bench.run_pasbench \
  --method pico \
  --pico-command "pico --task {task}" \
  --output-dir runs/pasbench_pico
```

当前批量 runner 会输出：

- `scores.json` / `scores.csv`：逐任务评分；
- `core_results.md` / `core_results.csv`：TSR、VER、TCR、CVR、AGR、HAL、Cost 表；
- `successful_reports.txt`：`pass_rate=1` 的 report 路径，用于后训练数据过滤。

从成功 trace 构造 SFT 数据：

先做 trajectory filtering，只保留可执行、可验证、可复现的轨迹：

```bash
python -m meteo_agent_da.post_training.filter_traces \
  --successful-reports-file runs/pasbench_50/successful_reports.txt \
  --allow-no-artifacts \
  --dry-run-expected \
  --output runs/pasbench_50/filtered_reports.txt
```

```bash
python -m meteo_agent_da.post_training.build_sft_data \
  --successful-reports-file runs/pasbench_50/filtered_reports.txt \
  --output runs/pasbench_50/post_training_sft.jsonl
```

构造 DPO-style preference 数据：

```bash
python -m meteo_agent_da.post_training.build_preference_data \
  --scores-json runs/pasbench_50/scores.json \
  --chosen-method heuristic_tool \
  --rejected-method text_only \
  --output runs/pasbench_50/post_training_preferences.jsonl
```

导出 GRPO-ready rollout reward 数据：

```bash
python -m meteo_agent_da.post_training.build_rollout_rewards \
  --scores-json runs/pasbench_50/scores.json \
  --output runs/pasbench_50/rollout_reward.jsonl
```

校验训练数据格式：

```bash
python -m meteo_agent_da.post_training.verify_training_data \
  --input runs/pasbench_50/post_training_sft.jsonl \
  --format sft
```

生成 tiny LoRA 训练计划。默认是 dry-run，不下载模型、不启动训练；加 `--execute` 后才会尝试调用本地 `transformers/peft/datasets/trl` 环境：

```bash
python -m meteo_agent_da.post_training.train_sft_lora \
  --train-file runs/pasbench_50/post_training_sft.jsonl \
  --output-dir runs/post_training/sft_lora

python -m meteo_agent_da.post_training.train_dpo_lora \
  --preference-file runs/pasbench_50/post_training_preferences.jsonl \
  --output-dir runs/post_training/dpo_lora
```

也可以直接使用脚本入口：

```bash
bash scripts/run_filter_traces.sh
bash scripts/run_build_sft.sh
bash scripts/run_build_dpo.sh
bash scripts/run_reward_export.sh
bash scripts/run_sft_tiny.sh
bash scripts/run_dpo_tiny.sh
```

## 后训练闭环

```text
PASBench-DA task
   ↓
Agent rollout
   ↓
trace.jsonl / report.json / artifacts
   ↓
verifier 检查
   ↓
reward breakdown
   ↓
successful trace → SFT data
bad/good report pair → DPO data
rollout + reward → GRPO-ready data
   ↓
mini SFT / DPO training plan
   ↓
before-after Agent eval
```

### Stage 1: Behavior Cloning / SFT

目标是让模型学习可执行工具轨迹，而不是学习泛泛的气象文本回答。数据来自 `filter_traces.py` 通过的成功 report 和 trace，输出为 `post_training_sft.jsonl`。

过滤规则会拒绝：

- 工具调用失败的 trace；
- 没有工具 evidence 的 text-only trace；
- 缺少 artifacts 的 trace，或在数据检查类任务中显式允许无 artifact；
- 科学检查不通过的 trace，例如 dry-run 下声称完成真实训练；
- 命令路径为空、split file 不可复现或包含不安全命令的 trace；
- 过长、重复、JSON 格式错误的 trace。

### Stage 2: Preference Optimization / DPO

`build_preference_data.py` 输出 `prompt / chosen / rejected`，并在 metadata 中标注偏好类型：

- `tool_grounded_vs_text_only`
- `verified_report_vs_unverified_report`
- `successful_recovery_vs_failed_recovery`
- `concise_reproducible_vs_verbose_hallucinated`

样例位于：

```text
examples/preference_cases/
examples/failure_traces/
```

### Stage 3: GRPO-ready Reward

`rewards.py` 返回可解释 reward breakdown，而不是单个不可解释分数：

```json
{
  "total_reward": 0.86,
  "tool_success_reward": 0.95,
  "artifact_reward": 1.0,
  "scientific_check_reward": 0.8,
  "format_reward": 0.9,
  "reproducibility_reward": 0.75,
  "hallucination_penalty": -0.1,
  "unsafe_command_penalty": 0.0
}
```

注意：当前项目提供的是 **GRPO-ready reward design** 和 rollout/reward 数据导出，不声称已经完成 GRPO 训练。

## Verifier 体系

`meteo_agent_da/verifiers/` 将后训练数据质量拆成可解释检查：

- `artifact_verifier`：检查 report 中声明的 csv/json/png/md/tex 是否真实存在；
- `command_verifier`：检查 PASNet 命令是否包含 `--output_dir`、`--split_file`、`--model` 等关键参数，并拒绝不安全命令；
- `metric_verifier`：检查 evaluator 输出是否有可解析 metric artifact；
- `report_verifier`：检查 final report 是否包含 summary、tool evidence、artifact/next-step 信息；
- `scientific_verifier`：检查 dry-run 下是否误称真实训练完成，是否有无来源数值指标。

## 当前范围

脚手架默认对昂贵的训练和评估命令使用 **dry-run**。只有在确认路径、GPU 分配和输出目录之后，才建议关闭 dry-run。

默认 PASNet 项目根目录为：

```text
/home/lrx/Unet/satellite_assimilation_v2
```

## 论文方向

暂定题目：

> MeteoAgent-DA: 面向卫星资料同化科研工作流的工具增强与后训练 LLM Agent

核心贡献：

1. 提出一个面向卫星资料同化科研的 agentic environment；
2. 构建 PASBench-DA，用于评测工具调用、实验规划、结果解释和论文写作能力；
3. 提出基于可执行科研轨迹的垂域 Agent 后训练数据、偏好学习和 GRPO-ready reward 流程；
4. 在 FY-3F MWTS-III / ERA5 / PASNet-DA 工作流上验证有效性。

扩展到更大资料同化任务空间的路线见：

- `docs/da_generalization_plan.md`
