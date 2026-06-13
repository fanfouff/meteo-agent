# MeteoAgent-DA

MeteoAgent-DA 是一个面向**卫星资料同化科研工作流**的气象垂域 Agent。
它聚焦你已有的主线资产：**FY-3F MWTS-III + ERA5 + PASNet-DA 温度廓线订正**。

这个项目刻意不做泛化的天气问答助手，而是服务于科研工作流。它的目标是帮助研究者：

- 检查 FY-3F/ERA5 `.npz` 数据集和数据划分文件；
- 生成 PASNet、Swin-UNet、FuXi-DA 等模型的训练或推理命令；
- 以 dry-run 或受控执行方式运行实验；
- 汇总 RMSE、MAE、垂直廓线误差和空间误差诊断；
- 生成可直接进入论文的表格、图注和结果分析段落；
- 收集成功工具调用轨迹，为 SFT/DPO/GRPO 等后训练流程提供数据。

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
  sandbox/            # 受控命令执行器
  bench/              # PASBench-DA 样例和评测辅助工具
  post_training/      # trace 到 SFT/preference 数据的转换工具
configs/
docs/
examples/
tests/
```

## 快速开始

在当前目录执行：

```bash
python -m meteo_agent_da.cli \
  --task "Compare PASNet and Swin-UNet on the 50pct split and generate a paper table" \
  --dry-run
```

配置文件默认读取：

```bash
configs/default.yaml
```

也可以用环境变量覆盖配置项，例如：

```bash
export METEO_AGENT_DA_PROJECT_ROOT=/path/to/satellite_assimilation_v2
export METEO_AGENT_DA_DEFAULT_DATA_ROOT=/path/to/npz
```

启用 OpenAI-compatible LLM planner：

```bash
export OPENAI_API_KEY=...
python -m meteo_agent_da.cli \
  --planner llm \
  --task "检查 50pct split，规划 PASNet 与 Swin-UNet 对比实验" \
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

构造 DPO-style preference 数据：

```bash
python -m meteo_agent_da.post_training.build_preference_data \
  --chosen runs/tool_agent/report.json \
  --rejected runs/text_only_report.json \
  --output post_training_preferences.jsonl
```

## 当前范围

脚手架默认对昂贵的训练和评估命令使用 **dry-run**。只有在确认路径、GPU 分配和输出目录之后，才建议关闭 dry-run。

真实执行 PASNet 命令需要三重显式开关：

```bash
python -m meteo_agent_da.cli \
  --task "训练 PASNet 100pct split" \
  --execute \
  --allow-risky \
  --run-commands
```

没有同时打开这些开关时，`pasnet_runner` 只生成命令和诊断信息。

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
3. 提出基于可执行科研轨迹的垂域 Agent 后训练流程；
4. 在 FY-3F MWTS-III / ERA5 / PASNet-DA 工作流上验证有效性。
