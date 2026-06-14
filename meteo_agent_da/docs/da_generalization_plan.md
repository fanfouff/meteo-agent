# 从 PASNet-DA 扩展到同化全域 Agent 的路线

当前 scaffold 已经能支撑一个窄而清晰的工作流：FY-3F MWTS-III / ERA5 / PASNet-DA 温度廓线订正。
下一阶段的目标不是把它改成泛天气问答助手，而是扩展成“资料同化科研工作流 Agent”。

## 总定位

MeteoAgent-DA 应该分成三层：

1. **Pico-style harness baseline**
   - 模型接入、工具调用、上下文治理、checkpoint/resume、运行 trace、评测回归。
   - 这层回答“agent 系统是否可靠、可复盘、可恢复”。

2. **Data-assimilation domain layer**
   - 同化任务本体、工具卡片、数据/实验 profile、verifier、领域知识库。
   - 这层回答“是否真的懂同化科研工作流”。

3. **Qwen post-training layer**
   - 使用 Qwen 作为 planner/chat LLM，先 SFT，再做 preference/DPO/GRPO。
   - 这层回答“模型是否学会了规划、用工具、解释证据、修复失败轨迹”。

## 同化全域任务树

扩展时建议按任务树逐步接入，而不是一次性支持所有算法和数据源。

### 1. 观测资料侧

- 卫星辐射资料：MWTS/MWHS/ATMS/IASI/AIRS、亮温、通道选择、云检测、bias correction。
- 雷达资料：反射率、径向风、质控、插值、降水个例。
- 常规资料：探空、地面站、飞机报、船舶浮标。
- GNSS RO：折射率、弯曲角、温湿廓线诊断。

### 2. 背景场与模式侧

- ERA5 / 再分析资料查询与裁剪。
- WRF / MPAS / CMA-GFS 等模式背景场。
- forecast lead time、cycle time、domain、vertical level、变量映射。

### 3. 同化方法侧

- 3DVar / 4DVar：代价函数、观测算子、背景误差协方差、窗口设置。
- EnKF / LETKF / Hybrid：ensemble size、localization、inflation、spread/skill。
- Nudging / OI / ML-DA：轻量 baseline 与神经订正/融合方法。
- OSSE / OSE / forecast-impact：敏感性实验与消融设计。

### 4. 诊断与论文侧

- O-B / O-A innovation 统计。
- RMSE、MAE、bias、ACC、spread、CRPS、vertical profile。
- 水平空间误差图、垂直层结误差图、时间序列、case study。
- 实验表格、figure caption、ablation summary、limitation wording。

## 代码结构建议

```text
meteo_agent_da/
  agent/
    llm_planner.py          # Qwen/OpenAI-compatible planner
    context_manager.py      # Pico-style context budget, memory recall
    checkpoint.py           # resume + workspace/data drift
  domains/
    profiles.py             # pasnet_satellite, wrf_3dvar, enkf_cycle, radar_da
    ontology.py             # observation/model/method/diagnostic taxonomy
  tools/
    catalog.py              # tool cards, schema, examples, risk level
    obs/                    # satellite/radar/conventional/GNSS tools
    models/                 # WRF/ERA5/forecast background tools
    assimilation/           # 3DVar/4DVar/EnKF/ML-DA command builders
    diagnostics/            # O-B/O-A, metric, plotting, paper artifacts
  bench/
    tasks/
      pasnet_satellite.jsonl
      wrf_3dvar.jsonl
      enkf_cycle.jsonl
      radar_da.jsonl
    verifiers.py
    run_pasbench.py
```

`PASNet-DA` 不应该被删除，而是作为第一个 domain profile：`pasnet_satellite`。
后续新增同化方向时，只增加 profile、工具卡片、bench task 和 verifier，不改 agent 主循环。

## Qwen Planner 接入原则

当前代码已支持：

```bash
python -m meteo_agent_da.cli \
  --planner qwen \
  --llm-base-url http://localhost:8000/v1 \
  --llm-model <your-qwen-model> \
  --task "检查 50pct split，并生成 PASNet 和 Swin-UNet 的 dry-run 训练命令" \
  --dry-run
```

也可以用于 chat：

```bash
python -m meteo_agent_da.cli \
  --chat \
  --planner qwen \
  --llm-base-url http://localhost:8000/v1 \
  --llm-model <your-qwen-model> \
  --dry-run
```

设计原则：

- Qwen 只做 planner/chat reasoning，不直接执行 shell。
- 工具执行、安全检查、trace、artifact 落盘仍由本地 harness 控制。
- 如果 Qwen endpoint 不可用，自动回退到启发式 planner，保证 benchmark 可跑。
- prompt 里必须约束“不要发明工具、不要声称未验证的路径/指标/产物”。

## 后训练数据路线

### Stage 1: SFT

数据来源：

- PASBench / DA-Bench 任务 instruction。
- Qwen/tool-agent 成功轨迹。
- tool call 参数、tool observation、反思修正、final report。

推荐格式：

```json
{
  "messages": [
    {"role": "system", "content": "You are MeteoAgent-DA..."},
    {"role": "user", "content": "科研任务"},
    {"role": "assistant", "content": "结构化规划 + 工具轨迹 + 最终回答"}
  ],
  "metadata": {
    "domain_profile": "pasnet_satellite",
    "tools": ["sanity_check", "data_indexer", "pasnet_runner"],
    "verifier_pass": true
  }
}
```

### Stage 2: Preference / DPO

chosen：

- 工具路径完整。
- 参数合法。
- 指标和 artifact 有证据。
- 最终回答不夸大、不幻觉。

rejected：

- text-only 无工具证据。
- 漏关键工具。
- 命令参数错误。
- 指标解释无来源。
- 对缺失数据/路径作确定性结论。

### Stage 3: RL-style / GRPO

奖励函数建议从 verifier 指标构造：

```text
reward =
  0.30 * verifier_pass
+ 0.20 * tool_recall
+ 0.15 * command_validity
+ 0.15 * artifact_recall
+ 0.10 * (1 - hallucination_rate)
+ 0.10 * cost_score
```

## DA-Bench 评测矩阵

方法对比：

```text
Text-only
Heuristic Tool Agent
Pico-style Harness Agent
Qwen Tool Agent
Qwen Tool Agent + Reflection
SFT-Qwen
SFT-Qwen + DPO/GRPO
```

指标：

- `TSR`: task success rate。
- `VER`: verifier pass rate。
- `TCR`: tool-call recall。
- `CVR`: command validity rate。
- `AGR`: artifact generation rate。
- `HAL↓`: hallucinated path/metric/tool rate。
- `Cost↓`: 平均 tool steps、token cost、运行时长。

任务分层：

- `in-domain`: PASNet satellite workflow。
- `near-domain`: 其他卫星/再分析/垂直廓线诊断。
- `cross-domain`: WRF/3DVar、EnKF cycle、radar DA。
- `OOD`: 缺失路径、错误 split、未知资料、不可执行命令。

## 最小可投递闭环

1. 保留 PASNet-DA 作为第一个可执行 profile。
2. 接入 Qwen planner，跑通 chat + dry-run tool planning。
3. 扩展 PASBench 到 50 条，覆盖 single-tool、multi-tool、negative/OOD。
4. 新增 executable verifier，而不是只看 required tools。
5. 收集 `pass_rate=1` 的 trace，生成 SFT 数据。
6. 用 text-only/失败轨迹生成 DPO pair。
7. 跑出第一张核心结果表：

```text
Method                  TSR   VER   TCR   CVR   AGR   HAL↓  Cost↓
Text-only
Heuristic Tool Agent
Qwen Tool Agent
SFT-Qwen
SFT-Qwen + DPO/GRPO
```

### 本轮已落地的评测闭环

- `examples/pasbench_da_50.jsonl`：50 条任务，覆盖 in-domain、negative/OOD 和 cross-domain profile planning。
- `meteo_agent_da/bench/verifiers.py`：可执行 verifier，检查路径、命令、metric、artifact、cost 和 artifact hallucination。
- `meteo_agent_da/bench/run_pasbench.py`：批量运行 `text_only | pico | heuristic_tool | qwen_tool | sft_qwen`，支持 `--attempts` 和核心结果表输出。
- `meteo_agent_da/domains/profiles.py`：保留 `pasnet_satellite` 为第一个可执行 profile，并登记 `wrf_3dvar`、`enkf_cycle`、`radar_da`、`gnss_ro`、`conventional_obs`、`diagnostics_paper`。
- `post_training/build_sft_data.py`：从 `trace.jsonl` 保留 plan、tool call、tool observation、reflection 和 final summary。
- `post_training/build_preference_data.py`：可从 runner 的 `scores.json` 自动构造 `heuristic_tool` vs `text_only` 的 DPO-style pair。

这条路线能同时服务三个叙事：

- 面试：你不是只会调 API，而是在做可评测、可恢复、可审计的垂域 agent harness。
- 后训练：你有可执行轨迹、过滤规则、preference pair 和 reward。
- 气象同化：你的贡献不是泛天气聊天，而是把同化科研工作流工具化、基准化、可训练化。
