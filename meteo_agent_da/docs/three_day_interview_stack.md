# MeteoAgent-DA 三天速成技术栈面试稿

这份文档只服务一个目标：三天内把 MeteoAgent-DA 讲成一个可信的 Agent 后训练项目，而不是泛泛地背“我做了一个气象 Agent”。

核心面试定位：

> MeteoAgent-DA 是一个面向气象资料同化科研工作流的 Tool-use Agent Harness。它把数据检查、命令生成、指标解析、论文 artifact 产出等动作封装成工具，记录 JSONL trace 和 report，再通过 verifier/reward 过滤和评分，构造 SFT、DPO 和 GRPO-ready 后训练数据，并用 PASBench-DA 做 Agent Eval。

## 简历技术栈重塑

原技术栈可以改得更像后训练/Agent 岗位：

```text
技术栈：
Python、Transformers / OpenAI-compatible LLM Planner、Tool-use Agent、
ReAct-style Trace、Tool Registry、Sandboxed Dry-run Execution、
SFT/DPO Preference Data、Reward Breakdown、Verifier、Agent Eval、
JSONL Trace、PASBench-DA、LoRA Tiny Training Demo、FY-3F / ERA5 / PASNet-DA
```

如果简历空间有限，压缩成：

```text
技术栈：Python、Transformers、Tool-use Agent、SFT/DPO 数据构建、Reward/Verifier、Agent Eval、JSONL Trace、PASBench-DA
```

更强的项目名：

```text
MeteoAgent-DA：面向气象资料同化科研 Agent 的后训练数据构建与可执行评测框架
```

## 简历 Bullet 重塑版

推荐版本：

```text
● 构建面向 FY-3F/ERA5/PASNet-DA 资料同化科研流程的 Tool-use Agent Harness，支持任务规划、工具注册、受控 dry-run 执行、JSONL trace 落盘、report/artifacts 生成与可复盘审计。
● 设计 Agent 后训练数据管线，将成功工具轨迹转换为 SFT messages，并基于 tool-grounded/text-only、verified/unverified、recovery/failure、concise/hallucinated 等对比构造 DPO preference pairs。
● 构建 executable workflow reward 与 verifier 体系，从工具成功率、artifact 完整性、命令可复现性、格式合规性、科学一致性和幻觉风险等维度评估 Agent 行为。
● 设计 PASBench-DA 评测集，对数据检查、命令生成、实验分析、论文表格生成和失败恢复能力进行量化评测，输出 TSR、VER、TCR、CVR、AGR、HAL、Cost 等 Agent Eval 指标。
```

更偏“后训练岗位”的版本：

```text
● 围绕气象资料同化科研任务构建 Agent post-training data pipeline，采集 plan/tool/observation/reflection/report 轨迹，并通过 verifier 过滤失败工具调用、缺失 artifact、dry-run 虚假训练声明和指标幻觉样本。
● 将可执行成功轨迹转为 SFT messages，将工具增强报告与 text-only/失败/幻觉报告构造成 DPO preference pairs，并导出 rollout-reward JSONL，为 GRPO/RLAIF 式优化提供 reward signal。
● 设计 PASBench-DA 与 executable workflow reward，量化工具调用成功率、命令有效性、artifact 真实性、科学一致性和幻觉率，用于对比 Base/SFT/DPO Agent 的行为改进。
```

## 60 秒面试开场

可以这样说：

> 我这个项目不是做泛天气问答，而是做气象资料同化科研 Agent 的后训练数据和评测框架。具体场景是 FY-3F MWTS-III 卫星观测、ERA5 背景场和 PASNet-DA 温度廓线订正。系统把数据检查、训练命令生成、指标解析、绘图和论文表格生成封装成工具，Agent 每次执行都会保存 JSONL trace、report 和 artifacts。然后我用 verifier 过滤可执行轨迹，构造 SFT 数据、DPO preference pairs 和 GRPO-ready reward breakdown，再用 PASBench-DA 评估工具调用、命令可复现、artifact 生成和幻觉率。重点不是只写一个 Agent demo，而是形成“任务 -> 轨迹 -> 后训练数据 -> reward/verifier -> eval”的闭环。

## 三天速成计划

### Day 1：跑通项目，讲清 Agent Harness

目标：你要能解释“Agent 是怎么工作的”，并现场跑最小 demo。

必须会跑：

```bash
cd /home/lrx/agent/meteo_agent_da
python -m unittest discover -s tests
python -m meteo_agent_da.cli \
  --task "Compare PASNet and Swin-UNet on the 50pct split and generate a paper table" \
  --dry-run
python -m meteo_agent_da.cli --chat --dry-run
```

必须看懂的文件：

```text
meteo_agent_da/agent/runtime.py
meteo_agent_da/agent/schemas.py
meteo_agent_da/agent/tool_registry.py
meteo_agent_da/tools/registry.py
meteo_agent_da/tools/pasnet_runner.py
```

当天要背下来的系统流程：

```text
AgentTask
  -> Planner 生成 AgentPlan
  -> ToolRegistry 查找 ToolSpec
  -> Tool Handler 执行领域工具
  -> ToolResult 返回结构化 observation
  -> Reflection 写入 trace
  -> AgentReport 汇总 summary/artifacts/next_steps
```

当天高频问题：

1. 为什么要 Tool Registry？
   - 防止 LLM 发明工具，把可执行动作限制在已注册工具内。
2. 为什么默认 dry-run？
   - 训练命令昂贵且有 GPU/路径风险，dry-run 先验证命令和路径。
3. 为什么保存 trace？
   - trace 能复盘中间决策，也能转成 SFT/DPO/RL 数据。

### Day 2：掌握后训练数据 SFT/DPO/Reward/Verifier

目标：你要能解释“为什么这是后训练项目”，而不是普通 Agent demo。

必须会跑：

```bash
bash scripts/run_filter_traces.sh
bash scripts/run_build_sft.sh
bash scripts/run_build_dpo.sh
bash scripts/run_reward_export.sh
```

必须看懂的文件：

```text
meteo_agent_da/post_training/filter_traces.py
meteo_agent_da/post_training/build_sft_data.py
meteo_agent_da/post_training/build_preference_data.py
meteo_agent_da/post_training/rewards.py
meteo_agent_da/post_training/build_rollout_rewards.py
meteo_agent_da/verifiers/
```

SFT 要这样讲：

```text
输入：通过 verifier 的成功 report + trace
输出：messages = system + user + assistant
assistant 内容：plan + tool trajectory + reflections + final summary
目的：让模型学习工具规划和执行过程，而不是只学习最终回答
```

DPO 要这样讲：

```text
chosen：工具证据完整、verifier 通过、能恢复失败、少幻觉
rejected：text-only、unverified、failed recovery、hallucinated report
```

四类 preference pair：

```text
tool_grounded_vs_text_only
verified_report_vs_unverified_report
successful_recovery_vs_failed_recovery
concise_reproducible_vs_verbose_hallucinated
```

Reward 要这样讲：

```text
total_reward =
  tool_success_reward
  + artifact_reward
  + scientific_check_reward
  + format_reward
  + reproducibility_reward
  - hallucination_penalty
  - unsafe_command_penalty
```

当天高频问题：

1. 为什么不是所有 trace 都能进 SFT？
   - 错误工具调用、假 artifact、dry-run 虚假声明会污染模型。
2. DPO rejected 怎么构造？
   - 从 text-only baseline、失败报告、缺 artifact 报告、指标幻觉报告中构造。
3. 你做 GRPO 了吗？
   - 不能说做完了。只能说完成 GRPO-ready reward design 和 rollout/reward JSONL 导出。

### Day 3：掌握 Agent Eval 和气象资料同化背景

目标：你要能回答“怎么证明有效”和“为什么是气象同化”。

必须会跑：

```bash
python -m meteo_agent_da.bench.run_pasbench \
  --tasks examples/pasbench_da_sample.jsonl \
  --method text_only heuristic_tool \
  --output-dir runs/interview_demo
```

必须看懂的文件：

```text
examples/pasbench_da_sample.jsonl
examples/pasbench_da_50.jsonl
meteo_agent_da/bench/pasbench.py
meteo_agent_da/bench/verifiers.py
meteo_agent_da/bench/run_pasbench.py
docs/interview_playbook.md
```

评测指标要背：

```text
TSR：task success rate
VER：verifier pass rate
TCR：tool-call recall
CVR：command validity rate
AGR：artifact generation rate
HAL：hallucination rate，越低越好
Cost：平均工具步数或成本，越低越好
```

气象同化要背：

```text
资料同化 = 背景场 + 观测 -> 分析场
x_a = x_b + increment
O-B = observation - background
O-A = observation - analysis
RMSE/MAE = 误差指标
FY-3F MWTS-III = 卫星微波温度探测亮温
ERA5 = 再分析资料，这里提供背景场和参考分析场
PASNet-DA = 用卫星观测和背景场做温度廓线增量订正的模型工作流
```

当天高频问题：

1. 怎么证明 Agent 真的变好？
   - 在 held-out PASBench-DA 上比较 Base/Text-only/Tool Agent/SFT/DPO 的 TSR、VER、TCR、CVR、AGR、HAL、Cost。
2. 这个和传统气象模型有什么关系？
   - 不是替代 NWP/同化系统，而是辅助科研工作流和 ML-DA 实验复现。
3. dry-run 结果能说明模型效果吗？
   - 不能。dry-run 只能说明命令、路径、工具流程可复现，不能说明 RMSE 提升。

## 技术栈逐项面试讲法

### 1. Transformers

这个项目里 Transformers 不是主角，但它是后续 SFT/DPO 的模型底座。

面试说法：

> 项目预留了 OpenAI-compatible/Qwen planner 和 LoRA tiny training 入口。当前重点不是训练大模型本身，而是构造高质量 Agent 轨迹数据和评测环境。后续可以用 Transformers/PEFT/TRL 接 SFT 或 DPO。

被问“你训练了吗”：

> 当前默认提供 dry-run training plan 和数据格式校验，没有夸大为已经完成大规模训练。

### 2. Tool-use Agent

面试说法：

> 我把资料同化科研任务拆成多个受控工具，例如数据检查、命令生成、指标解析和论文表格生成。LLM 只负责规划和解释，不直接执行任意 shell。

关键文件：

```text
agent/runtime.py
agent/tool_registry.py
tools/registry.py
```

### 3. JSONL Trace

面试说法：

> 每次 Agent run 都会写 `trace.jsonl`，里面包括 plan、tool、reflect、report。这个格式适合流式追加、失败恢复、离线审计和后训练数据构造。

为什么不用纯文本日志：

> JSONL 是结构化数据，可以直接解析成 SFT messages、DPO metadata 和 reward rollout。

### 4. SFT 数据构建

面试说法：

> SFT 样本不是普通问答，而是把 Agent 成功轨迹组织成 system/user/assistant。assistant 内容包含 plan、tool trajectory、observation、reflection 和 final summary。

一句话记忆：

> SFT 学的是“怎么做科研工具流”，不是“怎么写一段漂亮解释”。

### 5. DPO 数据构建

面试说法：

> DPO 用 chosen/rejected pair 学偏好。我不是只做 tool-agent vs text-only，而是设计了四类偏好：工具证据、验证通过、错误恢复、少幻觉可复现。

一句话记忆：

> DPO 学的是“什么样的 Agent 行为更值得偏好”。

### 6. Reward/Verifier

面试说法：

> Verifier 判断一个报告是否可信，reward 把可信度变成可优化信号。两者都尽量可解释，而不是给一个黑盒分数。

Verifier 五类：

```text
artifact_verifier：产物路径是否存在
command_verifier：命令参数和 split_file 是否可复现
metric_verifier：指标 artifact 是否可解析
report_verifier：summary/tool evidence/next steps 是否完整
scientific_verifier：dry-run 下是否虚假声称训练完成，是否指标幻觉
```

### 7. Agent Eval

面试说法：

> Agent Eval 不能只看文本相似度，要看任务是否真的完成。所以 PASBench-DA 评估工具召回、命令有效性、artifact 真实性、verifier pass 和幻觉率。

一句话记忆：

> 对 Agent 来说，能不能做对比说得像不像更重要。

### 8. FY-3F / ERA5 / PASNet-DA

面试说法：

> FY-3F MWTS-III 提供微波亮温观测，ERA5 提供背景场和参考分析场，PASNet-DA 学习从背景场到参考场的温度廓线增量订正。MeteoAgent-DA 服务的是这个科研工作流，而不是替代模型本身。

## 高频追问短答案

### Q：这个项目最大亮点是什么？

不是做了一个聊天 Agent，而是把气象资料同化科研流程变成了可执行、可验证、可后训练的数据闭环。

### Q：为什么说它是后训练项目？

因为它不只跑 Agent，还能把成功轨迹转 SFT，把好坏报告转 DPO，把 rollout 转 reward JSONL，并用 verifier 做数据过滤。

### Q：你和普通 SFT 项目区别是什么？

普通 SFT 多是 instruction-response；这里是 Agent trajectory SFT，包含工具调用、observation、reflection 和 artifacts。

### Q：你和普通 DPO 项目区别是什么？

普通 DPO 常比较两段文本；这里比较两条 Agent 行为轨迹，偏好可执行、可验证、可恢复、少幻觉的轨迹。

### Q：你和普通 Agent Demo 区别是什么？

普通 demo 只展示能跑；这个项目有 benchmark、verifier、reward、trace 和后训练数据导出。

### Q：如果模型乱调用工具怎么办？

Tool Registry 限制工具集合，unknown tool 会报错；verifier 和 PASBench 会惩罚错误工具调用；DPO 也会偏好正确工具轨迹。

### Q：如果模型编造 artifact 怎么办？

artifact verifier 检查路径是否真实存在，HAL 指标统计 hallucinated artifact/path。

### Q：如果模型声称训练完成但其实 dry-run 怎么办？

scientific verifier 会检查 dry-run 下的虚假训练声明，这类 trace 会被过滤。

### Q：这个项目目前最大的不足是什么？

当前重点是后训练数据和评测闭环，真实 SFT/DPO/GRPO 大规模训练还没有完成；后续要接本地 Qwen/LoRA，在 held-out PASBench 上做 before-after 对比。

### Q：三天准备时最该背什么？

背四句话：

```text
1. Agent Harness：规划、工具调用、受控执行、trace/report/artifacts。
2. Post-training Data：成功轨迹 -> SFT，好坏轨迹 -> DPO，rollout -> reward。
3. Verifier/Reward：工具、命令、artifact、指标、dry-run 诚实性和幻觉风险。
4. PASBench-DA：用 TSR/VER/TCR/CVR/AGR/HAL/Cost 证明行为改进。
```

## 面试时不能乱说的边界

不能说：

```text
我已经完成 GRPO 训练。
dry-run 证明模型 RMSE 提升。
Agent 已经真实训练了 PASNet/Swin-UNet。
没有 metric artifact 但可以报具体 RMSE 数值。
```

可以说：

```text
我完成了 GRPO-ready reward design 和 rollout/reward 数据导出。
dry-run 验证了命令构造、路径和工具流程。
真实模型效果需要执行训练并用 evaluator 验证。
当前项目重点是 Agent 后训练数据构建与评测闭环。
```

## 最后一天模拟面试提纲

按这个顺序练 20 分钟：

1. 60 秒项目介绍。
2. 画出 Agent runtime 流程。
3. 解释 trace 如何转 SFT。
4. 解释四类 DPO pair。
5. 解释 reward breakdown。
6. 解释 PASBench-DA 指标。
7. 解释 FY-3F/ERA5/PASNet-DA 场景。
8. 承认边界：当前是后训练数据与评测框架，不夸大成完整 RL 训练。

收束句：

> 这个项目的重点是把一个真实气象资料同化科研流程，变成 Agent 可以执行、系统可以验证、模型可以后训练的数据闭环。这正好覆盖 Tool-use Agent、SFT/DPO 数据、Reward/Verifier 和 Agent Eval 这些后训练岗位核心能力。
