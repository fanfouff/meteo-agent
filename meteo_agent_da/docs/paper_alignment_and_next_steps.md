# 论文对齐与后续改造清单

本文档说明 MeteoAgent-DA 当前代码与三篇参考论文的对应关系，并列出为了投递 Agent / 后训练岗位还需要补齐的实验闭环。

参考材料：

- `/home/lrx/agent/paper/工具调用bench.pdf`：ToolLLM / ToolBench
- `/home/lrx/agent/paper/agent metric1.pdf`：ScienceAgentBench
- `/home/lrx/agent/paper/SFT2023.pdf`：AgentTuning
- `/home/lrx/agent/paper/2-pico.pdf` 与 `https://gitee.com/htxoffical/pico.git`：pico 本地 agent harness 设计

## 当前已补的工程改动

| 改动 | 文件 | 对齐论文/部分 | 作用 |
|---|---|---|---|
| 多轮对话式科研 Agent | `meteo_agent_da/agent/interactive.py` | pico 的 runtime 主循环、session/history、trace 思路 | 从单次任务变成可持续对话的同化科研助手 |
| 会话状态与领域记忆 | `meteo_agent_da/agent/session.py` | pico 的结构化会话记忆与上下文治理 | 支持 follow-up 问题，减少重复说明 |
| text-only baseline | `meteo_agent_da/baselines/text_only.py` | ToolBench / ScienceAgentBench 的 baseline 对照 | 支撑“纯文本 vs 工具 Agent”的效果表 |
| PASBench-DA 指标扩展 | `meteo_agent_da/bench/pasbench.py` | ScienceAgentBench Section 2.3 Program Evaluation；ToolBench ToolEval | 从只看 required tools 扩展到工具召回、精确率、成功率、artifact、预算 |
| 多 report 评估脚本 | `meteo_agent_da/bench/evaluate_trace.py` | ScienceAgentBench 的多次运行和聚合评估思路 | 可以直接比较 text-only/tool-agent/SFT/DPO |
| DPO-style preference 数据 | `meteo_agent_da/post_training/build_preference_data.py` | AgentTuning 的 trajectory filtering；DPO 类偏好数据流程 | 将 chosen/rejected report 转成偏好学习样本 |
| README 使用入口 | `README.md` | 面试项目可复现要求 | 提供 chat、baseline、eval、preference data 命令 |

## 参照 ToolLLM / ToolBench 还需要做什么

ToolBench 的关键部分包括：API collection、instruction generation、solution path annotation、API retriever、ToolEval，以及 single-tool / multi-tool / OOD evaluation。

对应到 MeteoAgent-DA，建议补：

1. 工具卡片和工具检索器
   - 对应 ToolBench Figure 1 的 API Retriever。
   - 当前项目已有 `ToolRegistry`，但还没有“根据任务检索最相关工具”的独立模块。
   - 建议新增 `tools/catalog.py` 或 `agent/tool_retriever.py`，把每个领域工具描述成 tool card：名称、输入 schema、输出 artifact、风险等级、示例调用。

2. PASBench-DA 任务生成流程
   - 对应 ToolBench 的 instruction generation。
   - 当前只有 `examples/pasbench_da_sample.jsonl` 的 3 条样例。
   - 建议扩充到 100-300 条，覆盖：
     - single-tool：只查数据、只生成命令、只汇总指标；
     - intra-workflow multi-tool：数据检查 -> 命令生成 -> 论文表格；
     - cross-workflow multi-tool：结果指标 -> 绘图 -> 论文分析；
     - negative / unsolvable：split 不存在、指标文件缺失、GPU 不允许执行。

3. solution path 标注
   - 对应 ToolBench 的 solution path annotation。
   - 每条 PASBench 任务应给出标准工具序列，例如：
     `sanity_check -> data_indexer -> pasnet_runner -> evaluator -> paper_writer`。
   - 当前 `required_tools` 是弱标注，后续需要 `gold_tool_path` 和 `allowed_alternatives`。

4. ToolEval 风格评估
   - 对应 ToolBench Appendix A.5 的 pass rate / win rate。
   - 当前已新增 `tool_recall`、`tool_precision`、`tool_success_rate`、`pass_rate`。
   - 后续应加入 pairwise win rate：比较两个 agent 的轨迹，优先选择工具证据更完整、重复调用更少、artifact 更可复现的轨迹。

## 参照 ScienceAgentBench 还需要做什么

ScienceAgentBench 强调不要直接声称端到端科学发现，而要评估科学工作流中的可执行子任务。它的 Section 2.1 把任务拆成 instruction、dataset information、expert-provided knowledge、annotated program；Section 2.3 强调 execution result、success criteria、cost 和 rubric。

对应到 MeteoAgent-DA，建议补：

1. 任务实例四件套
   - `Task Instruction`：用户科研目标。
   - `Dataset Information`：FY-3F/ERA5 路径、split 文件、样本预览、stats 文件。
   - `Expert Knowledge`：PASNet-DA 参数约束、同化术语、指标解释、论文写法。
   - `Annotated Workflow`：标准工具序列、命令模板、预期 artifact。

2. 可执行 verifier
   - 当前 `verifier` 还是字符串，评估只看 report。
   - 建议新增 `bench/verifiers.py`：
     - path verifier：路径存在、split count 合法；
     - command verifier：命令参数完整、模型名合法、split_file 合法；
     - metric verifier：RMSE/MAE 字段存在，数值可解析；
     - artifact verifier：LaTeX 表格/图片文件生成；
     - cost verifier：工具步数、执行时间、API 费用。

3. 分层指标表
   - 对应 ScienceAgentBench 的 SR、VER、CBS、Cost 思路。
   - MeteoAgent-DA 可以采用：
     - `TSR`：task success rate；
     - `VER`：verifier pass rate；
     - `TCR`：tool-call recall；
     - `CVR`：command validity rate；
     - `AGR`：artifact generation rate；
     - `HAL↓`：hallucinated path/metric/tool rate；
     - `Cost↓`：平均工具步数、平均 token/API cost。

4. 多次运行与 best-of-k
   - ScienceAgentBench 对每个任务重复多次，并按指标选择 best run。
   - 当前项目尚未实现批量 runner。
   - 建议新增 `bench/run_pasbench.py`，支持 `--method text_only|heuristic_tool|llm_tool|sft|dpo` 和 `--attempts 3`。

## 参照 AgentTuning 还需要做什么

AgentTuning 的核心是 AgentInstruct：instruction construction、trajectory interaction、trajectory filtering，并用成功轨迹做混合 SFT，提升 planning、memory、tool utilization。

对应到 MeteoAgent-DA，建议补：

1. 成功轨迹采集
   - 当前 runtime 已写 `runs/<run_id>/trace.jsonl` 和 `report.json`。
   - 下一步要批量跑 PASBench，把 `pass_rate=1` 的报告收集成 `AgentInstruct-DA`。

2. SFT 数据升级
   - 当前 `post_training/build_sft_data.py` 只从 `report.json` 生成简化 messages。
   - 建议改成读取 `trace.jsonl`，保留：
     - planner 输出；
     - 每一步 tool call 的参数；
     - tool observation；
     - reflector 修正；
     - final report。

3. 轨迹过滤
   - 对应 AgentTuning 的 reward-based filtering。
   - 当前有 `post_training/rewards.py` 的雏形。
   - 建议把 PASBench 指标接进 reward：
     `reward = 0.35*VER + 0.25*TCR + 0.2*AGR + 0.1*(1-HAL) + 0.1*cost_score`。

4. DPO 数据
   - 当前已新增 `build_preference_data.py`，支持 chosen/rejected report。
   - 后续需要自动构造 rejected：
     - text-only 无工具证据；
     - 漏掉关键工具；
     - 命令参数错误；
     - 指标解释幻觉；
     - artifact 未生成。

5. 后训练实验表
   - 推荐实验矩阵：
     - Text-only；
     - Heuristic Tool Agent；
     - LLM Tool Agent；
     - LLM Tool Agent + Reflection；
     - SFT；
     - SFT + DPO。

## 参照 pico 还需要做什么

pico 的价值不在论文贡献，而在 agent harness 工程能力。当前 MeteoAgent-DA 已经补了交互式 session 和会话记忆，但还没有完整复制 pico 的上下文治理与恢复机制。

建议补：

1. ContextManager
   - 对应 pico 的 prefix、memory、relevant memory、history、current request。
   - MeteoAgent-DA 现在只有 `ConversationSession.context_text()` 的轻量实现。
   - 后续可以新增 `agent/context_manager.py`，为每段设置 token/字符预算，永远不裁当前请求。

2. Tool execution gate
   - 对应 pico 的 `run_tool()` 统一闸口。
   - 当前 `ToolRegistry.run()` 已有 risky 检查，但还缺：
     - 参数 schema 校验；
     - 重复调用拦截；
     - 敏感路径/环境变量脱敏；
     - tool result 裁剪；
     - partial success 分类。

3. Checkpoint / resume
   - 对应 pico 的 checkpoint 和 workspace drift 检查。
   - MeteoAgent-DA 目前有 session resume，但没有 checkpoint。
   - 后续建议保存：
     - project config hash；
     - tool signature；
     - dataset fingerprint；
     - split file mtime；
     - last report id。

4. 固定 benchmark 与回归
   - 对应 pico 的 harness regression、memory ablation、recovery ablation。
   - MeteoAgent-DA 应建立：
     - harness regression：工具合同不破；
     - memory ablation：有无会话记忆；
     - tool ablation：有无工具；
     - post-training ablation：SFT/DPO 前后。

## 当前项目最优先的下一步

1. 扩充 PASBench-DA 到 50 条，然后再到 100-300 条。
2. 新增 `LLMPlanner`，支持 OpenAI-compatible 或本地 vLLM，让模型真实决定工具调用。
3. 新增 `bench/verifiers.py` 和 `bench/run_pasbench.py`，把评估跑成表格。
4. 用成功 tool-agent 轨迹生成 SFT 数据。
5. 用 text-only/失败轨迹构造 DPO rejected。
6. 做第一张核心结果表：

```text
Method                  TSR   VER   TCR   CVR   AGR   HAL↓  Cost↓
Text-only               ...
Heuristic Tool Agent    ...
LLM Tool Agent          ...
+ Reflection            ...
+ SFT                   ...
+ SFT + DPO             ...
```

这张表跑出来以后，项目就从“agent scaffold”进入“可投递、可面试、可写论文”的阶段。
