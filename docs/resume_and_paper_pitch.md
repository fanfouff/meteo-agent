# Resume and Paper Pitch

## Resume Version

**MeteoAgent-DA: 面向卫星资料同化科研工作流的垂域 Agent 与后训练框架**

围绕 FY-3F MWTS-III/ERA5/PASNet-DA 温度廓线订正任务，设计并实现科研型 Agent Harness，接入数据索引、实验命令生成、评估指标解析、绘图和 LaTeX 论文产物生成工具，形成“实验规划-工具执行-错误反思-结果汇报-轨迹沉淀”的闭环。项目进一步构建 PASBench-DA 评测集，并将成功工具轨迹转化为 SFT/偏好学习数据，用于提升模型在资料同化科研任务中的工具调用成功率、实验复现率和结果解释可靠性。

## Interview Highlights

- **Agent Runtime**: 使用显式 tool registry 和 reflective loop，而不是纯 prompt 问答。
- **Domain Grounding**: 所有关键动作都落到 PASNet-DA 数据、训练脚本、指标文件和论文 artifact。
- **Evaluation**: 用 PASBench-DA 测工具召回、路径检查、命令生成、结果解释，而不是只展示 demo。
- **Post-training**: 将可执行科研轨迹转成 SFT/preference/RL reward 信号，贴合后训练岗位。

## Paper Abstract Sketch

Large language model agents have shown promise in scientific workflows, but broad weather agents often lack grounding in specialized data-assimilation procedures. We introduce MeteoAgent-DA, a tool-augmented and post-trained research agent for satellite data-assimilation workflows. MeteoAgent-DA exposes domain tools for FY-3F MWTS-III/ERA5 data indexing, PASNet-DA experiment construction, metric parsing, figure generation, and paper artifact writing through a controlled reflective runtime. We further construct PASBench-DA, a benchmark targeting executable research tasks including data inspection, experiment planning, command generation, error repair, result reasoning, and paper writing. Successful execution traces are converted into supervised and preference-style post-training data, enabling LLMs to learn domain procedures rather than only meteorological text. Experiments on PASNet-DA workflows evaluate tool success, reproducibility, scientific validity, and artifact generation.

## Contribution List

1. A controlled agentic environment for satellite data-assimilation research.
2. PASBench-DA, an executable benchmark for PASNet-style research workflows.
3. A trajectory-based post-training pipeline for domain research agents.
4. Validation on FY-3F MWTS-III / ERA5 / PASNet-DA temperature-profile correction.
