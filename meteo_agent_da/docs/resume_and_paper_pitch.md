# 简历与论文话术

## 简历版本

**MeteoAgent-DA：面向卫星资料同化科研工作流的垂域 Agent 与后训练框架**

围绕 FY-3F MWTS-III/ERA5/PASNet-DA 温度廓线订正任务，设计并实现科研型 Agent Harness，接入数据索引、实验命令生成、评估指标解析、绘图和 LaTeX 论文产物生成工具，形成“实验规划-工具执行-错误反思-结果汇报-轨迹沉淀”的闭环。项目进一步构建 PASBench-DA 评测集，并将成功工具轨迹转化为 SFT/偏好学习数据，用于提升模型在资料同化科研任务中的工具调用成功率、实验复现率和结果解释可靠性。

## 面试亮点

- **Agent Runtime**：使用显式 tool registry 和 reflective loop，而不是纯 prompt 问答。
- **Domain Grounding**：所有关键动作都落到 PASNet-DA 数据、训练脚本、指标文件和论文 artifact。
- **Evaluation**：用 PASBench-DA 评测工具召回、路径检查、命令生成和结果解释，而不是只展示 demo。
- **Post-training**：将可执行科研轨迹转成 SFT、preference 和 RL reward 信号，贴合后训练岗位。

## 论文摘要草稿

大语言模型 Agent 已经在科学工作流中展现出潜力，但宽泛的天气 Agent 往往缺少对专业资料同化流程的稳定约束和可执行 grounding。为此，我们提出 MeteoAgent-DA，一个面向卫星资料同化科研工作流的工具增强与后训练 Agent。MeteoAgent-DA 通过受控的反思式运行时，将 FY-3F MWTS-III/ERA5 数据索引、PASNet-DA 实验构造、指标解析、图表生成和论文产物写作封装为领域工具。进一步地，我们构建 PASBench-DA，用于评测数据检查、实验规划、命令生成、报错修复、结果推理和论文写作等可执行科研任务。成功执行轨迹会被转换为监督微调和偏好学习数据，使 LLM 学到的不只是气象文本知识，而是垂域科研流程。实验部分将在 PASNet-DA 工作流上评估工具调用成功率、实验可复现性、科学合理性和 artifact 生成质量。

## 论文贡献点

1. 提出一个面向卫星资料同化科研的受控 agentic environment。
2. 构建 PASBench-DA，用于评测 PASNet 风格科研工作流中的可执行任务能力。
3. 提出基于科研轨迹的垂域 Agent 后训练管线。
4. 在 FY-3F MWTS-III / ERA5 / PASNet-DA 温度廓线订正场景中验证方法有效性。
