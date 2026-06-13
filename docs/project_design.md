# MeteoAgent-DA 项目设计

## 项目定位

MeteoAgent-DA 不是一个宽泛的天气科学助手，而是一个面向离线卫星资料同化和温度廓线订正的科研工作流 Agent。项目有意把任务空间收敛到 PASNet-DA 主线：

- FY-3F MWTS-III 微波亮温观测；
- ERA5/NWP 背景温度场；
- 稀疏观测 mask；
- PASNet-DA 与对比基线模型实验；
- RMSE、MAE、垂直廓线误差和空间误差诊断；
- 可直接进入论文的 LaTeX 产物。

这样设计可以避开泛天气 Agent 框架的正面重合，让项目拥有更清晰的资料同化科研身份。

## 四层架构

### 1. Agent Runtime

运行时采用反思式闭环：

```text
规划 -> 调用工具 -> 观察结果 -> 反思修正 -> 生成报告
```

第一版实现是确定性的本地运行时。后续可以替换为 LLM planner，但不需要改变工具契约。

### 2. Domain Tools

领域工具层负责包装已有 PASNet-DA 项目能力：

- `data_indexer`：检查 `.npz` 数量、split 文件、统计量文件和数据根目录；
- `pasnet_runner`：生成受控的训练/评估命令；
- `evaluator`：汇总 metric JSON/CSV/NPY 等结果产物；
- `plotter`：生成或登记论文图表命令；
- `paper_writer`：生成 LaTeX 表格和结果分析段落；
- `sanity_checker`：检查数据泄漏、缺失文件、非法 split 比例和不安全运行选项。

### 3. Benchmark

`PASBench-DA` 评测的不是通用气象知识，而是可执行科研工作流能力：

- 数据查询；
- 实验规划；
- 命令生成；
- 报错修复；
- 结果解释；
- 论文写作。

### 4. Post-Training

成功运行轨迹会被转化为后训练数据：

- SFT 样本：用户任务 -> 工具轨迹 -> 最终报告；
- preference pair：有效实验计划/报告 vs 无效实验计划/报告；
- RL-style reward：工具成功率、指标正确性、可复现性和科学合理性。

## 初始里程碑

1. 将现有 PASNet 工作流工具化。
2. 实现带 trace 记录的反思式 Agent loop。
3. 构建 100 到 300 条 PASBench-DA 任务。
4. 收集成功工具调用轨迹。
5. 先做 SFT，再做 preference/RL 实验。

## 面试叙事

最强的面试表述是：

> 我没有做一个泛化聊天机器人，而是做了一个受控科研 Agent Harness，让 LLM 能够和卫星同化数据、模型训练脚本、评估程序和论文产物发生真实交互。随后我把可执行工作流轨迹作为后训练数据，使模型学到的不只是气象文本知识，而是完成 PASNet 类科研实验的操作过程。
