[English](README_EN.md) | 中文

# AUGUR：事件前历史证据与安全告警排序

代码、配置和结果摘要按目录提供；一键核验、逐表数据来源与当前复现范围见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

本制品整理 AUGUR 论文使用的核心状态读出、实体关系特征、分数融合、实验配置与部分复算结果。核心目标是在每条观察写入之前读取可用历史，并在会话判分时利用固定宽度的累计证据。

## 已包含

- state_124/：124 维累计历史读出的实现、字段说明和合成契约检查。
- current/：直接实体的 39 维事件前读出、主机与实体动作差值及三路同向合分代码。
- snapshots/：论文对应实验脚本快照，包括训练、受控表达编辑和局部判分的部分实现。脚本中的机器专属绝对路径已替换为占位符；原始算法未据此改写。
- legacy_auxiliary/：先前的反事实与局部证据代码，供追踪实现来源；具体版本和论文当前设置应以 configs/ 与 snapshots/ 为准。
- configs/ 与 results/：登记的实验方案、AIT 逐场景逐种子指标与 AIT、AutoLabel、CERT 的结果摘要。

## 快速核验

建议使用 Python 3.11 或更新版本与 NumPy 2.3.5。

    python state_124/state_contract_checks.py
    python test_current.py
    python recompute_ait.py

第一项和第二项使用合成输入，检查先读后写、有限关系和分数融合；它们不是安全检测效果实验。第三项依据已经导出的 AIT 场景指标，按场景和种子等权复算正文对应的 AP 与 Recall@1% FPR，不会重新训练模型。

## 数据与复现范围

AIT-ADS 的八个场景用于方法开发及场景轮换评估，每折训练、验证和测试按场景分配。6/1/1 与 4/2/2 各采用八折和五个随机种子。AutoLabel 与 CERT 使用各自任务的状态适配；它们并非直接运行 AIT 的完整三路集成。评价单位分别为会话、60 秒窗口和活动，不能直接将各任务的分数作为相同口径比较。

原始数据按提供方的访问方式和许可获取：

- AIT-ADS: https://zenodo.org/records/8263181
- AutoLabel: https://www.usenix.org/conference/usenixsecurity25/presentation/peng-yihao
- CERT Insider Threat Test Dataset: https://www.sei.cmu.edu/library/insider-threat-test-dataset/

**当前发布范围：** 本制品不包含原始数据、全部预处理缓存、训练模型或所有上游基线；snapshots/ 中部分脚本依赖这些缓存及先前实验目录，不能仅凭此仓库完成从原始数据到所有表格的端到端重跑。结果摘要提供可追踪的已导出指标，不应冒充完整的独立复现。复用第三方模型时请遵循相应原项目的许可证。

MANIFEST.sha256 记录发布文件的哈希。仓库中的模型快照、参数选择与适配口径请结合论文附录 B 阅读。

