[English](README_EN.md) | 中文

# AUGUR：持续历史状态与安全告警会话排序

本制品将持续维护的历史状态、数值判分路径和封存分数复算分开提供。
历史快照保留在 `snapshots/`；新增可移植入口在 `portable/`。

## 一键复现

Python 3.11 以上；完整本地验证环境为 Python 3.14。

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-reproduction.txt
python run_reproduction.py --output-dir reproduced --include-training
```

此命令执行：状态契约和流式回放测试；36 行历史结果复算；原发布分数的
240 个验证阈值复算；80 组直接实体开关复算；80 组原表达匹配对照复算；
一个 4/2/2 折/种子的数值推理；并从原准备矩阵重新训练该单元的三路主分类器和三个协议专家。
不加 `--include-training` 时省略重训。运行输出写入指定目录，不能用作原始封存结果。

## 已包含

- `state_124/`、`current/`：累计读出、直接实体读出、动作差值及融合实现。
- `portable/replay.py`：已适配、有序观察 → 主机/实体状态特征。持续状态不会因会话关闭而清零。
- `portable/model.py`、`infer.py`、`train.py`：原数值特征 → 三路模型、局部专家、实体开关与融合。
- `portable/data/`：80 组开关和80组原表达匹配对照的逐例验证/测试分数、来源哈希和数据署名。
- `portable/fixtures/numeric422/`：一个单元的冻结模型、数值训练/验证输入和测试推理输入。
- `portable/frozen_splits.json`：从封存队列核对的场景分配与五个固定种子。
- `portable/verification/`：本次干净目录验证的紧凑收据。
- `snapshots/`、`configs/`、`results/`：不改动的研究快照、登记配置与原有结果。

## 范围

提供的是三层可执行证据：有序适配观察的状态维护；一个数值单元的六个估计器重训/推理；
全部80组分数到指标和队列的复算。数值单元不是全部80组重新训练。
它从原准备矩阵开始，不包含原始 AIT 文件到文本编码/标签对齐的完整独立重建。
其余基线、AutoLabel/CERT 的逐例输入及完整服务部署也没有因此变成可复现。

AP 与 Recall@1% FPR 是排序指标；实际送审阈值仅从验证集选择并在测试前冻结。
4/2/2 同一场景测试两次，计数中的出现次数不等于独立会话或独立环境数量。
数据、命令、许可与逐项范围见 [portable/README.md](portable/README.md) 和 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。
