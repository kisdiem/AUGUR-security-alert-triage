# AUGUR v56 · AIT 历史状态代码

本包摘录论文第 3 章所依据的 AIT 状态实现。原源码按字节收录，来源与 SHA-256 见 `SOURCE_MANIFEST.json`。包内包含源码、字段说明、合成检查和回放核验摘要。

## 建议阅读顺序

1. `implementation/alerts.py` 和 `implementation/adapters.py` 将 AMiner、Wazuh 观察转换为事件，定位直接出现的实体并映射行为。
2. `implementation/minimal_state_engine.py` 的 `MinimalEnvironment.before()` 读取事件前状态，从直接实体及其已观察关系选择实体。接着由 `apply()` 更新累计动作、活动时间、近期记录和关系槽位。会话汇总器独立于实体状态。
3. 同文件的 `SessionAccumulator` 对实体层、会话层分别取均值和最大值。 `features('activity_actions')` 输出 124 维，包括 16 个活动及时龄数值、108 个累计动作数值。字段和块顺序见 `FIELD_DICTIONARY.md`。
4. `implementation/engine.py` 是丰富状态实现，配套的 `graph_features.py` 和 `representation.py` 可用于对照原数值投影。`rotation_features.py` 记录了 AIT 特征构建的顺序，按 `t,id` 从场景缓存读入，先读取，再按会话汇总，最后更新实体状态。`frozen_reproduction/run_activity_control.py` 可核对 124 个历史数值在冻结特征中的选列方式。
5. `reference/ait_soc_v1/src/backend_replay_v2.py` 是完整场景回放和三种历史维护方式的对照脚本，`backend_complete.json` 是对应的原归档回执。这里的 `state_engine.py` 与 `minimal_state_engine.py` 字节一致。源码开头的早期说明可结合实际完成回执阅读。

## 在解压目录检查状态逻辑

```powershell
python -m pip install -r requirements-core.txt
python -B -X utf8 state_contract_checks.py
```

依赖已安装时直接运行第二条。检查涵盖事件前读取、累计动作跨越近期窗口、成功关系与未确认尝试的区分，以及在线汇总和独立批量汇总的 124 维一致性。原检查末尾还调用了一组规范化辅助检查，因此保留了 `strict_cf_norm.py`。

`verification/real_prefix_receipt.json` 是本次使用原归档重放 Fox 场景 256 条规范化告警得到的核验输出。包内的 `real_prefix_checks.py` 为对应检查入口，重新运行它需要原归档中的真实前缀样本和哈希回执。全场景重放及冻结特征构建脚本沿用 Linux 实验环境和上游项目路径，需要对应的原始数据、场景缓存及模型依赖。包内无需数据即可执行上面的合成检查。
