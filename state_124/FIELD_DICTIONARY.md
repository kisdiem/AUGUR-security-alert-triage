# 主935维接口的字段和求值顺序

该文档说明已有实现，不定义新模型。可执行来源为`implementation/minimal_state_engine.py`、`graph_features.py`、`rotation_features.py`和`frozen_reproduction/run_activity_control.py`；对应字节哈希保留在输入清单。

## 1. 观察、身份与事件前读取

适配事件包含`event_id,timestamp_ns,actor,target,focus,entity_observations,action,outcome`，以及可选父实体、主机、账户、会话和来源。`focus`按直接实体的首次出现顺序去重，只有被当前观察明确提及且有实体描述的键更新次数。AIT实体键由场景、主机、类型和可观察标识派生；只有PID时不推断生命周期。

调用顺序固定为`keys, rows = env.before(event)`、将读出交给会话汇总、`env.apply(event)`。相同时间戳沿已登记的稳定来源顺序处理。标签不参与状态更新。

查询从`focus`按序开始，已存在的关系按维护器顺序至多扩展两跳，直接层最近两个有效对端也加入候选。队列去重且最多12个键；未观察键产生全零占位。最近明细仅在12小时内可见；累计计数持续维护。完整状态后端保存六条明细，紧凑累计维护器只保存维持相同选择器所需的两个对端。两者对124维读出的真实等价范围见附录F。

## 2. 每实体31维

索引0为是否已观察，未知实体全零。索引1为本实体观察次数；索引2为当前时间减首次观察的秒数；索引3为距最近观察的秒数。索引4—30为以下27类累计动作，顺序固定：

```text
unknown, exec, proc_spawn, file_read, file_write, file_create, file_delete,
object_open, object_read, object_write, object_create, object_delete, process_exit,
net_connect, net_send, net_receive, authentication, session_start, session_end,
account_observation, audit_path_observation, cwd_observation,
execution_arguments_observation, dns_query, dns_forward, dns_observation, http_request
```

除已见标志外，输入数值x先取非负，再按`min(log1p(x)/10,3)`缩放。节点向量按原缓存先转float16再转float32。计数仍保存在后端；投影本身经过量化和截断，不能据其反解任意完整历史。

动作由来源可核对字段映射：例如Linux x86-64审计的0/1/2或257/59/56—58/42分别映射读、写、打开、执行、派生和连接；PATH/CWD/EXECVE单列为路径、目录与参数观察。dnsmasq的query/forwarded映射查询/转发，HTTP访问和认证结果按具体记录提取。完整分支和未知映射保留在`adapters.py`，不把所有泛型告警自动解释为成功操作。

## 3. 从实体到124维历史摘要

每条告警对选中的实体31维分别求均值和最大值，得到62维。会话再对这些62维求均值和最大值。四个块的顺序为“会话均值×实体均值、会话均值×实体最大、会话最大×实体均值、会话最大×实体最大”。

先拼四块的前四个活动/时龄字段，得到16维；再拼四块的27类累计动作，得到108维。最终124维与当前811维拼接为935维。在线汇总使用float64累计和、float32最大值，输出按原缓存转为float32。功能样例检查在线与批量汇总相等；真实数据的等价性由另行封存的完整前缀核验支持。

丰富1643维原投影中的对应列也已固定：448维数值块从1195开始，四个节点池化偏移为1195、1276、1419、1500；活动取每个偏移的0—3，累计动作取8—34。主对照保留相同的28个分裂候选。

## 4. 当前811维与辅助路径

当前输入为384维签名向量、384维消息向量和43维DNS统计，来自当前可见会话，不是孤立的一条记录。冻结MiniLM、训练内IDF、最大长度和池化规则随正文第3.5节与附录F记录。DNS字段的具体计算函数在`features_dns.py`，局部20维选取及扫描40维在`extract_local_evidence.py`和`run_miss_experts.py`。

CF只在登记的两条DNS后缀规则及静态一致性条件下改写特征视图，不写回自然状态。规范化与CF代码、拒绝条件和非原地修改检查均在`strict_cf_norm.py`。这类检查验证程序条件，不证明真实目的端替换对攻击语义总是等价。
