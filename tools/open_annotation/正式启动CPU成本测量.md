# 正式启动前的 CPU 成本测量

`measure_native_startup_cost.py` 只测真实有限证明的读取与检查成本，不生成
正式准入、GPU 通过或任务接续许可。默认检查不会导入选型模块、创建输出
或启动子进程；必须显式 `--execute` 才开始测量。

输入计划格式为 `annotation-native-startup-cost-plan/v1`，精确包含：

- `selected`：实际 selected admission 的绝对路径及 SHA。
- `formal_contract`：实际正式合同的绝对路径及 SHA。
- `formal_bundle`：已审正式冻结包 `bundle_files.json` 的路径及 SHA。
- `code_root`：该正式冻结包的代码根。
- `output_root`：代码根旁边全新的 `cpu-startup-cost-*` 目录。
- `max_seconds`：1 至 900 的整数预算。
- `schema`：上述格式名。

三个来源 track 与 index 都先核摘要、分片计数守恒及每轨最多 4096 来源。
只有这些固定索引列出的有限 JSONL 分片可以读取；不会扫描正式全库。
所有代码成员逐 SHA 检查；禁止把音频、权重或数据库文件当作代码成员。
输出也不能在任一已有 queue、labels、resources 或源码目录内。

执行顺序固定，不能由计划减少次数：

1. 从已核代码根导入实际 `native_batch_admission` 和 `bounded_speed_runtime`。
2. 用 `NativeSelectionScope(enabled=True)` 完整检查选型一次。
3. 同一个 scope 连续复用八次，每次仍逐文件检查真实内容 SHA。
4. 用实际 `M._snapshot_verified_pins` 建立有限输入快照。
5. 用实际 `M._verify_snapshot` 连续检查八次，保留全部 SHA 和目录身份检查。

工具不构造 RuntimeLease、生产 memfd、锁或 claim。SQLite 标准库可以导入，
但连接数据库、加载模型、读取原音、回读原侧车、写入既有输入、派生进程和
网络连接均拒绝。子进程为单 CPU、nice15、CUDA 不可见；控制器只清理自己
尚未回收的精确子进程。阶段开始即保存进度，正常、异常和超时都保留日志。
不使用曾发生崩溃的 `dump_traceback_later` 异步栈采集。

维护版在 Popen 后保存 `child-identity.json`，绑定本次父子 PID、实际父关系、
`/proc/PID/stat` 的启动时钟及 boot ID。诊断只能依据这份自有身份读目标
进程，不能为找 PID 枚举全机；实际清理仍使用本次尚未回收的 Popen 子进程。
最初 `2f5e...` 运行没有这份身份，不在其执行副本上补写或猜测。

每阶段记录实际时间、唯一输入文件数、唯一字节数和累计峰值 RSS。
“唯一字节数”不是实际 I/O 总字节数，完整检查可能多次读取同一文件。
另记录八次串行总耗时和后七次底层检查耗时，帮助识别首个 worker 等待
20 秒 ACK 的风险。这个数不含 RuntimeLease 的其它小控制检查、父子身份、
锁、派生进程、GPU 上下文和 ACK 本身：小于 20 秒也不能声称真实启动通过，
更不能因此放宽 20、120 或 900 秒的既有时限。

先运行 `--plan <计划> --plan-sha <SHA>` 做默认检查。独立审阅并确认实际
材料后，再对同一计划加 `--execute`。失败时保留原目录，不覆盖、不自动重试。

第一次真实测量使用正式包 `763f44fc...`，完整选型检查在 196.554 秒后拒绝，
没有进入八次复用或底层校验，不能据此报告它们为零成本。父实际等待
198.443 秒，子进程用户 CPU 为 153.087 秒、系统 CPU 为 10.791 秒，峰值内存
3733504000 字节；不是 900 秒超时，也不是额外只读守卫拒绝。
完整窗口比较已经相同，随后发现原报告的 decision_counts 省略未出现的
failed 键，而读回辅助重建固定插入 failed:0，导致字典表示不等。原报告、
旧正式包与本次失败不改写；需另审辅助代码及真实零键反例后，再由操作者
明确安排新版本测量。

计数表示修复现已完成独立复验。`native_batch_readback.py` 的新版本
`8169fbcc...` 从实际记录生成 `dict(Counter(states))`，只允许
`candidate`、`unknown`、`failed` 三种真实状态；未出现的状态不插入零键。
两份报告的计数字段仍按完整 JSON 表示精确比较，不接受额外零键、未知键，
也不接受用浮点数或布尔值冒充整数。队列槽位原有的三键格式不改。
独立合成对照保留旧 `3412f748...` 的实际失败，并验证新版本通过；两边共
12 种缺失、错误、补零及类型伪装和 6 类非法状态均被拒绝。相关 120 项
CPU 测试在普通模式与 `-O` 模式均通过。它们证明这一表示错误已修复，
不代表第二次真实成本已经测过；新的正式包、选型材料和新输出目录仍须
另行固定，旧失败目录保持不变。

固定计划字段可参照[不可执行模板](templates/native-startup-cost-plan.example.json)，所有占位SHA须替换为实际已核原件；正式根下每次使用独立`cpu-startup-cost-*`目录，不覆盖旧失败。当前维护版已补父子PID、Linux启动时钟及boot身份，Popen后记录`child-identity.json`，便于只读定位自己的CPU诊断进程；首次2f5e副本没有这些字段，其原件保留，不事后补写。
