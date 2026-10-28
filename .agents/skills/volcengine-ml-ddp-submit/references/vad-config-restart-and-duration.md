# VAD 配置、重启与训练时长分析

## 目录

1. 配置变更范围
2. 运行中任务重启
3. 重启后验收
4. 全量 VAD 基模从 1×8/accu2 到 2×8/accu1
5. Epoch 时长估算
6. 结果表达

## 1. 配置变更范围

先确认平台任务入口、共享 worktree 和 `CONFIG` 指向的确切 TOML。不要把同名 PSE 分支副本当成任务实际配置。

调整 DataLoader 时：

1. 按用户要求决定只改 train，还是统一修改 train/valid/test；用户说“worker/prefetch 用某值”且未限定时，默认三者统一。
2. 同时检查 `batch_size`、`num_workers`、`prefetch_factor`、`drop_last`、`pin_memory`，避免机械替换其他数值。
3. 保持用户已确认的模型与优化参数不变，例如 `block_size`、每 rank batch、全局 batch、optimizer LR 和 scheduler `max_lr`。
4. 更新 TOML 内的实验描述，使 workers/prefetch 与实际值一致。
5. 解析 TOML 并对 train/valid/test 三组目标字段做显式断言；不要添加配置内容哈希、`asset_contract` 或通用 preflight。
6. 运行聚焦合同测试和 `git diff --check`。若测试只因硬编码旧配置值失败，明确区分“陈旧字面值断言”和功能失败；未经授权不要顺带修改测试。
7. 只暂存目标 TOML，使用中文 commit message，推送后证明 `HEAD == upstream` 且工作区干净。

运行任务已经引用的 TOML 不可原地修改。用户要求“调参数并重启同一任务”授权的是任务操作，不自动授权破坏实验身份；应新建 successor TOML 和独立 `root_dir`，让重试/重新运行的任务显式指向新配置。训练进程启动后通常已将 TOML 读入内存，文件变化本来也不会安全改变当前进程。

## 2. 运行中任务重启

按入口 Skill 的“先提交唯一后继任务，再依授权处理旧任务”顺序执行：

1. 在控制台确认完整旧任务 ID、实际 worker/rank 拓扑、状态和当前启动时间，记录代码版本、配置路径、最新 batch/epoch 与更替原因。
2. 核对稳定且兼容的 checkpoint、优化器/调度器与数据游标恢复语义；`RESUME=auto` 必须在所有 rank 一致。仅当实际 launcher 使用 lease 时检查该机制，不给无 lease 的入口强加要求。
3. 完成独立 successor 配置、输出根、校验，以及任务代码来源所要求的 commit/push 或部署快照核查；不原地修改已运行的 TOML。
4. 提交一个优先级为 `6` 的后继任务，取得唯一完整 ID，并从详情核对实际入口、successor 配置、资源、挂载和优先级。若共享输出，必须在提交前验证新入口会等旧写入者释放，或已有等效的写入互斥；无法确保时先解决，不以先停旧任务规避。
5. 只有当前会话授权包含停止精确旧任务时才停止，并核对最终状态、资源与全部写入者释放。停止备注准确记录更替原因；用户要求保留旧任务时保持原状态。
6. 核验新任务调度、恢复日志与真实 batch 推进；结果不确定时先刷新，不重复点击提交、重试或重新运行。

平台“停止 → 重试/重新运行”的旧操作路径不能替代上述顺序。若当前功能无法在停止旧任务前建立后继任务，保留旧任务并报告具体限制，不自行改用先停后试。新任务排队或已创建不等于训练已恢复。

## 3. 重启后验收

只有以下证据齐全时才报告重启成功：

- 已记录唯一后继任务 ID，以及与旧任务的更替关系；详情中的入口、配置和代码版本匹配预期；
- 全部预期 worker 和 rank 出现并运行；
- 任务详情、实际复制的 TOML 和日志显示相同的目标拓扑；对下节全量 VAD 后继配方为 `2/8/16`，其他任务按实际合同核对；
- rank-0 DataLoader 摘要显示目标 batch、workers 和 prefetch；
- `RESUME=auto` 在所有 rank 得到一致决定，并从预期 epoch/batch cursor 恢复；
- 没有 NCCL 初始化失败、挂载缺失；如果 launcher 使用 lease，再额外检查 lease 冲突。

同步任务描述中的代码版本、配置路径和 DataLoader 参数；若平台不允许编辑，交付时明确描述仍是旧值。

## 4. 全量 VAD 基模从 1×8/accu2 到 2×8/accu1

当前全量双伪标签基模的迁移保持每卡 batch 48：

`8 × 48 × 2 = 16 × 48 × 1 = global batch 768`。

这只证明有效 global batch 相等，不证明每 epoch step 数逐个相等。对当前
10,261,016 条 train 数据、DistributedSampler 和 `drop_last=true`，旧配置约为
13,361 optimizer step/epoch，新配置的实跑日志为 13,360 batch/step；差 1 来自
分片、drop-last 与尾累积取整。

使用独立 successor TOML、独立输出根、专用 2/8/16 launcher 和专用挂载入口。
launcher 必须拒绝其他拓扑，断言 accumulation=1、batch/rank=48；不调用带旧配置
或内容哈希/lease 的通用 launcher。rank 0 可先做有界真实数据 smoke，另一个 worker
在 rendezvous 等待。

开发机只有两卡时，不声称做过真实 2×8 NCCL。提交前用两个 node-rank dry-run，
再用一个已验收 train 样本完成 feature、模型、BCE、AMP backward 和 optimizer
step。提交后从任务详情确认两个 worker 各 8 张 A800，从复制的 TOML 确认
2/8/16、batch48、accumulation1，从 rank-0 持久日志确认 `/13360`、batch 连续增长、
loss/grad norm 有限且没有 OOM/CUDA/NCCL/traceback。

### 同步 WSOLA successor

全量双伪标签基模需要启用无 `affine_resample` 的速度扰动时，不修改已经运行的
无速度扰动 TOML 或 launcher。使用独立配置
`conformer_noncausal_newbase_fullaudio_dualpseudomask_berp_rirgate_simu0803_wsolasync_volcengine_2x8_noaccum.toml`
和独立入口：

`platform_entrypoint_2x8_newbase_fullaudio_rt60gate_wsolasync_noaccum.sh` →
`train_volcengine_2x8_newbase_fullaudio_rt60gate_wsolasync_noaccum.sh`。

用户 vePFS 挂载到 `/root` 时，外置组件
`/root/code/tools/sox_sync_wsola/bin/sox_sync_wsola` 会随代码盘在两个 worker
可见。即便开发机已验证，也必须让每个 worker 在 CUDA runtime 和 rendezvous
之前独立检查：路径可执行、版本精确等于 `sox_sync_wsola 1.0.0`、TOML train
模式为 `wsola_audio_driven_categorical_mask`、train 概率 `0.7`、valid 概率
`0.0`。任一节点不满足即 fail-closed，不能让单节点成功掩盖另一节点缺组件。

提交后除 2/8/16、batch48、accumulation1 外，还要从两个 worker 的日志分别
取得 `[wsola] node=0/2` 与 `[wsola] node=1/2` 证据，再从 rank-0 Dataset 日志
确认同步模式实际初始化。最后要求至少两个单调增加的训练日志点和有限 loss/grad
norm。2026-09-08 的一次已验证切换中，新任务先创建并取得 ID，再停止旧任务；
新任务两个 worker 均通过组件检查并运行到 Batch 0→10。该记录只证明当时环境，
不能替代后续任务的实时检查。

## 5. Epoch 时长估算

VAD 常见日志格式：

```text
Epoch 1, Batch 460/5395, Loss: ..., LR: ..., GradNorm: ..., Acc: ...
```

按以下步骤计算：

1. 先按 restart/run 分段，默认使用日志中最后一个连续 run，再取该 run 最后出现的 epoch 和 batch 单调增加段；不能用历史最大 epoch 代替最新 run。
2. 双层时间戳默认选择原始 rank-0 logger 的内层时间；平台外层可能是采集/展示时间。用 `--timestamp-source outer|inner|index:N` 明确选择，并用 `--line-filter` 固定一个 worker/rank，避免重复 batch 行。
3. 从日志中的 `/B` 直接取得每轮 batch 数，不用 scheduler 周期替代。当前 trainer 的 batch index 为 `0..B-1`；处理完 batch `b` 后剩余 `B-1-b`。
4. 日志每 10 batch 输出时，单个相邻间隔容易因 DataLoader 预取呈现快慢交替；不要使用这些间隔的普通中位数。
5. 计算全观察区间平均秒/batch、50-batch 滚动窗口中位数、100-batch 窗口中位数和最近约 200 batch 的长期平均。
6. 使用 `scripts/analyze_vad_log.py <log> --timestamp-source inner --line-filter '<rank0-pattern>'` 得到一致结果。

公式：

```text
epoch_seconds = total_batches × seconds_per_batch
remaining_seconds = (total_batches - current_batch) × seconds_per_batch
```

当 `gradient_accumulation_steps=1` 时 batch 与 optimizer step 一一对应；即使梯度累积大于 1，wall-clock epoch 仍按 dataloader batch 数估算，但学习率/保存频率要另按 optimizer step 解释。

轮末 validation、checkpoint、首次 DataLoader 初始化和平台调度不包含在纯训练速度内，要单独增加余量。

## 6. 结果表达

先给中心值和范围，再给证据：

- “一个 epoch 约 12 小时 20 分，建议按 12～12.7 小时估算”；
- 报告起止 batch、观察时长、长期秒/batch、窗口稳健值；
- 用最新 batch 计算剩余时间和预计完成时刻；
- 说明 checkpoint/validation 可能再增加几分钟；
- 日志区间过短或跨越重启时，只报告观测吞吐，不给虚假精确 ETA。

实测示例：Batch 460→990 共 530 batch、耗时约 4448 秒，长期平均约 8.39 秒/batch；最近 200 batch 约 8.00 秒/batch。对 5395 batch/epoch，纯训练约 12.0～12.6 小时。
