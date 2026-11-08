# VAD 配置、重启与训练时长分析

## 目录

1. 配置变更范围
2. 运行中任务重启
3. 重启后验收
4. 全量 VAD 基模从 1×8/accu2 到 2×8/accu1
5. Epoch 时长估算
6. 结果表达

## 1. 配置变更范围

先确认平台任务入口、共享 worktree 和 `CONFIG` 指向的确切 TOML；同名的 PSE 分支副本不是任务实际读取的那份配置。

调整 DataLoader 时：

1. 按用户要求决定只改 train，还是 train/valid/test 一起改。用户说“worker/prefetch 用某值”又没有限定范围时，默认三组统一。
2. 同时检查 `batch_size`、`num_workers`、`prefetch_factor`、`drop_last`、`pin_memory`，避免机械替换其他数值。
3. 用户已确认的模型与优化参数保持不变，例如 `block_size`、每 rank batch、全局 batch、optimizer LR 和 scheduler `max_lr`。
4. 更新 TOML 内的实验描述，使 workers/prefetch 与实际值一致。
5. 解析 TOML，对 train/valid/test 三组目标字段做显式断言；断言范围就到这三组字段，不加配置内容哈希、`asset_contract` 或通用 preflight。
6. 运行聚焦合同测试和 `git diff --check`。测试只是因为硬编码了旧配置值而失败时，要区分“陈旧字面值断言”和功能失败；未经授权时测试文件保持原样。
7. 授权包含提交和推送时，只暂存目标 TOML 和必要的支持文件，commit message 用中文，并核对远端 SHA。与本次无关的改动原样保留，不必为了工作区干净去处理它们。GitHub 写入只能在获准的服务器上实际执行，本机从不承担上传。

运行中的任务已经引用了 TOML，而训练进程启动后通常已经把 TOML 读进内存，改文件不会安全地影响当前进程，就地改还会让实验身份和实际跑的配置对不上。用户要求“调参数并重启同一任务”时，授权的是任务操作，不包括改动实验身份；做法是新建 successor TOML 和独立 `root_dir`，让重试或重新运行的任务显式指向新配置。

## 2. 运行中任务重启

按入口 Skill 的“先提交唯一后继任务，再依授权处理旧任务”顺序执行：

1. 在控制台确认完整旧任务 ID、实际 worker/rank 拓扑、状态和当前启动时间，记录代码版本、配置路径、最新 batch/epoch 与更替原因。
2. 核对 checkpoint 是否稳定且兼容，以及优化器、调度器和数据游标的恢复语义。`RESUME=auto` 要在所有 rank 上给出一致结果，各 rank 决定不同就说不清从哪一步恢复。launcher 实际使用 lease 时才检查该机制，没有 lease 的入口不需要额外加。
3. 完成独立 successor 配置、输出根和校验，并完成任务代码来源所要求的 commit/push 或部署快照核查；已运行的 TOML 不就地修改。
4. 提交一个优先级为 `6` 的后继任务，取得唯一完整 ID，再从详情核对实际入口、successor 配置、资源、挂载和优先级。新任务与旧任务共享输出时，提交前要验证新入口会等旧写入者释放，或者已有等效的写入互斥；确认不了就先解决这个互斥，先停旧任务不算替代方案。
5. 当前会话的授权只到提交后继任务时，旧任务保持原状；授权里明确包含停止这个具体旧任务时才停，停完核对最终状态、资源释放和全部写入者退出。停止备注如实写更替原因。
6. 核验新任务调度、恢复日志与真实 batch 推进。结果不确定时先刷新页面；重复点提交、重试或重新运行，只会多出任务。

平台的“停止 → 重试/重新运行”是旧操作路径，替代不了上面的顺序。当前功能做不到在停止旧任务之前建立后继任务时，保留旧任务并报告这条具体限制，不改用先停后试。新任务排队或已创建，都不等于训练已经恢复。

## 3. 重启后验收

证据齐全才算重启成功：

- 已记录唯一后继任务 ID，以及与旧任务的更替关系；详情中的入口、配置和代码版本匹配预期；
- 全部预期 worker 和 rank 出现并运行；
- 任务详情、实际复制的 TOML 和日志显示相同的目标拓扑；对下节全量 VAD 后继配方为 `2/8/16`，其他任务按实际合同核对；
- rank-0 DataLoader 摘要显示目标 batch、workers 和 prefetch；
- `RESUME=auto` 在所有 rank 得到一致决定，并从预期 epoch/batch cursor 恢复；
- 没有 NCCL 初始化失败、挂载缺失；launcher 使用 lease 时，再额外检查 lease 冲突。

同步任务描述中的代码版本、配置路径和 DataLoader 参数；平台不允许编辑时，交付里说明描述仍是旧值。

## 4. 全量 VAD 基模从 1×8/accu2 到 2×8/accu1

2026-09-08 全量双伪标签基模的迁移保持每卡 batch 48：

`8 × 48 × 2 = 16 × 48 × 1 = global batch 768`。

这只说明有效 global batch 相等，不说明每 epoch step 数逐个相等。对当前
10,261,016 条 train 数据、DistributedSampler 和 `drop_last=true`，旧配置约为
13,361 optimizer step/epoch，新配置的实跑日志为 13,360 batch/step；差的 1 来自
分片、drop-last 与尾累积取整。

改用独立 successor TOML、独立输出根、专用 2/8/16 launcher 和专用挂载入口。
launcher 只接受目标拓扑，其他拓扑不予启动，并断言 accumulation=1、batch/rank=48；带旧配置
或内容哈希/lease 的通用 launcher 不适用。rank 0 可以先做一次有界真实数据 smoke，另一个 worker
在 rendezvous 等待。

开发机只有两卡时，不声称做过真实 2×8 NCCL。提交前用两个 node-rank dry-run，
再用一个已验收 train 样本完成 feature、模型、BCE、AMP backward 和 optimizer
step。提交后从任务详情确认两个 worker 各 8 张 A800，从复制的 TOML 确认
2/8/16、batch48、accumulation1，从 rank-0 持久日志确认 `/13360`、batch 连续增长、
loss/grad norm 有限且没有 OOM/CUDA/NCCL/traceback。

### 同步 WSOLA successor

全量双伪标签基模要启用不带 `affine_resample` 的速度扰动时，已经运行的无速度扰动 TOML 和 launcher 都不动。改用独立配置
`conformer_noncausal_newbase_fullaudio_dualpseudomask_berp_rirgate_simu0803_wsolasync_volcengine_2x8_noaccum.toml`
和独立入口：

`platform_entrypoint_2x8_newbase_fullaudio_rt60gate_wsolasync_noaccum.sh` →
`train_volcengine_2x8_newbase_fullaudio_rt60gate_wsolasync_noaccum.sh`。

用户把 vePFS 挂到 `/root` 时，外置组件
`/root/code/tools/sox_sync_wsola/bin/sox_sync_wsola` 会随代码盘在两个 worker
上可见。开发机验证过也只覆盖开发机这一台；两个 worker 各自要在 CUDA runtime 和 rendezvous
之前独立检查：路径可执行、版本精确等于 `sox_sync_wsola 1.0.0`、TOML train
模式为 `wsola_audio_driven_categorical_mask`、train 概率 `0.7`、valid 概率
`0.0`。任一节点不满足即 fail-closed；单节点通过不能说明另一节点也装了组件。

提交后除 2/8/16、batch48、accumulation1 之外，还要从两个 worker 的日志分别
取得 `[wsola] node=0/2` 与 `[wsola] node=1/2` 的证据，再从 rank-0 Dataset 日志
确认同步模式实际初始化，最后要求至少两个单调增加的训练日志点和有限 loss/grad
norm。2026-09-08 的一次已验证切换中，新任务先创建并取得 ID，再停止旧任务；
两个 worker 都通过组件检查并跑到 Batch 0→10。这条记录只证明当时的环境，
替代不了后续任务的实时检查。

## 5. Epoch 时长估算

VAD 常见日志格式：

```text
Epoch 1, Batch 460/5395, Loss: ..., LR: ..., GradNorm: ..., Acc: ...
```

按以下步骤计算：

1. 先按 restart/run 分段，默认取日志中最后一个连续 run，再取该 run 最后出现的 epoch 和 batch 单调增加段。历史最大 epoch 代替不了最新 run。
2. 双层时间戳默认取原始 rank-0 logger 的内层时间；平台外层可能是采集或展示时间。用 `--timestamp-source outer|inner|index:N` 明确选择，并用 `--line-filter` 固定一个 worker/rank，避免重复的 batch 行。
3. 每轮 batch 数直接取日志里的 `/B`，不用 scheduler 周期替代。当前 trainer 的 batch index 为 `0..B-1`；处理完 batch `b` 后剩余 `B-1-b`。
4. 日志每 10 batch 输出一次时，单个相邻间隔容易因为 DataLoader 预取呈现快慢交替；普通中位数把两种间隔混在一起，不能代表训练速度。
5. 计算全观察区间平均秒/batch、50-batch 滚动窗口中位数、100-batch 窗口中位数和最近约 200 batch 的长期平均。
6. 用 `scripts/analyze_vad_log.py <log> --timestamp-source inner --line-filter '<rank0-pattern>'` 得到一致结果。

公式：

```text
epoch_seconds = total_batches × seconds_per_batch
remaining_seconds = (total_batches - 1 - current_batch) × seconds_per_batch  # 已处理0-based batch b后
```

当 `gradient_accumulation_steps=1` 时 batch 与 optimizer step 一一对应；梯度累积大于 1 时，wall-clock epoch 仍按 dataloader batch 数估算，但学习率和保存频率要另按 optimizer step 解释。

轮末 validation、checkpoint、首次 DataLoader 初始化和平台调度不包含在纯训练速度内，要单独留余量。

## 6. 结果表达

先给中心值和范围，再给证据：

- “一个 epoch 约 12 小时 20 分，建议按 12～12.7 小时估算”；
- 报告起止 batch、观察时长、长期秒/batch、窗口稳健值；
- 用最新 batch 计算剩余时间和预计完成时刻；
- 说明 checkpoint/validation 可能再增加几分钟；
- 日志区间过短或跨越重启时，只报告观测到的吞吐，不给虚假精确的 ETA。

实测示例：Batch 460→990 共 530 batch、耗时约 4448 秒，长期平均约 8.39 秒/batch；最近 200 batch 约 8.00 秒/batch。对 5395 batch/epoch，纯训练约 12.0～12.6 小时。
