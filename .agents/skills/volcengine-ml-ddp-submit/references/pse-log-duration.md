# PSE 训练日志时长分析

用于估算单轮/epoch 时长，或判断平台任务是否正常推进。

## 选择权威日志

优先使用共享输出文件系统上的 rank-0 持久日志，不依赖过期的 Edge 页面，也不使用平台任务总运行时直接计算。先在 worker 或代码开发机定位确切 run：

```bash
base=/root/group-shared/voiceprint/zhounan/workspace/speech_enhancement/pse/<experiment-root>
find "$base" -type f -path '*/logs/log-*.txt' -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort
```

选择开始时间和实验目录与当前任务一致的日志。一个实验目录可能同时包含旧日志、活跃日志、TensorBoard 文件和 `latest_model.tar`，不得混用不同 run 的时间戳。

## 解析正确时间戳

使用 `[TrainState] step=N` 行。它们按固定间隔写入并代表 global optimizer step。不要把同一步的 `grad norm` 行作为第二个样本；动态 batch sampler 和 resume offset 可能使 `Batch` 与 `Step` 不一致，因此不要用 `Batch` 推断吞吐。按文件顺序切分 restart/run，只分析最后一个连续段；该段不足两个单调样本时报告不足，不退回旧run冒充当前进度，绝不能按 step 全局排序后拼接多个 run。

至少报告：

- 第一个实际训练 step 与最新 step/时间；
- 全观察区间的“总耗时 ÷ step 差值”；
- 相邻日志样本每步时间的中位数或截尾中位数；
- 最近 1,000–5,000 step 的吞吐；
- 明显慢区间，以及是否对应 checkpoint、validation、NaN telemetry 或动态时长 batch。

短期稳健值通常取 50-step 区间的中位数；长期加权值更适合 ETA，因为包含周期性 data-loader 卡顿。两者差异明显时给出范围，不报告虚假的精确时间。

## 确定每轮 step 数

按以下优先级取证：

1. 启动 stdout 中的 `[DataLoader] ...` 或 `batches/epoch=`；
2. 同一日志中已观察到的 `Epoch=1` → `Epoch=2` 转换；
3. 仅当数据集、配置和 world size 可证明完全相同时，才参考历史 run。

PSE logger 文件可能包含模型参数和训练指标，但没有 Python 启动阶段的 `print()`。缺少 `batches/epoch` 时必须明确说明，并把 step 数标为估算值。

不要把 `first_cycle_steps` 当作每轮 step 数；它是学习率调度周期。除非配置明确将该周期定义为固定 epoch 数，也不要用它除以 `epochs`。

当前动态 sampler 在 `gradient_accumulation_steps=1` 时，optimizer step 约等于 dataloader batch 数。梯度累积大于 1 时使用：

```text
optimizer_steps_per_epoch = ceil(batches_per_epoch / gradient_accumulation_steps)  # 仅当trainer安全处理尾组时
```

如果trainer直接丢弃尾组、恢复了半组或存在跳过更新，按实际实现与日志修正，不仅凭这个除法认定训练等价。

## 估算完成时间

设每轮估算 optimizer step 为 `S_epoch`，当前 global step 为 `S_now`，当前 epoch 起点 global step 为 `S_start`，实测每步秒数为 `t`：

```text
remaining_seconds ≈ max(S_epoch - (S_now - S_start), 0) × t
epoch_seconds ≈ S_epoch × t + startup_overhead
```

如果不能从 checkpoint/日志证明 `S_start`，只报告吞吐，不用 `% S_epoch` 猜测。日志时间戳来自 naive `datetime.now()` 时，时区未知；不要无条件当 UTC。仅在容器/日志元数据证明后通过 `--log-timezone` 指定。

报告中心 ETA 和范围，并单独提示轮末验证、checkpoint 保存和平台日志刷新开销；常规 step 时间没有覆盖这些部分。多轮总时长可先乘纯训练估算，再说明验证和保存会延长 wall-clock 时间。

## 同时执行健康检查

时长不能单独证明任务健康。同步报告最新 `sv_arc_margin`、AMP scaler scale、`skipped_optimizer_updates`、`NanGradTelemetry` 和 epoch marker。少量早期 skipped updates 可能轻微改变平均值，但不是预加载失败证据；持续增长的 skip count 属于另一类数值问题。

运行：

```bash
python3 scripts/analyze_pse_log.py <log> \
  --steps-per-epoch <N> \
  --epoch-start-step <S_start> \
  --log-timezone <UTC-or-Asia/Shanghai>
```

不知道每轮 step 数或当前 epoch 起点时，省略对应参数并只报告观测吞吐，不能用 `first_cycle_steps` 自动补值。
