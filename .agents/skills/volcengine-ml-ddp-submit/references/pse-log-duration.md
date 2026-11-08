# PSE 训练日志时长分析

这些方法用来估算单轮/epoch 时长，或判断平台任务是否正常推进。

## 选择权威日志

时长以共享输出文件系统上的 rank-0 持久日志为准。过期的 Edge 页面不参与计算，平台任务的总运行时间也不直接拿来算。先在 worker 或代码开发机定位确切 run：

```bash
base=/root/group-shared/voiceprint/zhounan/workspace/speech_enhancement/pse/<experiment-root>
find "$base" -type f -path '*/logs/log-*.txt' -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort
```

选择开始时间和实验目录与当前任务一致的日志。一个实验目录可能同时包含旧日志、活跃日志、TensorBoard 文件和 `latest_model.tar`；混用不同 run 的时间戳，算出来的每步时间不对应任何一个 run。

## 解析正确时间戳

时间戳取 `[TrainState] step=N` 行：这些行按固定间隔写入，代表 global optimizer step。同一步的 `grad norm` 行不是第二个样本，不参与计算：同一步写两行会让样本数虚高。动态 batch sampler 和 resume offset 会让 `Batch` 与 `Step` 不一致，因此吞吐不从 `Batch` 推断。按文件顺序切分 restart/run，只分析最后一个连续段；该段不足两个单调样本时报告“不足”，不退回旧 run 冒充当前进度，也不按 step 全局排序后拼接多个 run。

至少报告五项：

- 第一个实际训练 step 与最新 step/时间；
- 全观察区间的“总耗时 ÷ step 差值”；
- 相邻日志样本每步时间的中位数或截尾中位数；
- 最近 1,000–5,000 step 的吞吐；
- 明显慢区间，以及是否对应 checkpoint、validation、NaN telemetry 或动态时长 batch。

短期稳健值通常取 50-step 区间的中位数。长期加权值更适合 ETA，因为它包含周期性 data-loader 卡顿。两者差异明显时给出范围，不报告看起来精确的单一时间。

## 确定每轮 step 数

每轮 step 数按以下优先级取证：

1. 启动 stdout 中的 `[DataLoader] ...` 或 `batches/epoch=`；
2. 同一日志中已观察到的 `Epoch=1` → `Epoch=2` 转换；
3. 参考历史 run，仅在数据集、配置和 world size 可证明完全相同时。

PSE logger 文件可能包含模型参数和训练指标，但没有 Python 启动阶段的 `print()`。缺少 `batches/epoch` 时，在报告里写明这一点，并把 step 数标为估算值。

`first_cycle_steps` 是学习率调度周期，不是每轮 step 数。配置没有明确把该周期定义为固定 epoch 数时，拿它除以 `epochs` 得到的每轮步数没有依据。

当前动态 sampler 在 `gradient_accumulation_steps=1` 时，optimizer step 约等于 dataloader batch 数。梯度累积大于 1 时按下式换算：

```text
optimizer_steps_per_epoch = ceil(batches_per_epoch / gradient_accumulation_steps)  # 仅当trainer安全处理尾组时
```

如果 trainer 直接丢弃尾组、恢复了半组或存在跳过更新，按实际实现与日志修正；这个除法只在 trainer 安全处理尾组时成立，单凭它认定训练等价会算错每轮步数。

## 估算完成时间

设每轮估算 optimizer step 为 `S_epoch`，当前 global step 为 `S_now`，当前 epoch 起点 global step 为 `S_start`，实测每步秒数为 `t`：

```text
remaining_seconds ≈ max(S_epoch - (S_now - S_start), 0) × t
epoch_seconds ≈ S_epoch × t + startup_overhead
```

不能从 checkpoint 或日志证明 `S_start` 时，只报告吞吐，不用 `% S_epoch` 猜。日志时间戳来自 naive `datetime.now()` 时时区未知，不默认按 UTC 处理；只有容器或日志元数据能证明时区时，才通过 `--log-timezone` 指定。

报告中心 ETA 和范围，并单独提示轮末验证、checkpoint 保存和平台日志刷新这三项开销：常规 step 时间没有涵盖它们。ETA 由日志里实测的每步时间推算；多轮总时长可以先用纯训练估算乘轮数，再说明验证和保存会把 wall-clock 拉长。

## 同时执行健康检查

时长本身不能证明任务健康。同一个报告里还要给出最新 `sv_arc_margin`、AMP scaler scale、`skipped_optimizer_updates`、`NanGradTelemetry` 和 epoch marker。少量早期 skipped updates 可能轻微改变平均值，但不构成预加载失败的证据；持续增长的 skip count 属于另一类数值问题。

运行：

```bash
python3 scripts/analyze_pse_log.py <log> \
  --steps-per-epoch <N> \
  --epoch-start-step <S_start> \
  --log-timezone <UTC-or-Asia/Shanghai>
```

不知道每轮 step 数或当前 epoch 起点时，省略对应参数，只报告观测吞吐；自动补值不行，`first_cycle_steps` 不是每轮步数。
