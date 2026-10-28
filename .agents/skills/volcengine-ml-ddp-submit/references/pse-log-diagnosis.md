# PSE 启动日志诊断

## `sv_ce` 排查顺序

首个 batch 的 `speaker/sv_ce` 很高时：

1. 读取完整继承后的有效 TOML；
2. 检查预加载摘要中 `sv_arc_margin.weight` 的状态；
3. 核对 checkpoint 的 speaker 数量与 class-ID vocabulary；
4. 分开报告原始 `speaker/sv_ce` 与加权后的 `speaker/sv_ce`，warmup 可能让加权项很小；
5. 检查 `sv_margin` 调度；从更晚 step 才开始的 margin ramp 不能解释 margin 为零时的高 CE。

当类别数约为 22,281 时，均匀预测基线为 `ln(N)≈10.01`。首值约 12–13 与随机或不兼容的分类头相符，但不能单独证明 extraction 网络损坏。日志中的 `sv_arc_margin.weight:True` 只表示 `requires_grad=True`。

## 完整预加载能证明什么

加载父 PSE checkpoint 可能保留良好的重建起点，同时让 Shared SV ArcFace 头保持随机。此时可能出现：

- reconstruction 指标看起来已继承；
- 第一批 `sv_ce` 激增；
- 在线 speaker condition 与 ArcFace 目标不再处于父模型坐标系；
- 非严格加载不抛出异常。

应修复后继配置并按授权重启/重提；不要通过原地修改正在运行任务使用的 TOML 来补救。

## 其他信号的边界

- `identity/target_cos`、`identity/interferer_cos` 和 `identity/margin_violation_rate` 诊断的是 frozen verifier 路径，不等价于 ArcFace 分类头加载状态。
- `identity/to_reconstruction_ratio` 可能因为带符号 SNR reconstruction 总和接近零而激增；它不是梯度范数比。
- `requires_grad=True` 对联合训练的 extraction/SV 参数是预期值，既不能证明加载成功，也不能证明数值健康。
- FP16 speaker backward 出现非有限梯度时，先检查 GradScaler scale 与 skipped updates；不能只凭这一点归因于 checkpoint 加载失败。

## 重提任务的最低验收标准

在新任务启动段捕获并确认：

```text
effective preloaded_embedding_skip_prefixes = []
sv_arc_margin.weight is loaded
no unexpected missing embedding keys
the first sv_ce is interpreted together with n_speakers and warmup
```
