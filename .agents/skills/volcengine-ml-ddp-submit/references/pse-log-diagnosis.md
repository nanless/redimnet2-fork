# PSE 启动日志诊断

## `sv_ce` 排查顺序

首个 batch 的 `speaker/sv_ce` 很高时，按下面的顺序查：

1. 读取完整继承后的有效 TOML；
2. 检查预加载摘要中 `sv_arc_margin.weight` 的状态；
3. 核对 checkpoint 的 speaker 数量与 class-ID vocabulary；
4. 原始 `speaker/sv_ce` 与加权后的 `speaker/sv_ce` 分开报告：warmup 期间加权项本身很小，两个值合在一起看不出原始值偏高；
5. 检查 `sv_margin` 调度；margin ramp 从更晚的 step 才开始时，它在这一段是零，这段的高 CE 与 ramp 无关。

当类别数约为 22,281 时，均匀预测基线为 `ln(N)≈10.01`。首值约 12–13 来自日志，与随机初始化或不兼容的分类头相符；这是推断，单凭这个数值证明不了 extraction 网络损坏。日志中的 `sv_arc_margin.weight:True` 只表示 `requires_grad=True`。

## 完整预加载能证明什么

加载父 PSE checkpoint 可能保留良好的重建起点，同时让 Shared SV ArcFace 头保持随机。这时会出现下面四种现象：

- reconstruction 指标看起来已继承；
- 第一批 `sv_ce` 激增；
- 在线 speaker condition 与 ArcFace 目标不再处于父模型坐标系；
- 非严格加载不抛出异常。

修复后继配置后按授权重启或重提任务。原地改正在运行任务使用的 TOML 补不了这个问题：改动到不了已经启动的进程，只会让磁盘配置和正在跑的训练不一致。

## 其他信号的边界

- `identity/target_cos`、`identity/interferer_cos` 和 `identity/margin_violation_rate` 诊断的是 frozen verifier 路径，与 ArcFace 分类头的加载状态不是一回事。
- `identity/to_reconstruction_ratio` 可能因为带符号 SNR reconstruction 总和接近零而激增；它与梯度范数的比值无关。
- `requires_grad=True` 对联合训练的 extraction/SV 参数是预期值，既证明不了加载成功，也证明不了数值健康。
- FP16 speaker backward 出现非有限梯度时，先检查 GradScaler scale 与 skipped updates；只有这一条证据不足以把原因归到 checkpoint 加载失败。

## 重提任务的最低验收标准

新任务启动段要捕获并确认：

```text
effective preloaded_embedding_skip_prefixes = []
sv_arc_margin.weight is loaded
no unexpected missing embedding keys
the first sv_ce is interpreted together with n_speakers and warmup
```
