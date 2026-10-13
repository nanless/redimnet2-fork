# 06 · 术语表与常见问题

上一篇：[本机数据与开训检查](07-本机数据与开训检查.md) · [实验配方与复现](05-recipes-and-repro.md) · 回到：[中文导读首页](README.md)

---

## 1. 术语表（按字母 / 拼音混排）

| 术语 | 含义 |
|------|------|
| **AAM-Softmax / ArcFace** | 在角度空间对目标类加 margin：`cos(θ+m)`，拉开类间 |
| **ASTP** | Attentive Statistics Pooling；带注意力的均值+标准差池化 |
| **AS-Norm** | Adaptive Score Normalization；对余弦分数做自适应归一化（评估技巧，本训练栈本地验证未必启用） |
| **B0–B6** | ReDimNet2 从轻到重的规格；由 `C` / `stages_setup` 等决定 |
| **Chunk / num_frms** | 训练时随机裁的固定长度片段；帧数约等于秒数×100（10 ms hop） |
| **CN-Celeb / CNC2** | 中文说话人数据集；混合配方中的 `C_` 命名空间 |
| **data.list** | 训练用的样本清单，每行一条 JSON。字段是 `key` 样本编号、`spk` 说话人、`wav` 音频绝对路径 |
| **dataset_dir** | 增强用的噪声与混响目录；当前配方直接读文件夹，不用键值库 |
| **DDP / torchrun** | 多 GPU 数据并行启动方式 |
| **dual_agg** | stage 内 2D→1D 与 1D-att 双路上采样再聚合 |
| **EER** | 等错误率；FAR=FRR 时的错误率，越低越好 |
| **Embedding** | 固定维说话人向量；本仓库常见 192 维 |
| **exp_dir** | 实验输出目录，权重、日志与验证表都写在这里 |
| **FAR / FRR** | 虚警率 / 拒识率 |
| **GMAC** | 乘加次数量级，衡量算力 |
| **LM / do_lm** | Large-Margin fine-tuning；大间隔微调阶段 |
| **LMDB** | 一种键值库。旧流程曾把 MUSAN/RIR 打进去再随机读；**当前配方直接读 wav 目录**，LMDB 仅旧配置可选 |
| **Mel / n_mels / F** | Mel 滤波器组；频率维大小，配方里常 `F=72` |
| **Margin scheduler** | 训练中逐步增大分类间隔 |
| **Mixup / Cutmix / OverlapMix** | batch 内混合波形与软标签 |
| **MUSAN** | 噪声 / 音乐 / 语音噪声集合，用于加噪 |
| **Pareto front** | 精度–效率权衡前沿；一点支配另一点时更优 |
| **PTN** | Pre-TraiNing；主训练阶段（相对 LM） |
| **raw_fast** | 训练时直接 seek 读 wav 的数据模式 |
| **ReDim** | Dimension Reshaping：`(B,C,F,T) ↔ (B,C·F,T)` |
| **RIR** | Room Impulse Response；房间脉冲，用于混响 |
| **Speaker verification** | 判断两段语音是否同一人 |
| **Speed perturbation** | 0.9/1.0/1.1 变速；常把类别数扩为 ×3 |
| **SphereFace2** | 本仓库默认投影/损失头类型之一 |
| **stages_setup** | 各 stage 的 stride、块数、扩展、注意力压缩比等 |
| **TCM** | Time Context Module；1D 上的长时上下文块 |
| **utt2spk / spk2id** | 语句→说话人；说话人→整数类号 |
| **VoxBlink2 / VB2** | 大规模说话人数据；混合配方中的 `B_` |
| **VoxCeleb1/2** | 常用说话人基准；Vox2 训练、Vox1 测试 |
| **weigth1d** | 多路 1D 特征的可学习 softmax 融合（类名拼写以代码为准） |
| **x-vector** | 早期深度说话人嵌入做法的代称；现在也常泛指标量嵌入 |

---

## 2. FAQ

### Q1. 我是语音新生，最短路径是什么？

1. 读 [01 背景](01-background.md) + [首页快速上手](README.md)，用 hub 抽两条语音的 embedding，算余弦。  
2. 读 [02](02-architecture.md) 前三节，只抓住「2D↔1D」和「时间池化」。  
3. 打开 `redim_structural.py` 与 `ReDimNet2Wrap.forward` 对照。  
4. 有机器再按 [05](05-recipes-and-repro.md) 跑 B0 冒烟。

### Q2. `ptn` 和 `lm` 权重我该用哪个？

推理优先用 **`lm`**。`ptn` 是大间隔微调前的主训练结果；`dis` 是部分型号的蒸馏版。

### Q3. hub 加载很慢或失败？

检查网络访问 GitHub Release；也可浏览器下载对应 `.pt` 后，用 `redimnet2.load_custom` / 本地 `torch.load` + `model_config` 自行构建（逻辑见 `redimnet2/__init__.py`）。

### Q4. 训练 OOM 怎么办？

优先：减小 `dataloader_args.batch_size`、换更小型号（B0/B3）、关闭 `compile_backbone`、混合精度保持开启。当前配方读 wav 目录做增强，一般不必碰 LMDB；只有仍用旧 LMDB 增强时，再设 `lmdb_to_ram: false`。多卡时注意「每卡 batch × GPU 数」带来的有效 batch 与学习率缩放。

### Q5. 为什么 PTN 的 `num_class` 像说话人数 ×3？

因为 **语速扰动** 把 0.9/1.1 也当成新类。LM 关掉变速后，类数回到真实说话人数。YAML 里写死的数字可能只是注释性默认值——**运行时会按 utt2spk 覆盖**，不要手工改乱。

### Q6. 本地 EER 和 README 表格差一截？

可能原因：训练未充分、数据子集、评估协议（是否 cleaned、是否 AS-Norm）、随机种子、增强强度、是否只跑了主训练（PTN）没跑大间隔微调（LM）。先保证数据句数等与官方 prepare 脚本一致，再按同样的评估设定复测。

### Q7. `ReDimNet2` 和 `ReDimNet2Wrap` 区别？

- **Wrap**：波形进、embedding 出（前端+骨干+池化+Linear）。  
- **ReDimNet2**：骨干本身，吃频谱特征图。hub 与发布权重走 Wrap。

### Q8. 可以只用 CPU 推理吗？

可以，但大模型会慢。`model.eval()` + `torch.no_grad()`；注意 `torchaudio` 后端与 Mel 实现一致。

### Q9. 我想换成自己的数据集？

准备同样的 `data.list` + `utt2spk`，验证集可先关掉或改 `executor` 评估逻辑；投影头类别数会随 `spk2id` 变。增强路径、采样率 16 kHz、语言域差异都会影响迁移效果。

### Q10. 中英文文档冲突以谁为准？

以 **根目录英文 README、论文与代码** 为准。本 `docs/` 旨在降低入门门槛，若上游更新请对照 diff。

### Q11. `weigth1d` 是不是拼写错了？

代码里类名就是 `weigth1d`（缺了字母）。阅读/检索时用代码中的拼写，写论文叙述可用 “weighted 1D fusion”。

### Q12. 和原版 WeSpeaker 什么关系？

`wespeaker_lite` 是面向本仓库的轻量训练栈：数据协议改成 `raw_fast` JSONL，骨干对接 `redimnet2`，保留 WeSpeaker 风格的 DDP 训练与 Vox1 试验评估习惯。

### Q13. 改了配置里的某个键，训练却没反应？

先怀疑这个键根本不生效。已确认**训练代码里没有任何地方读**下面这几个键（详表见 [04 第 8 节](04-训练流程.md)）：

- `gpus` —— 用哪几张卡由命令行参数决定（`launch.sh` 的第二个位置参数、以及 `CUDA_VISIBLE_DEVICES`）。
  注意 `vox2/recipes/redimnet2/b3/lm.yaml` 里写着 `gpus: [0]`，它不会把训练限制在单卡。
- `grad_accum_steps` —— 训练栈没有实现梯度累积，每步都直接反向与更新。
- `eval_step_interval` —— 验证是按轮触发的，不是按步。
- `scheduler_args.scale_ratio` —— 启动时被 `卡数 × 每卡批大小 / 64` 覆盖。
- `projection_args.num_class` —— 被实际读到的说话人表覆盖（见 Q5）。

还有 `fbank_args` 里的 `num_mel_bins`、`dither` 在数据侧也不起作用，梅尔前端由 `model_args` 里的 `F`、`hop_length`、`spec_params` 决定。

### Q14. 大间隔微调中断了，重新启动能接着训吗？

**不能，会从头开始。** 微调配置里写了 `model_ckpt` 指向主训练的权重，而训练代码的逻辑是"配置给了 `model_ckpt` 就优先用它"，所以重启时骨干和分类头都会**重新从主训练权重初始化**，只有起始轮号来自最新的那份文件。

微调本身只训 5 轮（vox2 线）或 1 轮（混合线），建议一次跑完；一旦中断，清空 `exp_dir` 重跑更稳妥。主训练不写 `model_ckpt`，不受这个影响，中断后正常续训。细节见 [04 第 6 节](04-训练流程.md)。

### Q15. `num_avg` 是取最后 N 轮的权重平均吗？

**不是。** 代码里没有做任何参数平均。`num_avg`（默认 10）只决定"保留最后这几轮的权重文件别删掉"，以及让完整版评测在最后这几轮开启。训练结束建的 `final_model.pt` 直接软链到最后一轮的那份权重。

### Q16. 我自己训出来的权重，能直接用 `scripts/eval_*.py` 评测吗？

**不能直接喂，键名对不上。** 两套检查点的键名不同：

- **官方发布的权重**：顶层就 `model_config` 和 `state_dict` 两个键，`state_dict` 的键名直接对应 `ReDimNet2Wrap`。
- **训练栈存下来的权重**：是 `SpeakerModel` 的权重，键名带 `backbone.`（骨干）和 `projection.`（损失头）前缀。

两个评测脚本要求前者。要用自己的权重，得先把 `backbone.` 前缀剥掉、丢掉 `projection.*`，再包成含 `model_config` 的字典。另外脚本**不做重采样**，喂进去的音频必须是 16 kHz，采样率不对会直接报错。

### Q17. 训练输出把盘写满了？

本机**容器根只剩约 2 GB**，训练日志和 `exp/` 下的权重默认很容易落到那里。把 `exp_dir` 改到共享盘，或改到 `/dev/shm`（本机约 325 GB）。完整的数据现状与磁盘余量见 [07 本机数据与开训检查](07-本机数据与开训检查.md)。

---

## 3. 一张「概念 → 代码」总表

| 概念 | 代码锚点 |
|------|----------|
| 维度重塑 | `layers/redim_structural.py` |
| 时间池化 + 多 stage | `ReDimNet2.run_stage` / `forward` |
| 1D 上下文 | `TimeContextBlock1d` |
| 段级嵌入 | `layers/poolings.py` → `ASTP` |
| 发布加载 | `hubconf.py`, `load_custom` |
| 训练一步 | `utils/executor.py` → `run_epoch` |
| 损失头 | `models/projections.py` |
| Vox2 数据 | `examples/vox2/scripts/prepare_data.py` |
| 混合数据 | `examples/vb2_vox2_cnc2/scripts/*.sh` |

---

## 4. 延伸阅读

1. 论文：[arXiv:2603.11841](https://arxiv.org/abs/2603.11841)  
2. 前作 ReDimNet 仓库：[IDRnD/redimnet](https://github.com/IDRnD/redimnet)  
3. 英文总览与表格：[根 README](../README.md)  
4. 结构示意 / Pareto 图：`assets/scheme.png`、`assets/pareto.png`

读完本页，建议回到 [首页的一周学习路径](README.md#建议学习路径约一周)，按天执行比一次性读完更有效。
