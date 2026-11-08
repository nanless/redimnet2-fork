# 03 · 代码走读：文件、关键函数、输入输出、被谁调用

上一篇：[架构](02-architecture.md) · 下一篇：[训练流程](04-训练流程.md)

这篇文档解决一个问题：**拿到一个文件，能不能在五分钟内弄清它负责什么、入口函数在哪、吃什么、吐什么、谁在调它、改了会影响谁。** 每个条目都给文件路径与行号区间；行号只对 2026-11-08 那次核对成立——**具体提交号不写死**，用 `git rev-parse --short HEAD` 查（或看 [STATUS 第 1 节](STATUS.md)），改代码后按函数名重新定位。

读法：先看第 1 节的入口清单确定自己在哪条链上，再跳到对应小节；第 8 节是读代码时实际发现的问题清单，建议一并看。

---

## 1. 六个入口，先确定自己在哪条链上

| 入口 | 命令/调用方式 | 落在哪个文件 | 属于哪条链 |
|---|---|---|---|
| 推理取向量 | `torch.hub.load('PalabraAI/redimnet2', 'redimnet2', ...)` 或 `import redimnet2; redimnet2.load_custom(...)` | `hubconf.py:10-26` → `redimnet2/__init__.py:18-47` | 推理 |
| 训练 | `applications/speaker_rec/launch.sh <yaml> [卡数] [覆盖项…]` | `applications/speaker_rec/launch.sh:1-170` → `wespeaker_lite/bin/train.py:885` | 主训练 / 大间隔微调 |
| 仓库级评测 | `python scripts/eval_vox1_checkpoints.py …`、`python scripts/eval_ood_checkpoints.py …` | `scripts/eval_vox1_checkpoints.py:182`、`scripts/eval_ood_checkpoints.py` | 评测 |
| 标注 | `python tools/open_annotation/gpu_launch.py --execute --bundle-sha … --contract …` | `tools/open_annotation/gpu_launch.py:348-359` | 标注 |
| 接续包 | `python -B -m tools.ptn_resume.build --spec … --output-dir …` | `tools/ptn_resume/build.py:167-172` | 主训练接续 |
| LM 启动 | 平台入口执行包内 `runtime/lm_launch.sh` | `tools/ptn_resume/lm_launch/run_lm.sh:1-132` | 大间隔微调 |

旁线还有三个入口（不在本次重点，但同属仓库）：`word_ft/train.py`（完整字微调 v1）、`word_ft_v3/train.py`（v3）、`duration_ft/train.py`（整数秒裁片微调），启动包装在 `scripts/launch_word_two_gpu.py`、`scripts/launch_word_v3.py`、`scripts/launch_duration_two_gpu.py`。

---

## 2. `redimnet2/`：模型库（12 个文件，3,674 行）

### 2.1 文件职责

| 文件 | 行数 | 职责 | 关键定义（行号） |
|---|---:|---|---|
| `__init__.py` | 53 | 对外加载：下载权重、自描述实例化、`eval()` | `load_custom`（18）、`from_pretrained`（38）、`redimnet2`（42）、`ReDimNet2` 兼容壳（50） |
| `redimnet2.py` | 688 | 骨干装配 + 前端 + 池化 + 投影 | `FreqEncoder`（50）、`_parse_agg_gnorm`（83）、`ReDimNet2`（131）、`ReDimNet2.build`（208）、`run_stage`（395）、`forward`（423）、`ReDimNet2Wrap`（508）、`Wrap.forward`（611）、`ReDimNet2Custom`（687） |
| `layers/redim_structural.py` | 138 | 2D↔1D 维度重塑与多路加权 | `to1d`（33）、`to2d`（40）、`MaskedTo1d`（77）、`MaskedTo2d`（93）、`weigth1d`（109） |
| `layers/masking.py` | 348 | 掩码生成、传递、归一化与零化 | `make_sample_mask`（43）、`frame_count`（83）、`downsample_frame_mask`（147）、`MaskDownConv2d`（218）、`MaskResample`（239）、`MaskedGroupNorm`（268）、`rezero`（299） |
| `layers/blocks.py` | 371 | 2D 卷积块与 1D 时间上下文块 | `ConvBlock2d`（17）、`build_internal_1d_tcm_block`（74）、`TimeContextBlock1d`（168）、`SymmetricalDWMixer`（207）、`TimeContextBlock1dV2`（284） |
| `layers/attention.py` | 462 | 注意力与 Transformer 层 | `RotaryEmbedding`（28）、`MultiHeadAttention`（60）、`TransformerEncoderLayer`（285）、`StableTransformerEncoderLayer`（352）、`TransformerEncoderLayerV2`（404） |
| `layers/convnext.py` | 161 | ConvNeXt 风格块与权重标准化卷积 | `WSConv2d`（67）、`WSConv1d`（74）、`ConvNeXtLikeBlock`（89） |
| `layers/resblocks.py` | 131 | 基础残差块与频率维 SE | `fwSEBlock`（43）、`ResBasicBlock`（68） |
| `layers/features.py` | 260 | torchaudio 路线声学前端 | `NormalizeAudio`（35）、`PreEmphasis`（56）、`FbankAug`（77）、`MelBanks`（135）、`STFT` |
| `layers/features_tf.py` | 571 | 纯 torch 时频前端（可学习/固定 DFT） | `get_filterbanks`（52）、`SpectralFeaturesTF`（86）、`TFMelBanks`（397）、`TFSpectrogram`（479） |
| `layers/poolings.py` | 391 | 帧级 → 段级池化 | `TAP`（30）、`TMAP`（53）、`TSDP`（79）、`TSTP`（99）、`ASTP`（147）、`MHASTP`（244）、`MQMHASTP`（318） |
| `layers/layernorm.py` | 100 | 通道在前/在后的 LayerNorm | `LayerNorm`（34）、`MaskedLayerNorm`（70） |

### 2.2 `redimnet2.py` 逐段

**`ReDimNet2.__init__`（`redimnet2.py:131-206`）**：把字符串开关翻译成内部标志，再调 `build()`。

- `causal` 三态：`'full'` 给 1D 与 2D 块名都加 `-causal`；`'only_1d'` 只给 1D；`'none'` 都不加（`:171-181`）。
- `block_1d_type` 前缀编码：`v2_` 走 `TimeContextBlock1dV2`，`sdwm_` 走 `SymmetricalDWMixer`（`:183-189`）。
- `stages_setup` 每项语义 `((sf,st), num_blocks, conv_exp, kernel_sizes, att_block_red)`；`att_block_red=None` 表示该 stage 不做 1D 块。

**`build()`（`:208-394`）**：实际建层。输入 `F`（梅尔维数）、`C`（起始通道）、`out_channels`、`stages_setup` 等，输出挂到 `self` 上的 `stem`、`stage{i}`、`fin_wght1d`、`fin_to2d`、`head`。它同时算出两个关键属性：`self.time_stride`（所有 stage 时间步长的乘积）与 `self.freq_stride`（频率步长乘积）——这两个值被 `Wrap.forward` 用来对齐截断、被 `out_channels` 推导用来算池化输入维。

**`run_stage()`（`:395-422`）**：跑一个 stage，返回“本 stage 新增的 1D 特征列表”。

- 普通 stage：`stage(prev_outs_1d[, mask])`（`:418-421`）。
- `dual_agg=True` 的 stage：`pre` → 两条上采样分支，各自返回**与自己输出对齐的掩码**（`:396-411`）。这里的注释记录了一个已修的长度不匹配崩溃。

**`forward()`（`:423-507`）**：

| 输入 | 处理 | 输出 |
|---|---|---|
| `inp`：`(B, C, F, T)`；`mask`：`(B,1,1,T)` 或 `(B,1,T)` | 非子网模式先截时间到 `time_stride` 整数倍（`:451-472`）；`stem` → 逐 stage → `fin_wght1d` 融合 → `fin_to2d` → `head` | 默认 `(B, C·F, T)`；`return_all_outputs=True` 时再加全部 1D 特征 |

**`ReDimNet2Wrap.forward()`（`:611-686`）**：外部只调这一个。

| 步骤 | 行号 | 说明 |
|---|---|---|
| 生成掩码 | 620-627 | `lengths` 为 `None` 时 `mask=None`，与旧行为逐位一致 |
| 可选右侧补零 | 628-634 | `pad_right_samples` 不为 `None` 时补波形**并给掩码补 0** |
| 声学前端 | 635-651 | `self.spec(x)` 或 `self.spec(x, mask)`；3 维输出补一个通道维 |
| 对齐截断 | 649-671 | 把 `(x, mask)` 一起截到 `time_stride` 整数倍；太短（`T_keep==0`）则保留原样 |
| 骨干 | 670-674 | `self.backbone(x, mask)` |
| 池化前整形 | 676-681 | 4 维输出展平为 `(B, C·F, T)`；`before_pool_offset` 可裁掉头部若干帧 |
| 池化与投影 | 682-686 | `bn(pool(out))` → `linear` → 可选 `bn2` |

**被谁调用**：`models/speaker_model.py:32-36`（训练）、`scripts/checkpoint_utils.py:30-85`（评测）、`scripts/export_checkpoint.py:19-43`（导出）、`redimnet2/__init__.py:18-35`（推理）。

**改动影响面**：默认路径（`lengths=None`）被 20 个已发布权重与官方数字依赖，**逐位不变**是前提；改动后用 `tests/test_mask_input_contract.py` 与 `tests/test_varlen_masking.py` 回归。

### 2.3 `ASTP` 与掩码

`ASTP.forward(x, mask=None)`（`poolings.py:181-241`）：

| 输入 | 形状 | 输出 |
|---|---|---|
| `x` | `(B,C,F,T)` 或 `(B,C·F,T)` | `(B, 2·in_dim)`；有掩码时额外返回掩码 |
| `mask` | `(B,1,T)` 或 `(B,T)` | 同上 |

三条实现细节：4 维输入先展平（`:188-190`）；`global_context_att=True` 时拼 `[x, 均值, 标准差]` 再算注意力（`:192-210`）；**无掩码支用 `torch.var`（无偏），有掩码支把有偏矩乘 `n/(n-1)` 对齐**（`:195-207` 的注释解释了两种算法为什么要对齐，否则“只是传了个 `lengths`”就会改变输出）。

---

## 3. `applications/speaker_rec/`：训练栈（32 个文件，12,012 行）

### 3.1 `launch.sh`（170 行）

| 段 | 行号 | 做什么 |
|---|---|---|
| 参数解析 | 4-17 | 第 1 个参数是配方 yaml，第 2 个（可选、纯数字）是卡数，其余透传给 `train.py` |
| 解释器解析 | 34-60 | 优先级：`REDIMNET2_PYTHON` → `REDIMNET2_ENV` 指定的 conda 环境 → 名为 `redimnet2` 的环境 → PATH 上的 `python`（并告警） |
| torchrun 解析 | 107-113 | 优先用同一个环境里的 `torchrun`，找不到才用 PATH |
| 启动 | 155-165 | `torchrun --nproc_per_node=$NUM_GPUS ... train.py --config ...` |

这个脚本是“实际用哪个解释器”的唯一决定点。装了两份 torch 时，选错环境不会报错，只会静默跑在另一份上。

### 3.2 `wespeaker_lite/bin/train.py`（886 行）

按执行顺序切成 10 段：

| 段 | 行号 | 关键动作 | 读/写的对象 |
|---|---|---|---|
| ① 分布式初始化 | 176-193 | 取 `RANK`/`WORLD_SIZE`/`LOCAL_RANK`，`cuda.set_device`，`init_process_group('nccl')`，建 `models/` | 环境变量、`exp_dir/models` |
| ② 同步与恢复判定 | 195-207 | `dist.barrier` 后再 `select_resume_checkpoint()` | `models/model_*.pt` |
| ③ 配置与随机种子 | 208-222 | 打日志、`set_seed(seed+rank)`、把 `num_workers`/`seed` 传给子进程环境 | `configs` |
| ④ 数据统计与分类头参数 | 224-252 | 读 `utt2spk` → `spk2id`；`projection_args.num_class = len(spk2id)`；主训练且开变速时 `num_class ×= 3`；LM 阶段强制关变速 | `train_label` |
| ⑤ 数据集与加载器 | 254-390 | 可选把噪声/混响 LMDB 拷到内存盘（`stage_lmdb_to_ram`，117-166）；`Dataset()`；变长时换 `collate_varlen`；可选 `BucketedLengthDataset`；算 `epoch_iter` 并断言 ≥1 | `data.list`、`dataset_args` |
| ⑥ 验证集 | 392-430 | 每个验证集建两个加载器：8 秒快评（`Vox1TestEvalDatasetSimple`）与整段全评（`Vox1TestEvalDataset`） | `validation` 里的 `vox1` 根目录 |
| ⑦ 模型与恢复 | 432-590 | `get_speaker_model()` + `get_projection()` → `SpeakerModel`；决定 `ckpt_to_load`（本阶段检查点优先于 `model_ckpt`）；分类头按说话人名重组；跨阶段时丢形状不符的优化器状态 | 检查点、`spk2id.json` |
| ⑧ 损失/优化器/调度 | 592-647 | `criterion`、两组参数、`scheduler`、`margin_scheduler`；打印“实际峰值学习率” | `configs` |
| ⑨ 落配置与训练 | 649-783 | 写 `exp_dir/config.yaml`；`GradScaler`；`training_loop()` | `exp_dir` |
| ⑩ 主函数 | 885-886 | `fire.Fire(train)` | 命令行 |

`training_loop()`（`:785-856`）：逐轮 `run_epoch` → 验证 → 写 `val_log.csv` → 按 `save_epoch_interval` 存 `model_N.pt` → `final_model.pt` 软链。**恢复语义**写在 `:446-465` 的注释里：本阶段检查点恢复优化器/缩放器状态；从 `model_ckpt` 冷启动只载权重。

### 3.3 `wespeaker_lite/utils/executor.py`（575 行）

| 函数 | 行号 | 输入 | 输出 | 调用方 |
|---|---|---|---|---|
| `Mixup.forward` | 49-70 | 波形 + one-hot | 混合波形 + 混合标签 | `run_epoch` |
| `to_ohe` | 71-77 | 类别数 + 标签 | one-hot | `run_epoch` |
| `_check_training_numerics` | 78-184 | 波形/向量/损失/轮次/优化器/缩放器 | 无（异常即抛） | `run_epoch` |
| `run_epoch` | 185-330 | 加载器、模型、损失、优化器、两个调度器、缩放器、混合增强 | 无（副作用：更新参数、打印） | `train.py:training_loop` |
| `EER` | 331-341 | 标签 + 分数 | `(eer, thresh)` | `eval_epoch*` |
| `eval_epoch` | 365-462 | 整段评测加载器 | 指标字典 | `train.py:857` |
| `eval_epoch_simple` | 463-562 | 8 秒快评加载器 | 指标字典 | `train.py:857` |
| `cos_sim_mat` | 563-573 | 两组向量 | 余弦矩阵 | 两个 eval |
| `set_bn_eval` | 574-575 | 模块 + 标志 | 无 | 评测前 |

`run_epoch` 的每步顺序（顺序是语义的一部分）：`cur_iter=(epoch-1)*epoch_iter+i` → `scheduler.step(cur_iter)` → `margin_scheduler.step(cur_iter)` → 取 `lengths` → 可选混合增强（变长时取两条真实长度的较大值，`:246-253`）→ 前向（带取证上下文）→ 投影头（可能自带损失）→ 数值检查 → 每 16 步记一次损失/准确率 → `zero_grad` / `scaler.scale(loss).backward()` / `step` / `update` → 到 `epoch_iter` 就 `break`。

### 3.4 `wespeaker_lite/utils/schedulers.py`（320 行）

| 类 | 行号 | 输入 | 行为 |
|---|---|---|---|
| `MarginScheduler` | 20-94 | `epoch_iter`、起始/固定轮、初始/最终间隔、`increase_type` | `step(current_iter)` 按绝对迭代推进；`get_margin()` 读投影头当前间隔 |
| `BaseClass` | 97-199 | 优化器、轮数、每轮步数、`scale_ratio` 等 | 公共：`get_lr`/`set_lr`/`step`/多进程系数 |
| `ExponentialDecrease` | 200-223 | 同上 | 指数衰减 |
| `ExponentialDecayPlateauWithWarmup` | 224-320 | 同上 + 预热轮数、平台轮数、衰减轮数/率 | 三语料 B6 用的就是这个（`official_repro_ptn.yaml:190`） |

这些类**没有** `state_dict()`/`load_state_dict()`，所以检查点里不存调度器状态（`utils/checkpoint.py:147-158` 的注释说明了这件事）。续训能复现轨迹，是因为它们只依赖绝对迭代数。

### 3.5 `wespeaker_lite/utils/checkpoint.py`（199 行）

| 函数 | 行号 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| `select_resume_checkpoint` | 31-56 | `models/` 目录、可选显式路径 | 检查点路径或 `None` | 显式路径优先；否则按**轮数**取最大，不按 mtime（拷贝会改 mtime） |
| `resume_start_epoch` | 57-77 | 路径 + 载荷里的 `epoch` | 起始轮 | 文件名与载荷轮数两处都核；跨阶段从 1 重新计 |
| `validate_same_stage_resume` | 78-102 | 状态字典、模型、检查点、说话人表 | 说话人表路径 | 同阶段续训前核“键集合与形状”与“映射未变” |
| `is_training_checkpoint` | 110-120 | 任意对象 | 布尔 | 区分“训练检查点字典”与“纯状态字典” |
| `extract_state_dict` | 121-133 | 检查点 | `(state_dict, extra)` | 两种格式都吃 |
| `load_checkpoint` | 134-140 | 模型 + 路径 | 无 | `strict=False` |
| `save_checkpoint` | 141-199 | 模型 + 路径 + 可选优化器/缩放器/轮次 | 无 | 原子写：临时文件 + `os.replace`；无优化器时写纯状态字典（历史格式） |

### 3.6 其余工具文件

| 文件 | 行数 | 关键定义 | 作用 |
|---|---:|---|---|
| `utils/utils.py` | 215 | `parse_config_or_kwargs`（49）、`spk2id`（105）、`save_spk2id`（116）、`load_spk2id`（123）、`reassemble_projection_head`（130）、`seed_worker`（204） | 配置解析与说话人表重组；**LM 的分类头按名字重组就在这里** |
| `utils/numerical_failure.py` | 371 | 取证导出上下文管理器 | 只在非有限值时写私有取证材料；47 轮接续包里它是“新增”的那个文件 |
| `utils/meters.py` | 176 | 各类计分器 | 损失/准确率统计 |
| `utils/file_utils.py` | 47 | 读写小工具 | 被数据与日志代码调用 |
| `models/speaker_model.py` | 36 | `get_speaker_model`（19）、`SpeakerModel`（26） | 骨干 + 投影头的容器，`forward` 直接转发 |

### 3.7 `dataset/dataset.py`（944 行）

| 定义 | 行号 | 输入 | 输出 | 被谁调用 |
|---|---|---|---|---|
| `Processor` | 36-69 | 迭代数据集 + 处理函数 + 参数 | 迭代数据集 | `Dataset()` |
| `DistributedSampler` | 70-133 | 数据集 | 采样器 | `DataList` |
| `DataList` | 134-180 | 清单文件列表 | 无限流（可重复） | `Dataset()` |
| `is_varlen_enabled` / `get_varlen_length_bounds` | 181-233 | `dataset_args` | 布尔 / `(min,max)` 秒 | `train.py`、`Dataset()` |
| `is_vad_segments_enabled` / `get_vad_segment_config` | 234-309 | `dataset_args` | 布尔 / 配置字典 | `Dataset()` |
| `rttm_path_for_audio` | 277-309 | 音频路径 | RTTM 路径 | `vad_select_segment` |
| `Dataset` | 325-510 | `data_type`、清单、`dataset_args`、`spk2id` | 处理链 | `train.py:288` |
| `collate_func_single` / `collate_func_mult` | 511-515 / 595-599 | 样本批 | 批字典 | 验证加载器 |
| `collate_varlen` | 600-681 | 样本批 + 最大批秒数 + 采样率 | `{'key','label','wav','lengths'}` | 训练加载器（变长时） |
| `PreBatched` / `BucketedLengthDataset` | 693-722 / 723-898 | 数据集 + 批大小 + 桶参数 | 已按长度分桶的整批 | `train.py:340` |
| `Vox1TestEvalDataset(Simple)` | 516-594 / 899-943 | 试验根目录、`eval_dur` | 评测样本 | `train.py` 验证段 |

**`Dataset()` 的处理链顺序**（`:325-510`，顺序是语义的一部分）：读清单并洗牌 → `parse_raw_fast_varlen`/`parse_raw_fast_and_filter`（逐条读音频）→ 样本级洗牌 → `spk_to_id` → `resample` → 可选 `vad_select_segment` → 可选 `speed_perturb` → 定长/变长裁剪 → 可选增强（新格式 `augs.deserialize_augmentor`，旧格式 `add_reverb_noise_gain`）。

**顺序不能换的原因**（`:396-409` 的注释）：`vad_select_segment` 会回磁盘按绝对时间读原始音频，如果变速在它前面，变速结果会被整段覆盖，而标签已经被 `speed_perturb` 改成了“另一个说话人”。实测确认过波形逐位回到原样。

### 3.8 `dataset/processor.py`（1,529 行，全仓最长）

| 函数 | 行号 | 输入 | 输出/副作用 |
|---|---|---|---|
| `parse_rttm_segments` / `merge_segments` | 129-189 | RTTM 路径 / 段列表 | 语音段列表 |
| `select_vad_span` | 201-426 | 段列表、目标/最小/最大秒、切分策略 | 选中的一段（含切点与回退原因） |
| `_equal_spans` | 427-451 | 候选段、目标秒 | 等长切分方案 |
| `read_audio_section` / `read_audio_span` | 452-549 / 761-794 | 文件、采样率、起止秒 | 波形（按绝对时间读盘） |
| `parse_raw_fast_and_filter` | 550-651 | 一行样本 JSON、`chunk_len`、`data_type`、过滤参数 | 定长片段 |
| `parse_raw_fast_varlen` | 652-760 | 同上 + 长度上下界 | 变长片段 |
| `vad_select_segment` | 795-1060 | 样本、RTTM 参数、`min/max/target_sec`、各类策略 | 按语音活动边界切出的片段 |
| `shuffle` / `spk_to_id` / `resample` | 1061-1145 | 样本流 | 洗牌 / 加标签 / 重采样 |
| `speed_perturb` | 1146-1180 | 样本流、说话人数 | 变速并把标签映射成“另一个说话人” |
| `random_chunk` / `random_chunk_range` | 1234-1290 | 样本流、固定长度 / 长度区间 | 裁剪 |
| `apply_augmentor` / `add_reverb_noise_gain` | 1296-1470 | 样本流、增强器 / 噪声混响数据 | 增强后的波形 |

统计口径也在这里：`_drop_counter_incr`/`report_drop_counters`（76-115）按进程计数，`report_vad_stats`（116-128）统计“选了几段、切了几次、回退了几次”。**这些计数是按进程的**，开多 worker 时主进程通常看不到（`executor.py:302-330` 的注释解释了这一点）。

### 3.9 增强与数据源

| 文件 | 行数 | 关键定义 | 说明 |
|---|---:|---|---|
| `models/mix_augs.py` | 328 | `Cutmix`（27）、`Mixup`（81）、`MixCutUp`（116）、`OverlapMix`（180） | 批内混合；训练器只在 `mixup_alpha` 非空时启用 |
| `dataset/augs.py` | 526 | `AudioSource`（138）、`Augmentor`（227）、`OneOf`（243）、`Sequential`（256）、`CustomNoise`（272）、`Reverb`（338）、`CodecAug`（376）、`LowPassAug`（441） | 新版可组合增强树；`LEGACY_AUG_BEHAVIOR` 开关控制是否退回官方旧行为 |
| `dataset/lmdb_data.py` | 145 | LMDB 读取 | 老配方的噪声/混响库；新配方用原始 wav 目录 |

### 3.10 `models/projections.py`（536 行）

| 类 | 行号 | 前向输出 | 说明 |
|---|---|---|---|
| `SphereFace2` | 67-217 | `(logits, loss)` 或 `(logits, 逐样本损失)` | **自己算损失**（`:201-211`）；三语料 B6 用的就是它 |
| `ArcMarginProduct` | 219-289 | `(logits, loss)` | 标准大间隔 |
| `ArcMarginProduct_intertopk_subcenter` | 290-413 | `(logits, loss)` | 子中心 + topk |
| `AddMarginProduct` / `SphereProduct` / `Linear` | 414-536 | logits（`Linear` 返回 `(logits, loss)`） | 对照与冒烟 |
| `get_projection` | 文件末尾 | 投影头实例 | 按 `project_type` 分发 |

**共同接口**：`forward(input, label, target_mask=None)` + `update(margin)`（有 `update` 才能被 `MarginScheduler` 驱动）。

### 3.11 VAD 标注脚本 `scripts/vad/`

| 文件 | 行数 | 作用 |
|---|---:|---|
| `annotate_firered_offline.py` | 1,470 | 用 FireRedVAD 离线给训练音频写 RTTM。三种模式：`infer`（按行号分片、可续跑）、`validate`（只读复查）、`oracle`（批处理与逐条对照）。关键函数：`extract_feature`（183）、`plan_batches`（230）、`forward_batch_resilient`（330）、`write_sidecar`（416）、`validate_sidecar`（479）、`run_infer`（720）、`run_validate`（1041） |
| `firered_rttm_common.py` | 757 | RTTM 读写、来源指纹、校验的共用件 |
| `test_firered_rttm.py` | 266 | 上述两者的测试入口 |
| `run_firered_rttm_cloud_8gpu.sh` / `..._local_2gpu.sh` | 23,319 / 19,719 字节 | 八卡 / 两卡启动包装 |

### 3.12 云端脚本 `scripts/cloud/`

| 文件 | 作用 | 备注 |
|---|---|---|
| `train_b6_mix_cloud_8gpu.sh`（34 KB） | 三语料八卡训练主脚本 | 被 `run_b6_mix_reproduction.sh` 与 `train_b6_mix_cloud_8gpu_lm.sh` 调用；实际训练逻辑都在它里面 |
| `run_b6_mix_reproduction.sh`（23 KB） | 复现总入口（`MODE=formal` 会重训 PTN） | LM 阶段**不能**用它（`lm_launch/README.md:17`） |
| `train_b6_mix_cloud_8gpu_lm.sh` | `STAGE=lm` 的薄包装 | 默认按 mtime 取最新权重（会拿到 120），所以 LM 走 `run_lm.sh` |
| `check_b6_lm_init.py` | LM 初始化门：核 `model_115` 与说话人表，写 `lm_initialization.json` | 被 `run_lm.sh` 真跑一遍 |
| `check_b6_mix_data.py` | 三语料数据核对 | 训练前只读 |
| `make_b6_smoke_recipe.py` | 生成短测配方 | 短测 |
| `probe_b6_ddp.py` | 多卡连通性探针 | 冒烟 |
| `prepare_b6_cloud_paths.sh` | 云端路径准备 | 部署 |
| `evaluate_b6_mix_reproduction.sh` | 正式评测：训练权重 vs 官方参考权重 | 读 `lm/models/model_1.pt` 与 `lm/config.yaml` |

### 3.13 示例与配方

| 路径 | 内容 |
|---|---|
| `examples/vox2/scripts/prepare_data.py`（1,276 行） | 只练 VoxCeleb2 的数据准备：`download → extract → convert → lists → [lmdb]`；产出 `data.list`、`utt2spk` 等 |
| `examples/vb2_vox2_cnc2/scripts/*.sh` | 三语料拼表；`gen_datalists.py`（211）、`join_datalists.py`（199）、`filter_datalist_by_duration.py`（249）、`infer_durations_sf.py`（117） |
| `examples/*/recipes/redimnet2/{b0..b6}/*.yaml` | 32 个配方，见 02 文档第 4.6 节 |

---

## 4. `tools/open_annotation/`：标注系统（顶层 90 个文件，21,431 行）

### 4.1 按层归类（顶层文件与行数）

| 层 | 文件（行数） |
|---|---|
| ① 来源与清单 | `source_registry.py`(191)、`manifest_builder.py`(288)、`source_filter.py`(90)、`audit_source_filter.py`(35)、`source_clock.py`(27)、`collect_sample_audio.py`(76)、`portable_source_identity.py`(306) |
| ② 合同与记录 | `contracts.py`(270)、`engine_protocol.py`(231)、`schema.py`(413)、`sidecar.schema.json`、`freeze_contract.py`(167)、`freeze_legacy_successor.py`(166) |
| ③ 执行与资源 | `gpu_launch.py`(367)、`gpu_bootstrap.py`(134)、`gpu_challenge.py`(131)、`gpu_common.py`(216)、`gpu_vendor/resources.py`(250)、`gpu_vendor/topology.py`(47)、`worker.py`(310)、`device_preflight.py`(127)、`validate_two_gpu.py`(340)、`probe_known_worker_cpu_io.py`(96)、`probe_retained_process_parent_death.py`(202)、`diagnose_cpu_stack_capture.py`(93) |
| ④ 模型与解码 | `audio_windows.py`(257)、`qwen_backend.py`(553)、`alignment_observer.py`(148)、`normalization.py`(83)、`native_transformers_batch.py`(240)、`native_b1_feature_padding.py`(238)、`vllm_backend.py`(365)、`vllm_live_probe.py`(362)、`asr_stage_store.py`(196) |
| ⑤ 队列与证据 | `sidecar.py`(106)、`work_queue.py`(263)、`report.py`(193)、`audit_outputs.py`(194)、`quality_axes.py`(165)、`union_coverage.py`(778)、`union_consumer.py`(85)、`readonly_progress.py`(187)、`bounded_metadata_progress.py`(121)、`finalize_remaining_roles.py`(208) |
| ⑥ 准入与交接 | `cloud_gate.py`(249)、`make_admission_spec.py`(45)、`make_cloud_launcher.py`(90)、`make_platform_observation_v4r2.py`(115)、`verify_cloud_mounts.py`(34)、`verify_portable_successor_mounts.py`(168)、`native_batch_admission.py`(484)、`native_batch_readback.py`(214)、`native_batch_queue_handoff.py`(406)、`queue_handoff.py`(471)、`bounded_speed_handoff.py`(380)、`bounded_speed_queue.py`(423)、`bounded_speed_queue_handoff.py`(362)、`bounded_speed_runtime.py`(715)、`run_portable_successor.py`(480)、`resume_existing_annotation.py`(422) |
| ⑦ 试验与对照 | `run_native_batch_pilot.py`(233)、`run_native_partial_recovery_probe.py`(304)、`run_native_cpu_sealer.py`(271)、`run_bounded_speed_pilot.py`(366)、`prepare_native_batch_validation.py`(173)、`prepare_native_partial_recovery_control.py`(138)、`prepare_bounded_speed_validation.py`(156)、`prepare_speed_validation.py`(127)、`prepare_speed_continuation.py`(526)、`prepare_vllm_live_inputs.py`(158)、`prepare_normalization_ab.py`(162)、`prepare_sample_audio_review.py`(85)、`prepare_boundary_context_review.py`(70)、`audit_native_transformers_batch.py`(673)、`audit_normalization_ab.py`(288)、`audit_speed_comparison.py`(391)、`audit_engine_comparison.py`(262)、`compare_native_b1_bridge.py`(249)、`benchmark_retained_reads.py`(109)、`bounded_cpu_audio_timing.py`(169)、`measure_native_startup_cost.py`(228)、`estimate_speed_capacity.py`(92)、`sample_audio_review.py`(620) |
| ⑧ 绘图与文档产物 | `quality_figures.py`(304)、`plot_saved_boundary_review.py`(301)、`plot_normalization_ab.py`(80)、`plot_native_feature_benchmark.py`(87)、`plot_synthetic_feature_padding.py`(57)、`plot_annotation_stage_timing.py`(66)、`plot_annotation_eight_gpu_window.py`(76)、`diagnose_native_feature_padding.py`(149) |
| ⑨ 其它 | `__init__.py`(1)、`word_adapter.py`(384)、`launch_portable_successor.sh`、`launch_queue_handoff.sh`、`parent_sources.json`、`README.md`、若干中文说明 md |

`tests/` 76 个文件 13,221 行，是这套系统的回归网；`legacy_snapshots/production_20261025/` 20 个文件 3,689 行是历史生产包的只读快照（**不是**现行代码）。

### 4.2 重点模块逐个

#### `contracts.py`（270 行）——所有身份的来源

| 定义 | 行号 | 作用 |
|---|---|---|
| `ENGINE_MODULES` | 16-19 | 14 个引擎文件名元组（清单见 02 文档第 9.1 节） |
| `NATIVE_BATCH_POLICY` / `ASR_B1_FEATURE_POLICY` / `NATIVE_FORMAL_DECLARATION` | 20-25 | 两种新策略的字符串标识与“正式声明默认关闭”的结构 |
| `asr_batch_profile(contract)` | 27-52 | 校验合批声明：策略名、批大小只能 1/4/8、固定排序、`max_batch==1`（Ultra 走单窗） |
| `asr_feature_profile(contract)` | 53-67 | 校验“官方 B1 特征再合批”声明 |
| `engine_modules_for_contract(contract)` | 68-71 | 在 14 个基础上按合同追加 1–2 个文件，最多 16 |
| `need` / `digest` / `canonical_bytes` / `native_json` | 73-108 | 断言、摘要、统一字节形式、JSON 规范化 |
| `safe_join` / `relative_path` / `stat_identity` | 109-135 | 防路径穿越、相对路径校验、文件身份四元组 |
| `atomic_new` | 136-161 | 不可覆盖发布，是全系统“不覆盖”的基础 |
| `validate_execution_policy` / `asr_compute_profile` | 162-193 | 执行策略与精度/注意力实现的一致性 |
| `read_contract` | 194-241 | 读合同 + 核 SHA + 核时间映射与当前 `TIME_MAPPING` 相等（不等则要求重新冻结） |
| `verify_models` / `model_provenance` / `job_key` | 242-270 | 模型文件核对、来源摘要、任务键 |

**被谁调用**：几乎所有模块。**改动影响面**：它在 `ENGINE_MODULES` 里，改它要重新打包、重新准入。

#### `freeze_contract.py`（167 行）——唯一的冻结入口

`freeze()`（32-107）：核基础合同 SHA 与用途 → 复制源码到仓库外 → 补 `task_names.py` → 覆盖 `boundary_mapping`、`normalization_profile` → 重算 `engine_source_files` → 生成两份合同 → 生成成员清单与收据。`make_native_formal_contract()`（110-132）只生成“正式合同数据”，`default_enabled=False`、`formal_runtime_admitted=False`。

**调用方**：人工命令行（`main`，135-164）。**输出**：`contracts/*.json`、`bundle_files.json`、`freeze_receipt.json`。

#### `worker.py`（310 行）——单卡单进程的主循环

| 函数 | 行号 | 输入 | 输出 |
|---|---|---|---|
| `timed_call` / `progress_performance` | 27-63 | 计数器 + 阶段 + 函数 | 阶段耗时 / 性能快照 |
| `authenticate_resource` | 71-93 | 资源票、证明、合同/索引 SHA | 已核验的票（含 `bundle_sha256`） |
| `_descriptor` / `_stage_inputs` | 96-110 | 合同 + 窗口 | 侧车里的来源描述 / 阶段输入摘要 |
| `process_group` | 113-232 | 一批来源 + 领取句柄 + 侧车库 + 缓存 + 后端 | 回执列表 |
| `main` | 235-304 | 命令行参数 | `worker_summary.json` |

`main` 的顺序：读合同 → 核用途与后端匹配 → 准备输出目录（新目录）→ 读运行准入（`bounded_speed_runtime.runtime_options`）→ 建队列/侧车库/阶段缓存 → 验资源票 → 核 `source_files` 摘要 → 验解码库 → 建后端 → 循环领取分片 → 写汇总。

**被谁调用**：`gpu_bootstrap.py` 在已准入的 GPU 上下文里以子进程方式启动它（命令行在 `gpu_launch.py:183-199` 拼装）。

#### `gpu_launch.py`（367 行）——GPU 父进程

| 函数 | 行号 | 职责 |
|---|---|---|
| `worker_ready` | 28-56 | 核 worker 的终态就绪文件与汇总，写 `terminal_ack.json` |
| `ensure_alias` | 59-66 | 建立/校验 `/root/group-shared` 别名 |
| `validate_output_paths` | 69-82 | 目录互斥规则（见 02 文档第 5.3 节） |
| `selected_uuids` | 85-96 | `auto2`（仅试验）/`auto8`（正式八卡）/两个显式 UUID |
| `_run` | 99-318 | 主流程：核成员清单与合同 → 资源双探针 → 持锁 → 逐卡起子进程 → 监控（挑战/模型/排空）→ 归还资源 → CPU 全量核对 |
| `run` | 321-345 | 新 schema 入口：有界运行规格下需持租约才能跑 |
| `main` | 348-359 | 命令行 |

**被谁调用**：人工/平台入口。**它不 import torch**（`:115` 注释），所有模型动作都在子进程里。

#### `gpu_bootstrap.py` / `gpu_challenge.py` / `gpu_common.py` / `gpu_vendor/`

| 文件 | 行数 | 职责 |
|---|---:|---|
| `gpu_bootstrap.py` | 134 | 在“只看得见一张卡”的环境里：验票 → 建 CUDA 上下文 → 跑显存挑战 → 调 `worker.main()` → 保持上下文直到父进程终态确认 |
| `gpu_challenge.py` | 131 | 七阶段显存挑战，证明“这张卡上只有本进程在用”；明确声明**不做**宿主机/容器 PID 翻译 |
| `gpu_common.py` | 216 | 纯 CPU 身份与原子控制：`sha`、`write`、`pidfd`、进程启动时刻、自有成员枚举、清理、排空状态 |
| `gpu_vendor/resources.py` | 250 | 显卡快照解析（36-55）、双探针（90-109）、物理卡锁（128-149）、资源票（150-195）、允许的进程组（196-207） |
| `gpu_vendor/topology.py` | 47 | 卡的拓扑信息 |

#### `audio_windows.py`（257 行）——把“一条音频”变成“若干窗口”

| 函数 | 行号 | 输入 | 输出 |
|---|---|---|---|
| `inspect_source` | 49-83 | 来源字典 | `(identity, errors)`；identity 含整文件 SHA、PCM SHA、采样率、声道、帧数、时长、前后 stat |
| `reuse_vad` | 111-174 | 来源 + identity | `{status, reused, segments, errors}` |
| `plan_windows` | 175-207 | identity + 来源 + 运行策略 + VAD | 窗口列表（`window_id`、拥有区、上下文区、时钟映射、是否被边界限制） |
| `read_window` | 208-231 | 来源 + identity + 窗口 | 单声道 16 kHz 波形 + `input_pcm_sha256` |
| `units_from_windows` | 232-257 | 窗口列表 + identity | `(字单元列表, 异常列表)` |

**窗口规则**：核心窗 + 左右上下文，总长 ≤30 秒（`:190`）；VAD 静音点可以把切点往前挪最多 1 秒（`:186-190`）；窗口首尾相接铺满整段（`:202-204` 断言）。

#### `source_clock.py`（27 行）

`sample_bounds(start_seconds, end_seconds, sample_rate, mapping)`：v2 映射用 `Fraction(str(x))` 把十进制秒当有理数，起点向下取整、终点向上取整；v1 保留仅为复核历史坐标。**被谁调用**：`audio_windows.units_from_windows`、`schema` 的几何检查。

#### `qwen_backend.py`（553 行）——模型侧唯一入口

| 函数/类 | 行号 | 职责 |
|---|---|---|
| `native_token_rows` | 25-51 | 纯 CPU 核多行生成的 token 几何（提示矩形、EOS 位置、EOS 后 token 是否只有填充） |
| `length_batches` | 53-62 | 按长度排序后分批，控制批内最长/最短比 |
| `model_configurations` / `apply_asr_compute` / `loaded_compute_facts` | 64-105 | 递归找子配置并设置精度/注意力；读回实际生效值 |
| `ASRAttentionTrace` | 110-183 | 观测实际用了哪种注意力实现 |
| `QwenBackend` | 185-538 | `__init__`（加载分词器/模型，187-330）、`transcribe`（373-392）、`_transcribe_original_rows`（397-443）、`align`（444-491）、`snapshot_stats`/`metrics`（492-538） |
| `FixtureBackend` | 539-553 | CPU 夹具后端，合成数据用 |

**被谁调用**：`worker.py:263-267` 选后端；`worker.py:164,182` 调 `transcribe`/`align`。

#### `native_transformers_batch.py`（240 行）——批次文件事务

`NativeTransformersBatchStore`（100-240）：`identity.json` 定生产者/合同/模型/批大小 → `plan.json` 定 cohort 切分 → `batches/<cohort_sha>.json` 存整批结果 → `attempts/*.json` 存失败现场。`run_batch()`（192-238）是唯一入口：**先整批落盘并读回，才逐行返回**；未提交的 cohort 提供原全部行，否则被拒。

**被谁调用**：`qwen_backend.py`（合批路径）、`run_native_batch_pilot.py`、`audit_native_transformers_batch.py`、`native_batch_readback.py`、两个 `prepare_*` 脚本。

#### `native_b1_feature_padding.py`（238 行）

逐条按官方 B1 取特征与提示，再在特征层合批；`validate_source_bindings()` 被 `freeze_contract.py:55-57` 与 `contracts.asr_feature_profile()` 调用，用来钉住 7 份来源/配置绑定。

#### `work_queue.py`（263 行）

| 定义 | 行号 | 职责 |
|---|---|---|
| 完成位图编解码 | 21-37 | 每分片一个定长槽位 |
| `stable_lock` / `ProductionRunLock` | 51-68 | 稳定文件锁 |
| `read_index` | 70-85 | 读分片索引并核 SHA |
| `WorkQueue` | 86-198 | `__init__`（90-111）、`receipt`（114-139）、`read_shard`（129-139）、`write_completion`（140-155）、`claim_next`（156-192）、`is_complete`（193-198） |
| `Claim` | 199-258 | 上下文管理器：`assert_current`、`sources`、`journal`、`finish` |
| `prepare_queue` | 260-263 | 首次建队列（只能由父进程调用） |

**被谁调用**：`worker.py:253-256`（正常路径）、`gpu_launch.py:140-141`（建队列）、`bounded_speed_queue.py`（提速路径）。

#### `sidecar.py`（106 行）

`SidecarStore`（10-69）：`path()` 强制镜像映射（`dataset/相对路径.annotation.json`）；`publish()` 不可覆盖 + 回读 + 补 commit；`lookup()` 语义回读 + 源 stat 复核。`StageCache`（77-106）按输入摘要寻址，命中即重验。

#### `schema.py`（413 行）

| 定义 | 行号 | 作用 |
|---|---|---|
| `SCHEMA` / `SUCCESS` / 各类正则 | 17-23 | 记录版本、成功状态集合、SHA 与来源 ID 形式 |
| `quality_reasons` | 130-219 | 生成“为什么不能当候选”的原因列表（三十余条） |
| `build_record` | 220-234 | 组装侧车并给判定：`failed`/`unknown`/`candidate`，三者都 `usable_for_training=False` |
| `validate_record` / `assert_valid_record` | 235-413 | 结构、类型、有限性、跨字段一致性 |

**改动影响面**：改判定条件等于改数据资格，属于合同变更。

#### `quality_axes.py`（165 行）

`quality_axes(record)`（113-160）：输出可重建的分轴计数（文字、边界、语言、来源、身份各自的待审数量），供报告与图使用；**不产生人工真值**。

#### `report.py`（193 行）

`verify_run(contract, index, queue_root, output_root, full=False, …)`（41-175）：GPU 退出后的核对——分片回执、侧车数量、完成位图、孤儿文件与软链接清单。`main`（176-193）是命令行。**调用方**：`gpu_launch.py:309-315`、`audit_outputs.py:24,57`。

#### `union_coverage.py`（778 行）

`validate_union(spec_path, spec_sha, mode='plan', …)`（735-770）：核心是 `_validate()`（550-734），用磁盘索引保证“每个来源只被读一次、不重不漏”。内部 `_select_old_source`（537-549）决定某源该用旧结果还是该重算，`_terminal`（334-373）核终态。**调用方**：`cloud_gate.py:209-212`、`union_consumer.py`、`finalize_remaining_roles.py`。

#### `bounded_speed_runtime.py`（715 行）

新提速生产者的运行准入。关键：`runtime_options`（250-257）解析命令行里的运行规格；`guard_native_formal_route`（258-276）拦“用试验身份跑正式路径”；`compile_runtime`（400-509）把规格编译成带文件钉子的运行上下文；`RuntimeLease`（568-700）在进程内持租约并用密封内存文件（memfd）传递证明。**调用方**：`worker.py:248-250`、`gpu_launch.py:328-343`。

#### `queue_handoff.py` 一族（471 / 380 / 423 / 362 / 406 行）

四个交接实现是同一套结构：`check()` 只核固定输入 → `request_value()`/`observation()` 生成并核“请求/观察” → `publish()` 原子写 `runtime-spec.json` + `reply.json`。

| 文件 | 定位 |
|---|---|
| `queue_handoff.py` | 排队后等真实平台观察，再交给“可移植后继”运行器 |
| `bounded_speed_queue_handoff.py` | 提速版的队列交接（含专用显卡来源校验） |
| `bounded_speed_handoff.py` | 提速版的材料交接（按分片登记已封账、持有租约） |
| `native_batch_queue_handoff.py` | 原生合批 v2 的交接（额外核 B1 特征依赖） |

这四个文件函数名高度重复，见第 8 节第 4 条。

#### 准入相关

| 文件 | 关键函数（行号） | 说明 |
|---|---|---|
| `cloud_gate.py` | `verify`（117）、`check_union_data`（202）、`main`（236） | 纯 CPU 重读两卡证据，决定是否允许生成八卡启动件 |
| `make_admission_spec.py` | 45 行全文 | 生成 30/3000/512 三段规格 |
| `make_cloud_launcher.py` | 90 行全文 | 渲染可审阅的八卡启动脚本（不部署） |
| `verify_cloud_mounts.py` | 34 行全文 | 只核持久挂载，不建目录 |
| `native_batch_admission.py` | `check_quality`（180-206）、`check_formal_bundle`（210-232）、`_full_check_native_selected_pilot`（234-313）、`check_native_selected_pilot`（400-431）、`prepare_formal_code`（432-474） | 正式准入的唯一判定处；`REVIEWABLE_CONTROL` 在 207 行 |

---

## 5. `tools/ptn_resume/`：接续与 LM（27 个文件，6,658 行）

### 5.1 构建侧

| 文件 | 行数 | 关键定义（行号） | 输入 → 输出 |
|---|---:|---|---|
| `build.py` | 173 | `legacy_files`（17）、`verify_source`（30）、`config_from_legacy`（53）、`validate_materials`（66）、`yaml_config_bytes`（103）、`build`（116）、`main`（167） | 私有 spec → 新包目录（`source/`、`runtime/`、`resume.yaml`、`resume-contract.json`、`source_delta.json`、`source_manifest.json`、`bundle_files.json`） |
| `build_lm_package.py` | 410 | `SPEC_KEYS`（38 起），构建流程同 `build.py` 但方向相反 | LM spec → LM 冻结包（单轮、冷优化器/冷缩放器） |
| `contracts.py` | 124 | `BASE`/`ORIGINAL`/`PREDECESSOR`/`CHECKPOINT`（11-20）、`PATCH_NAMES`（22-23）、`RUNTIME_NAMES`（24-25）、`CLOCK`（26-28） | 固定身份常量；被 `state_gate.py` 与 `runtime/*` 导入 |
| `state_gate.py` | 113 | `verify_package`（35）、`verify_restored`（66） | 已构造并载入的 CPU 对象 → 精确状态证明（`runtime_admission:false`） |

`state_gate.verify_restored()` 的检查清单（`:76-104`）：成员清单与 190 成员源树 → 两个新鲜终态观察 → 无 CUDA 上下文 → `model47` 的 SHA/大小 → 载荷只允许 4 个键 → 823 状态/584 动量 → 配置与 `resume.yaml` 全等 → 说话人表身份 → 头形状 `[357822,192]` → 逐张量精确比较 → 585 参数与 `[583,2]` 分组顺序 → `SGD` → 动量形状/有限性 → 缩放器 → **实际调用冻结调度器与间隔到 705047，核 `lr=0.72`、`margin=0.2`** → 复查检查点 stat。

### 5.2 证据侧

| 文件 | 行数 | 关键定义（行号） | 作用 |
|---|---:|---|---|
| `three_segment_lm_check.py` | 1,334 | `checkpoint_evidence`（351）、`speaker_map`（411）、`lm_label`（430）、`reviewed_events_scope`（460）、`scan_successor_log`（481）、`resource_return`（499）、`source_identity`（525）、`original_189_index`（587）、`original_95_index`（598）、`lm_cold_start_contract`（678）、`package_lm_contract`（767）、`platform_terminal`（894）、`evaluate`（913）、`summarise`（1111）、`exit_code`（1124） | 三段主训练 + LM 冷启动的合同生成（`build`）与检查（`check`） |
| `engineering_two_gpu.py` | 552 | 两卡工程验证编排 | 只验证工程，不授予正式资格 |
| `inspect_engineering_artifacts.py` | 243 | 读两卡产物生成摘要 | 复核 |
| `live_process_probe.py` | 213 | 只读 `/proc` 核 8 个 rank + 192 个加载进程 | 真实容器内取证 |
| `refresh_terminal_refs.py` | 127 | 刷新控制面终态引用 | 排队后时效修复 |

### 5.3 `runtime/`（包内执行器，仓库侧源件）

| 文件 | 行数 | 职责 |
|---|---:|---|
| `entry.py` | 493 | **唯一父入口**：显式执行、稳定锁、CPU 恢复、空闲八卡检查、零自动重试；`FATAL` 正则（26 行）扫日志致命行；`write_new`（33-43）原子发布 |
| `worker.py` | 126 | 单 rank 入口：先持全部稳定锁的共享锁 → 核 8 卡 → `entry.worker_ready_before_cuda(rank)` → `admitted_worker`；所有 rank 齐了才解除前向/优化门 |
| `strict_restore.py` | 141 | 公共门：`configure(contract, contract_sha, package_sha)` 绑定新包，导出 `need`/`CONFIG`；引入 `state_gate` 与 `ptn_contracts` |
| `cpu_check.py` | 53 | 真实 `model_47` 的 CPU 完整恢复，不前向、不建输出、不初始化 CUDA |
| `epoch_guard.py` | 34 | 每轮核来源与 192 个加载进程 |
| `loader_worker_audit.py` | 71 | 只读采集 fork worker 证据（在任何批次交给训练器之前） |
| `predecessor_gate.py` | 25 | 等同一控制面的稳定新材料 |
| `launch.sh` | — | 容器内入口：`exec .../python -B "$PACKAGE_ROOT/runtime/entry.py"`（12 行） |

这些文件是**仓库侧源件**，冻结包里的同名文件由 spec 逐份钉 SHA 提供（`INTERFACE.md:19`）；两者内容未必相同，仓库里的文件不能当作包内同名文件的等价物。

### 5.4 `lm_launch/`（LM 启动门）

| 文件 | 行数 | 作用 |
|---|---:|---|
| `run_lm.sh` | 132 | 唯一入口：7 项前置检查（目录为新目录、不在代码目录内、`model_115.pt` 名字与 SHA、`spk2id.json` 存在、关键环境变量不被覆盖、未设 `SKIP_*`）→ 真跑 `check_b6_lm_init.py --expected-epoch 115` → 写 `lm_initialization.json` → 交给 `train_b6_mix_cloud_8gpu.sh` |
| `lm_stage_gate.py` | 323 | 常量：`NUMBER=13075`、`BATCH=48`、`WORLD=8`、`MARGIN=0.3`、冷启动标记、禁止出现的日志行（37-51）；`verify`（119）、`verify_speaker_map`（170）、`verify_recipe`（217）、`run`（248）、`postflight`（280） |
| `submit.conf.yaml` | 97 | 平台任务配置草案（镜像/队列/规格/挂载留 TODO） |
| `env.template` | 49 | 环境模板 |
| `README.md` | 89 | 为什么是这层包装、怎么用、还缺哪些输入 |

### 5.5 `tools/` 其余目录

| 目录/文件 | 行数 | 说明 |
|---|---:|---|
| `lm_continuation/readiness.py` | 203 | 历史两段接续的 CPU 准备判据；**保持原判据，继续拒绝失败任务**（`README.md:85`） |
| `ptn_reference/audit_reference.py` | 107 | 只核参考源码字节与语法 |
| `ptn_reference/archive/` 5 个文件 | 344 | 旧启动文件只读归档（`epoch_guard`/`loader_worker_audit`/`predecessor_gate`/`strict_restore`/`worker`） |
| `ptn_diagnostics/inspect_numeric_cpu.py` | 182 | 检查点量级 CPU 扫描，不建模型 |
| `tools/task_names.py` | 27 | 任务命名（`zhounan_` 前缀） |
| `tools/ptn_forward_probe.py` | 213 | 前向探针 |

---

## 6. `scripts/`、`tests/`、旁线微调

### 6.1 `scripts/`（20 个文件，2,673 行）

| 文件 | 关键定义 | 作用 |
|---|---|---|
| `checkpoint_utils.py` | `embedding_payload`（30）、`load_embedding_checkpoint`（86） | 从“纯权重”或“训练检查点”两种格式取出可推理的骨干 |
| `export_checkpoint.py` | `export_checkpoint`（19）、`main`（44） | 训练检查点 → 推理权重（附 `model_config`） |
| `eval_vox1_checkpoints.py` | `eer_percent`（80）、`extract_embeddings`（136）、`load_checkpoint`（161）、`eval_one`（165）、`main`（182） | Vox1 O/E/H 等错误率 |
| `eval_ood_checkpoints.py` | 同类 | 域外试验 |
| `eval_duration_matrix.py` / `eval_word_matrix.py` / `eval_duration_cpu_cell.py` | 时长/词级矩阵评测 | 旁线 |
| `check_repository.py` | 仓库自检 | 只读 |
| `align_training_units.py` / `prefill_word_cpu.py` / `prepare_word_eval.py` / `prepare_duration_finetune.py` / `train_word_finetune.py` / `train_duration_finetune.py` / `launch_word_two_gpu.py` / `launch_word_v3.py` / `launch_duration_two_gpu.py` / `run_cpu_checks.py` / `summarize_duration_results.py` / `test_checkpoint_export.py` | 旁线训练/评测/启动 | 完整字与整数秒裁片实验 |

### 6.2 `tests/`（20 个文件，4,003 行）

按守护对象分四组：

| 组 | 文件 |
|---|---|
| 模型与掩码 | `test_mask_input_contract.py`、`test_varlen_masking.py`、`test_cold_start_state.py` |
| 训练与调度 | `test_schedule_and_launch_guards.py`、`test_training_numeric_sampling.py`、`test_training_failure_capture.py`、`test_vad_segment_boundary.py` |
| 词级 | `test_word_align.py`、`test_word_eval.py`、`test_word_schema.py`、`test_word_train.py`、`test_word_gpu_resources.py`、`test_word_v3.py` |
| 时长与接续 | `test_duration_data.py`、`test_duration_eval.py`、`test_duration_metrics.py`、`test_duration_resources.py`、`test_duration_resume_guard.py`、`test_duration_train.py`、`test_lm_continuation.py` |

### 6.3 旁线微调包

| 包 | 文件数/行数 | 入口 | 说明 |
|---|---|---|---|
| `word_ft/` | 8 / 2,153 | `word_ft/train.py` | 完整字微调 v1；`schema.py` 定义 `word-complete-units/v1` |
| `word_ft_v3/` | 4 / 808 | `word_ft_v3/train.py` | v3（`topology.py` 管数据拓扑，`resources.py` 管卡） |
| `duration_ft/` | 8 / 1,867 | `duration_ft/train.py` | 整数秒裁片微调（**旧诊断路线**，不能替代完整字输入） |

---

## 7. 一张“改这里会影响谁”的表

| 你改的文件 | 直接影响 | 需要同步/重跑 |
|---|---|---|
| `redimnet2/redimnet2.py` 默认路径 | 推理、训练、评测、所有已发布权重 | `tests/test_mask_input_contract.py`、`tests/test_varlen_masking.py`；已发布数字要重算 |
| `redimnet2/layers/poolings.py` | 同上 | 同上 |
| `wespeaker_lite/bin/train.py` | 所有训练与微调 | 短测；接续包里的执行器若引用它需重打包 |
| 训练栈 `utils/executor.py` 或 `numerical_failure.py` | 主训练；**接续包把这两个文件作为“允许的差异”** | `tools/ptn_resume` 的 spec 与审阅 |
| 任一配方 yaml | 该实验 | 短测；若已被冻结包引用则重打包 |
| `tools/open_annotation/` 的 16 个引擎文件之一 | 标注系统身份 | 重新冻结包 + 重跑 pilot + 重新准入 |
| `tools/open_annotation/` 的 `REVIEWABLE_CONTROL` 四文件之一 | 只影响 CPU 辅助路径 | 附独立审阅原件，且引擎文件全等 |
| `tools/ptn_resume/contracts.py` 的身份常量 | 接续包与状态门 | 新版本包，旧包作废 |
| `tools/ptn_resume/lm_launch/*` | LM 启动门 | 重新冻结 LM 包并更新两个哈希 |
| `docs/*.md` | 无运行时影响 | `docs/reviews/check_document_links.py`、`check_report_tables.py` |

---

## 8. 可疑点与已知不一致（读代码时实际发现的）

按对判断的影响排序。前三条会让人得出错误结论，后几条只是维护负担。

### 8.1 配方里的 `loss:` 在两个正式配方中不生效

`official_repro_ptn.yaml:84-85` 写 `loss: CrossEntropyLoss`、`loss_args: {}`，`official_repro_lm.yaml` 同理；但 `project_type: sphereface2` 的投影头**自己算损失并返回元组**（`models/projections.py:201-211`），训练器见元组就取第二项（`utils/executor.py:263-268`），`criterion` 从头到尾没被调用（`train.py:596` 只是构造了它）。后果：改 `loss`/`loss_args` 不会有任何效果，也不会报错。读配方时容易误以为损失是交叉熵。

### 8.2 `_stage_inputs()` 有一个从未使用的参数

`worker.py:104-110` 的签名是 `_stage_inputs(stage, window, context, contract, contract_sha, models)`，函数体完全没用到 `context`；两个调用点（`:158`、`:178`）都传了它。这是死参数，不会出错，但会让人误以为阶段输入摘要包含来源上下文。

### 8.3 注释里明确标出的“地雷”分支，当前走不到

`redimnet2.py:456-472` 注释说：骨干里的掩码截断写成了 `mask[:, :, :T_keep]`，切的是频率轴（长度 1）等于没切；因为 `Wrap.forward` 已经提前把两者一起截到 `time_stride` 整数倍（`:649-671`），今天走不到这里。注释自己写了“这是一颗留着的地雷，改成 `...` 才是对的”。这段分支不属于已修代码。

### 8.4 四个交接模块是同构实现

`queue_handoff.py`（471）、`bounded_speed_queue_handoff.py`（362）、`bounded_speed_handoff.py`（380）、`native_batch_queue_handoff.py`（406）里，后两者与前两者共享整套函数名与调用顺序（`build_controller`、`verify_outer`、`check`、`validate_resource_snapshot`、`request_value`、`observation`、`_expected_runtime`）。改一份不会自动改另一份；判断“哪个是当前路径”只能看调用方或对应中文说明。

### 8.5 两处“看起来像现行代码”的历史副本

- `tools/open_annotation/legacy_snapshots/production_20261025/`：23 个文件的历史生产包快照。与当前同名的 17 个文件中，**多数字节不同**（`audio_windows.py`、`contracts.py`、`gpu_bootstrap.py`、`gpu_launch.py`、`normalization.py`、`quality_axes.py`、`qwen_backend.py`、`report.py`、`schema.py`、`sidecar.py`、`work_queue.py`、`worker.py` 这 12 个不同，`alignment_observer.py`、`gpu_challenge.py`、`gpu_common.py`、`manifest_builder.py`、`source_registry.py` 5 个相同）。目录 README 声明它是历史身份、不是现行入口，但文件名相同，容易被误 import 或误 diff。
- `tools/ptn_reference/archive/`：5 个旧启动文件（`strict_restore.py` 等）与 `tools/ptn_resume/runtime/` 同名。前者是归档，后者是新实现；`build.py` 从 spec 取新文件，不从 archive 取。

### 8.6 仓库侧 `runtime/` 与包内 `runtime/` 不是同一份

`tools/ptn_resume/runtime/` 是仓库维护的新执行器源件（`launch.sh:12` 指向 `$PACKAGE_ROOT/runtime/entry.py`），而冻结包里的同名文件由 spec 的 `runtime_files` 逐份提供并核 SHA（`INTERFACE.md:19`、`state_gate.py:62`）。比对时以包的 `bundle_files.json` 为准；仓库里的文件不等于包内身份。

### 8.7 配方注释与字段的历史残留

`official_repro_ptn.yaml:24-25` 写着 `grad_accum_steps` 已删除（全仓检索无命中，确认训练栈不支持梯度累积），`:81-82` 写着 `gpus:` 不被读取。这些是**已经澄清过的**残留说明，不是死代码；但复制配方做新实验时容易把这两行当成可配置项。

### 8.8 文档与代码不一致的已修项

旧版本 02 文档里的“B0–B6 参数量表”在包内没有对应字典（`redimnet2/` 检索 `b0`–`b6` 无命中），型号完全由配置决定；旧版本把处理链写成“先增强后取段”，与 `dataset.py:396-441` 的实际顺序（先按边界读盘取段、再变速）相反。本次重写已改正，这里留一条记录，避免有人按旧版记忆改代码。

---

## 9. 未验证

| 项 | 状态 |
|---|---|
| 任何显卡上的实际行为 | 未验证。本次只读源码与仓库内记录，没有执行训练或标注 |
| 各配方的实际数据规模与说话人数 | 未验证。需要读 `data.list` / `utt2spk` 并统计 |
| `tools/open_annotation/tests/`（76 个文件）是否全部通过 | 未运行；只读了测试清单与目录结构 |
| `docs/` 其余文档（04/05/08/09/10/12/13/15/16）与代码的一致性 | 未逐条复核 |
| `applications/speaker_rec/scripts/cloud/*.sh` 的完整分支 | 只读了作用与入口，未逐行审 |
| `word_ft/`、`word_ft_v3/`、`duration_ft/` 的内部实现 | 只看入口与文件清单，未逐函数读 |
