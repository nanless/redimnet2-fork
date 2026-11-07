# 02 · 架构：分层、模块职责、数据流与外部边界

上一篇：[背景与动机](01-background.md) · 下一篇：[代码走读](03-code-walkthrough.md)

这篇文档解决一个问题：**这个仓库里到底有哪几块东西、它们之间的依赖方向是什么、数据从哪进哪出、哪些部分已经冻住不能动。** 目标是三个月后接手的人能凭这篇文档找到文件、知道改哪个文件会影响哪条链路，而不是凭目录名猜。

核对口径：2026-11-08 只读核对 `dev_a800_2gpus:/root/code/github_repos/redimnet2-fork`。**仓库状态不写死**——HEAD 与未提交改动数用 `git rev-parse --short HEAD`、`git status --porcelain | wc -l` 查，或看自动生成的 [STATUS 第 1 节](STATUS.md)。本文引用行号的地方只对当时的提交成立，改动后请以函数名重新定位。**没有实际读到的内容一律标注“未验证”**，不靠目录名推断。

---

## 1. 仓库体量实测

这一节解决“这个仓库有多大、力气该花在哪”。数字全部来自本次 `find ... -name "*.py" | xargs wc -l` 的实测。

| 目录 | py 文件数 | 总行数 | 里面是什么 |
|---|---:|---:|---|
| `tools/` | 238 | 48,412 | 标注系统、接续包、诊断与对照工具；**行数占全仓 63%**，是维护重心 |
| `applications/speaker_rec/` | 32 | 12,012 | 训练栈（精简 WeSpeaker）、VAD 标注、云端启动脚本、示例配方 |
| `tests/` | 20 | 4,003 | 仓库级回归测试（变长掩码、调度与启动门、词级与时长微调） |
| `redimnet2/` | 12 | 3,674 | 模型本体：声学前端、骨干、池化 |
| `scripts/` | 20 | 2,673 | 权重导出、Vox1/域外评测、词级/时长微调入口 |
| `word_ft/`、`word_ft_v3/`、`duration_ft/` | 8 / 4 / 8 | 2,153 / 808 / 1,867 | 完整字微调（v1、v3）与整数秒裁片微调两条旁线 |
| `docs/`、`.agents/`、仓库根 | 7 / 3 / 1 | 630 / 518 / 26 | 文档自检脚本、技能脚本、`hubconf.py` |
| **合计** | **353** | **76,776** | |

`tools/` 内部再拆：

| 子目录 | py 文件数 | 行数 | 说明 |
|---|---:|---:|---|
| `tools/open_annotation/` | 189 | 38,639 | 其中顶层 90 个（21,431 行）、`tests/` 76 个（13,221 行）、`gpu_vendor/` 3 个（298 行）、`legacy_snapshots/` 20 个（3,689 行） |
| `tools/ptn_resume/` | 27 | 6,658 | 主训练 47→48 接续包构建、状态门、三段证据检查、LM 冻结包与启动门 |
| `tools/lm_continuation/` | 6 | 1,293 | 历史两段接续的 CPU 准备判据（仍保留原判据，见 `readiness.py`） |
| `tools/ptn_reference/` | 11 | 1,252 | 旧参考源码的只读归档与字节身份核对（`archive/` 下 5 个旧启动文件） |
| `tools/ptn_diagnostics/` | 3 | 330 | 检查点量级 CPU 扫描等只读诊断 |
| `tools/` 根 | 2 | 240 | `task_names.py`（任务命名）、`ptn_forward_probe.py`（前向探针） |

行数最大的 10 个 py 文件（`wc -l` 排序，全仓 353 个文件里）：

| 行数 | 文件 | 一句话 |
|---:|---|---|
| 1,529 | `applications/speaker_rec/wespeaker_lite/dataset/processor.py` | 样本清单逐条取样、切窗、变速、按语音活动取段 |
| 1,470 | `applications/speaker_rec/scripts/vad/annotate_firered_offline.py` | FireRedVAD 离线逐条标注，写 RTTM 侧车 |
| 1,334 | `tools/ptn_resume/three_segment_lm_check.py` | 三段主训练 + LM 冷启动的证据合同生成与检查 |
| 1,276 | `applications/speaker_rec/examples/vox2/scripts/prepare_data.py` | 只练 VoxCeleb2 的数据准备（下载→解压→转码→清单） |
| 944 | `applications/speaker_rec/wespeaker_lite/dataset/dataset.py` | `Dataset()` 拼装、变长补零、按长度分桶 |
| 886 | `applications/speaker_rec/wespeaker_lite/bin/train.py` | 训练主程序 |
| 778 | `tools/open_annotation/union_coverage.py` | 旧结果与剩余输入的联合覆盖核对 |
| 757 | `applications/speaker_rec/scripts/vad/firered_rttm_common.py` | RTTM 读写与校验的共用件 |
| 715 | `tools/open_annotation/bounded_speed_runtime.py` | 新提速生产者的运行准入（锁、密封证明、租约） |
| 697 | `tools/ptn_resume/tests/test_three_segment_lm_check.py` | 上表的正反例测试 |

---

## 2. 五层结构与依赖方向

这一节解决“谁可以 import 谁”。仓库不是按包安装的库，而是靠 `PYTHONPATH` 与目录约定拼接，**依赖方向错了会在导入期直接炸**，所以先记住方向。

```text
L4 文档与证据    docs/  checkpoints/reports/  docs/plans/  docs/reviews/
                 （只描述，不参与运行）
                        ▲
L3 运维与接续    tools/open_annotation/   tools/ptn_resume/   tools/lm_continuation/
                 （合同、冻结、GPU 父进程、队列、侧车、接续包、状态门）
                        ▲ 只读调用
L2 旁线实验      word_ft/  word_ft_v3/  duration_ft/  scripts/
                 （完整字微调、整数秒裁片、导出与评测）
                        ▲
L1 训练栈        applications/speaker_rec/
                 launch.sh → torchrun → wespeaker_lite/bin/train.py
                 → utils/executor.py → dataset/ → models/ → 配方 yaml
                        ▲ import
L0 模型库        redimnet2/  （只依赖 torch / torchaudio / numpy，不认识训练栈）
```

实测的导入事实（不是约定，是代码里就这么写的）：

| 事实 | 证据 |
|---|---|
| `redimnet2/` 不 import 任何 `applications/` 或 `tools/` 下的东西 | `redimnet2/__init__.py:1-4` 只 import torch 与本包；`redimnet2/redimnet2.py:1-48` 只 import torch 与 `layers.*` |
| 训练栈靠 `import redimnet2` 拿到骨干 | `applications/speaker_rec/wespeaker_lite/models/speaker_model.py:17-24`：`get_speaker_model()` 只接受以 `ReDimNet2` 开头的名字，用 `getattr(redimnet2, name)` 反射 |
| 标注系统的父进程与队列代码不 import torch，也不 import 训练栈（torch 只在 worker 子进程里被 `qwen_backend` 用到） | `tools/open_annotation/contracts.py:1`（“stdlib-only helpers”）、`gpu_launch.py:115` 注释“No torch import in this parent” |
| 接续包构建器不 import 训练器 | `tools/ptn_resume/build.py:116-166` 只做文件复制与摘要，不导入 `train.py`；`build_lm_package.py:16-27` 写明 “never imports the trainer” |
| 标注系统唯一接触 torch 的地方在 worker 进程内部 | `worker.py:263-267` 才 `from qwen_backend import QwenBackend`；GPU 父进程与队列代码都是纯 CPU |

依赖方向的实用含义：**改 `redimnet2/` 会同时影响推理、训练、评测和所有冻结包**；改 `tools/open_annotation/` 里被合同钉住的文件会作废既有准入（见第 8 节），但不影响训练。

---

## 3. 第 0 层：`redimnet2/` 模型库

这一节解决“模型是怎么把一个波形变成一个 192 维向量的”，以及“为什么变长批处理需要一路传掩码”。

### 3.1 三个对外入口

| 入口 | 位置 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| `load_custom(model_name, train_type, dataset)` | `redimnet2/__init__.py:18-35` | 型号/阶段/数据集三字符串 | 已 `eval()` 的 `ReDimNet2Wrap` | 从 GitHub Release 下载 `{model}-{dataset}-{train_type}.pt`（`__init__.py:6,19-21`），用文件里的 `model_config` 自描述实例化（`:23`），`missing_keys`/`unexpected_keys` 非空即抛错（`:25-29`） |
| `from_pretrained(...)` / `redimnet2(...)` / `hubconf.redimnet2(...)` | `__init__.py:38-47`、`__init__.py:50-53`、`hubconf.py:10-26` | 同上 | 同上 | 三个名字同一行为；`hubconf.py` 是 `torch.hub` 的入口壳 |
| `ReDimNet2Wrap(**model_config)` | `redimnet2/redimnet2.py:508-610` | 配置字典 | `nn.Module` | 训练与自定义实验的正式入口，`ReDimNet2Custom` 只是它的别名（`:687-688`） |

**外部边界**：`load_custom` 的权重来源是硬编码的 GitHub Release 地址（`__init__.py:6`）。离线机器上它必然失败，这不是 bug，是仓库对“公开权重”的边界。仓库内训练/评测不经过它，而是直接构造 `ReDimNet2Wrap` 再 `load_state_dict`。

### 3.2 形状与数据流

`ReDimNet2Wrap.forward(x, lengths=None)`（`redimnet2/redimnet2.py:611-686`）一条路走完：

```text
波形 (B, T) 或 (B, 1, T)
  │  [可选] make_sample_mask(lengths) → (B, 1, T)           masking.py:43
  ▼ spec(x[, mask])
梅尔/时频特征 (B, 1, F, T)   F=72 默认                     features.py:135-260 或 features_tf.py:397-478
  │  时间维截到 time_stride 的整数倍（有掩码时才截）        redimnet2.py:649-671
  ▼ backbone(x, mask)                                        redimnet2.py:423-507
逐 stage 的 1D 特征列表 → fin_wght1d 软加权 → (B, C·F, T)
  ▼ 池化（ASTP 默认）                                       poolings.py:147-241
(B, 2·C·F) → BatchNorm1d → Linear(→ embed_dim=192)
  ▼ 可选 bn2（emb_bn）
embedding (B, 192)
```

配置默认值在 `ReDimNet2Wrap.__init__` 的签名里（`redimnet2.py:509-560`）：`F=72`、`C=24`、`embed_dim=192`、`pooling_func='ASTP'`、`feat_type='pt'`、`global_context_att=True`、`hop_length=160`。三语料 B6 正式配方把这些覆盖成 `C=64`、`out_channels=224`、`return_2d_output=True`、`agg_gnorm=True`、`feat_type='tf'`（`applications/speaker_rec/examples/vb2_vox2_cnc2/recipes/redimnet2/b6/official_repro_ptn.yaml:97-169`）。

**型号 B0–B6 不在包内**：包内没有 B0–B6 的字典或分支（对 `redimnet2/` 全目录检索 `b6`/`B6` 无命中），规模完全由配置里的 `C`、`out_channels`、`stages_setup` 决定。所以“这是 B6”只能从配方或检查点里的 `model_config` 读出来。

### 3.3 掩码契约：三处必须同时成立

这是全仓最容易改错的地方。传 `lengths` 之后，掩码要一路穿到池化，任何一处漏掉都会静默改变结果（不是报错）。

| 环节 | 位置 | 职责 |
|---|---|---|
| 生成 | `layers/masking.py:43-77` `make_sample_mask` | `(B,)` 真实采样点数 → `(B,1,T)` 的 0/1 掩码 |
| 前端 | `layers/features.py:222`、`features_tf.py:449` 的 `forward(x, mask=None)` | 归一化、预加重、均值统计都只在真实样本上算 |
| 骨干入口 | `redimnet2.py:443-451` | 时间维统一**截断**到 `time_stride` 整数倍；掩码同步截（用 `...` 保留最后一维） |
| 下采样 | `layers/masking.py:147-162` `downsample_frame_mask`、`:218-238` `MaskDownConv2d`、`:239-267` `MaskResample` | 每次时间步长变化时掩码跟着变 |
| 归一化 | `layers/masking.py:268-298` `MaskedGroupNorm` | 组归一化只用真实帧 |
| 池化 | `layers/poolings.py:181-241` | 上下文均值/方差、注意力 logits、加权矩都排除补零帧 |
| 截断一致性 | `redimnet2.py:649-671` | `Wrap.forward` 提前把 `(x, mask)` 一起截到 `time_stride` 整数倍，避免骨干里两条分支帧数不同 |

两个已写进代码注释的坑，接手时必须知道：

1. **“传了 `lengths`”不等于“和逐条推理逐位相同”**。只要批里有补零，边界帧会进入卷积感受野，余弦会掉到 0.92–0.98（`redimnet2.py:466-495` 的实测记录；`tests/test_varlen_masking.py` 记的是下限）。文档里不许写成“等于 1.0”。
2. **补零却不传 `lengths`** 是错误用法，同一说话人的余弦会掉到 0.326 量级（同上注释）。

### 3.4 池化与分类头

| 模块 | 位置 | 作用 |
|---|---|---|
| `ASTP` | `layers/poolings.py:147-241` | 注意力加权均值 + 加权标准差；`mask_aware=True`；无掩码支用 `torch.var`（无偏），有掩码支把有偏矩换算成无偏，两路对齐（`:195-207`） |
| `TAP`/`TMAP`/`TSDP`/`TSTP`/`MHASTP`/`MQMHASTP` | 同文件 `:30/53/79/99/244/318` | 其它池化，配方可换 |
| 分类头 | **不在 `redimnet2/`**，在 `applications/speaker_rec/wespeaker_lite/models/projections.py` | 训练期的大间隔分类器与损失，推理时丢弃 |

这条边界很重要：`redimnet2/` 到 embedding 为止；**“embedding 之后算什么损失”属于训练栈**。所以导出的推理权重里没有 `projection.*`，而训练检查点里有（见第 5.4 节）。

---

## 4. 第 1 层：`applications/speaker_rec/` 训练栈

这一节解决“一次训练从敲命令到写出 `model_N.pt` 中间发生了什么”。

### 4.1 启动与进程模型

```text
applications/speaker_rec/launch.sh <config.yaml> [num_gpus] [overrides...]
  ├─ 解析 Python 解释器：REDIMNET2_PYTHON → REDIMNET2_ENV → 名为 redimnet2 的 conda 环境 → PATH   (launch.sh:34-60)
  ├─ 从同一环境取 torchrun（避免 python 与 torchrun 来自两个环境）                              (launch.sh:107-113)
  └─ torchrun --nproc_per_node=N wespeaker_lite/bin/train.py --config <yaml> ...                  (launch.sh:162)
        └─ train.py 读 RANK / WORLD_SIZE / LOCAL_RANK → dist.init_process_group('nccl')            (train.py:176-186)
```

要点：

- **卡数不由配置决定**。`launch.sh` 第二个参数是唯一来源；配方里写 `gpus:` 不会被读（`official_repro_ptn.yaml:81-82` 的注释就是这么写的）。
- 每个 rank 各自读同一份样本清单，`spk2id()` 在**每个 rank 上分别**从 `utt2spk` 重建（`train.py:226-232`）——所以 `utt2spk` 的行序必须稳定，否则各 rank 的类别编号会不一致。
- `models/` 的创建与恢复判定之间有 `dist.barrier`（`train.py:195-207`），防止非 0 rank 先列目录、误判“没有可恢复检查点”而从第 1 轮重新开始。

### 4.2 数据面

| 步骤 | 位置 | 输入 | 输出 |
|---|---|---|---|
| 读样本清单 | `train.py:224-232` | `train_label`（`utt2spk`） | `(utt, spk)` 列表 + `spk2id_dict` |
| 组装数据集 | `dataset/dataset.py:325-510` `Dataset()` | `data_type='raw_fast'`、`data.list`、`dataset_args` | `IterableDataset` 链 |
| 逐条取样 | `dataset/processor.py:550-651`（定长）或 `:652-760`（变长） | 一行 JSON + 配置 | 波形片段 + 说话人 |
| 增强 | `dataset/augs.py:227-525`（噪声/混响/低通/编解码）、`models/mix_augs.py:27-236`（Cutmix/Mixup/MixCutUp/OverlapMix） | 波形 | 增强波形 |
| 合批 | `dataset/dataset.py:511-515`（定长）、`:600-681` `collate_varlen`（变长） | 样本列表 | `{'key','label','wav'[,'lengths']}` |
| 分桶（可选） | `dataset/dataset.py:723-898` `BucketedLengthDataset` | 数据集 | 同一长度桶内的整批 |

`collate_varlen` 产出的 batch 字典键就是训练器读取的契约：`key`（语句名）、`label`（整数类别）、`wav`（右侧补零后的波形）、以及**仅变长时存在**的 `lengths`。`executor.run_epoch` 用 `batch.get('lengths', None)` 判断走哪条路（`utils/executor.py:225-231`），所以**老配方不加这个键就完全走旧路径**。

已写进代码的硬约束：

- 打开按语音活动取段（`dataset_args.vad_segments.enable`）必须同时打开变长（`dataset_args.varlen.enable`），否则 `Dataset()` 直接抛错（`dataset/dataset.py:346-355`），因为按语音活动取出的片段长度本来就是变的。
- 分桶打开时，`DataLoader` 的 `batch_size` 必须置 `None`、`drop_last` 必须去掉（`train.py:340-357`），否则会把已经成批的数据再乘一次批大小。
- 每轮结束的步数上限来自 `sample_num_per_epoch // world_size // batch_size`，为 0 时直接断言失败（`train.py:371-390`），因为训练循环用 `if (i+1) == epoch_iter: break` 收尾，为 0 会永不结束。

### 4.3 优化面

| 组件 | 位置 | 关键事实 |
|---|---|---|
| 模型 + 分类头 | `train.py:441-448` | `SpeakerModel(backbone, projection)`；`SpeakerModel.forward` 直接转发给骨干（`models/speaker_model.py:26-36`） |
| 损失 | `train.py:596` | `getattr(torch.nn, configs['loss'])`。**注意**：三语料 B6 配方用 `project_type: sphereface2`，该投影头自己算损失并返回 `(logits, loss)`（`models/projections.py:201-211`），训练器见元组就取第二项（`utils/executor.py:263-268`）。因此配方里的 `loss: CrossEntropyLoss` 在这条路上**不生效**（见 03 的可疑点清单） |
| 参数组 | `train.py:601-608` | 两组：骨干用 `optimizer_args.weight_decay`，投影头用 `loss_weight_decay_mult × weight_decay` |
| 学习率调度 | `utils/schedulers.py:97-320`；装配在 `train.py:616-640` | 纯 Python 类，不是 `torch.optim.lr_scheduler`；由**绝对迭代数**驱动（`executor.py:213-220` 每步 `scheduler.step(cur_iter)`），所以续训能自行复现轨迹 |
| 学习率缩放 | `train.py:617-633` | 默认 `lr_scaling: legacy`：配方峰值学习率会被 `world_size × batch_size / 64` 乘一遍。八卡每卡 96 条时倍率 12，配方写 0.06 实际跑 0.72（`official_repro_ptn.yaml:192` 有同样注释）。要“写多少跑多少”就写 `lr_scaling: base_lr` |
| 间隔调度 | `utils/schedulers.py:20-94` `MarginScheduler`；装配在 `train.py:641-647` | 三段式：`increase_start_epoch` 前固定 `initial_margin`，之后按 `exp` 或线性升到 `final_margin`，`fix_start_epoch` 之后固定 |
| 混合精度 | `train.py:681`、`executor.py:249-275` | `torch.cuda.amp.GradScaler(enabled=enable_amp)`；反向/更新都在 scaler 下 |
| 分布式 | `train.py:588` | `DistributedDataParallel(find_unused_parameters=False)` |
| 数值故障取证 | `executor.py:244-269` 调 `utils/numerical_failure.py` | 只在检测到非有限值时导出私有取证材料；这是 47 轮接续包替换/新增的两个文件之一（见第 6.6 节） |

### 4.4 磁盘产物

一次训练在 `exp_dir` 下写出这些东西，**外部工具（接续包、评测脚本、LM 启动门）都按这个布局读取**：

| 路径 | 内容 | 谁写 | 谁读 |
|---|---|---|---|
| `<exp_dir>/config.yaml` | 实际生效的完整配置（含推导值） | `train.py:650-654` | `scripts/eval_vox1_checkpoints.py:161`、`scripts/export_checkpoint.py:19`、LM 启动门 |
| `<exp_dir>/models/model_N.pt` | 训练检查点：`{'epoch','model','optimizer','scaler'}` 或纯 `state_dict` | `utils/checkpoint.py:141-199` | 续训扫描（`:31-56`）、LM 冷启动 |
| `<exp_dir>/models/final_model.pt` | 指向最后一轮的 12 字节软链 | 训练末尾 `train.py:778-782` | 评测脚本 |
| `<exp_dir>/spk2id.json` | 说话人名 → 类别号 | `train.py:566` 调用（定义在 `utils/utils.py:116-129`） | LM 重组分类头（`utils/utils.py:130-203`） |
| `<exp_dir>/val_log.csv` | 每轮 `vox1-eer-8s` 等指标 | 训练循环 `train.py:849`（读取在 `:394-400`） | 选轮、报告、三段证据检查 |

检查点写入是**原子**的：先写同目录 `model_N.pt.tmp.<pid>` 再 `os.replace`（`utils/checkpoint.py:186-197`），并且临时名刻意不匹配 `model_(\d+)\.pt`，免得被续训扫描当成最新轮。

### 4.5 评测面

训练过程中的验证是**每轮两次**：8 秒快评（`Vox1TestEvalDatasetSimple`，`eval_dur=8.0`）与整段全评（`Vox1TestEvalDataset`），见 `train.py:404-430`；指标计算在 `utils/executor.py:331-341`（`EER` 用 `roc_curve` + `brentq`）。

仓库级评测脚本（不训练）：

| 脚本 | 作用 |
|---|---|
| `scripts/eval_vox1_checkpoints.py` | 用给定权重在 VoxCeleb1 的 O/E/H 试验清单上算等错误率，与官方表对比 |
| `scripts/eval_ood_checkpoints.py` | 域外试验（Vox1-B、VOICES、CN-Celeb） |
| `scripts/export_checkpoint.py:19-43` | 把训练检查点剥成推理权重（只留骨干，附 `model_config`） |
| `applications/speaker_rec/scripts/cloud/evaluate_b6_mix_reproduction.sh` | 三语料复现的正式评测包装（训练权重 + 官方参考权重对比） |

**口径纪律**：`val_log.csv` 里的 `vox1-eer-8s` 只读前 8 秒，**不能与官方整段协议的 0.26% 直接相减比较**（AGENTS.md 第 9.1 节已写明）。同口径比较只能拿整段评测脚本重跑。

### 4.6 配方家族

`applications/speaker_rec/examples/` 下 32 个 yaml，按用途分三类：

| 家族 | 路径 | 用途 |
|---|---|---|
| 只练 VoxCeleb2 | `examples/vox2/recipes/redimnet2/{b0..b6}/{ptn,lm}.yaml` | 公开权重口径，b0 是冒烟首选 |
| 三语料（VB2+Vox2+CNC2） | `examples/vb2_vox2_cnc2/recipes/redimnet2/{b3,b6}/*.yaml` | 复现论文主表；`official_repro_ptn.yaml` / `official_repro_lm.yaml` 是本次正式实验的依据 |
| 变体 | 同目录 `varlen_*.yaml`、`vadseg_*.yaml` | 变长训练、按语音活动取段训练；必须成对打开（见 4.2） |

三语料 B6 主训练配方的关键取值（`official_repro_ptn.yaml`，行号为该文件内行号）：`data_type: raw_fast`（:16）、每卡 96 条（:18）、200 帧定长裁剪（:71）、`compile_backbone: null`（:15，本次明确关闭 `torch.compile`）、`num_epochs: 120`（:173）、`SGD lr 0.06 momentum 0.9 nesterov`（:174-179）、`project_type: sphereface2 scale 32 t 3 lanbuda 0.7`（:180-188）、`save_epoch_interval: 1`（:189）、间隔 `0.0→0.2`、第 15 轮开始升、第 30 轮固定（:87-94）。

三语料 B6 大间隔微调配方（`official_repro_lm.yaml`）：`do_lm: true`（:17）、`num_epochs: 1`（:187）、`model_ckpt: .../ptn_ha/models/model_115.pt`（:24）、`spk2id_map_path: .../ptn_ha/spk2id.json`（:25）、间隔固定 0.3（:177-183）、`save_epoch_interval: 1`（:206）。**从第 115 轮而不是第 120 轮起步**来自官方 `lm.yaml`，目录名里的 `100ep` 不是轮数依据（`official_repro_ptn.yaml:8-9` 已写明这处冲突）。

---

## 5. 第 2 层：标注系统 `tools/open_annotation/`

这一节解决“逐音频标注是怎么组织的：谁持有 GPU、谁写盘、什么条件下才允许正式跑”。这是全仓最复杂的部分，90 个顶层模块，但结构其实只有六层。

### 5.1 六层划分

| 层 | 模块 | 职责 | 是否碰 GPU |
|---|---|---|---|
| ① 来源 | `source_registry.py`、`manifest_builder.py`、`source_filter.py`、`source_clock.py` | 登记来源路径与角色、把大清单切成分片、筛出三个训练域、把十进制秒换成精确采样点 | 否 |
| ② 合同 | `contracts.py`、`engine_protocol.py`、`schema.py`、`sidecar.schema.json` | 冻结的身份与字段校验；引擎模块清单；侧车记录结构与交叉校验 | 否 |
| ③ 执行 | `gpu_launch.py`、`gpu_bootstrap.py`、`gpu_challenge.py`、`gpu_common.py`、`gpu_vendor/`、`worker.py` | 资源快照与双探针、物理卡锁、七阶段显存挑战、子进程身份、单个 worker 的主循环 | 是（仅 worker 与 bootstrap） |
| ④ 模型 | `qwen_backend.py`、`alignment_observer.py`、`normalization.py`、`native_transformers_batch.py`、`native_b1_feature_padding.py` | Qwen ASR + 强制对齐的离线封装；原生合批的文件事务；官方 B1 特征逐条取再合批 | 是（被 ③ 调用） |
| ⑤ 证据 | `sidecar.py`、`work_queue.py`、`report.py`、`audit_outputs.py`、`quality_axes.py`、`union_coverage.py`、`readonly_progress.py`、`bounded_metadata_progress.py` | 侧车不可覆盖发布、分片领取与完成位图、跑后核对、联合覆盖、只读进度 | 否 |
| ⑥ 准入与交接 | `cloud_gate.py`、`make_admission_spec.py`、`make_cloud_launcher.py`、`native_batch_admission.py`、`queue_handoff.py`、`bounded_speed_*`、`run_portable_successor.py` | 两卡证据重读、八卡启动件生成、挂载核对、正式准入判定、排队后新鲜观察交接 | 否 |

### 5.2 冻结与成员清单

`freeze_contract.py:32-107` 是唯一的冻结入口，做四件事：

1. 校验外部基础合同的 SHA 与用途（`:36-40`：只接受 `pilot_validation` / `silver_annotation`，且 `model_API_calls_allowed == 0`、`release_training_labels == False`）。
2. 把当前维护源码复制到**仓库之外**的新目录，只复制 `.py` 与 `sidecar.schema.json`，跳过 `tests/` 与 `__pycache__`（`:62-73`）。
3. 生成两份合同（试验 / 全量）并重算 `engine_source_files`（`:87-98`）。
4. 生成成员清单 `bundle_files.json` 与冻结收据 `freeze_receipt.json`（`:99-106`）。成员清单覆盖“复制目录里除 `contracts/` 外的所有文件”，所以**新增一个 .py 文件也会改变包身份**。

### 5.3 一次运行的目录布局与互斥规则

`gpu_launch.py:69-82` `validate_output_paths()` 把规则写成了断言，接手时不要放宽：

| 参数 | 必须满足 |
|---|---|
| `--run-root` | 已存在的目录，且不等于代码目录、也不是代码目录的祖先（`:72`） |
| `--device-lock-root` | 与 `run-root` 互不包含（`:74`）——所有协作任务共用一套卡锁，不能每个任务自造 |
| `--output` / `--queue-root` / `--output-root` | 都在 `run-root` 之内、都在代码目录之外，三者两两互不包含（`:75-82`） |
| `--output` | 必须不存在（`:78`），失败后换新目录，旧目录留证 |

运行期还会在 `queue_root` 下放一把 `.parent.lock`（`gpu_launch.py:129-138`，`work_queue.py:61-67` 同名实现），保证同一队列只有一个父进程；worker 通过 `pass_fds` 继承这把锁与各卡锁的 fd，并在 `authenticate_resource()` 里核对 fd 指向的 inode（`worker.py:88-90`）。

### 5.4 单个来源的处理顺序（worker）

`worker.py:113-232` `process_group()` 是标注系统的“主循环体”，顺序不能换：

```text
来源行 (source dict: source_id / dataset / relative_path / audio_path)
  ├─ ① 查上一代生产者回执 → 有就保留、不重算             worker.py:120-131
  ├─ ② 查本轮侧车缓存 → 有就复用、只记账                   :122-135
  ├─ ③ inspect_source：整文件 SHA + 逐帧解码 PCM SHA      audio_windows.py:49-83
  ├─ ④ reuse_vad：严格复用旧语音活动结果，异常留证          :111-174
  ├─ ⑤ plan_windows：按源时钟切「核心窗 + 上下文窗」        :175-207
  ├─ ⑥ read_window：按采样点 seek、下混单声道、重采样 16k   :208-231
  ├─ ⑦ ASR（可合批）→ 强制对齐                            qwen_backend.py:373-491
  ├─ ⑧ units_from_windows：原生时间 → 采样点区间，按拥有区过滤  audio_windows.py:232-257
  └─ ⑨ 发布前复查源文件 stat → build_record → 侧车不可覆盖发布 → 记日志  worker.py:211-232
```

三处容易忽略的细节：

- **窗口是“拥有区 + 上下文”两段**：`owned_start/end` 决定一个字属于哪个窗，`context_start/end` 是实际送模型的音频；跨窗重复的字按中点归属（`audio_windows.py:245-250`）。
- **时间不是浮点乘采样率**：v2 映射把十进制秒当有理数处理（`source_clock.py:16-27`），避免“恰好共享边界”被算成 1 个采样点的重叠。
- **发布前必须复查源文件**：`worker.py:216-219` 在发布前重读 stat，变了就把该源降级为 `decode_failed` 且不产出字单元——这是“音频在跑的过程中被替换”的唯一防线。

### 5.5 队列、侧车与质量轴

| 组件 | 位置 | 机制 |
|---|---|---|
| 分片领取 | `work_queue.py:156-192` `claim_next()` | `broker.json` 记游标与租约，`claims/<shard>.lock` 用 `flock` 独占；每个分片一个代次号（`attempt_state/*.json` 的 `generation`） |
| 完成位图 | `work_queue.py:21-37,140-155` | 固定长度二进制位图，每分片一个槽位，写入后立即 `pread` 回读比对 |
| 分片回执 | `work_queue.py:114-139` `receipt()` | 逐条核对 `schema`、索引 SHA、合同 SHA、分片 SHA、来源数与来源 ID 集合 |
| 侧车发布 | `sidecar.py:56-69` `publish()` | 路径镜像源目录（`dataset/relative_path + '.annotation.json'`），`atomic_new` 不可覆盖，写完回读比对，再补 `.commit.json` |
| 侧车查找 | `sidecar.py:32-54` `lookup()` | 语义回读 + 源 stat 复核；只有“侧车已写、commit 未写”这一种崩溃可以修 commit，其余一律拒绝 |
| 阶段缓存 | `sidecar.py:77-106` `StageCache` | 按输入摘要寻址，命中时重新校验输入与输出摘要；已有胜者永不被覆盖 |
| 质量轴 | `quality_axes.py:113-160`、`schema.py:130-219` | 机器可重建的分轴计数 + 保守的“为什么还不能当候选”的原因列表 |
| 判定 | `schema.py:220-234` `build_record()` | `decision.state` 只有三种：`failed`（源解码失败）、`unknown`（有原因）、`candidate`（无原因）；**三者都写 `usable_for_training=False`** |

`schema.py:130-219` 的原因列表是这套系统的“契约条文”，例如：源未完整解码、stat 前后不一致、窗口没有铺满整段、ASR 未观测到结束符、对齐目标与 ASR 文本不一致、对齐文本覆盖不上、源单位重叠或乱序。**改判定条件等于改数据资格**，必须当作合同变更处理。

### 5.6 原生合批的文件事务

`native_transformers_batch.py:100-240` `NativeTransformersBatchStore` 解决“一次生成多行、但只有整批落盘后单行才可用”：

```text
固定来源组（≤256 行）
  → identity.json（生产者/合同/模型 SHA + 批大小 1/4/8）      :102-120
  → plan.json（把行切成 cohort，每 cohort 一份意图摘要）        :134-160
  → 每个 cohort 一个批次文件 batches/<cohort_sha>.json（内容+摘要）  :174-184
  → 单行结果只能从已落盘批次里读，并带 durable_native_batch 引用  :192-238
```

硬规则：**未提交的 cohort 必须提供原来的全部行与原顺序**，不允许“拿剩下的缓存未命中行另起一批”（`:196-214` 的 `need`）；失败会把整批原样写成 `attempts/*.json` 再抛出（`:185-190`）。

### 5.7 准入链

```text
两卡小批/3000 条基线（validate_two_gpu.py）
  → cloud_gate.py（纯 CPU 重读两卡证据）
  → make_admission_spec.py / make_cloud_launcher.py（生成八卡启动件，不提交）
  → native pilot（批 1/4/8 三轨，run_native_batch_pilot.py）
  → native_batch_admission.py：check_quality → check_formal_bundle → _full_check_native_selected_pilot
  → 正式八卡（gpu_launch.py --target-uuids auto8）
```

规模门是硬的：正式用途要求 `n ≥ 3000` 且恢复轨 `rc == 512`（`native_batch_admission.py:246-248`，AGENTS.md 第 9.3 节同款表述）。300 条 pilot 不算准入。

### 5.8 交接与提速族

外部平台会排队，排队期间拿到的“资源空闲”证据会过期，所以有一族**只读交接**模块：先固定输入，再等真实的平台新观察，最后原子发布一份 `runtime-spec.json` + `reply.json` 给下一个环节。四个近亲模块（`queue_handoff.py`、`bounded_speed_queue_handoff.py`、`bounded_speed_handoff.py`、`native_batch_queue_handoff.py`）是同一模式的不同阶段变体，函数名高度重名，改动时要看清目标文件（见 03 的可疑点清单）。

---

## 6. 第 3 层：接续与 LM `tools/ptn_resume/`

这一节解决“主训练中断后怎么受控接上、以及 LM 凭什么能启动”。

```text
build.py（47→48 接续包）
  ├─ 读私有 spec：旧 16 成员包、原 189 文件树、model_47 与说话人表、两个终态观察、新输出目录
  ├─ 逐份核旧成员字节 → 复制成新 190 成员树（只允许两处变化：换 executor.py、加 numerical_failure.py）  build.py:30-52 核源树、`:116-166` 组装新包
  ├─ 生成 resume.yaml（只用 safe_dump 写、再读回比对类型）                                              :103-115
  ├─ 复制本仓 state_gate.py / contracts.py 为 runtime/                                                INTERFACE.md:21
  └─ 写 resume-contract.json / source_delta.json / source_manifest.json / bundle_files.json            :116-166

state_gate.verify_restored()（真实执行前必须调用）
  ├─ verify_package：成员集合、逐文件摘要、源树 190 成员、两处差异                              state_gate.py:35-63
  ├─ 重读 model47（大小、SHA、只允许 4 个键、823 状态 / 584 动量）                              :79-83
  ├─ 逐张量精确比较 model / optimizer / scaler                                                  :91-100
  ├─ 用冻结调度器与间隔推进到 705047，核 lr=0.72、margin=0.2                                    :102-103
  └─ 返回证明，但 `runtime_admission` 恒为 false                                               :112

three_segment_lm_check.py（三段证据合同）
  ├─ build：读三段真实任务/run/CSV/检查点/异常/来源身份 → 合同                          :913-1073
  └─ check：按合同重读原件并逐项判定，输出 17/18 项计数与退出码                        :1111-1134

build_lm_package.py（LM 冻结包）
  └─ 与 build.py 同规矩，但方向相反：单轮、重算分类头、冷优化器/冷缩放器          build_lm_package.py:1-27

lm_launch/（启动门，容器内执行）
  ├─ run_lm.sh：7 项前置 + 真跑 LM 初始化门 + 冻结身份 → 交给既有云端训练脚本      lm_launch/README.md:21-33
  └─ lm_stage_gate.py：核 clock/批大小/margin/说话人表/配方/日志标记                    :37-51,119-279
```

两个“设计上就该是 false / 就该校验”的点，别当故障修：

- `ready_for_lm_execution` 恒 `false`（AGENTS.md 第 9.4 节）：它是“准备情况”检查器的输出，不是执行许可。
- `verify_restored` 返回的 `runtime_admission: false`（`state_gate.py:112`）：单份精确状态不证明锁、资源、八 rank 一致。

---

## 7. 外部系统边界

这一节解决“哪些东西不在这个仓库里，但链路依赖它”。接手时如果这些边界变了，症状通常不是报错而是“数字变了”。

| 外部系统 | 接触点 | 边界性质 |
|---|---|---|
| GitHub Release 权重 | `redimnet2/__init__.py:6` | 硬编码 URL；离线环境必然失败，训练/评测不走这里 |
| 共享盘挂载 | `/root/group-shared/...`（配方、标注来源）；`gpu_launch.py:59-66` 建立 `/shared/group-shared` 到 `/root/group-shared` 的别名并要求两者同源 | 路径写死在配方与合同里；别名不成立就直接拒绝 |
| 训练语料（VoxCeleb2 / VoxBlink2 / CN-Celeb2 / MUSAN / RIRS） | 配方 `dataset_dir`、`train_data`、`train_label` | 只读；不在仓库内，也不该进仓库 |
| conda 环境 | `launch.sh:34-60` 按 `REDIMNET2_PYTHON`/`REDIMNET2_ENV` 解析 | 解释器选错会静默跑在别的 torch 上 |
| 火山引擎机器学习平台 | `applications/speaker_rec/scripts/cloud/*.sh`、`tools/ptn_resume/lm_launch/submit.conf.yaml` | 仓库只生成提交件与启动门，不提交；任务名统一 `zhounan_` 前缀 |
| Qwen ASR / 强制对齐模型与推理源码 | `qwen_backend.py:187-330`、合同里的 `models` / `source_files` / `decoder` | 模型文件、第三方源码、解码库三者都由外部合同逐项钉住 SHA；`worker.py:260` 在跑前重核 |
| 物理 GPU | `gpu_vendor/resources.py:36-109`（快照与双探针）、`worker.py:80-81`（只认同一个 UUID） | 只接受 A800、≥80 GB；未知进程占卡时拒绝共享 |
| 审核控制面（任务终态） | `queue_handoff.py`、`state_gate.py:77`、`three_segment_lm_check.py:894-912` | 只读外部观察文件；不接受布尔值代替原件 |

---

## 8. 三条数据流（文字版）

### 8.1 主训练（PTN）

```text
data.list（训练清单）+ utt2spk（语句→说话人）
  → Dataset()/Processor 依次：读样本清单并全局洗牌 → raw_fast 解析（逐条读音频）→ 样本级洗牌 → spk_to_id → 重采样 16k → [可选] 按语音活动取段 → [可选] 变速 → 定长裁剪或变长取段 → [可选] 噪声/混响/低通/编解码增强
  → collate → batch{'key','label','wav'[,'lengths']}
  → SpeakerModel(ReDimNet2Wrap + 投影头)
  → 前向出 embedding → 投影头出 logits（及自带损失）→ AMP 反向 → SGD 更新
  → 每轮：8 秒快评 + 整段全评 → val_log.csv；按 save_epoch_interval 写 models/model_N.pt
  → exp_dir 留下 config.yaml / spk2id.json / models/
  → scripts/export_checkpoint.py 或 cloud/evaluate_b6_mix_reproduction.sh 导出/评测
```

### 8.2 大间隔微调（LM）

```text
PTN 的 models/model_115.pt（650,778,662 B，sha 65497b6a…）
  → 门 1：spk2id 门（存在 → sha256 → 非空 → 整数编号 → len==119,274 → LM 标签 115,362 → 全覆盖）
  → 门 2：分类头按说话人名字重组（utils/utils.py:130-203），不是按索引切
  → 冷启动：只载模型参数，优化器与缩放器从零开始（train.py:446-465 的注释与分支）
  → 1 轮训练（official_repro_lm.yaml:187），间隔固定 0.3
  → models/model_1.pt + config.yaml
  → evaluate_b6_mix_reproduction.sh：与官方参考权重在同口径下对比
```

### 8.3 逐音频标注

```text
源清单（三个训练域）→ source_filter/source_registry 固定来源与角色 → manifest_builder 切成不可变分片
  → 合同（外部 JSON + 引擎源 SHA）→ freeze_contract 复制出冻结副本（成员清单 + 收据）
  → gpu_launch：资源双探针 → 物理卡锁 → 逐卡子进程（gpu_bootstrap → 显存挑战 → worker）
  → worker 领取分片 → 解码/切窗 → Qwen ASR（默认单窗；原生合批走固定 cohort 文件事务）
  → 强制对齐 → 文字规范化 → 质量轴与判定
  → 侧车不可覆盖发布 + 分片回执 + 完成位图
  → report/audit_outputs 跑后核对 → union_coverage 核“旧结果 + 剩余输入”的联合覆盖
  → 下一位接续者（queue_handoff 一族）等真实平台观察后再发下一段运行说明
```

---

## 9. 冻结与可改

这一节解决“我能不能改这个文件”。判断标准不是“看起来重要不重要”，而是“它的 SHA 有没有被别人当作身份钉住”。

### 9.1 已冻结、不可改（改了就要重新打包、重新准入）

| 对象 | 范围 | 证据 |
|---|---|---|
| 冻结包成员 | 冻结目录下所有 `.py` + `sidecar.schema.json` + 自动补入的 `task_names.py` | `freeze_contract.py:67-77,99-101` |
| 引擎文件（14 个基础） | `contracts.py`、`audio_windows.py`、`qwen_backend.py`、`worker.py`、`schema.py`、`sidecar.py`、`work_queue.py`、`report.py`、`source_registry.py`、`manifest_builder.py`、`normalization.py`、`alignment_observer.py`、`quality_axes.py`、`source_clock.py` | `contracts.py:16-19` |
| 引擎文件（+2） | 走原生合批时加 `native_transformers_batch.py`；走 B1 特征协议时再加 `native_b1_feature_padding.py`，合计 16 | `contracts.py:68-71` |
| 正式准入的等式 | `check_formal_bundle()` 对 16 个引擎文件做**无条件相等比较** | `native_batch_admission.py:210-232` |
| `REVIEWABLE_CONTROL` 四文件 | `native_batch_admission.py`、`native_batch_readback.py`、`native_batch_queue_handoff.py`、`bounded_speed_runtime.py`——**只有引擎文件全等之后**才有资格走“可审阅替换”，且要附独立审阅原件 | `native_batch_admission.py:207,219,446,462` |
| 模型 47 身份 | SHA `91c1b76a…`、650,777,251 B、823/584/585、头形状 `[357822,192]`、时钟 705047/lr 0.72/margin 0.2 | `tools/ptn_resume/contracts.py:11-28`、`state_gate.py:11-15` |
| 三段证据合同 | `build` 出的合同一旦被启动门钉住，只能换版本，不能改旧件 | `three_segment_lm_check.py:913-1073`、AGENTS.md 第 9.2 节 |
| 已发布的历史身份 | 旧 95/189 成员清单、`model_47`、失败任务的终态观测 | `INTERFACE.md:13-16,29` |

### 9.2 可改（改完按对应链路自测即可）

| 对象 | 影响面 | 自测方式 |
|---|---|---|
| `redimnet2/` 新增可选参数 | 推理一致性；已发布权重依赖的默认路径**必须逐位不变** | `tests/test_mask_input_contract.py`、`tests/test_varlen_masking.py` |
| 训练栈新增可选配置项 | 老配方不写就走老路径 | `tests/test_schedule_and_launch_guards.py`、`tests/test_training_numeric_sampling.py` |
| 新增配方 yaml | 只影响该实验 | 短测（`make_b6_smoke_recipe.py` 一类） |
| `tools/open_annotation/` 的非引擎工具（绘图、只读进度、对照审计） | 不影响合同身份 | `python3 -m unittest discover -s tools/open_annotation/tests` |
| `tools/ptn_resume/tests/`、`engineering_two_gpu.py` 等证据工具 | 不影响已冻结包 | `python3 -m unittest discover -s tools/ptn_resume/tests` |
| 文档 `docs/*.md` | 无运行时影响 | `docs/reviews/check_document_links.py`、`check_report_tables.py` |

判定口诀：**先看它是不是 `engine_source_files` 里的一员，再看它是不是某个包里被 SHA 钉住的一员，最后看它是不是公式本身。** 三者都不是，才是普通代码。

---

## 10. 三个月后接手：按问题找文件

| 你想知道 | 先看 |
|---|---|
| 模型怎么从波形出向量 | `redimnet2/redimnet2.py:611-686` + `layers/poolings.py:147-241` |
| 变长批处理为什么结果不完全一致 | `redimnet2/redimnet2.py:466-495`、`docs/08-变长训练与长度分桶.md` |
| 训练为什么卡住/不收敛 | `train.py:371-390`（步数断言）、`:616-647`（学习率与间隔）、`executor.py:213-289` |
| 检查点能不能续训 | `utils/checkpoint.py:31-101`；**四键齐不齐**比 SHA 更重要 |
| 某个配方到底跑的是哪套数据 | 配方 `train_data`/`train_label` 指向的 `data.list`/`utt2spk`；不要凭 `exp_dir` 目录名推断 |
| 标注为什么停了 | `docs/reviews/` 里最近一份、`worker.py:280-295` 的 chunk 事件、`report.py:41-175` |
| 标注能不能改代码 | 第 9.1 节；改了引擎文件就必须重新打包 + 重跑 pilot + 重新准入 |
| LM 能不能启动 | `lm_launch/README.md`、`tools/ptn_resume/三段LM准备协议.md`、AGENTS.md 第 9.2 节 |
| 三段证据从哪来 | `three_segment_lm_check.py` 的 `build` 子命令与它读的原件清单 |

---

## 11. 未验证与已知限制

这一节列出本文**没有**证明的事，避免被当成结论引用。

| 项 | 状态 |
|---|---|
| B0–B6 各自参数量 | 未复算。旧版本文档里的参数量表属估算，本次没有构造模型统计 |
| 训练能否在本机复跑 | 未验证。本次只读代码与静态结构，没有启动训练 |
| 标注系统的 GPU 行为 | 未验证。本文依据源码、合同与仓库内报告，没有执行 GPU |
| `tools/open_annotation/legacy_snapshots/production_20261025/` 的 23 个文件与当前模块的差异原因 | 只确认“有差异”（17 个同名文件中 9 个字节不同），没有逐个溯源；该目录 README 说四个部署差异是 `ENGINE.md`、`gpu_launch.py`、`gpu_vendor/resources.py`、`gpu_vendor_provenance.json` |
| `docs/` 旧文档与代码的一致性 | 本次重写 02/03；其余文档（04/05/08/09/10/12/13/15/16）未逐条复核 |
