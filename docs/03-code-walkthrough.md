# 03 · 代码地图与关键模块

上一篇：[网络结构](02-architecture.md) · 下一篇：[训练流程](04-训练流程.md)

---

## 1. 两套入口：推理 vs 训练

```mermaid
flowchart TB
  subgraph A["推理路径（轻）"]
    H[hubconf.py] --> L[redimnet2.load_custom]
    L --> W[ReDimNet2Wrap]
  end

  subgraph B["训练路径（重）"]
    Launch[launch.sh / torchrun] --> T[wespeaker_lite/bin/train.py]
    T --> SM[SpeakerModel]
    SM --> BB[redimnet2 backbone]
    SM --> PJ[projections.*]
  end
```

| 你想… | 从哪进 |
|--------|--------|
| 加载官方权重抽向量 | `torch.hub` → `hubconf.redimnet2` |
| 改网络结构做实验 | `redimnet2/redimnet2.py` + `layers/` |
| 改增强 / 损失 / 优化器 | `applications/speaker_rec/wespeaker_lite/` |
| 改数据与超参复现论文 | `applications/speaker_rec/examples/**/recipes/**/*.yaml` |
| **核对发布的权重和官方表差多少** | `scripts/eval_vox1_checkpoints.py`、`scripts/eval_ood_checkpoints.py` |

注意最后一行：这两个脚本在仓库根目录 `scripts/` 下，不在 `applications/` 里。
它们不训练，只加载 `checkpoints/pretrained/` 里的权重算等错误率，结果与官方数字对比后写成表格。

---

## 2. `redimnet2/` 包：文件职责

```text
redimnet2/
├── __init__.py          # load_custom / from_pretrained
├── redimnet2.py         # ReDimNet2, ReDimNet2Wrap
└── layers/
    ├── redim_structural.py   # to1d / to2d / weigth1d  ★先读
    ├── blocks.py             # 2D/1D 积木工厂
    ├── attention.py          # MHA / Transformer 层
    ├── convnext.py           # ConvNeXt-like + WSConv
    ├── resblocks.py          # ResBasicBlock + fwSE
    ├── features.py           # MelBanks 等（torchaudio）
    ├── features_tf.py        # 可学习/固定 DFT 的 TF 前端
    ├── poolings.py           # ASTP 等
    └── layernorm.py          # channels_first / last
```

### 建议阅读顺序（约半天）

1. **`redim_structural.py`** — 弄懂 reshape，后面全是这套语言。  
2. **`ReDimNet2.forward` / `run_stage`** — 看 stage 如何堆。  
3. **`TimeContextBlock1d` + `build_internal_1d_tcm_block`** — 1D 里有什么。  
4. **`ReDimNet2Wrap`** — 前端 + 池化 + Linear。  
5. **`__init__.py` 的 `load_custom`** — 权重如何自描述实例化。

---

## 3. 关键类速查

### `ReDimNet2Wrap`

- **输入**：`(B, samples)`，16 kHz。  
- **输出**：`(B, embed_dim)`。  
- **重要参数**：`feat_type`、`embed_dim`、`pooling_func`、`hop_length`、`spec_params`、以及透传给骨干的 `C`、`F`、`stages_setup`…  

`feat_type` 常见取值：

| 值 | 含义 |
|----|------|
| `pt` / `pt_mel` | torchaudio Mel |
| `tf` / `tf_mel` | TF 实现的 Mel（训练 recipe 里常见） |
| `tf_spec` | 更偏谱图的 TF 前端 |

### `ReDimNet2`

- 纯骨干：吃 `(B,1,F,T)`（或等价），吐出融合后的特征图。  
- 旋钮：`stages_setup`、`block_1d_type`、`block_2d_type`、`dual_agg`、`agg_gnorm`、`fm_weigthing_type`、`causal` 等。详见 [02](02-architecture.md)。

### 训练侧别名

`wespeaker_lite` 通过 `get_speaker_model("ReDimNet2Custom")` 从 `redimnet2` 包取类（名称以 YAML 的 `model:` 字段为准）。骨干只负责 embedding；**投影头不在 `SpeakerModel.forward` 里**，由 `executor` 单独调用。

---

## 4. `hubconf.py` 与权重契约

```python
# 伪代码逻辑
weight_name = f"{model_name}-{dataset}-{train_type}.pt"
# 例：b6-vox2-lm.pt
#     b6-vb2+vox2+cnc2_v0-lm.pt
url = f"https://github.com/PalabraAI/redimnet2/releases/download/v1.0.0/{weight_name}"
ckpt = torch.hub.load_state_dict_from_url(url)
model = ReDimNet2Wrap(**ckpt["model_config"])
model.load_state_dict(ckpt["state_dict"])
```

含义：

- **配置跟着权重走**，换 B0/B6 不必手写结构。  
- `pretrained=False` 在 hub 入口会直接报错；自定义结构请直接构造 `ReDimNet2Wrap(...)`。

---

## 5. `applications/speaker_rec/` 地图

```text
applications/speaker_rec/
├── launch.sh / launch.md     # torchrun 封装；在仓库根调用
├── pyproject.toml            # Poetry 依赖（package-mode=false）
├── wespeaker_lite/
│   ├── bin/train.py          # 主入口
│   ├── dataset/              # IterableDataset 数据处理
│   ├── models/               # SpeakerModel + projections + mix_augs
│   └── utils/                # executor / schedulers / checkpoint
└── examples/
    ├── vox2/                 # 仅 VoxCeleb2
    └── vb2_vox2_cnc2/        # Vox2 + VoxBlink2 + CN-Celeb2
```

### 模块速览

| 路径 | 作用 |
|------|------|
| `dataset/dataset.py` | 拼数据处理步骤；当前训练 assert `raw_fast` |
| `dataset/processor.py` | 读 wav、变速、切 chunk、旧版加噪混响 |
| `dataset/augs.py` | 可组合增强树（Sequential / OneOf / …） |
| `dataset/lmdb_data.py` | 读 MUSAN/RIR 的旧版 LMDB；**当前配方不用**，增强直接读 wav 目录 |
| `models/projections.py` | SphereFace2、ArcMargin… |
| `models/mix_augs.py` | GPU 上 Mixup / Cutmix / OverlapMix |
| `utils/executor.py` | `run_epoch`、EER 评估 |
| `utils/schedulers.py` | LR 与 **margin** 调度 |
| `utils/checkpoint.py` | 存盘时剥掉 DDP / compile 前缀 |

---

## 6. 张量形状「随身小抄」

假设 `B=2, C=24, F=72, T=200`（数字仅示意）：

| 位置 | 形状 |
|------|------|
| 波形 | `(2, 32000)`（约 2 s @16 kHz） |
| Mel | `(2, 1, 72, T')` |
| stem → to1d | `(2, 24*72, T')` = `(2, 1728, T')` |
| 某 stage 频 stride=2 后 | `C`↑、`F`↓，常维持 `C·F` 量级便于融合 |
| ASTP 后 | `(2, 2·in_dim)` |
| Linear 后 | `(2, 192)` |

调试时优先打印这几处，比通读所有层更快定位 bug。

---

## 7. 常见「在哪改」

| 目标 | 改哪里 |
|------|--------|
| 换 Mel bins / hop | `model_args` 里 `F`、`hop_length`、`spec_params` |
| 加深某个 stage | YAML `stages_setup` 的 `num_blocks` |
| 关掉注意力省算力 | 对应 stage `att_block_red: null`，或改 `block_1d_type` |
| 换损失 | `projection_args.project_type` |
| 换增强强度 | `dataset_args.aug_setup` |
| 只训骨干、冻投影 | 需自己在 `train.py` / optimizer 参数组动手（仓库默认未封装开关） |

---

## 8. 依赖分层

| 场景 | 依赖 |
|------|------|
| 仅推理 | 根目录 `requirements.txt`：`torch torchaudio scipy numpy` |
| 训练 | `applications/speaker_rec/pyproject.toml` + 系统 `ffmpeg`、`libsndfile`；当前配方增强直接读原始 wav 目录。只有仍用旧配置的 LMDB 并打开 `lmdb_to_ram` 时，才需留意 `/dev/shm` 空间 |

下一篇把训练一步步展开：[04 训练流程](04-训练流程.md)。
