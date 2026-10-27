# ReDimNet2 中文导读

> 面向入门研究生的中文文档体系。根目录 [README.md](../README.md) 保持英文官方说明；这里用平实中文把「做什么 → 怎么想 → 代码在哪 → 怎么训 → 怎么复现」串起来。

**论文**：[ReDimNet2: Scaling Speaker Verification via Time-Pooled Dimension Reshaping](https://arxiv.org/abs/2603.11841)（Interspeech 2026）  
先看这里：把语音变成固定长度的「说话人指纹」（embedding），并在精度与算力之间走出更好的 Pareto 曲线。

---

## 本次实验维护入口（2026-10-27整理）

历史章节中的机器状态以各自日期为准。[实验维护与验收边界](16-实验维护与验收边界.md) 区分主训练接续、自动文字与时间标注、官方模型完整字自然变长微调三项工作，并说明仓库代码与外部证据怎样对应。新迁入代码的 CPU 测试与历史 GPU 实测分别登记，不能混用。

- [转写提速实现与八卡容量测算](reviews/2026-10-27-转写提速实现与八卡容量测算.md)：新计算配置、阶段计时、对照与去重接续工具；11:22—11:34只读实据及显存/排期估算。按用户后续要求未做GPU实测，未切换任务。
- [标注吞吐瓶颈与提速验证安排](reviews/2026-10-26-标注吞吐瓶颈与提速验证安排.md)：10月26日历史阶段耗时和进度、转写优先优化的依据，以及尚未执行的两卡测速与质量验证；10月27日文档整理不代表新巡检。
- [10月27日文档与技能同步记录](reviews/2026-10-27-文档与技能同步记录.md)：本轮更新范围、入口、实际检查及待执行工作。
- [PTN启动时效修复与八卡接续巡检](reviews/2026-10-26-PTN启动时效修复与八卡接续巡检.md)：前次启动材料过期原因、同冻结包后继、实际8rank/192worker/首SGD、真实容器八卡与检查点边界。
- [全仓二次审查与任务故障核验](reviews/2026-10-26-全仓二次审查与任务故障核验.md)：保留先前549项CPU回归与PTN48轮前向NaN的当时快照；后续修复和新的八卡任务以10月26日后续记录为准，候选质量未自动放行。
- [PTN双卡恢复与完整故障取证实测](reviews/2026-10-26-PTN双卡恢复与完整故障取证实测.md)：真实模型47完整恢复、实际优化步连续与恢复比较、独立NaN注入；W2工程证据不能冒称W8接续验收。
- [标注时钟及撇号对照](reviews/2026-10-26-标注时钟及撇号对照.md)：固定16来源、同原生ASR的两轨真实FA对照，保留未知、原生与FIX时间及公平文字span比较。
- [完整取证接续与标注时钟修复计划](plans/2026-10-26-PTN完整取证接续与标注时钟修复.md)
- [从model47构建独立接续包与真实状态门](../tools/ptn_resume/README.md)
- [PTN前向故障的隔离诊断方案](plans/2026-10-26-PTN前向非有限值隔离诊断.md)

- [代码归仓与全面检查计划](plans/2026-10-25-全仓检查与实验代码归仓.md)
- [文档和项目技能检查计划](reviews/文档与技能检查计划.md)
- [归仓版本独立复审与未验项目](reviews/2026-10-25-独立复审与未验项目.md)
- [项目维护技能源码](../.agents/skills/redimnet2-reproduction-ops/SKILL.md)
- [主训练接续只读参考与静态检查](../tools/ptn_reference/README.md)：含历史启动动作，不导入或执行；缺外部原件，不能用静态检查授予运行资格。

---

## 你将学到什么

| 问题 | 答案在哪 |
|------|----------|
| 说话人验证是什么？为何需要专门网络？ | [01 背景与动机](01-background.md) |
| ReDimNet2 比普通 CNN 多了什么？ | [02 网络结构](02-architecture.md) |
| 仓库里每个文件对应什么概念？ | [03 代码地图](03-code-walkthrough.md) |
| 数据怎么进模型、loss 怎么算？ | [04 训练流程](04-训练流程.md) |
| 如何准备数据、跑 B0→B6？ | [05 实验配方与复现](05-recipes-and-repro.md) |
| **本机数据在哪、开训前要处理什么？** | [05 第 2 节](05-recipes-and-repro.md) 与 [07 本机数据与开训前检查](07-本机数据与开训检查.md) |
| EER、PTN、LM、ASTP 是什么？ | [06 术语表与 FAQ](06-glossary-faq.md) |
| 想让一批里各句长度不同、少补点零？ | [08 变长训练与长度分桶](08-变长训练与长度分桶.md) |
| 想先检测有声段并保存时间标注？ | [09 用语音活动检测给音频加时间标注](09-VAD标注与RTTM.md)；完整字还需文字、字界与审核，见 [16 实验维护与验收边界](16-实验维护与验收边界.md) |
| 想在一台 8 卡机器上跑 b6 混合数据训练？ | [10 在火山引擎上用 8 张卡训练](10-云端八卡训练.md) |
| 三语料 B6 的官方配方、学习率和第 115 轮选择如何核实？ | [12 三语料 B6 官方复现配方核查](12-三语料B6官方复现配方核查.md) |
| 训练权重怎样导出，并与官方权重使用同一评测入口？ | [13 训练权重导出与评测](13-训练权重导出与评测.md) |
| 云端数据和增强目录怎么挂载、怎样验证？ | [14 三语料 B6 云端数据路径](14-三语料B6云端数据路径.md) |
| 本次八卡短测修复了什么，正式任务在哪里？ | [15 八卡短测与正式训练记录](15-三语料B6八卡短测与正式训练记录.md) |
| 官方权重复现得怎样？EER 差多少？ | [评测报告入口](../checkpoints/reports/README.md) |

---

## 仓库鸟瞰

```text
redimnet2-fork/
├── redimnet2/                 # 模型本体（推理也可只用这一包）
│   ├── redimnet2.py           # ReDimNet2 + ReDimNet2Wrap
│   └── layers/                # 特征前端、2D/1D 块、注意力、池化
├── hubconf.py                 # torch.hub 入口
├── applications/speaker_rec/  # 训练栈（wespeaker_lite + examples 下的配方）
│   ├── wespeaker_lite/        # 数据 / 损失头 / 训练循环
│   └── examples/              # vox2、vb2_vox2_cnc2 配方
├── scripts/                   # 对官方预训练权重做复现评测的两个脚本
├── checkpoints/               # 官方权重、评测结果表、中文评测报告
├── assets/                    # 结构示意、Pareto 图
└── docs/                      # ← 你正在读的中文体系
```

```mermaid
flowchart TB
  subgraph Infer["只用模型做推理"]
    W[16 kHz 波形] --> Hub[torch.hub / ReDimNet2Wrap]
    Hub --> Emb[说话人 embedding<br/>默认 192 维]
  end

  subgraph Train["完整训练"]
    DL[data.list + utt2spk] --> Pipe[wespeaker_lite 数据处理]
    Pipe --> SM[SpeakerModel<br/>backbone = ReDimNet2]
    SM --> Proj[SphereFace2 / AAM 等投影头]
    Proj --> Opt[SGD + Margin 调度 + DDP]
    Opt --> Val[Vox1-O 余弦打分 → EER]
  end
```

---

## 5 分钟上手：抽 embedding

依赖（推理最小集）：`torch`、`torchaudio`、`scipy`、`numpy`。

```python
import torch
import torchaudio

# 默认：VoxCeleb2 上训好的 B6，large-margin 权重（推荐）
model = torch.hub.load(
    "PalabraAI/redimnet2", "redimnet2",
    model_name="b6", train_type="lm", pretrained=True,
)
model.eval()

waveform, sr = torchaudio.load("audio.wav")  # (channels, samples)
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)
if waveform.size(0) > 1:
    waveform = waveform.mean(0, keepdim=True)

with torch.no_grad():
    emb = model(waveform)  # (1, embed_dim)，常见 embed_dim=192
```

常用旋钮：

| 参数 | 取值 | 含义 |
|------|------|------|
| `model_name` | `b0` … `b6` | 从小到大，参数量约 1.1M → 12.3M |
| `train_type` | `ptn` / `lm` / `dis` | 预训练 / 大间隔微调（推荐）/ 蒸馏（部分型号） |
| `dataset` | `vox2`（默认）、`vb2+vox2_v0`、`vb2+vox2+cnc2_v0` | 训练数据组合 |

完整模型表与指标见根目录英文 [README.md](../README.md#results)。

---

## 结果直觉（VoxCeleb2-dev 训练）

EER 越低越好。同一协议下，模型越大通常越准，但算力也上去：

| 型号 | 参数量 | GMACs | Vox1-O EER (%) |
|:----:|:------:|:-----:|:--------------:|
| B0 | 1.1M | 0.33 | 1.04 |
| B3 | 4.1M | 2.70 | 0.42 |
| B6 | 12.3M | 13.05 | 0.29 |

混合更大数据（VoxBlink2、CN-Celeb2）后，跨数据集表现通常更好。Pareto 曲线见图：[`assets/pareto.png`](../assets/pareto.png)。

---

## 建议学习路径（约一周）

```mermaid
flowchart LR
  D1[Day1<br/>背景 + 抽 embedding] --> D2[Day2<br/>读 to1d/to2d]
  D2 --> D3[Day3<br/>走通 ReDimNet2.forward]
  D3 --> D4[Day4<br/>训练栈与 SphereFace2]
  D4 --> D5[Day5<br/>vox2 数据 + B0 冒烟]
  D5 --> D6[Day6–7<br/>B3 全流程 / 读混合配方]
```

| 天数 | 做什么 | 文档 |
|:----:|--------|------|
| 1 | 弄清任务与指标；用 hub 抽向量、算余弦相似度 | [01](01-background.md) |
| 2 | 理解「维度重塑」：`(B,C,F,T) ↔ (B,C·F,T)` | [02](02-architecture.md) |
| 3 | 对照代码：stem → stages → ASTP → Linear | [03](03-code-walkthrough.md) |
| 4 | 看 `train.py`、增强、投影头、PTN→LM | [04](04-训练流程.md) |
| 5 | 核对本机数据在哪、把 `/data` 路径问题解决掉 | [07](07-本机数据与开训检查.md) |
| 5 | 跑 `prepare_data.py` + B0 单卡 1 epoch | [05](05-recipes-and-repro.md) |
| 6–7 | 完整 B3；有条件再看混合数据配方 | [05](05-recipes-and-repro.md)、[06](06-glossary-faq.md) |

> 第 5 天分两步：**先处理路径再做冒烟**。本机没有 `/data` 目录，不先解决这一步，
> 训练会在读样本清单时直接失败。详见 [07 本机数据与开训检查](07-本机数据与开训检查.md)。

---

## 设计哲学（先建立直觉）

传统说话人网络常在 **2D Mel 频谱图**上做卷积。ReDimNet 系列的核心想法是：

1. **同一特征两种看法**：看成 `(通道, 频率, 时间)` 的 2D 图，或看成 `(通道×频率, 时间)` 的 1D 序列。
2. **2D 擅长局部频时模式**；**1D + 注意力擅长长时上下文**。
3. ReDimNet2 进一步在 **1D 通路上对时间做池化**，算力降下来后，可以把通道开得更宽——于是「同样算力更准」或「同样精度更省」。

结构示意见：[`assets/scheme.png`](../assets/scheme.png)。细讲见 [02 网络结构](02-architecture.md)。

---

## 文档与代码的对应关系

| 文档 | 主要代码 |
|------|----------|
| 02 架构 | `redimnet2/redimnet2.py`, `layers/redim_structural.py`, `layers/blocks.py` |
| 03 代码地图 | `redimnet2/layers/*`, `hubconf.py`, `scripts/eval_*.py` |
| 04 训练 | `applications/speaker_rec/wespeaker_lite/**` |
| 05 复现 | `applications/speaker_rec/examples/**` |
| 07 本机数据 | `AGENTS.md` 第 3、6 节、`checkpoints/reports/05-训练数据检查.md` |
| 08 变长训练与长度分桶（含按语音边界取样，第 9 节） | `wespeaker_lite/dataset/{dataset,processor}.py`、`utils/executor.py` |
| 09 用语音活动检测给音频加时间标注 | `applications/speaker_rec/scripts/vad/**` |
| 10 在火山引擎上用 8 张卡训练 | `applications/speaker_rec/scripts/cloud/*.sh` |
| 16 实验维护与验收边界 | 三条实验线、维护入口、版本身份及历史证据的适用范围 |
| 评测报告 | `checkpoints/eval_*.csv`、`checkpoints/reports/**` |

> 编号从 01 到 16，但**没有 11**。11 这个编号在整理文档时被并入 10（云端八卡训练）
> 与 12（配核查），为了不打乱已有链接就空着没重排。看到 10 直接跳 12 不是漏了一篇。

---

## 引用

```bibtex
@inproceedings{redimnet2_2026,
  title={ReDimNet2: Scaling Speaker Verification via Time-Pooled Dimension Reshaping},
  author={Yakovlev, Ivan and Okhotnikov, Anton},
  booktitle={Proceedings of Interspeech 2026},
  year={2026},
  note={Accepted to Interspeech 2026},
  eprint={2603.11841},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  url={https://arxiv.org/abs/2603.11841}
}
```

许可证：MIT。英文官方说明以仓库根目录为准；若中英文表述冲突，以英文 README 与代码为准。


## 2026-10-25 仓库检查与代码维护

本次现行实验代码、脚本、合成测试与技能源已归入仓库。修复内容、479个CPU用例及各项独立复验范围见[全仓检查与修复报告](reviews/2026-10-25-全仓检查与修复报告.md)。该报告不代替新鲜云端进度或真实GPU、完整字质量验收。
