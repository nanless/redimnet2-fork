<!-- lint-allow: queue_state,fixed_claim,exec_bit -->
<!-- 本索引有两处要"引用旧说法"：第 1 节说明 LM 曾经被写成「排队中」，第 3 节整节是
     "不能照做的说法"清单。引用≠断言，故整篇豁免这两条规则；其余规则仍然生效。 -->

# ReDimNet2 中文文档总索引

> 这是 `docs/` 的总入口，回答三个问题：现在到哪一步了、遇到什么问题该读哪一篇、哪一篇已经过时。
> 根目录英文 [README.md](../README.md) 是对外说明；本目录是给学生看的中文材料。中英文冲突时，以英文 README 与代码为准。

**论文**：[ReDimNet2: Scaling Speaker Verification via Time-Pooled Dimension Reshaping](https://arxiv.org/abs/2603.11841)（Interspeech 2026）
任务本身很简单：把一段语音变成一个固定长度的「说话人向量」，再用余弦相似度判断两段语音是不是同一个人。

本索引最后修订：2026-11-08。这一版按真实性审计的结果修掉 `docs/` 里的假断言，并建立了防漂机制：易变事实只写在自动生成的 [STATUS.md](STATUS.md)，其余文档写“用这条命令查”。规则与逐篇可信度见 [18 号文档](18-文档索引与状态.md)。

---

## 1. 现在到哪一步了（2026-11-08 核对）

> 本节只写哪条线走到了哪一步。具体数字（轮数、步数、sha、侧车条数、任务号、权限位、测试计数）看 [STATUS.md](STATUS.md)：它由脚本生成，抄到本页就会过期。

| 工作线 | 状态 | 细节 |
|---|---|---|
| 三语料 B6 主训练（PTN） | 已完成 120/120 轮（2026-11-03 结束），八卡已归还 | 末轮 `vox1-eer-8s` 只读每段音频开头 8 秒，不能和官方 0.26% 的整段协议直接比 |
| 大间隔微调（LM） | 运行目录已跑完 1 轮，不是「排队中」 | 见下面「LM 已跑完，任务号归属本地核不了」 |
| 三训练域自动文字与时间标注 | 2026-11-01 13:13:34（北京）停摆，累计约 52.64% 的源 | 续跑前要重新取得准入；**v4 那三代先后失败**，形态见 [16 号](16-实验维护与验收边界.md) |
| 完整字自然变长微调 / 长度交叉评测 | **未启动** | 目前只有 CPU 上的九格长度矩阵，见 [13 号](13-训练权重导出与评测.md) 第 6 节 |

### LM 已跑完，任务号归属本地核不了

LM 已经跑过了。2026-11-06 收口时各篇文档都写「已提交、排队中」，那是当晚的观测，现在不成立：

- 运行目录 `/root/experiments/redimnet2/b6-mix-official-20261019/lm-from-ptn115-20261106-v1/lm` 在 11-07 03:05–05:37 UTC 之间写满了 1 轮：`13075` 次优化更新（`train.log` 两行都是这个数）、`models/model_1.pt`、`val_log.csv` 记 `vox1-eer-8s = 0.293`、回执 `training_exit_code: 0`。大小与 sha 见 [STATUS 第 4 节](STATUS.md)。
- **但回执的 `ok` 仍是 `false`**：冷启动标记只写进 `cloud_launch_lm.log`，回执却去读 `train.log`，于是判成"没看到冷启动"。仓库源码已改成两份日志都读，**r3 冻结包没重打**，那份回执不会自己变对。看到 `ok:false` 先去日志找标记，再判断训练是否失败。
- 六组正式评测**还没对 LM 权重跑过**（运行目录下没有 `released/`、`evaluation/`）。

任务号只有平台侧记录。运行产物里没有 `task_id` 字段，CLI 又缺凭据读不到任务，所以本地只能证到“运行目录这一层”。平台侧记录里完成者是 `t-20261106110454-vd8pq`，本地无该 ID 的留档；更早的 `t-20261106195855-rq2b7`（入口多带 `--execute`，被 gate 拒绝，从未运行）与 `t-20261107053934-6lcmt`（冻结包内 `launch.sh` 缺执行位）均已作废。判断 LM 跑没跑，看运行目录与回执，与任务号对不对无关。

> 带日期的历史段落保留原时刻，不改写历史。任何「现在」的判断以 [STATUS.md](STATUS.md) 与当次命令输出为准；`AGENTS.md` 第 9 节仍留着 11-06 的「排队中」写法，读到时按本节理解。

---

## 2. 按问题找文档

### 入门：先建立直觉（半天到两天）

| 文档 | 解决什么问题 / 什么时候读 | 状态 |
|---|---|---|
| [01 背景与动机](01-background.md) | 说话人验证在做什么、EER 怎么读、波形怎么变成 Mel 频谱。刚接触这个方向时先读它 | 现行 |
| [02 网络结构](02-architecture.md) | 仓库分层与数据流、时间池化、ASTP、B0–B6 的差别。想知道模型为什么这样设计时读 | 现行（2026-11-08 整篇重写） |
| [03 代码地图](03-code-walkthrough.md) | 每个文件对应哪个概念；变长批量与逐条推理的真实差距。准备读或改代码前读 | 现行 |
| [06 术语表与 FAQ](06-glossary-faq.md) | 缩写、常见误解、`-eer-8s` 与官方表的口径差。遇到看不懂的词或数字时查 | 大部分现行；Q14 的「能续训」对本次 LM 不适用 |

> 02、03 两篇在 2026-11-08 被整篇改写（02 → 仓库分层与数据流，03 → 六个入口与逐文件代码走读），上表已按新版本写。行数与最后修改时间不抄，用 18 号里的循环命令取。

### 配方：数据和超参怎么配（要自己训模型时）

| 文档 | 解决什么问题 / 什么时候读 | 状态 |
|---|---|---|
| [05 实验配方与复现](05-recipes-and-repro.md) | 两条训练线、数据准备、B0–B6 差异、启动命令、常见坑。要自己跑一个模型时从头读 | **部分过时**：「改 `lm.yaml` 再启 LM」已被冻结包入口取代；「混合线可直接开训」是按 10 月的状态写的 |
| [08 变长训练与长度分桶](08-变长训练与长度分桶.md) | 变长开关、长度分桶、按语音边界取段的全部配置与实测。想少补零或按完整语音段取样时读 | 现行。文中「原有那 18 份固定长度配方没被改」是**对的**（32 = 18 + 6 + 6 + 2） |
| [12 三语料 B6 官方复现配方核查](12-三语料B6官方复现配方核查.md) | 复现官方三语料 B6 的参数依据、学习率核算、LM 初始化检查门。要解释「为什么这样配」时读 | 现行（LM 状态已改为「运行目录跑完 1 轮」，任务号归属另注） |

### 执行：真正跑起来（训练、标注、评测）

| 文档 | 解决什么问题 / 什么时候读 | 状态 |
|---|---|---|
| [04 训练流程](04-训练流程.md) | 一条样本从磁盘到 loss、增强、验证口径、存档与续训、「写了但不起作用的键」。读训练代码或排查「改了没反应」时读 | 现行（已按「32 份里有 30 份」写对，两份官方复现配方是 `compile_backbone: null`） |
| [09 用语音活动检测给音频加时间标注](09-VAD标注与RTTM.md) | 用 FireRedVAD 给训练音频生成 `.rttm` 的完整做法、续跑、复查和实测速度。要跑或复查 RTTM 标注时读 | 现行（VAD 这一条线；文字与字界标注是另一条线，见 16 号） |
| [10 在火山引擎上用 8 张卡训练](10-云端八卡训练.md) | 通用八卡脚本、环境变量、开训前检查、失败对照、控制台提交通道。要在平台上跑任务时读 | **部分过时**：配方总数已改对；通用脚本「自动取最新轮」与正式入口固定第 115 轮的差别仍在 |
| [13 训练权重导出与评测](13-训练权重导出与评测.md) | 训练检查点怎么导出成发布权重、六组评测怎么跑、九格长度矩阵能说明什么 | 现行；但九格数字是 CPU 历史评测、未重跑，LM 六组正式评测仍未做 |
| [15 三语料 B6 八卡短测与正式训练记录](15-三语料B6八卡短测与正式训练记录.md) | 10-19 起每次任务的编号、失败原因、修复和状态补充。查「某次任务发生了什么」时读 | **历史记录（按小节日期读）**：LM 状态已更新为「跑完 1 轮」；「LM 完成后自动导出并评测」是 10-19 入口的设计 |

### 运维：核对环境、判断结论能不能用

| 文档 | 解决什么问题 / 什么时候读 | 状态 |
|---|---|---|
| [07 本机数据与开训检查](07-本机数据与开训检查.md) | 本机数据在哪、磁盘余量、逐项计数、开训前检查清单。换机器或开训前对路径时读 | **部分过时**：第 1 节「现在能不能开训」是 10-14 的结论；数据就绪不等于现在应该启动 |
| [14 三语料 B6 云端数据路径](14-三语料B6云端数据路径.md) | 共享盘怎么挂、绝对路径怎么准备、数据检查程序怎么跑。云端提交前核对挂载时读 | 现行 |
| [16 实验维护与验收边界](16-实验维护与验收边界.md) | 三条实验线的维护入口，以及「哪些结论能说、哪些不能说」。要判断一个说法站不站得住时读 | **现行**（判断结论能不能用时最该读的一篇）；标注失败一节已按 v2/v3/v4 拆开重写 |
| [18 文档索引与状态](18-文档索引与状态.md) | 防漂规矩、逐篇可信度、不能照做的说法。怀疑某篇过时或准备改文档前读 | 现行 |
| [STATUS.md](STATUS.md) | **全部易变事实**：HEAD、权限位、pytest 摘要、产物 sha、最近三次提交 | 自动生成，勿手改 |

### 历史：只作追溯

| 位置 | 内容 | 怎么用 |
|---|---|---|
| [`plans/`](plans/) | 按日期命名的执行计划（10-25 至 10-28） | 说明当时打算怎么做；**不代表现在的状态或授权**。份数用 `ls docs/plans/*.md | wc -l` |
| [`reviews/`](reviews/) | 文字记录、JSON 摘要和图片（10-25 至 11-07） | 每次操作的证据与边界；带日期，按日期读。**未逐篇复核**，链接没死不等于内容正确 |
| [15 号前半](15-三语料B6八卡短测与正式训练记录.md) | 10-19 至 10-20 的短测与任务更替 | 历史事实；后半有 11-06 状态补充 |

> 历史记录里出现「运行中」「已通过」「可启动」时，都只对应它自己的日期与源码；不能当作今天的状态。

---

## 3. 明确过时或已被替代的说法（改文档前先看）

> 定位用关键词，不用行号——行号一改就漂。每行都给了复核命令。

| 说法（关键词） | 现在应该怎么理解 | 复核 |
|---|---|---|
| 「LM 已提交、排队中」 | 运行目录已跑完 1 轮（13,075 步、exit 0、0.293） | `tail -2 <lm>/val_log.csv`；`cat <lm>/lm_receipt.json` |
| 「LM 完成者是 `rq2b7`」 | `rq2b7` 从未运行（`--execute` 被拒）；完成者是 `vd8pq`，**本地无留档** | `grep -rl "t-2026110" <lm> \|\| echo 无任务号留档` |
| 「回执 `ok:false`，所以训练失败」 | 假阴性：回执只读 `train.log`，冷启动标记在 `cloud_launch_lm.log` | `grep -c "cold start" <lm>/cloud_launch_lm.log` |
| 「当前共 30 份配方」 | 32 份：18 原始 + 6 `varlen_*` + 6 `vadseg_*` + 2 `official_repro_*` | `ls applications/speaker_rec/examples/*/recipes/redimnet2/*/*.yaml \| wc -l` |
| 「所有配方都写 `compile_backbone: default`」 | 两份 `official_repro_*.yaml` 是 `null`（本次固定不编译） | `grep -rc "compile_backbone" …/b6/official_repro_*.yaml` |
| 「5 个 `.sh` 是 644、没有可执行位」 | 用 `test -x` 当场核，当前值见 [STATUS 第 2 节](STATUS.md)。真丢过执行位的是**冻结包内**的 `launch.sh`（`launcher_missing`） | `test -x tools/open_annotation/launch_queue_handoff.sh && echo ok` |
| 「所有 argparse 入口都真跑过 `--help`」 | 只抽样跑过；`gpu_launch.py --help` 会打 usage 但非零退出（`SystemExit: 0` 被 `except BaseException` 抓住） | `/root/miniforge3/envs/redimnet2/bin/python tools/open_annotation/gpu_launch.py --help; echo rc=$?` |
| 「v4 已经按新包 + 新 pilot + 新 admission 修好了」 | 短窗护栏只进了冻结包与 17:59 重打的 pilot；v5 准入与 v4 生产的 `formal-code` **没有**；v4 生产死在 CUDA 初始化 | `grep -c MIN_WINDOW_FRAMES <各包>/worker.py` |
| 「PTN 跑完后把 `lm.yaml` 的 `model_ckpt` 指到权重，再启 LM」 | 本次只能走「冻结包 + 合同」入口，两个 sha256 要对上 | 见 [16 号](16-实验维护与验收边界.md) 与技能 LM 受控启动 |
| 「混合线可以直接开训」 | 数据与清单是齐的；但 PTN 已跑完 120 轮，再跑就是新实验、新输出目录 | — |
| 「微调中断了重启能接着训」 | 通用训练栈确实这样；本次 LM 入口要求 `new_run_root` 不存在，见到 `resumed optimizer state` 就停 | — |
| 「18 份固定长度配方应为 20 份」 | **上一版本索引判错了**：`08` 的「原有的 18 份」是对的（32 − 6 − 6 − 2 = 18），只是加上 2 份 `official_repro_*` 后固定长度配方总计 20 份 | `ls …/b6/*.yaml \| wc -l` |

> 每条的证据与复核命令见 [18 号文档](18-文档索引与状态.md)。本表只列会影响动手做实验的说法，不列文字小毛病。

---

## 4. 5 分钟上手：抽一个说话人向量

依赖（推理最小集）：`torch`、`torchaudio`、`scipy`、`numpy`。

```python
import torch
import torchaudio

# 默认：VoxCeleb2 上训好的 B6，大间隔微调权重（推荐）
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
|---|---|---|
| `model_name` | `b0` … `b6` | 从小到大，参数量约 1.1M → 12.3M |
| `train_type` | `ptn` / `lm` / `dis` | 主训练 / 大间隔微调（推荐）/ 蒸馏（部分型号） |
| `dataset` | `vox2`（默认）、`vb2+vox2_v0`、`vb2+vox2+cnc2_v0` | 训练数据组合 |

完整模型表与指标见根目录英文 [README.md](../README.md#results)。

---

## 5. 建议学习路径（约一周）

| 天数 | 做什么 | 文档 |
|:----:|--------|------|
| 1 | 弄清任务与指标；用 hub 抽向量、算余弦相似度 | [01](01-background.md)、本文第 4 节 |
| 2 | 理解「维度重塑」：`(B,C,F,T) ↔ (B,C·F,T)` | [02](02-architecture.md) |
| 3 | 对照代码：stem → stages → ASTP → Linear | [03](03-code-walkthrough.md) |
| 4 | 看 `train.py`、增强、投影头、PTN→LM | [04](04-训练流程.md) |
| 5 | 核对本机数据在哪、把路径问题解决掉；跑 B0 单卡 1 轮冒烟 | [07](07-本机数据与开训检查.md)、[05](05-recipes-and-repro.md) |
| 6–7 | 完整 B3；有条件再看混合数据配方 | [05](05-recipes-and-repro.md)、[06](06-glossary-faq.md) |

> 本机没有 `/data` 目录。不先处理路径，训练会在读样本清单时直接失败，详见 [07](07-本机数据与开训检查.md)。

---

## 6. 文档与代码的对应关系

| 文档 | 主要代码 |
|---|---|
| 02 网络结构 | `redimnet2/redimnet2.py`、`redimnet2/layers/redim_structural.py`、`layers/blocks.py` |
| 03 代码地图 | `redimnet2/layers/*`、`hubconf.py`、`scripts/eval_*.py` |
| 04 训练流程 | `applications/speaker_rec/wespeaker_lite/**` |
| 05 配方 | `applications/speaker_rec/examples/**/recipes/**` |
| 07 本机数据 | `AGENTS.md` 第 3、6 节、`checkpoints/reports/05-训练数据检查.md` |
| 08 变长与分桶 | `wespeaker_lite/dataset/{dataset,processor}.py`、`utils/executor.py` |
| 09 VAD 标注 | `applications/speaker_rec/scripts/vad/**` |
| 10 云端八卡 | `applications/speaker_rec/scripts/cloud/*.sh` |
| 12 官方复现配方 | `.../recipes/redimnet2/b6/official_repro_{ptn,lm}.yaml`、`scripts/cloud/check_b6_lm_init.py` |
| 13 权重导出与评测 | `scripts/export_checkpoint.py`、`scripts/eval_{vox1,ood}_checkpoints.py` |
| 14 云端数据路径 | `scripts/cloud/{prepare_b6_cloud_paths,check_b6_mix_data}.sh|py` |
| 16 维护与验收 | 三条实验线、维护入口、版本身份与历史证据的适用范围 |
| 评测报告 | `checkpoints/eval_*.csv`、`checkpoints/reports/**` |

---

## 7. 编号说明

- 编号从 01 到 16，但**没有 11**。11 这个编号在整理时被并入 10（云端八卡训练）与 12（配方核查），为了不打乱已有链接就空着。看到 10 直接跳 12 不是漏了一篇。
- 本次新建 18。另有一处同名风险：`/root/codex_artifacts/repo-handover-20261107/server-untracked-docs/` 下还有一套未入库的 `docs/17-…`、`docs/18-FLAC定位异常与接续检查.md`、`docs/16-训练巡检与阶段接续.md`，与仓库里的 16 号、本次的 18 号**重名**。若将来把那批文档合并进来，要先统一编号，否则会互相覆盖。

---

## 8. 引用

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

---

## 9. 维护这份索引的方法

1. 加一篇文档：在本文对应分组加一行，写清「解决什么问题 / 什么时候读」和状态；同时在 [18 号](18-文档索引与状态.md) 登记可信度与已知问题。
2. 改一篇文档：同步看 18 号里这篇的「冲突点」是否已消掉；消掉了就把那条从 18 号删掉。
3. 涉及「现在」的判断不写进正文。易变事实写进 `docs/STATUS.md`（由脚本生成），正文写命令或指路；确实要写具体数字时，标「截至 <日期> 的观测」并紧跟复核命令。
4. 改完后在仓库根跑这三条：

```bash
python3 tools/gen_status.py          # 刷新 docs/STATUS.md（会真跑一次 pytest）
python3 tools/gen_status.py --check  # STATUS.md 是否已过期（过期退出码 1）
python3 tools/gen_status.py --lint   # 正文有没有又写死 HEAD/权限位/测试计数（命中退出码 1）
```

5. 复查命令（都在仓库根执行）：

```bash
# 每篇文档最后一次被改的日期与提交
for f in docs/*.md; do printf '%s\t%s\n' "$f" "$(git log -1 --format=%ad --date=short -- "$f")"; done

# 文档内部链接是否都落得到文件（0 表示没有死链）
python3 - <<'PY'
import re, os, glob
bad = []
for f in sorted(glob.glob("docs/*.md")):
    for link in re.findall(r"\]\(([^)]+)\)", open(f, encoding="utf-8").read()):
        if link.startswith(("http", "#", "mailto")):
            continue
        p = os.path.normpath(os.path.join(os.path.dirname(f), link.split("#")[0]))
        if p and not os.path.exists(p):
            bad.append((f, link))
print(len(bad), bad)
PY
```
