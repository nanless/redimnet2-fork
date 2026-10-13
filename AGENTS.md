# 在本仓库里写字、改代码时请遵守的说明

这份文件给两类读者看：

1. **刚接触说话人验证的研究生**：用来弄清本仓库在做什么、数据放哪、训练怎么接起来。  
2. **在本仓库里协助改代码或写文档的自动助手**：必须按下面的文字习惯和事实来写，不要另造一套说法。

根目录的英文 `README.md` 仍是对外的英文说明；面向学生的中文导读在 `docs/`，评测数字在 `checkpoints/reports/`。  
**写给学生看的中文材料时，以本文的文字习惯为准。**

---

## 1. 怎么写字（强制）

### 1.1 总原则

- 用**完整、好懂的中文句子**。句子可以长一点，但要把因果说清楚。  
- 假定读者是**入门研究生**：学过一点深度学习，但不一定熟悉说话人验证工程习惯。  
- 宁可多写两句解释，也不要用只有业内或工具圈才懂的词来「显得专业」。  
- **仓库里我们能改的阶段名、说明用文件名、文档名：不要为「显得专业」或「和旧习惯一致」而刻意保留黑话。** 能改成白话或直白英文的，就改掉，并同步改链接与正文。  
- 第三方库、磁盘上已有的大数据路径、别人仓库里的文件名，以及真正写进命令行/配置里的键名，正文里可以照实写出（因为磁盘或接口上就是那些字）；**叙述时仍要用中文说明它们干什么**。  
- 第一次出现缩写时，先写中文全称，再在括号里写常见缩写。例如：等错误率（EER）。

### 1.2 不要用的说法

下面这些词和写法**不要出现在给学生看的文档、报告、以及本仓库相关回复里**。  
配置键、外部数据路径、第三方库自带的名字可以照实出现，但**不要借「路径/配置除外」去保留本仓库自己起的黑话文件名或阶段名**。

**工程黑话 / 生造中英夹杂（举例，不限于此）：**

| 不要写 | 可以改成 |
|--------|----------|
| manifest | 训练用的样本清单；或直接写文件名，如 `data.list`、`utt2spk`；命令阶段名用 `lists` |
| canonical（路径） | 约定好的主目录；请以某某路径为准 |
| 落地 / 闭环 / 对齐（空泛地用） | 写完了、改好了、数字和官方表一致 |
| blocker / residual / nit | 会挡住训练的问题 / 剩下的小问题 / 文字上的小毛病 |
| PASS / FAIL（当口号甩） | 检查通过 / 检查未通过，并写清差在哪 |
| leverage / spin up / ship / land | 使用 / 启动 / 提交或发布 / 改完 |
| pipeline（不解释就甩） | 训练流程、数据处理步骤 |
| artifact | 生成的文件、中间结果、评测表格 |
| scaffold / boilerplate | 现成模板、固定套话代码 |
| 多 agent / subagent（对学生正文） | 不需要提工具分工；只写结论和依据 |
| 一句话结论（营销腔） | 可以写「先看这里」，但内容要具体 |

本仓库数据准备脚本 `prepare_data.py` 的命令阶段：`download` → `extract` → `convert` → **`lists`**（写 `data.list`、`utt2spk` 等样本清单）→ 可选的 `lmdb`。  
**不要**再用 `manifests` 这类黑话当阶段名。

**故弄玄虚的翻译或半中半英：**

- 不要写「跑一下 alignment」「看看 delta 是否 OK」而不解释。  
- 不要把「样本清单」写成「清单文件（manifest）」再反复用英文词。  
- 不要堆砌「赋能、抓手、颗粒度、链路、范式」这类空词。

**自动助手腔（一律禁止）：**

- 「我来帮你把 xxx 搞定」「已为您」「轻松一键」  
- 「根据我的分析」「作为 AI」  
- 无必要的英文小标题夹在中文段落里（如 `## Verdict`、`## Key Takeaways`）

### 1.3 推荐写法

- 先说**读者现在能不能训练 / 数字能不能信**，再说细节。  
- 路径写全，并说明**这个目录里是什么**。  
- 数字写清单位：句数、说话人数、文件数、等错误率用百分点。  
- 未做完的事单独列，写**差哪一步**，不要用含糊的「进行中」敷衍。

---

## 2. 这个仓库在做什么

**说话人验证**：给两段语音，判断是不是同一个人。  
常用做法是：把每段语音变成一个固定长度的向量（常叫 embedding，可说「说话人向量」），再算余弦相似度；用等错误率（EER）衡量好坏，**越低越好**。

本仓库实现的是论文 **ReDimNet2** 的模型与训练代码：

- 模型代码主要在 `redimnet2/`。只做推理时，看根目录英文 README 和 `docs/` 里的上手说明即可。  
- 完整训练在 `applications/speaker_rec/`，里面是精简后的 WeSpeaker 风格训练栈（目录名 `wespeaker_lite`）和若干实验配置（目录名 `recipes`）。

型号从轻到重常记作 B0、B1、…、B6。每个型号通常有两个训练阶段：

- **主训练**（配置文件名里常有 `ptn`）：时间更长，常用语速扰动等。  
- **大间隔微调**（配置文件名里常有 `lm`）：在主训练权重上再训较短一段时间，推理时一般优先用这一阶段的权重。

---

## 3. 两条训练线分别是什么

### 3.1 只练 VoxCeleb2（目录 `examples/vox2`）

- 训练语音：VoxCeleb2 的开发集（dev）。  
- 训练过程中的验证：VoxCeleb1 上的标准试验列表（常见文件名 `veri_test2.txt`，对应常说的 Vox1-O）。  
- 数据增强用的噪声与混响：MUSAN、以及 RIRS 里的模拟房间脉冲（`simulated_rirs`）。  
- 配置：`recipes/redimnet2/b0` … `b6`，每个型号有 `ptn.yaml` 和 `lm.yaml`。

在本机已经准备好时，配置里默认会读：

- 训练样本清单：`/data/wespeaker_data/vox2_dev/data.list`  
- 语句到说话人：`/data/wespeaker_data/vox2_dev/utt2spk`  
- 噪声目录：`/data/wespeaker_data/raw/musan/music`（以及 `noise`、`speech`）  
- 混响目录：`/data/wespeaker_data/raw/rirs/simulated_rirs`

### 3.2 混合数据（目录 `examples/vb2_vox2_cnc2`）

在上一节的基础上，再把下面两套语料拼进同一份训练清单：

- VoxBlink2（大规模说话人数据，配置与清单里常用前缀 `B_`）  
- CN-Celeb2（中文说话人数据，常用前缀 `C_`）  
- VoxCeleb2 在混合清单里常用前缀 `A_`  

当前配置指向（需要事先生成这些文件）：

- 主训练：`/data/wespeaker_data/vox2_vb2_cnc2_v0/mix_join/data.list` 与同目录下的 `utt2spk`  
- 大间隔微调：同根目录下的 `mix_join_lm/`  

生成脚本在：

- `applications/speaker_rec/examples/vb2_vox2_cnc2/scripts/prepare_vox2_vb2_cnc2_data.sh`  
- `applications/speaker_rec/examples/vb2_vox2_cnc2/scripts/prepare_vox2_vb2_cnc2_lm_data.sh`

---

## 4. 「训练用的样本清单」是什么（不要叫别的神秘名字）

音频本身是大量 `.wav`（或 `.flac`）文件。  
训练程序不会自己把整个硬盘扫一遍当训练集，而是先读几张**表**，表里写明：有哪些句子、文件在哪、属于哪个说话人。

本仓库最常见的几张表：

| 文件 | 作用（白话） |
|------|----------------|
| `data.list` | 每行一条训练样本，一般是 JSON，至少包含样本编号、说话人、音频路径 |
| `utt2spk` | 每一句对应哪个说话人 |
| `spk2utt` | 每个说话人有哪些句子（有的脚本会生成，便于检查） |
| `wav.scp` | 样本编号到音频路径（Kaldi / WeSpeaker 习惯；增强数据准备时也可能生成） |

可以记成：**表告诉程序「读谁」；磁盘上的 wav 才是「声音本身」。**  
缺表时，即使 wav 都在，训练配置也跑不起来。

准备 VoxCeleb2 那条线的表，用：

`applications/speaker_rec/examples/vox2/scripts/prepare_data.py`

默认会做：下载（可选）→ 解压 → 转成 wav → **写上面这些表**。  
**默认不会**再去做一种旧的键值库打包（见下一节）。

---

## 5. 数据增强：直接读原始音频，不要先打成键值库

以前有些流程会把 MUSAN、RIRS 先打进 LMDB（一种便于随机读的键值库），再在配置里写 `noise_lmdb_file`、`reverb_lmdb_file`。

**本仓库当前的 ReDimNet2 实验配置已经改成：增强时直接从文件夹读 wav。**

- 配置里用 `dataset_dir` 指向目录。  
- 全部 18 个 `recipes/redimnet2/**/*.yaml` **不应再出现** `lmdb_file`、`noise_lmdb_file`、`reverb_lmdb_file`、`lmdb_to_ram`。  
- `prepare_data.py` 默认阶段不含 `lmdb`；只有有人显式传入 `--stages lmdb` 才会构建那种旧格式。  
- 代码里仍保留读 LMDB 的能力，只为兼容旧配置；**新实验不要依赖它。**

本机约定的增强目录（可以是软链接，指向共享盘上的真实数据）：

```text
/data/wespeaker_data/raw/musan/music
/data/wespeaker_data/raw/musan/noise
/data/wespeaker_data/raw/musan/speech
/data/wespeaker_data/raw/rirs/simulated_rirs
```

说明：`prepare_data.py` 若从官方压缩包解压 RIRS，原始目录名可能是 `raw/RIRS_NOISES/simulated_rirs`。  
脚本会再准备好配置里用的 `raw/rirs/simulated_rirs`（通常是软链接），避免路径对不上。

---

## 6. 本机数据现状（供写报告、开训前核对）

下面数字以 **2026-10-13** 前后在本机核对为准。若以后又拷贝或重转数据，请重新计数后再改文档。

### 6.1 请以哪套 VoxCeleb 为准

历史上有过不完整的拷贝（句数少一截）。  
**现在请以这套为准：**

```text
/root/group-shared/voiceprint/data/speech/speaker_verification/voxceleb/
```

其中：

- `vox2/wav/`：从完整 AAC 压缩包装换得到的 16 kHz 单声道 wav  
- `vox1/wav`：指向已有的 VoxCeleb1 全量 wav（软链接）  
- 说明文件：同级目录下的 `VoxCeleb约定主目录说明.txt`（写清以哪套 VoxCeleb 为准）  
- 拷贝与转换记录：`checkpoints/reports/voxceleb_copy_status.txt`

不要再把下面这类**不完整**的旧树当成训练主数据：

```text
/aispeech-cfs/voiceprint/data/voxceleb/vox2/wav
```

### 6.2 数量核对（期望值）

| 数据 | 期望或本机结果 | 说明 |
|------|----------------|------|
| VoxCeleb2 开发集 | **1,092,009** 句，**5,994** 个说话人 | 训练主数据 |
| VoxCeleb2 测试集 | 36,237 句 | 一般不拿来做本仓库这条训练 |
| VoxCeleb2 说话人目录 | 盘上 6,112 个；名单里测试说话人可能多写 2 个 | 缺的是测试集目录名，**不影响开发集句数** |
| MUSAN | music 660 + noise 930 + speech 426 = **2,016** | 增强用 |
| RIRS 模拟房间脉冲 | **60,000** 个 wav | 增强用 |
| CN-Celeb2 | 1,996 个说话人；flac 与 wav 各约 **524,787** 句 | 混合线用 |
| VoxBlink2 | `audio/` 下约 **111,284** 个说话人目录；已有解压完成标记 | 混合线用；全量句数若要写进报告需再统计 |

### 6.3 训练入口是否齐（截至写本文时）

| 项目 | 是否齐 |
|------|--------|
| `/data/wespeaker_data/vox2_dev/data.list` 与 `utt2spk`（各 1,092,009 行） | 齐 |
| `/data/wespeaker_data/raw/musan/...` 与 `raw/rirs/simulated_rirs` | 齐 |
| `/data/wespeaker_data/vox1-test/`（验证用试验列表及用到的 wav） | 齐（按试验列表链接了用到的句子；不是单独官方 test 小包的每一个文件都链全，但不影响按 `veri_test2.txt` 验证） |
| `/data/wespeaker_data/vox2_vb2_cnc2_v0/mix_join/` | **还没有**；混合线要先跑第 3.2 节的脚本 |

因此：

- **只练 VoxCeleb2：数据表和增强目录已经齐，配置也已改成读原始音频。**  
- **混合线：原始语料基本齐，还缺混合后的 `mix_join` / `mix_join_lm` 样本清单。**

更细的检查表见：`checkpoints/reports/05-训练数据检查.md`（若与本文冲突，以重新计过数的结果为准，并同时改两处）。

---

## 7. 评测报告在哪里

对**官方发布的预训练权重**做过复现评测，中文说明在：

```text
checkpoints/reports/
```

建议阅读顺序写在该目录的 `README.md`。  
原始汇总表在 `checkpoints/` 下若干 `.csv` 文件中。

写评测相关文字时：

- 说清测了哪些试验协议（例如 VoxCeleb1-O / E / H）。  
- 等错误率用百分点；和官方表比较时写「我们的数减去官方的数」。  
- 不要用「对齐得很好」代替具体数字。

---

## 8. 改代码时注意什么

1. **不要把训练配方改回必须使用 LMDB。** 新改动应保持 `dataset_dir` 读文件夹。  
2. **不要引入第二套神秘路径昵称。** 需要「以哪份数据为准」时，写绝对路径，并说明为什么。  
3. **大文件、日志、权重、整库音频不要塞进 git。** 报告里写路径和计数即可。  
4. **改训练配置后**，核对：增强目录是否存在、`data.list` 行数是否合理、验证列表路径是否可读。  
5. **给学生看的文档**若要更新，同步遵守本文第 1 节；并尽量同步改 `docs/` 与 `checkpoints/reports/` 里过时的说法。  
6. 自动助手若开多个并行任务改同一批文件，合并后要再检查一遍，避免有的文件改回旧 LMDB 写法。

代码阅读入口（中文）：

- `docs/03-code-walkthrough.md`：文件和概念对应关系  
- `docs/04-训练流程.md`：数据如何进模型、损失怎么算  
- `docs/05-recipes-and-repro.md`：如何准备数据与启动训练  

---

## 9. 常见缩写对照（正文里请先写中文）

| 缩写 | 建议在正文中的说法 |
|------|--------------------|
| EER | 等错误率 |
| PTN | 主训练阶段（对应 `ptn` 配置 / 权重） |
| LM | 大间隔微调阶段（对应 `lm` 配置 / 权重） |
| MUSAN | 噪声与音乐等增强用音频集合（直接写 MUSAN 亦可，但要说明用途） |
| RIR / RIRS | 房间脉冲响应 / 该数据集 |
| VB2 | VoxBlink2 |
| CNC2 | CN-Celeb2 |
| DDP | 多卡数据并行 |
| embedding | 说话人向量（固定维） |

模型结构里的专有名词（如 ASTP、ReDim）在 `docs/06-glossary-faq.md` 有表；写给学生看时仍应先用白话解释「做什么」，再挂缩写。

---

## 10. 给自动助手的额外约束（学生可以当「仓库规矩」读）

- 回答本仓库问题时，**默认用第 1 节的文字习惯**；不要对学生使用工具圈黑话。  
- 涉及数据是否齐全时，**先查路径和行数/文件数**，再下判断；不要只复述旧报告。  
- 用户明确禁止某类操作时（例如「不要转成 LMDB」），后续改动不得再偷偷加回去。  
- 未要求提交 git 时，不要擅自 `git commit` 或推远程。  
- 需要写长篇中文说明时，优先写入仓库文档（如 `docs/`、`checkpoints/reports/`、本文），而不是只在对话里丢一堆口号式结论。

---

## 11. 相关文件索引

| 路径 | 内容 |
|------|------|
| `AGENTS.md` | 本文：文字习惯 + 训练/数据事实 |
| `docs/README.md` | 中文导读总入口 |
| `docs/01-background.md` … `06-glossary-faq.md` | 背景、结构、代码、训练、复现、术语 |
| `checkpoints/reports/README.md` | 官方权重复现评测入口 |
| `checkpoints/reports/05-训练数据检查.md` | 训练语料核对表 |
| `applications/speaker_rec/examples/vox2/README.md` | 只练 VoxCeleb2 的操作说明 |
| `applications/speaker_rec/examples/vb2_vox2_cnc2/README.md` | 混合数据线的操作说明 |

若你发现文档里又出现第 1.2 节禁止的说法，请改成白话，并尽量在同一批修改里改干净。
