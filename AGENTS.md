# 本仓库的中文写法与改代码约定

## GitHub 写入限制（2026年10月28日起）

- 本机（本地 Mac）对 GitHub 只读：浏览、下载、clone、fetch、pull 都可以，上传和推送不走。文件、代码、文档、数据、提交对象和分支都在限制范围内，网页、API、gh、自动同步也不作为例外。
- 本机 `remote.*.pushurl=disabled://github-push-blocked-by-policy` 是这条限制的落地方式，保留原样即可。改 URL、remote、端口，或者用代理、隧道让本机继续上传，效果上等于把限制绕过去。
- 用户后来放宽了一步：任务需要的仓库可以同步到 `dev_a800_2gpus`，由服务器自身的仓库或进程连接 GitHub，执行任务需要的提交和推送。本机可以用 SSH 发起远程命令；在本机 git push 再把数据流转出去不行。此前“服务器中转也禁止”的笼统限制，已被这次服务器许可替代。
- 已删除的 `nanless/redimnet2-fork` 远端 `codex/repository-audit-20261025` 分支不恢复、不重建。本机 26 仓库的禁推配置保持不变，这些仓库也不必重新扫描。
- 本地编辑、commit 可以做，作者与提交者仍是 `nanless <francis7999@outlook.com>`。需要写 GitHub 时走上面的服务器方式；GitLab 仍按用户指定的目的地协作。规则维护本身不代表仓库已经迁移或推送过，历史操作保持历史含义。

## 大数据落点硬规则（2026-11-09 起）

用户 2026-11-09 的硬性规定原文：**“大数据默认一律 `/root/group-shared/voiceprint` 下，不许写 `/root` 其他目录。”** 这条与上面的 GitHub 写入限制同级，是仓库规矩，不因任务方便、旧脚本写死或目录已经存在而例外。

| 数据类型 | 落点 | 例子 |
|------|------|------|
| 实验产物：训练 run、检查点、日志、评测的输入与导出、批量推理输出 | `/root/group-shared/voiceprint/experiments/redimnet2/<实验名>/` | `experiments/redimnet2/lm-from-ptn115-20261106-v1/` |
| 工具产物、准入材料、账本、评测原件：任务表单证据、日志截取、诊断 JSON、ETA、claim 归档、交接脚本 | `/root/group-shared/voiceprint/artifacts/redimnet2/<用途>/` | `artifacts/redimnet2/eval-official-20261108/`（评测）、`artifacts/redimnet2/native-speed-successor-20261028-v1/controls/`（准入与合同）、`artifacts/redimnet2/retired-claims-20261108/`（账本归档）、`artifacts/redimnet2/device-locks/`（设备锁）、`artifacts/redimnet2/repo-handover-20261107/`（交接脚本）、`artifacts/redimnet2/volcengine-submission/<task_id>/`（提交核验） |
| 开源标注侧车：逐音频标签、分片、租约与队列回执 | `/root/group-shared/voiceprint/open_annotation/redimnet2-silver-<日期>/` | `open_annotation/redimnet2-silver-20261109/handoff-v4r2-20261108-v6/run-v6/` |

- **`/root` 只放代码仓库与临时日志。** 代码在 `/root/code/github_repos/...` 与独立实验树；临时日志用完即弃。`/root/experiments` 已于 2026-11-09 整树迁到 `/root/dsh-workspaces-mac/02_实验与标注_ReDimNet2`（旧路径不再存在），`/root/codex_artifacts` 现为指向 `01_正式产物档案` 的兼容软链，`/root/annotations` 不存新产物：这些位置只作 provenance 只读引用，并在文档里标成“历史路径”；不往里写新批，也不建旧路径别名。
- **为什么不能先写 `/root` 再挪过去**：训练容器里 `/root` 是容器本地盘，`/shared/group-shared`（容器内通常另挂成 `/root/group-shared`）是 vePFS，两者是不同挂载点。跨挂载点的 `mv` 会退化为“复制 + 删除”：几百 GB 的检查点要先占双份空间，复制慢，中途失败还会留下半份文件，看起来像“已经移动”。所以新数据一开始就写 group-shared，不做“先落 `/root` 再搬”。
- **运行中的任务不因这条规则重启。** 2026-11-09 起没有继续写旧根的任务；旧根已整体迁走，已经写在其中的训练目录、侧车和冻结包只读，文档里用“历史路径”注明。要切换读取位置，先把文件按字节搬到新根并核 sha，再改引用；迁移不重跑实验、不改变判据和统计口径。容器里 group-shared 若是只读挂载，先按平台工作流调成读写或申请权限，不退回 `/root` 本地盘。

自检（提交任务或写报告前各跑一遍）：

```bash
# 1) 代码里不该再出现旧落点
grep -rn "/root/experiments" --include='*.py' tools/ | head
grep -rn "/root/codex_artifacts" --include='*.py' tools/ | head

# 2) 指导性文档里，旧落点只允许出现在标了“历史路径”或 storage-policy-allow 的行
grep -rn -e "/root/experiments" -e "/root/codex_artifacts" -e "/root/annotations" \
  AGENTS.md docs .agents/skills | grep -v -e "历史路径" -e "storage-policy-allow" | head
```

第 2 条会把本节自己的规则原文也列出来（规则要写清禁止的目录名，绕不开），这类命中不用处理；剩下只看历史记录之外的新命中。确实要留旧路径的运行中写法，在 Markdown 行尾加 `<!-- storage-policy-allow: 原因 -->`，在 shell 脚本里加 `# storage-policy-allow`。

这份文件给两类读者看：

1. **刚接触说话人验证的研究生**：用来弄清本仓库在做什么、数据放哪、训练怎么接起来。
2. **在本仓库里协助改代码或写文档的自动助手**：按下面的文字习惯和本文件记下的事实来写。同一件事在文档和对话里出现两套说法时，读者没法判断哪套作数，所以两边用同一套事实和同一个说法。

根目录的英文 `README.md` 仍是对外的英文说明；面向学生的中文导读在 `docs/`，评测数字在 `checkpoints/reports/`。写给学生看的中文材料，按本文文字习惯来。

第 3—6 节的数据与机器状态主要记录于 2026-10-13 至 10-14。那里的“已齐”“可启动”只表示当时按路径核查过，不是之后任意机器或任务的启动许可：第 6 节的“齐”说明数据和路径在，说明不了训练、标注或微调现在能不能跑。阶段走到哪一步、启动前要逐条验什么，看第 9 节（2026-11-09 核对）；动手前重新检查实际文件、资源与任务来源。第 9 节写判断和门，不抄每次提交都会变的数字：HEAD、权限位、测试摘要、产物 sha 看自动生成的 [仓库状态](docs/STATUS.md)（`python3 tools/gen_status.py` 生成，`--check` 判过期、`--lint` 查正文硬编码），标注进度看 [标注看板](docs/ANNOTATION-PROGRESS.md)。[实验维护与验收边界](docs/16-实验维护与验收边界.md) 说明三条实验线各用什么代码入口，带日期的历史记录保持原时刻的写法。

---

## 1. 怎么写中文（本仓库的统一约定）

### 1.1 总原则

- 用完整、好懂的中文句子。句子可以长一点，但因果要交代清楚。
- 假定读者是入门研究生：学过一点深度学习，不一定熟悉说话人验证的工程习惯。
- 宁可多写两句解释，也不用只有业内或工具圈才懂的词来显得专业：读者看不懂时，省下的那半句要他自己去查。
- 仓库里我们能改的阶段名、说明用文件名、文档名，不为了显得专业或跟旧习惯一致刻意保留黑话；能改成白话或直白英文的，就改掉，并同步改链接与正文。
- 第三方库、磁盘上已有的大数据路径、别人仓库里的文件名，以及真正写进命令行或配置的键名，正文里照实写出（磁盘或接口上就是那些字）；叙述时仍用中文说明它们干什么。
- 缩写第一次出现，先写中文全称，再在括号里写常见缩写，例如等错误率（EER）。

### 1.2 不用的说法

下面这些词和写法不进给学生看的文档、报告和本仓库相关回复。它们把具体对象藏起来：读者看到“manifest”，不知道指的是哪张表，也不知道该去哪个文件核对。配置键、外部数据路径、第三方库自带的名字可以照实出现；但“路径/配置除外”不能拿来保留本仓库自己起的黑话文件名或阶段名。

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

本仓库数据准备脚本 `prepare_data.py` 的命令阶段是 `download` → `extract` → `convert` → **`lists`**（写 `data.list`、`utt2spk` 等样本清单；阶段名原来的叫法是 `manifests`，已停用）→ 可选的 `lmdb`。

**故弄玄虚的翻译或半中半英：**

- 写「跑一下 alignment」「看看 delta 是否 OK」时，把要跑的对象和要看的东西写出来。
- 不要把「样本清单」写成「清单文件（manifest）」再反复用英文词。
- 不堆砌「赋能、抓手、颗粒度、链路、范式」这类空词：它们没有指出任何模块、动作或指标。

**自动助手腔（这类句子不含信息，删掉即可）：**

- 「我来帮你把 xxx 搞定」「已为您」「轻松一键」
- 「根据我的分析」「作为 AI」
- 无必要的英文小标题夹在中文段落里（如 `## Verdict`、`## Key Takeaways`）

### 1.3 推荐写法

- 先说读者现在能不能训练、数字能不能信，再说细节。
- 路径写全，并说明这个目录里是什么。
- 数字写清单位：句数、说话人数、文件数、等错误率用百分点。
- 没做完的事单独列，写差哪一步，不用含糊的「进行中」敷衍。

---

## 2. 这个仓库在做什么

**说话人验证**：给两段语音，判断是不是同一个人。常用做法是把每段语音变成一个固定长度的向量（常叫 embedding，可说「说话人向量」），再算余弦相似度；用等错误率（EER）衡量好坏，**越低越好**。

本仓库实现的是论文 **ReDimNet2** 的模型与训练代码：

- 模型代码主要在 `redimnet2/`。只做推理时，看根目录英文 README 和 `docs/` 里的上手说明即可。
- 完整训练在 `applications/speaker_rec/`，里面是精简后的 WeSpeaker 风格训练栈（目录名 `wespeaker_lite`）和若干实验配置（目录名 `recipes`）。

型号从轻到重常记作 B0、B1、…、B6。每个型号通常有两个训练阶段：

- **主训练**（配置文件名里常有 `ptn`）：时间更长，常用语速扰动等。
- **大间隔微调**（配置文件名里常有 `lm`）：在主训练权重上再训较短一段时间，推理时一般优先用这一阶段的权重。本仓库这一次的 LM 是**冷启动**（只载权重，优化器与缩放器从零开始），入口是“冻结包 + 合同”的专用入口；改个配置直接跑 `train.py` 走不通，见 9.2。

---

## 3. 两条训练线分别是什么

### 3.1 只练 VoxCeleb2（目录 `examples/vox2`）

- 训练语音：VoxCeleb2 的开发集（dev）。
- 训练过程中的验证：VoxCeleb1 上的标准试验列表（常见文件名 `veri_test2.txt`，对应常说的 Vox1-O）。
- 数据增强用的噪声与混响：MUSAN、以及 RIRS 里的模拟房间脉冲（`simulated_rirs`）。
- 配置：`recipes/redimnet2/b0` … `b6`，每个型号有 `ptn.yaml` 和 `lm.yaml`。

本机准备好之后，配置默认读下面这些位置（就是配方里现在写的真实路径，本机可直接用）：

- 训练样本清单：`/root/group-shared/voiceprint/data/wespeaker_data/vox2_dev/data.list`
- 语句到说话人：同目录下的 `utt2spk`
- 噪声目录：`/root/group-shared/voiceprint/data/wespeaker_data/raw/musan/music`（以及 `noise`、`speech`）
- 混响目录：`/root/group-shared/voiceprint/data/wespeaker_data/raw/rirs/simulated_rirs`

这些地方早期写的是 `/data/wespeaker_data/...`，本机没有 `/data` 目录，现已改成上面的真实位置。

### 3.2 混合数据（目录 `examples/vb2_vox2_cnc2`）

在上一节的基础上，再把下面两套语料拼进同一份训练清单：

- VoxBlink2（大规模说话人数据，配置与清单里常用前缀 `B_`）
- CN-Celeb2（中文说话人数据，常用前缀 `C_`）
- VoxCeleb2 在混合清单里常用前缀 `A_`

当前配置指向下面这些位置（需要事先生成这些文件；路径就是配方里写的真实位置）：

- 主训练：`/root/group-shared/voiceprint/data/wespeaker_data/vox2_vb2_cnc2_v0/mix_join/data.list` 与同目录下的 `utt2spk`
- 大间隔微调：同根目录下的 `mix_join_lm/`

生成脚本在：

- `applications/speaker_rec/examples/vb2_vox2_cnc2/scripts/prepare_vox2_vb2_cnc2_data.sh`
- `applications/speaker_rec/examples/vb2_vox2_cnc2/scripts/prepare_vox2_vb2_cnc2_lm_data.sh`

---

## 4. 「训练用的样本清单」是什么

音频本身是大量 `.wav`（或 `.flac`）文件。训练程序不会把整个硬盘扫一遍当训练集，它先读几张**表**，表里写明有哪些句子、文件在哪、属于哪个说话人。

本仓库最常见的几张表：

| 文件 | 作用（白话） |
|------|----------------|
| `data.list` | 每行一条训练样本，一般是 JSON，至少包含样本编号、说话人、音频路径 |
| `utt2spk` | 每一句对应哪个说话人 |
| `spk2utt` | 每个说话人有哪些句子（有的脚本会生成，便于检查） |
| `wav.scp` | 样本编号到音频路径（Kaldi / WeSpeaker 习惯；增强数据准备时也可能生成） |

表里只有指针，音频内容始终在磁盘上的 wav 里。缺表时，即使 wav 都在，训练配置也跑不起来。

准备 VoxCeleb2 那条线的表，用：

`applications/speaker_rec/examples/vox2/scripts/prepare_data.py`

默认做的是：下载（可选）→ 解压 → 转成 wav → **写上面这些表**，不会再顺带做一种旧的键值库打包（见下一节）。

---

## 5. 数据增强直接读原始音频，不先打成键值库

以前有些流程会把 MUSAN、RIRS 先打进 LMDB（一种便于随机读的键值库），再在配置里写 `noise_lmdb_file`、`reverb_lmdb_file`。本仓库当前的 ReDimNet2 实验配置改成了增强时直接从文件夹读 wav。

- 配置里用 `dataset_dir` 指向目录。
- 全部 `recipes/redimnet2/**/*.yaml` 里已经不再出现 `lmdb_file`、`noise_lmdb_file`、`reverb_lmdb_file`、`lmdb_to_ram` 这几个键：原有 18 份，加上变长训练新增的 6 份 `varlen_*.yaml`、按语音边界取段的 6 份 `vadseg_*.yaml`，以及后加的 2 份 `official_repro_*.yaml`，当前共 32 份；2026-10-25 重新逐份检查，这 32 份都不含这几个键。
- `prepare_data.py` 默认阶段不含 `lmdb`；只有显式传入 `--stages lmdb` 才会构建那种旧格式。
- 代码里仍保留读 LMDB 的能力，只为兼容旧配置；新实验不必依赖它。

本机约定的增强目录（是软链接，指向共享盘上的真实数据）：

```text
/root/group-shared/voiceprint/data/wespeaker_data/raw/musan/music
/root/group-shared/voiceprint/data/wespeaker_data/raw/musan/noise
/root/group-shared/voiceprint/data/wespeaker_data/raw/musan/speech
/root/group-shared/voiceprint/data/wespeaker_data/raw/rirs/simulated_rirs
```

`prepare_data.py` 若从官方压缩包解压 RIRS，原始目录名可能是 `raw/RIRS_NOISES/simulated_rirs`；脚本会再准备好配置里用的 `raw/rirs/simulated_rirs`（通常是软链接），避免路径对不上。数这些目录里的 wav 个数时要跟随软链接（`find -L`），不加 `-L` 会数成 0。

关于路径：18 份配方原先统一写 `/data/wespeaker_data/...`，本机没有 `/data` 目录；现在配方已改成真实位置（`/root/group-shared/voiceprint/data/wespeaker_data/`），本机可以直接启动，不需要再建软链接。换机器时用 `--config_overrides` 覆盖文件改路径，不直接改配方。详见 `docs/07-本机数据与开训检查.md`。

---

## 6. 本机数据现状（写报告、开训前核对用）

下面数字以 **2026-10-13 到 10-14** 在本机核对为准。以后又拷贝或重转数据，重新计数后再改文档。本轮核对同时更正了两处过时说法：`/data` 路径在本机不存在、旧 VoxCeleb 树其实是齐的。

### 6.1 以哪套 VoxCeleb 为准

历史上有过不完整的拷贝（句数少一截）。现在以这套为准：

```text
/root/group-shared/voiceprint/data/speech/speaker_verification/voxceleb/
```

其中：

- `vox2/wav/`：从完整 AAC 压缩包装换得到的 16 kHz 单声道 wav
- `vox1/wav`：指向已有的 VoxCeleb1 全量 wav（软链接）
- 说明文件：同级目录下的 `VoxCeleb约定主目录说明.txt`（写清以哪套 VoxCeleb 为准）
- 拷贝与转换记录：`checkpoints/reports/voxceleb_copy_status.txt`

下面这棵旧树的说法要改：

```text
/aispeech-cfs/voiceprint/data/voxceleb/vox2/wav
```

以前这里写的是“不完整，别再当训练主数据”。但在 2026-10-13 到 10-14 实测，这棵旧树的计数与约定主目录完全一致（各 6,112 个说话人目录、1,128,246 句，抽查文件大小逐字节相同）。所以“少一截”的旧状态现在不成立 —— 可能是后来补齐了，也可能当初描述的是更早的时刻。

按 `VoxCeleb约定主目录说明.txt` 的约定，仍建议以约定主目录为准；现在写它“不完整”是错的。实测细节见 `docs/07-本机数据与开训检查.md` 第 5 节。

### 6.2 数量核对（期望值）

| 数据 | 期望或本机结果 | 说明 |
|------|----------------|------|
| VoxCeleb2 开发集 | **1,092,009** 句，**5,994** 个说话人 | 训练主数据 |
| VoxCeleb2 测试集 | 36,237 句 | 一般不拿来做本仓库这条训练 |
| VoxCeleb2 说话人目录 | 盘上 6,112 个；名单里测试说话人可能多写 2 个 | 缺的是测试集目录名，**不影响开发集句数** |
| MUSAN | music 660 + noise 930 + speech 426 = **2,016** | 增强用 |
| RIRS 模拟房间脉冲 | **60,000** 个 wav | 增强用 |
| CN-Celeb2 | 1,996 个说话人；flac 与 wav 各约 **524,787** 句 | 混合线用 |
| VoxBlink2 | `audio/` 下 **111,284** 个说话人目录；**9,904,382** 句 | 混合线用；句数来自生成清单后逐行核对 |

### 6.3 训练入口是否齐（截至 2026-10-13 实测）

配方里现在写的就是共享盘上的真实位置：

```text
/root/group-shared/voiceprint/data/wespeaker_data/
```

早期配方写的是 `/data/wespeaker_data/...`，本机没有 `/data` 目录，照原样跑会失败；现在已全部改成上面的真实位置，本机可直接启动。

| 项目 | 是否齐 | 说明 |
|------|------|------|
| `vox2_dev/data.list` 与 `utt2spk` | 齐 | 各 1,092,009 行；说话人 5,994 |
| `raw/musan/...` 与 `raw/rirs/simulated_rirs` | 齐 | 是软链接，分别落到共享盘的 musan 与 RIRS 目录 |
| `vox1-test/` | 齐 | 试验列表 37,611 行；按列表链接了用到的 4,708 个 wav，逐条核对无缺失 |
| `vox2_vb2_cnc2_v0/mix_join/` | 齐 | 主训练清单 **11,521,178** 句，两张表逐行核对 0 处不一致 |
| `vox2_vb2_cnc2_v0/mix_join_lm/` | 齐 | 大间隔微调清单 **5,021,010** 句（按最短 5 秒过滤后拼接），说话人是主训练那份的子集 |

因此：

- **只练 VoxCeleb2：数据表、增强目录与路径都已就绪，本机可直接启动。**
- **混合线：数据与清单都已就绪，启动前不用再准备数据。** 但“数据就绪”不等于“现在可以启动”：三语料 B6 主训练已经跑完 120 轮（见 9.1），再跑就是新实验、新输出；大间隔微调只能走冻结包 + 合同（见 9.2），改个配置直接跑不行。

更细的检查表见 `checkpoints/reports/05-训练数据检查.md` 与 `docs/07-本机数据与开训检查.md`。几处说法冲突时，以重新计过数的结果为准，并把这几处同时改掉。

---

## 7. 评测报告在哪里

对**官方发布的预训练权重**做过复现评测，中文说明在：

```text
checkpoints/reports/
```

该目录的 `README.md` 写了建议阅读顺序；原始汇总表是 `checkpoints/` 下的若干 `.csv` 文件。

写评测相关文字时：

- 说清测了哪些试验协议（例如 VoxCeleb1-O / E / H）。
- 等错误率用百分点；和官方表比较时写「我们的数减去官方的数」。
- 不用「对齐得很好」代替具体数字：读者要看的是差了几个百分点。

---

## 8. 改代码时注意什么

1. 训练配方保持用 `dataset_dir` 读文件夹，不回退到 LMDB：改回去，配方就和上面「数据增强直接读原始音频」那节的检查结论对不上，下一个读代码的人也会以为还有两套读法。
2. 需要「以哪份数据为准」时，写绝对路径并说明理由；另起一套路径昵称，下一个人对不上它指哪个目录。
3. 大文件、日志、权重、整库音频不进 git，报告里写路径和计数即可。
4. 改训练配置后核对三件事：增强目录是否存在、`data.list` 行数是否合理、验证列表路径是否可读。
5. 给学生看的文档若要更新，同步遵守本文第 1 节；`docs/` 与 `checkpoints/reports/` 里过时的说法尽量一起改。
6. 自动助手开多个并行任务改同一批文件时，合并后再检查一遍：有的文件可能被改回旧 LMDB 写法。

代码阅读入口（中文）：

- `docs/03-code-walkthrough.md`：文件和概念对应关系
- `docs/04-训练流程.md`：数据如何进模型、损失怎么算
- `docs/05-recipes-and-repro.md`：如何准备数据与启动训练

---

## 9. 阶段现状与执行门（2026-11-09 核对，动手前先读）

第 3—6 节讲“数据在哪、配方怎么写”，这一节讲“现在到哪一步了、再往前走要验过什么”。凡“当前”“运行中”“尚未完成”“可以直接启动”这类状态说法，以本节为准；带日期的历史观测仍按原时刻理解，不改写历史。

本节的状态是 2026-11-09 00:20（北京）的快照。训练和标注在云端跑，开发机只读共享盘，两边不能用一台的进程或 GPU 代表另一台。标注进度每小时约增加 48,600 源（约 0.42 个百分点），所以本节数字隔一两小时就对不上，属正常；动手前用本节给的命令取新值，任务进度看 [标注看板](docs/ANNOTATION-PROGRESS.md)，HEAD、权限位、测试摘要、产物 sha 看 [仓库状态](docs/STATUS.md)。

下面说的“门”，是启动、续跑或提交前要逐条验过的检查项。门没过就先停下来查：把判据放宽不算过门，改完代码也不等于门过了。

### 9.1 三个阶段各自到哪了

| 阶段 | 现在的状态 | 写报告时不能混的两件事 |
|------|------------|------------------------|
| 主训练（PTN，三语料 B6） | **已完成 120/120 轮。** 48→120 段是从 `model_47` 受控接续的：2026-10-26 07:51:27 起（首个真实梯度更新，绝对迭代 705,047），2026-11-03 10:12:02 止（均为北京时间）；八卡已于 11-03 02:12Z 归还 | **完成 120 轮 ≠ 第 48 轮前向 NaN 的根因已证明**：`ptn_success.json` 里 `historical_numeric_root_cause_proven` 仍是 `false`，写成“NaN 已查清”是错的 |
| 大间隔微调（LM） | 受控包与合同已冻结；**2026-11-07 03:05–05:37 UTC 已跑完 1 轮**：13,075 步、`training_exit_code=0`、`vox1-eer-8s`=0.293%。**冷启动**：只载 `model_115.pt` 的模型参数，优化器与缩放器从零开始。平台侧记录完成者是 `t-20261106110454-vd8pq`，本仓库没有这个 ID 的本地留档（运行产物里没有 `task_id` 字段），任务号归属只能到这一层 | 只能按 9.2 的入口启动；回执 `lm_receipt.json` 的 `ok:false` 是门禁只读 `train.log` 的假阴性（`7d03d3c` 已修源码，r3 冻结包未重打）；此前两次失败：`rq2b7` 入口误带 `--execute` 从未运行、`6lcmt` 因包内 `launch.sh` 缺执行位在 1 分 01 秒退出。**六列正式评测已跑完**，结果与口径见 9.1.1；权重体积、sha 与软链指向看 STATUS 第 4 节，不在正文抄 |
| 自动标注 | **2026-11-08 已复工，当前在跑**：任务 `t-20261108144507-mrvsh`（单机八卡、优先级 6、自动重试关闭），运行目录 `.../handoff-v4r2-20261108-v6/run-v6`。到 2026-11-09 00:20（北京）本轮完成 450,974 源 / 1,805 分片，8 个 worker、8 个在途租约，最近一小时 48,652 源；折算总进度 2,068,411 / 11,521,178 = **17.95%**，按这个速率还剩约 8.1 天（数据分布不变时的条件估计）。目录里没有 `failure.json`，也没有 `quarantine.json` | 17.95% 与 11-01 停摆时的 52.64% **不是同一个口径**：52.64% 含 1028-v1 已完成的 4,447,318 源，那批不在本 lineage 内、会被重算，所以新口径从 1,617,437 起算（见 `docs/ANNOTATION-PROGRESS.md` 第 4 节）。两个数不能相减，也不能说进度退回去了。此前 v2/v3 没有本地留档、v4 生产在 worker 初始化或 CUDA 分配阶段超时（`labels` 0 条）、v5 的 `run-v5` 是空目录；v6 是第一个真正写出分片的正式生产。`publish-work/terminal-t-*.json` 里的 `task_id` 错绑旧任务 `t-20261025054739-4kzhp`，不能当 v6 的终态用 |

**体积差的来源**：48—99 轮检查点是 650,777,251 B，100—120 轮是 650,778,662 B，差的 1,411 B 全部来自 zip 条目名 `model_99.pt.tmp`→`model_100.pt.tmp` 多一个字符、乘 1,411 个条目；条目集合与解压后总字节（650,502,933）都相同，模型和代码中途没有改过。

**评测口径**：最后一轮 `vox1-eer-8s` 是 **0.399%**，这个口径只读前 8 秒；第 65 轮 0.845%、第 48 轮 1.749%。官方 0.26% 是整段协议，两者不能直接相减比较，比较时写清“同口径”。

#### 9.1.1 六列正式评测（2026-11-07 18:58–22:52 UTC 实测，2026-11-09 复核）

协议：整段音频、eager float32、L2 归一化、cosine 相似度；CN-Celeb 用多段注册向量平均。单位是百分点，越低越好。

先校准再比差值：官方权重在六列上逐列落回 README，最大偏差 +0.015 pp（O 0.255/0.26、E 0.409/0.41、H 0.790/0.79、B 1.153/1.15、VOICES 3.029/3.03、CN-Celeb 3.605/3.59）。这说明评测链本身没有口径误差。

| 列 | README | 官方实测 | PTN115 | PTN120 | 本实验 LM |
|---|---:|---:|---:|---:|---:|
| Vox1-O | 0.26 | 0.255 | 0.420 | 0.399 | **0.287** |
| Vox1-E | 0.41 | 0.409 | 0.530 | 0.529 | **0.391** |
| Vox1-H | 0.79 | 0.790 | 0.983 | 0.974 | **0.735** |
| Vox1-B | 1.15 | 1.153 | 1.405 | 1.510 | **1.235** |
| VOICES | 3.03 | 3.029 | 4.731 | 4.755 | **3.147** |
| CN-Celeb | 3.59 | 3.605 | 3.717 | 3.768 | **3.378** |

读法：LM 在 E / H / CN-Celeb 三列低于官方，在 O / B / VOICES 三列略高，平均绝对偏差约 0.09 pp。PTN 到 LM 六列全部降低，VOICES 降幅最大（4.755 → 3.147）——这是阶段差，不是 PTN 的缺陷。完整表、逐格命令与解释见 13 号文档第 7 节。

SITW 没测：全库唯一的 `sitw` 目录 3.5 KB，只有 `ACCESS_REQUIRED.txt` 和一次 403 gated 的下载残留，0 个 wav、0 条试验。卡的是数据集授权，不是代码。

8 秒口径单独看，不与 README 相减：官方 0.239、PTN115 0.404、PTN120 0.399、本实验 LM 0.293。这个口径系统性偏高，只能同口径内部比较。

原件在 `/root/codex_artifacts/eval-official-20261108/csv/<权重>__{vox1OEH,ood,vox1O_8s}.csv`（历史路径；新评测产物按顶部《大数据落点硬规则》写 `/root/group-shared/voiceprint/artifacts/redimnet2/eval-official-<日期>/`，这轮原件保留在原处、只读引用）：四份权重（官方、PTN115、PTN120、LM）各三份，共 12 格，每格退出码 0。同目录 `csv/equiv__*.csv` 里的 E=0.28 来自被 `head` 截成 5,000 行的试验列表，不能引用（正确值 0.409），这两份只用来说明“共享嵌入版不改结果”。

### 9.2 LM 的执行门：冻结包 + 合同，两个哈希都不能省

LM 不是写条命令跑 `train.py`，而是先核两个哈希，再交给包内脚本。平台“入口命令”字段（多行）按这个形状写：

```bash
/bin/bash -lc 'set -euo pipefail
export LM_PACKAGE_SHA=<冻结包的 sha256>
export LM_CONTRACT_SHA=<冻结合同的 sha256>
exec /bin/bash <包根>/runtime/lm_launch.sh'
```

本次冻结的四个值（换版本时四处一起更新：只改一处，哈希与包就对不上）：包 `08a34b68ec26515bbb424f258498b2bc8a3bc0451ce1601f99fade79b99aa003`、合同 `bf437dc8fffe69822401c2f3f6b70205fec569a32d0a4d1bba3ec0bbd69230d0`、`resume.yaml` `e5d20447…`、`runtime/lm_launch.sh` `7fc15817…`；包共 208 个成员。

三条硬规则，每条后面都写了违反的后果：

1. 两个哈希要等于已冻结的那一份。换包或换合同就是新版本，重新冻结、重新记哈希；在旧哈希上放行，跑出来的就不是冻结过的那份包，“只改一个文件”也一样。
2. `new_run_root` 要指向不存在的目录，入口拒绝写已有输出。上次失败的目录留着接着用不行：新一次就是新目录，旧目录原样留作失败现场。
3. 平台自动重试关掉。失败要人看日志定因；自动重跑会冲掉现场，也可能在错误状态上继续写。

**首屏自检**：`sha256sum: OK` 出现两次 → `LM_GATE_VERIFIED`（带 `model_115` 哈希与 `clock:13075`）→ `LM_GATE_DELEGATE` → `this gpu 0`；T+2—5 分钟应见 `Epoch: 1` 与 `cold start: model_ckpt loads model parameters only; optimizer/scaler start fresh`。

**见到 `LM_GATE_REFUSED` 或 `resumed optimizer state` 就停**：前者是门没过，后者说明它以为在续训（本次不该续训）。停下来留日志；重提任务只是碰运气，现场还会被覆盖。

### 9.3 自动标注的改动门：动了引擎源就要重新准入

`worker.py` 等 16 个文件属于合同的 `engine_source_files`，`check_formal_bundle()` 对它们做的是**无条件等式比较**。函数在 `tools/open_annotation/native_batch_admission.py`，定位用 `grep -n "def check_formal_bundle" tools/open_annotation/native_batch_admission.py`；行号会随改动漂移，正文不引用行号。由此有下面几条：

- 改 `engine_source_files` 里任何一个文件，都要打新包、用新包重跑 native pilot、拿到新的 admission（准入凭据）。改完直接续跑，或者拿旧 admission 配新包，等式比较过不了。
- `REVIEWABLE_CONTROL` 白名单只有 4 个 CPU 辅助脚本（`native_batch_admission.py`、`native_batch_readback.py`、`native_batch_queue_handoff.py`、`bounded_speed_runtime.py`），而且要引擎源全等之后才轮到它；改了这 4 个文件并不免检。
- pilot 的规模门在 `_full_check_native_selected_pilot()`：production 要 **n ≥ 3000 且 rc == 512**，300 条不满足，小 pilot 不能当“已准入”。2026-11-08 的 pilot 实际跑了 9 步并全部结束：`screen` 三档（b1/b4/b8，各 30 源）、`main` 三档（各 3,000 源）、`recovery` 三档（各 512 源），每步都记 `resource_returned:true`，结果在 `.../pilot-20261108/run-v9/result.json`。这份结果自己写着 `GPU_runtime_verified:false`、`production_switch_allowed:false`：pilot 全过说明的是同一份代码在 pilot 规模上跑得通，正式生产的资格要看它自己的首片和标签。
- 准入凭据是 `.../admission-20261109/selected-admission-attempt-v6.json`（`passed:true`，正式包与 pilot 包的清单 sha 都是 `69f009e2…`）。准入通过不等于能跑：11-07 的 v5 准入在 14:55 `passed:true`，一小时后 v4 生产在 worker 初始化或 CUDA 分配阶段超时、`labels` 0 条。新包仍要自己的 pilot、admission 和首片证据。
- 2026-10-28 那次换生产者时，8 个在途 cohort 改名到了 `.../native-speed-successor-20261028-v1/run-v1/labels/.native_torch_cohorts.attempt1-crashed/`（8 个 cohort 目录、共 101 个组目录，旁边的 `MANIFEST.json` 写着回滚到原名所需命令）。改名是那次续跑的必要条件：`plan.json` 在崩溃前已写盘并绑着旧身份，不改名续跑会撞 `Refused: no partial retry`；当垃圾删掉则失去“崩在哪一组”的现场。2026-11-08 的 v6 用自己的新 run root，没有继承这批 cohort，这份归档只作现场留证。

**`st_dev` 不进内容身份等式。** device 由挂载命名空间决定：同一批侧车在开发节点读到 `57`，在八卡容器里记录为 `2097190`。判断“内容有没有变”用 `inode / bytes / mtime_ns / ctime_ns`（以及已存的 `file_sha256` / `pcm_sha256`）；device 漂移只记录、不拒。开发节点的 `st_dev` 也推不出云端容器的值：要真值只能在目标容器里读，侧车只记录产出侧的 device，使用侧的 device 只落在回执与日志里。

### 9.4 两个代码级执行门，以及一个不是绿灯的标志（LM 包 r3 已加）

1. **`spk2id.json` 要校验**：存在 → sha256 → 非空 → 整数编号 → `len == 119,274` → 读 LM 标签重数 `115,362` → 全覆盖。原因是缺它时 `train.py` 会从“按说话人名字重组分类头”退化成“按索引切前 115,362 行”，形状合法、`strict=False`，**会静默训错而且不报错**——这种错跑完才发现，比直接崩更贵。
2. **`RECIPE` 显式传包内 `resume.yaml`，并断言相等。** 包外的、或被改过的 recipe 都不行。
3. **`ready_for_lm_execution` 恒为 `false`，这是设计，不是没做完。** 它来自 `tools/ptn_resume/three_segment_lm_check.py`（单文件、纯标准库，有 `build` 与 `check` 两个子命令），只是“准备情况”检查器的输出，**不授权执行**。真实数据下 `build` 17/17、`check` 18/18 都 exit 0（2026-11-06 实测），也仍然不等于该跑；这个返回值不改，`false` 也不解释成“坏了”。单独跑 `check` 要带 `--contract`，不带时只回一条缺参错误（2026-11-09 实测），那不是检查结论。

### 9.5 提交任务的通道（2026-11-08 复测）

- **服务器上有官方 CLI，但现在用不了**：`/root/.volc/bin/volc`（v1.2.55）与 `mlp` 都报 `AccessKeyId or SecretAccessKey is empty`（2026-11-08 复测：`volc ml_task get -i t-20261106195855-rq2b7` 与 `volc ml_task list` 都打 `volc ml_task: exit status 255`，但 shell 的 `$?` 不是 255——`get` 实测 1，`list` 经管道实测 0。记录时把整段输出连 `rc` 一起写，只抄一个数字会错）；`~/.volc/credentials` 不存在，只有一个 `access_token`（CLI 不认）。用时先 `export PATH=/root/.volc/bin:$PATH`，但查任务用不上它，也读不到终态——任务的真实终态只能看控制台详情页。
- `ml_task list` **只列非终态任务**，查历史要用 `ml_task get -i <id>`；`list` 里没有不代表任务不存在。
- **当前唯一可用通道是控制台网页**。已验证的流程：任务列表页 → 目标行操作列点 **`复制`** → 打开预填表单（URL 形如 `/customTask/create?Id=<task>`）→ 改任务名称、描述、入口命令 → 点 **`复制任务`** 提交。
- **用 AX 元素动作，不用坐标点击**：`AXPress` 点按钮，`set_value` 填 `AXTextField` / `AXTextArea`。页面顶部“所有产品服务”大菜单展开时会整体遮挡，坐标点击会偏移到相邻行；先看 AX 树确认，或先按 Escape 关掉菜单。
- AX 快照 **60 秒过期**，每次重新采样索引会移位——采样完立刻动作；攒着一起点，会用到已经移位的索引。
- 数字输入框（`AXIncrementor`）**不吃 `set_value`，也不吃合成键入**（会被清成 0）。“最长运行时间”留空平台仍接受；LM 提交时该字段就是空的。
- 任务名以 `zhounan_` 开头（见第 11 节）；提交前把 9.2 / 9.4 的门验一遍，提交后回读任务 ID、优先级 6、单机 8×A800、自动重试关闭、两处挂载是否与预期一致。

### 9.6 验收纪律：三条最容易误判的等式

1. **元数据稳定 ≠ 可恢复。** 检查点文件 stat 前后一致、SHA 对得上，只说明这个文件没变。能不能续训要看四键齐不齐（`model` / `epoch` / `optimizer` / `scaler`）、张量数与元素数、动量、缩放器标量、学习率是否与 `val_log.csv` 逐位一致。2026-11-06 的 CPU 恢复演练结果：823 个张量 / 81,422,833 个元素（含 78 个 int64）、584 份动量、`scaler=16384.0`、1,329 个浮点张量 0 非有限、体积闭合 0 字节误差。即使这些都过，也只能写“能加载”，不能写“能续训”——检查点里没有 per-rank 随机状态和数据加载游标。
2. **检查通过 ≠ 真执行。** 两个真踩过的坑：
   - `torch.cuda.amp.GradScaler` 在没有 GPU 时**自动进入 Disabled**：`load_state_dict` 变成空操作、`get_scale()` 恒返回 `1.0`，脚本不报错、日志看起来“正常”。所以在 CPU 上验缩放器要用 `torch.amp.GradScaler("cpu")`，否则验的是一个壳，结论无效。
   - `read_bytes=0` **不等于没读**：页缓存加 mmap 的读取不计入 `rchar`。在用 IO 计数证明“没读文件 / 没读原音”之前，先确认读法；读法不同，计数当不了反证。
3. **CPU 通过 ≠ GPU 通过；自动候选 ≠ 可用标签。** 训练、标注、评测三条线分别验收（第 11 节同款规则）。mock、夹具、合成数据上跑通，只能写“CPU 工程验证通过”，不能写成“新版本已通过”。

---

## 10. 常见缩写对照（正文里先写中文）

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

模型结构里的专有名词（如 ASTP、ReDim）在 `docs/06-glossary-faq.md` 有表；写给学生看时，先用白话解释它做什么，再挂缩写。

---

## 11. 给自动助手的额外约束（学生可以当「仓库规矩」读）

- 回答本仓库问题时，默认用第 1 节的文字习惯，不给学生用工具圈黑话。
- 涉及数据是否齐全时，先查路径和行数/文件数再下判断；只复述旧报告，会把记录当时的状态当成现在。
- 状态类事实先查再写：HEAD、权限位、pytest 摘要、产物 sha、任务进度这类数字写进正文就会过期。查 `docs/STATUS.md`（`python3 tools/gen_status.py` 生成，`--check` 判过期、`--lint` 查正文硬编码）与 `docs/ANNOTATION-PROGRESS.md`；指向代码位置用函数名加 `grep`，不用“文件:行号”，改一行就全错。
- 用户明确禁止过某类操作时（例如「不要转成 LMDB」），后续改动不再加回去：加回去等于覆盖用户已经排除的方案，也和数据增强那节的检查口径冲突。
- 没要求提交 git 时，不做 `git commit`，也不推远程：提交历史里混进没被要求过的改动，事后分不清哪次是任务要的。
- 本仓库自己的新增提交使用作者与提交者 `nanless <francis7999@outlook.com>`，通过仓库本地 Git 配置设置，不改其他仓库的全局身份。用户要求提交时，提交标题与正文用详细中文，说明问题、修改、验证结果和尚未完成的工作。
- 用户于2026年10月27日明确要求所有任务名以 `zhounan_` 开头。生成或填写火山平台任务名称时按这个前缀来；名称前缀不替代优先级6、唯一任务查询和实际运行验收。已有任务若支持名称编辑，可在保留任务ID和运行配置的情况下原地添加前缀。报告保留提交时的历史名称并记录后续改名；平台生成的 task ID、worker ID、进程名或冻结文件路径不加这个前缀。
- 本项目持续维护的代码、脚本、配置示例、测试和说明进入本仓库：实验目录只保存只读证据和部署副本，现行实现只留在仓库里，这样部署记录才对得上 Git 提交。部署时记录 Git 提交、文件摘要、配置与外部输入；修改源码后另建版本，改旧冻结清单冒充同版本会让人以为跑的是旧代码。
- 项目技能的维护源是 `.agents/skills/redimnet2-reproduction-ops/`。已安装技能只作为同步副本：先改仓库源、检查引用，再由维护者同步安装并保留同步摘要。真实业务样本名单、标注原件、完整日志、模型或凭据不进公开技能。
- 新产物按顶部《大数据落点硬规则》落盘：实验产物进 `experiments/redimnet2/`，工具产物、准入材料、账本与评测进 `artifacts/redimnet2/`，开源标注侧车进 `open_annotation/redimnet2-silver-<日期>/`；`/root` 只放代码仓库和临时日志，旧根里的数据只读引用并标“历史路径”。
- 训练、标注与评测分别记录验收。CPU 合成测试通过不能写成新仓库版本已通过 GPU；自动候选边界不能自动转成训练或正式评测的可用标签。维护代码入库也不要求重启、迁移或改写正在运行的任务。
- 需要写长篇中文说明时，写进仓库文档（如 `docs/`、`checkpoints/reports/`、本文）；只在对话里丢一堆口号式结论，过几天就没人找得到依据。

---

## 12. 相关文件索引

| 路径 | 内容 |
|------|------|
| `AGENTS.md` | 本文：文字习惯 + 训练/数据事实 + 阶段现状与执行门（第 9 节） |
| `docs/16-实验维护与验收边界.md` | 三条实验线的维护入口、恢复与发布的验收边界（第 9 节的详细版） |
| `docs/17-脚本与入口总表.md` | 全部入口的真实参数、输入输出和风险 |
| `docs/18-文档索引与状态.md` | 防漂规矩、逐篇可信度、不能照做的说法 |
| `docs/19-今日工作与后续入口.md` | 2026-11-08/09 做了什么、现在什么在跑、下周盯什么、出问题先看哪里 |
| `docs/STATUS.md` | 会漂的状态：HEAD、权限位、pytest 摘要、产物 sha（生成器见 `tools/gen_status.py`，不手改） |
| `docs/ANNOTATION-PROGRESS.md` | 标注本轮完成量、折算总进度、速率与预计（`tools/annotation_progress.py` 生成，不手改） |
| `docs/README.md` | 中文导读总入口 |
| `docs/01-background.md` … `06-glossary-faq.md` | 背景、结构、代码、训练、复现、术语 |
| `docs/07-本机数据与开训检查.md` | 本机数据在哪、`/data` 路径问题怎么办、开训前逐条检查 |
| `checkpoints/reports/README.md` | 官方权重复现评测入口 |
| `checkpoints/reports/05-训练数据检查.md` | 训练语料核对表 |
| `scripts/eval_vox1_checkpoints.py` | 用官方权重测 VoxCeleb1-O/E/H 的等错误率 |
| `scripts/eval_ood_checkpoints.py` | 用官方权重测域外试验（Vox1-B、VOICES、CN-Celeb） |
| `applications/speaker_rec/examples/vox2/README.md` | 只练 VoxCeleb2 的操作说明 |
| `applications/speaker_rec/examples/vb2_vox2_cnc2/README.md` | 混合数据线的操作说明 |

若发现文档里又出现第 1.2 节列的说法，改成白话，并在同一批修改里改干净。
