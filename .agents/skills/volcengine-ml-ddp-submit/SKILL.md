---
name: volcengine-ml-ddp-submit
description: 检查、准备、提交、修复和诊断火山引擎机器学习平台上的 PyTorch DDP 训练与批量推理任务，覆盖通用平台表单、PSE、se_train 非标注SFT的VAD批量mask推理、MS-SWIFT Qwen Omni、TLive-Omni、checkpoint/resume、共享存储、远程Git、排队、日志与有界重试。标注VAD SFT的配置派生、OOM后继和验收使用 volcengine-vad-sft-task。
---

# 火山引擎多机多卡训练与批量推理运维

按“代码与配置检查 → 平台任务准备/提交 → 运行证据诊断”三步做，每一步都留下可复现的证据。通用 PyTorch DDP 自定义任务和 `se_train` 的 PSE 专项流程都走这套流程。

本技能的维护源在服务器仓库 `dev_a800_2gpus:/root/code/github_repos/redimnet2-fork` 的 `.agents/skills/volcengine-ml-ddp-submit/`（origin `nanless/redimnet2-fork`，进目录先用 `git rev-parse --show-toplevel` 与 `git remote -v` 核对）。本机没有该仓库的工作副本，所以 `~/.codex/skills/volcengine-ml-ddp-submit` 这份安装态副本就是本机唯一可改的一处：改动直接在这里做，动手前先 `tar czf` 打包整个目录以便回滚，改完需要回仓库时再同步进服务器那份检出。当前安装副本 17 个文件（`find ~/.codex/skills/volcengine-ml-ddp-submit -type f | wc -l` 实测：`SKILL.md`、`agents/openai.yaml`、12 份 `references/*.md`、3 个 `scripts/*.py`），两边是否一致用 `diff -rq` 现查。项目特定的训练和数据合同归对应项目技能，本次状态或授权不外推到其他项目。

**GitHub 写入限制（2026年10月28日起）**：本机 Mac 对 GitHub 只读，可以浏览、下载、clone、fetch、pull；上传文件、代码、文档、数据、提交或分支都不行，git push、网页/API/gh 写入和自动同步同样不走。落地方式是保留本机 `disabled://github-push-blocked-by-policy`，改 remote、URL、端口，或走代理、隧道，效果上都是绕过这条限制。用户明确允许把任务所需仓库同步到 `dev_a800_2gpus`，由服务器自身仓库和进程实际连接 GitHub 提交／推送；本机可以 SSH 发起远程命令，本机 push 透传仍不行。已删除的 `nanless/redimnet2-fork:codex/repository-audit-20261025` 分支不恢复，也不重新审计 26 个本机禁推仓库。GitLab 操作按其独立授权，GitLab 端口故障处理不构成本机 GitHub 禁写的例外。

## 先按任务加载资料

只读当前任务需要的资料。

| 场景 | 必读资料或工具 |
| --- | --- |
| 登录、开发机检查、镜像、资源、队列、挂载、表单、提交、排队诊断 | [平台 DDP 任务工作流](references/platform-ddp-workflow.md) |
| 在控制台**复制既有任务**提交（当前唯一可用通道）、复制来源任务、入口短脚本与 AX 读回、会话/锁屏/网络中断、首屏自检 | [控制台复制任务提交](references/console-clone-submission.md) |
| 排队位次波动、八卡同节点、失败三步定位（日志→落盘→侧车）、交接 5 分钟窗口与 waiter 1800 秒 | [排队、失败定位与交接窗口](references/queue-and-failure-triage.md) |
| 平台事实多久会变、改值时到哪重核、每类事实由哪份文档维护 | [平台事实的时效与复测方法](references/platform-facts-and-refresh.md) |
| PSE TOML 继承、完整预加载、checkpoint 兼容性 | [PSE 配置与预加载](references/pse-preload-and-config.md) |
| PSE 2×8 launcher、平台变量、专项挂载、远程 Git | [PSE DDP 与仓库操作](references/pse-ddp-and-repository.md)；提交任务时还要读平台工作流 |
| PSE 首批 `sv_ce`、ArcFace、重建指标、AMP/NaN 判断 | [PSE 启动日志诊断](references/pse-log-diagnosis.md) |
| PSE 每步速度、单轮时长、完成时间估算 | [PSE 训练时长分析](references/pse-log-duration.md)，并运行 `scripts/analyze_pse_log.py` |
| VAD 2×8 配置变更、全量双伪标签基模、同步 WSOLA 外置组件、无梯度累积提交、运行中任务切换、epoch ETA | [VAD 配置、重启与时长分析](references/vad-config-restart-and-duration.md)；全量数据/标签/速度扰动合同先读 `$se-train-project` 的 `Full-audio pseudo-mask VAD base training`；ETA 运行 `scripts/analyze_vad_log.py` |
| `se_train` 千万级 VAD mask 批量推理、精确分片、三件套验收、随机图与配对音频 | 先用 `$se-train-project` 并读其 `Bulk VAD inference and artifact delivery`；提交/监控时同时读[平台 DDP 任务工作流](references/platform-ddp-workflow.md) |
| 复制参考 VAD 任务、创建标注 SFT、OOM 后继配置及专项验收 | 改用 `$volcengine-vad-sft-task`；本 skill 只补通用平台表单/认证/日志机制 |
| MS-SWIFT/Qwen Omni Full SFT、ZeRO、model-only 续训、累计训练进度 | [MS-SWIFT Qwen Omni DDP](references/ms-swift-qwen-omni-ddp.md) |
| TLive-Omni 纯音频 Full SFT、独立环境、共享部署快照、2×8 launcher | [TLive-Omni DDP](references/tlive-omni-ddp.md)；模型和数据合同同时使用 `$tlive-omni-speaker-count-sft` |
| ReDimNetV2 三语料 B6 PTN→LM、开源文字字界标注、训练巡检、检查点和评测 | 用 `$redimnet2-reproduction-ops` 确定实验合同与证据，进度查询读其 `references/task-status-and-readonly-progress.md`；平台登录、表单、资源、挂载与状态仍读[平台 DDP 任务工作流](references/platform-ddp-workflow.md) |
| MMS-LLaMA 的 LRS3-only 或 LRS3+Vox2 AVSR 训练 | 用 `$avsr-reproduction-ops` 对应的 `references/mms-lrs3-training.md` 或 `references/mms-lrs3-vox2-training.md` 确定数据、模型和 batch 合同；表单、挂载、队列及状态仍读[平台 DDP 任务工作流](references/platform-ddp-workflow.md) |

混合任务读所有对应资料。要提交平台任务时，平台工作流是必读的：表单字段、挂载方案和提交前闸门都写在那里。

## 提交通道的现实约束（2026-11-09 复测）

- **服务器上有官方 CLI，但调不通**：`/root/.volc/bin/volc`（自报 1.2.55，提示最新 1.2.57）与同目录 `mlp`（70,664,479 B）。两者都不在 `PATH`，不补 `PATH` 时 `volc ml_task` 先报 `exec: "mlp": executable file not found in $PATH`（`volc` 只是外壳，真正干活的是 `mlp`）。
- **缺 AK/SK**：`~/.volc/credentials` 不存在，该目录下只有 `access_token`（68 B），CLI 不认这种凭据。实测 `volc ml_task get -i <task_id>`、`volc ml_task list`、`mlp job list` 都返回 `AccessKeyId or SecretAccessKey is empty, please check $HOME/.volc/credentials or config it by using volc configure`，外层报 `exit status 255`。读到 `access_token` 不等于能调用；复制它换不出可用的 AK/SK，只是多留一份凭据副本。
- **因此：控制台“复制任务”是当前唯一可用通道**。逐字段手册见 [控制台复制任务提交](references/console-clone-submission.md)，当前复制来源是 `t-20261026074833-ctjsk`。
- **入口写成一行短脚本**：网页编辑器对长命令会丢首字符（用户观测，2026-11-08 至 11-09 多次复现），所以入口是 `/bin/bash <脚本绝对路径>`，环境变量和执行细节放在共享盘的脚本里。提交前用 AX 读回编辑器的完整值逐字比对，提交后从详情页再读一次。
- **`ml_task list` 只列非终态任务**（厂商自述）：任务一旦完成/失败/停止就不在 `list` 里，查历史要用 `get -i <task_id>`。这条语义目前无法实测，因为 `list` 本身返回 255。`list` 里看不到，推不出“任务不存在”。
- **控制台会话会过期**：连续操作 3–4 小时后需要重新认证（用户观测，未独立复测时长）。重新走当前账号的认证路径，不退出账号——退出会把其他登录一起清掉。
- **锁屏时 Computer Use 不可用**：macOS 锁屏后取不到 screenshot 与 AX 树，报 `CGSSessionScreenIsLocked`。这时解锁后重新采样，不在锁屏状态下反复点或改用坐标点击。
- **网络中断先用内网 ping 判**：`ping 10.43.6.199` 100% 丢包而公网正常，是 VPN 断了，不是平台故障（2026-11-09 实测当前 3/3 通、平均 25.4 ms）。顺序是先 ping 内网、再 curl 公网、最后才刷新控制台。
- 以上是**通道现状**，不改变证据要求：平台任务状态、详情页字段与持久日志仍是三种独立证据（见“统一工作流”第 9、11 条）。CLI 不可用不等于核验标准可以降。

## 操作原则

- 复用用户现有的 Edge/Chrome 登录状态；为切换登录方式而退出账号，会把其他正在用的登录一起清掉。`企业联邦登录 → 钉钉登录` 只是当前组织曾验证的认证 profile，仅在实时页面提供且用户属于对应租户时使用，否则按当前账号的实际认证路径走。只在必须由用户完成授权或验证码时暂停。
- 区分只读检查、可逆编辑和外部写入：查看任务、日志、表单和存储不改外部状态；提交、复制、重试、停止、删除任务以及 Git commit/push 都会。依据当前会话的用户授权执行必要动作，某一项目的历史授权不外扩到其他任务或无关操作。
- 提交前从实时控制台、代码仓库或用户请求核实每个值。镜像标签、资源规格、队列、可用区、挂载路径、配置、commit 或任务 ID 都不凭印象填——这些值错了，任务要么起不来，要么跑在错的环境上。
- 用户规定：所有火山引擎实验任务提交时优先级设为 `6`，提交前核对表单选中值，提交后核对详情页实际值。平台设不了 `6` 时，停在提交前说明阻碍，不用任务名里的 `p6` 代替——名字里的标记平台不读。
- 用户于2026年10月27日明确要求所有任务名以 `zhounan_` 开头。生成或填写火山平台任务名称时按这个前缀来；名称前缀不替代优先级6、唯一任务查询和实际运行验收。已有任务若支持名称编辑，可在保留任务ID和运行配置的情况下原地添加前缀。报告保留提交时的历史名称并记录后续改名；平台生成的 task ID、worker ID、进程名或冻结文件路径不加这个前缀。
- 创建后继实验时先提交新任务并取得唯一 ID、核对其详情，再处理旧任务：顺序反过来，新任务提交失败就只剩一个已停的旧任务。只有用户已授权停止精确旧任务时才执行停止；用户要求旧任务继续运行或排队时保留。无论旧任务状态如何，都不先停旧再提交新的。新旧任务不并发写同一输出——两个写入者会互相覆盖；共享输出时，先建立可验证的写入互斥或启动闸门。
- 已运行或已用于实验的 TOML 默认当不可变处理；需要调整实验时新建后继配置，除非用户明确授权修改历史配置。
- 区分开发机 smoke 与火山引擎训练容器。开发机可以验证当前模型的真实前反向与 checkpoint 恢复；它证明不了云端镜像、挂载或完整拓扑已经跑过。用户指定云端只提交正式长训时，不另建云端 smoke；正式任务启动后从日志核对容器和 rank。
- `requires_grad=True` 证明不了 checkpoint 已加载：参数默认就是要梯度的，这个标志说明不了权重来自哪里。要同时检查最终生效配置、checkpoint 键/形状和加载摘要。
- 一次只提交一个已核验任务。任务排队时查一次容量或队列信息并报告；重复提交只会多出几个抢同一份资源的副本。排队位次是瞬时值，2026-11-09 观测在 112/120 与 107/120 之间来回变；八卡任务要求同一节点空出 8 张卡，两台各空 4 张拼不起来。位次回退不是任务被重置，详见[排队与失败定位](references/queue-and-failure-triage.md)。
- 先分清瞬态基础设施失败和确定性的代码/配置失败：前者重试可能过，后者只会把同一个错误再跑一遍。遇到可重复的 OOM、配置不兼容或路径错误，先识别自动重试链并准备不可变的后继配置；先提交并核对新任务，再按用户授权处理旧任务及错误重试。用户要求保留旧任务时不停止，也不重复提交同一个错误。
- 平台任务状态、持久训练日志状态和共享存储 checkpoint 状态是三种独立证据；只凭其中一种就说任务在运行、已停止或保存完成，会把排队、日志滞后和存储延迟当成结论。
- 长训巡检同时核对当前进程与后续阶段依赖：从实际入口和环境变量确认未来会重新打开的源码校验清单、smoke 凭证、数据审计及参考权重路径，并在实际容器里验证可读性和身份。活跃 PTN 有进度，不证明 LM/导出/评测能接续；不持有文件描述符，也不证明目录可安全改名。详情见[平台工作流的长训依赖核查](references/platform-ddp-workflow.md#长训的未来阶段依赖)。
- 状态检查先刷新详情并核对完整任务 ID；页面上的运行时计数可能过期。停止任务读取时间线原因及操作账号，`ByUser` 不当成训练崩溃；查看进度不构成重新启动授权。浏览器会话访问故障时，现有登录态的回退方法见平台工作流。
- `save_only_model` 产物不等同于完整 Trainer 状态。缺少 optimizer/scheduler 分片时，只能按权重续训的含义处理，并说明学习率、步数和数据游标是否重置。

## 统一工作流

1. **确认范围与授权。** 判断用户要的是检查、修复、准备表单、提交、重提、运行诊断还是训练 ETA；据此加载上表中的资料，并明确哪些外部写入已经获得授权。
2. **检查真实代码源。** 在目标开发机读取分支、commit、工作区状态、launcher、配置及完整继承链；搜索所有数据、输出、checkpoint 和挂载路径。远端拉取代码时确认所需 commit 已推送；共享 vePFS 直接提供代码时，核对任务实际可见的 checkout 和关键文件身份。无关项目不加 `asset_contract` 或通用 preflight；已有的数据摘要按各自的实验合同核验。
3. **审计有效配置。** 对 PSE 用仓库自己的 `load_config` 读取合并结果，打印关键 `[meta]`、`[distributed]`、梯度累积和保存间隔。对 VAD 同时核对 train/valid/test DataLoader、模型 chunk、batch、LR 和拓扑。只读子 TOML 或只改一处分区，看到的都不是最终生效值。
4. **验证 checkpoint 和启动契约。** 核对 checkpoint 可读性、索引引用的全部分片、文件写入稳定性、模型键和形状、speaker vocabulary、DDP rank/world-size 变量、所有节点可见的代码/数据/输出路径，以及选中 launcher 实际使用的 resume/lease 行为。没有 lease 的专用 launcher 不补一个。运行与目标拓扑相称的聚焦测试、`git diff --check` 和 launcher 的 `bash -n`；单节点任务不需要两个 node-rank dry-run。云端没验证过的镜像/挂载/rank，提交后继续核，开发机 smoke 代替不了。
5. **设计非重叠挂载。** 先做“开发机源 → 任务容器目标 → 读写模式 → 代码用途”映射。容器目标不能互相包含（平台会拒绝嵌套挂载）。按现有路径从平台资料里选一种中性挂载方案，只创建必要、可验证、不会覆盖非空目录的兼容软链接。输出根按《大数据落点硬规则》放共享组盘（`/root/group-shared/voiceprint/experiments/...` 或 `.../artifacts/...`）：该挂载在表单里是只读时先改成读写，不把输出降级到 `/root` 本地盘。
6. **审计完整任务表单。** 核对区域、私有镜像、代码来源、入口命令、环境变量、PyTorch DDP 框架、worker 数、单机 GPU、队列/AZ、优先级 `6`、运行时、重试、挂载、任务名和描述。仓库由共享 PFS 提供时，不再同时使用陈旧的网页上传代码快照。
7. **执行提交前闸门。** 按平台资料中的清单逐项复核。入口、配置、commit、镜像、必需挂载或权限有一项无法证实，就停在提交按钮之前并说明缺口。入口这一项还要看两件事：它是一行短脚本而不是长命令；编辑器的 `AXValue` 读回与预期逐字相同（丢失或追加都能在这一步发现）。
8. **按授权执行外部写入。** 代码改动只提交确切的新文件或修改；任务只提交一次。运行中任务的配置变更生成可追溯的后继版本；任务从远端 Git 拉代码就先提交并推送，从共享 vePFS 读代码就核对容器将读到的文件身份。后继任务先提交优先级 `6` 的新任务并核对其 ID/详情，只有已有授权明确覆盖停止操作时才停精确目标旧任务。用户指定保留旧任务时不停止；不擅自克隆、重试或删除。
9. **提交后核验。** 从任务列表和详情页记录完整任务名、ID、状态、框架、队列、节点/GPU、镜像、入口、配置和 commit。`排队中` 只证明创建成功；`部署中`、`运行中` 只说明平台状态。5 分钟内做一次首屏自检，抓“启动即错”。判断训练链路启动并保存，要看日志确认有效配置、拓扑、真实 batch 和有限 loss/grad，再从共享存储确认 checkpoint。走队列交接的任务，创建后 5 分钟内要在开发机上起 `auto_publish`；`auto_publish` 轮询 360×5 秒，waiter 的 `wait_seconds` 被断言在 1800 秒以内，错过窗口这一代任务只能作废。
10. **控制失败传播。** 确定性失败时读取父任务的重试记录，记录所有 child task ID，准备独立输出和修复配置。按上述顺序先提交并核对唯一后继任务；用户已授权清理旧重试任务时，再停止精确旧任务并确认剩余错误重试为 0。用户指定保留旧任务时不执行停止。新任务提交受阻时，保留旧任务状态并报告，不通过先停旧任务改变顺序。失败记录保留；除非用户明确要求，不删任务历史。
11. **按证据诊断。** 任务显示失败时先按平台日志最后几行 → `run-*` 目录落盘情况 → 侧车是否 0 的顺序走一遍，三步各留一条证据再定性（见[排队与失败定位](references/queue-and-failure-triage.md)）。PSE 启动异常先核对有效 skip 配置和加载摘要，再解释 `sv_ce`、重建、ArcFace、AMP 或 NaN。VAD OOM 先读每 rank 的真实分配/占用，再调 micro-batch 和 accumulation。估算时长读 rank-0 持久日志；VAD 的稀疏 batch 日志用 50/100-batch 窗口，不用单个日志间隔或平台总运行时外推。

## VAD 批量 mask 推理补充

- 这是推理与数据交付任务，不套用训练 epoch、optimizer、accumulation 或 checkpoint-resume 语义。核对推理 TOML、固定 checkpoint、EMA/precision、阈值、输入树、输出树和逐 rank manifest。
- 云端提交前，在目标代码版本上跑过真实单卡、多卡不重复分片和 `missing=0` 二次运行；有一项没完整通过，就停在提交按钮之前——分片有没有重复，只有跑过才知道。
- 平台任务使用精确 clean Git worktree、明确配置与 checkpoint。不自行加入内容哈希或 `asset_contract`。提交后从日志确认所有 rank 加载各自唯一 shard，并从共享存储确认三类文件同步增长。
- 任务完成分“推理完成”和“逐文件终验通过”两个阶段。后者以全量 validation attestation 为准；平台显示运行完成、或只看文件存在，都不够。
- GPU 显存占用低不等于任务异常。小模型纯推理常受音频解码、共享存储小文件读取和每样本多文件写入限制。调大 batch 前，先用真实时长分布做吞吐、峰值显存、padding 浪费和输出一致性测试，不直接改在跑的任务。
- 长尾录音要单独审计：一条几十分钟音频可把整批 padding 放大数十倍，无下采样的 noncausal 全局 attention 还会随帧长近似平方增长。普通批次剔除长样本并按真实长度重新裁剪；超长样本走经过 full-vs-stitched 数值对照的单文件回拼路径，不静默跳过，也不产出 `_segNNNN` 替代品。异常路径不进 NCCL barrier，否则 watchdog SIGABRT 会掩盖原始 OOM。
- 失败后的 post-scan 已经原子生成下一轮精确 missing shards，直接复用；在下一 pass 顶部再扫一遍全量，等于把千万级目录走两遍。平台日志至少打印扫描开始/完成，把共享存储元数据阶段和 GPU 卡死分开。
- 同一输出树的后继恢复任务可以显式复用上次已发布的 `mode=missing` shards，省掉再次遍历千万文件，但先匹配 base/output/manifest、输入与输出合同、limit、rank/world-size，逐个解析 shard，并核对路径、计数及 exact union/disjoint ownership。旧清单只能当当前 missing 的超集用：DataLoader 在解码前按 NPY+JSON+RTTM 三件套过滤，推理器继续逐样本防覆盖；任务结束后仍做新鲜 post-scan 和全量终验。不能复用的清单：中断扫描的隐藏临时文件、新输出树、`mode=all`、合同/拓扑不一致、元数据损坏的。
- 从慢扫描任务切换到快速恢复任务时，先提交一个锁定明确代码版本且优先级为 `6` 的后继任务，取得 ID 并核对详情；仅在已有停止授权时停旧任务并确认其写入者已释放。复用同一输出树时，新入口要在旧写入者退出前保持只读等待，或用其他可验证的互斥机制阻止并发写入；做不到就停在提交前解决。运行摘要和任务描述记录父任务、复用开关与 manifest 验证证据。GPU 碎片导致排队时，不靠重复提交绕开队列。
- 可视化和配对音频写到生产输出树之外，否则会被 no-orphan 终验判为污染；复制音频后解码两端，核对大小、采样率、采样点数和波形一致性，不移动源文件。

## PSE 完整预加载的最低要求

对架构和 speaker vocabulary 兼容的 PSE 权重做完整预加载，应在后继 TOML 中显式设置：

```toml
preload_embedding_skip_front = false
preloaded_skip_prefixes = []
preloaded_embedding_skip_prefixes = []
preloaded_extraction_skip_prefixes = []
```

其中 `preloaded_embedding_skip_prefixes` 优先于通用 fallback，是 `sv_arc_margin` 被静默随机初始化的常见原因。最终还要证明 checkpoint 含有 `embedding_model["sv_arc_margin.weight"]`，且形状、说话人数量和词表映射一致。

## 失败处理

- CloudFS 因服务关联角色或存储权限预检失败时，账号级设置不擅自开：改的是整个账号的权限面。先证明训练是否使用该挂载；确认未使用才省略，并记录差异。
- 平台拒绝嵌套挂载时，改用中性目标和安全软链接；训练必需的共享文件系统不删。
- GPU 池不足时，保留唯一任务处于排队状态并报告容量约束；不重复提交。
- 同配置、同阶段或同类输入上重复出现的 CUDA OOM，按确定性资源/配置失败止损；等平台按原配置把自动重试跑完，只是把同一个 OOM 重放几遍。首次 OOM 先记录失败阶段、输入长度、其他进程、allocated/reserved/free/requested 和碎片化证据。需要后继实验时保留原 TOML 和失败任务证据，新建独立 `root_dir`，并按任务专项合同检查 micro-batch、累积、scheduler 和尾梯度语义。
- 保持 global batch 时，同时核对 scheduler 的 step 口径、checkpoint 保存间隔和每 epoch optimizer-step 数；只改 `batch_size` 就宣称训练等价，算不出同一个训练量。
- 已获授权的服务器推送或独立获授权的 GitLab 操作因仓库 SSH 端口超时时，可以先验证同一 Git 服务的 22 端口，再使用一次性 `GIT_SSH_COMMAND`；本机 GitHub 禁写不是连接故障，换端口、URL、remote、代理或隧道都不解决它。未知主机或凭据问题也不靠跳过校验处理。
- 高首批 `sv_ce` 不是整模型损坏的充分证据；先区分随机 ArcFace 头、词表不兼容、原始与加权损失以及 warmup。
- 不在已运行进程里通过原地修改 TOML“修复”加载问题；创建后继配置并按授权重启或重提。
- 配置变更导致合同测试只因为旧字面值断言而失败时，不能写成测试全绿；确认它不是功能失败，再按用户授权范围同步测试或如实记录该项陈旧断言。

## 交付要求

对中文用户用中文，先报告结果，再给证据和限制。至少说明：

- 检查或修改的文件、最终有效配置和 checkpoint/加载证据；
- 执行的测试、未执行的真实多机多卡预检及原因；
- commit/push 结果（若获授权执行）；
- 是否提交任务，以及任务名、完整 ID、当前状态和关键资源；
- 若发生失败与重提，列出父任务、自动重试 child、被停止的错误任务、后继任务及各自最终状态；
- 相比开发机环境采用的挂载、入口或 CloudFS 调整；
- 对批量推理说明源总数、每 rank 所有权、三类输出计数、抽查结果、最终全量 attestation，以及可视化/音频交付目录是否与生产树隔离；
- 诊断或 ETA 的数据来源、估算范围与不确定性。

## Artifact locations

2026-11-09 起，服务器上保存的新材料一律落在 `/root/group-shared/voiceprint` 下，`/root` 只放代码仓库与临时日志。仓库 `redimnet2-fork` 的 `AGENTS.md` 顶部《大数据落点硬规则》是同一条规定，含三类落点与自检命令。本技能的提交核验材料（任务表单证据、日志截取、诊断 JSON、ETA 报告）写 `/root/group-shared/voiceprint/artifacts/redimnet2/<用途>/`，提交类材料用 `volcengine-submission/<task_id_or_run_id>/`；训练与推理输出写 `/root/group-shared/voiceprint/experiments/redimnet2/<实验名>/`，具体子目录仍由实际 launcher/TOML 的输出根决定。用户选择“仅保留本机”时，报告与日志快照留在本机任务目录、不上传；这次选择不变成以后所有任务都本地保存的规矩。

不先写 `/root` 再 `mv`：训练容器里 `/root` 是容器本地盘，`/shared/group-shared`（容器内通常另挂成 `/root/group-shared`）是 vePFS，跨挂载点 `mv` 退化为复制，大检查点占双份空间且中途失败会留半份。容器里 group-shared 是只读挂载时，按平台工作流调成读写再提交，不把输出改回 `/root` 本地盘。旧根 `/root/codex_artifacts`、`/root/experiments` 里的既有材料原样保留、只读引用，文档里标“历史路径”。

归拢旧产物或解释主机状态时读 [产物位置与统一根](references/artifact-locations.md)。

## 服务器产物的当前实体位置

> 服务器产物位置与老路径解析：见
> `../codex-server-artifact-storage/references/current-location-and-resolver.md`
> （唯一事实源；本条不再复制具体路径，避免多处漂移。）
