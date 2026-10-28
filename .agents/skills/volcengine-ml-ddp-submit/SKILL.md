---
name: volcengine-ml-ddp-submit
description: 检查、准备、提交、修复和诊断火山引擎机器学习平台上的 PyTorch DDP 训练与批量推理任务，覆盖通用平台表单、PSE、se_train 非标注SFT的VAD批量mask推理、MS-SWIFT Qwen Omni、TLive-Omni、checkpoint/resume、共享存储、远程Git、排队、日志与有界重试。标注VAD SFT的配置派生、OOM后继和验收使用 volcengine-vad-sft-task。
---

# 火山引擎多机多卡训练与批量推理运维

围绕“代码与配置检查 → 平台任务准备/提交 → 运行证据诊断”建立可复现的闭环。既支持通用 PyTorch DDP 自定义任务，也支持 `se_train` 的 PSE 专项流程。

本技能当前维护源为ReDimNet2仓库`.agents/skills/volcengine-ml-ddp-submit/`，安装副本从该目录同步。项目特定训练和数据合同仍归对应项目技能，不把本次状态或授权推广到其他项目。

**2026年10月28日起的用户全局GitHub铁律：本机Mac只能浏览、下载、clone、fetch、pull，任何时候不得向GitHub上传文件、代码、文档、数据、提交或分支；本机git push、网页/API/gh写入和自动同步均禁止。保留本机`disabled://github-push-blocked-by-policy`，不得用remote、URL、端口、代理或隧道绕过。用户明确允许将任务所需仓库同步到`dev_a800_2gpus`，由服务器自身仓库和进程实际连接GitHub提交／推送，本机可SSH发起远程命令；本机push透传仍禁止。已删除的`nanless/redimnet2-fork:codex/repository-audit-20261025`分支不恢复，不重新审计26个本机禁推仓库。GitLab操作按其独立授权，不能把GitLab端口故障处理当成本机GitHub禁写的例外。**

## 先按任务加载资料

只读取当前任务需要的资料：

| 场景 | 必读资料或工具 |
| --- | --- |
| 登录、开发机检查、镜像、资源、队列、挂载、表单、提交、排队诊断 | [平台 DDP 任务工作流](references/platform-ddp-workflow.md) |
| PSE TOML 继承、完整预加载、checkpoint 兼容性 | [PSE 配置与预加载](references/pse-preload-and-config.md) |
| PSE 2×8 launcher、平台变量、专项挂载、远程 Git | [PSE DDP 与仓库操作](references/pse-ddp-and-repository.md)；提交任务时还要读取平台工作流 |
| PSE 首批 `sv_ce`、ArcFace、重建指标、AMP/NaN 判断 | [PSE 启动日志诊断](references/pse-log-diagnosis.md) |
| PSE 每步速度、单轮时长、完成时间估算 | [PSE 训练时长分析](references/pse-log-duration.md)，并运行 `scripts/analyze_pse_log.py` |
| VAD 2×8 配置变更、全量双伪标签基模、同步 WSOLA 外置组件、无梯度累积提交、运行中任务切换、epoch ETA | [VAD 配置、重启与时长分析](references/vad-config-restart-and-duration.md)；全量数据/标签/速度扰动合同先读 `$se-train-project` 的 `Full-audio pseudo-mask VAD base training`；ETA 运行 `scripts/analyze_vad_log.py` |
| `se_train` 千万级 VAD mask 批量推理、精确分片、三件套验收、随机图与配对音频 | 先用 `$se-train-project` 并读取其 `Bulk VAD inference and artifact delivery`，提交/监控时同时读取[平台 DDP 任务工作流](references/platform-ddp-workflow.md) |
| 复制参考 VAD 任务、创建标注 SFT、OOM 后继配置及专项验收 | 改用 `$volcengine-vad-sft-task`；本 skill 只补充通用平台表单/认证/日志机制 |
| MS-SWIFT/Qwen Omni Full SFT、ZeRO、model-only 续训、累计训练进度 | [MS-SWIFT Qwen Omni DDP](references/ms-swift-qwen-omni-ddp.md) |
| TLive-Omni 纯音频 Full SFT、独立环境、共享部署快照、2×8 launcher | [TLive-Omni DDP](references/tlive-omni-ddp.md)；模型和数据合同同时使用 `$tlive-omni-speaker-count-sft` |
| ReDimNetV2 三语料 B6 PTN→LM、开源文字字界标注、训练巡检、检查点和评测 | 使用 `$redimnet2-reproduction-ops` 确定实验合同与证据，进度查询读其`references/task-status-and-readonly-progress.md`；平台登录、表单、资源、挂载与状态仍读[平台 DDP 任务工作流](references/platform-ddp-workflow.md) |
| MMS-LLaMA 的 LRS3-only 或 LRS3+Vox2 AVSR 训练 | 使用 `$avsr-reproduction-ops` 对应的 `references/mms-lrs3-training.md` 或 `references/mms-lrs3-vox2-training.md` 确定数据、模型和 batch 合同；表单、挂载、队列及状态仍读[平台 DDP 任务工作流](references/platform-ddp-workflow.md) |

混合任务应读取所有对应资料。涉及提交平台任务时，必须读取平台工作流。

## 操作原则

- 复用用户现有的 Edge/Chrome 登录状态；不要为了切换登录方式而退出账号。`企业联邦登录 → 钉钉登录` 只是当前组织曾验证的认证 profile，仅当实时页面提供且用户属于对应租户时使用；否则遵循当前账号的实际认证路径。只在必须由用户完成授权或验证码时暂停。
- 区分只读检查、可逆编辑和外部写入。查看任务、日志、表单和存储是只读；提交/复制/重试/停止/删除任务以及 Git commit/push 都会改变外部状态。依据当前会话的用户授权执行必要动作，不把某一项目的历史授权扩展到其他任务或无关操作。
- 提交前从实时控制台、代码仓库或用户请求中核实每个值。不要虚构镜像标签、资源规格、队列、可用区、挂载路径、配置、commit 或任务 ID。
- 用户规定：所有火山引擎实验任务提交时优先级设为 `6`，提交前核对表单选中值，提交后核对详情页实际值。若平台无法设置 `6`，停在提交前说明阻碍，不以任务名中的 `p6` 代替。
- 用户于2026年10月27日明确要求所有任务名以 `zhounan_` 开头。以后生成或填写火山平台任务名称时必须遵守；名称前缀不替代优先级6、唯一任务查询和实际运行验收。已有任务若支持名称编辑，可在保留任务ID和运行配置的情况下原地添加前缀。报告保留提交时的历史名称并记录后续改名；不要给平台生成的 task ID、worker ID、进程名或冻结文件路径强行改前缀。
- 创建后继实验时先提交新任务并取得唯一 ID、核对其详情。只有用户已授权停止精确旧任务时才执行停止；用户要求旧任务继续运行或排队时必须保留。无论旧任务状态如何，都不先停旧再提交新任务。确保新旧任务不会并发写同一输出；若共享输出，先建立可验证的写入互斥或启动闸门。
- 已运行或已用于实验的 TOML 默认不可变；需要调整实验时新建后继配置，除非用户明确授权修改历史配置。
- 区分开发机 smoke 与火山引擎训练容器。开发机可以验证当前模型的真实前反向与 checkpoint 恢复；它不能证明云端镜像、挂载或完整拓扑已运行。若用户指定云端只提交正式长训，不要另建云端 smoke；正式任务启动后再从日志核对容器和 rank。
- 不要用 `requires_grad=True` 证明 checkpoint 已加载。必须同时检查最终生效配置、checkpoint 键/形状和加载摘要。
- 一次只提交一个已核验任务。任务排队时检查一次容量或队列信息并报告，不要通过重复提交制造副本。
- 区分瞬态基础设施失败与确定性代码/配置失败。发现可重复的 OOM、配置不兼容或路径错误后，先识别自动重试链并准备不可变的后继配置；先提交并核对新任务，再依用户授权处理旧任务及错误重试；用户要求保留旧任务时不得停止，也不要重复提交同一错误。
- 将平台任务状态、持久训练日志状态和共享存储 checkpoint 状态作为三种独立证据；不能只凭其中一种宣称任务运行、停止或保存完成。
- 长训巡检同时核对当前进程与后续阶段依赖：从实际入口和环境变量确认未来会重新打开的源码校验清单、smoke 凭证、数据审计及参考权重路径，并在实际容器验证可读性和身份。活跃 PTN 有进度不证明 LM/导出/评测能接续；不持有文件描述符也不证明目录可安全改名。详情见[平台工作流的长训依赖核查](references/platform-ddp-workflow.md#长训的未来阶段依赖)。
- 状态检查先刷新详情并核对完整任务 ID；页面运行时计数可能过期。停止任务读取时间线原因及操作账号，`ByUser` 不应解释为训练崩溃；查看进度不构成重新启动授权。浏览器会话访问故障的现有登录态回退方法见平台工作流。
- `save_only_model` 产物不等同于完整 Trainer 状态。缺少 optimizer/scheduler 分片时，只能按权重续训语义处理并明确说明学习率、步数和数据游标是否重置。

## 统一工作流

1. **确认范围与授权。** 判断用户要的是检查、修复、准备表单、提交、重提、运行诊断还是训练 ETA；据此加载上表中的资料，并明确哪些外部写入已经获得授权。
2. **检查真实代码源。** 在目标开发机读取分支、commit、工作区状态、launcher、配置及完整继承链；搜索所有数据、输出、checkpoint 和挂载路径。远端拉取代码时确认所需 commit 已推送；共享 vePFS 直接提供代码时核对任务实际可见的 checkout 和关键文件身份。不要为无关项目强加 `asset_contract` 或通用 preflight；已有的数据摘要应照其实验合同核验。
3. **审计有效配置。** 对 PSE 使用仓库自己的 `load_config` 读取合并结果，打印关键 `[meta]`、`[distributed]`、梯度累积和保存间隔。对 VAD 同时核对 train/valid/test DataLoader、模型 chunk、batch、LR 和拓扑。不能只读子 TOML 或只改一处分区。
4. **验证 checkpoint 和启动契约。** 核对 checkpoint 可读性、索引引用的全部分片、文件写入稳定性、模型键和形状、speaker vocabulary、DDP rank/world-size 变量、所有节点可见的代码/数据/输出路径，以及选中 launcher 实际使用的 resume/lease 行为。没有 lease 的专用 launcher 不应被迫补一个。运行与目标拓扑相称的聚焦测试、`git diff --check` 和 launcher 的 `bash -n`；单节点任务不要求两个 node-rank dry-run。未在云端验证的镜像/挂载/rank 必须在提交后继续核验，不能用开发机 smoke 冒充。
5. **设计非重叠挂载。** 先制作“开发机源 → 任务容器目标 → 读写模式 → 代码用途”映射。容器目标不得互相包含。根据现有路径选择平台资料中的一种中性挂载方案，并只创建必要、可验证且不会覆盖非空目录的兼容软链接。
6. **审计完整任务表单。** 核对区域、私有镜像、代码来源、入口命令、环境变量、PyTorch DDP 框架、worker 数、单机 GPU、队列/AZ、优先级 `6`、运行时、重试、挂载、任务名和描述。仓库由共享 PFS 提供时，避免同时使用陈旧的网页上传代码快照。
7. **执行提交前闸门。** 按平台资料中的清单逐项复核。入口、配置、commit、镜像、必需挂载或权限有一项无法证实时，停止在提交按钮之前并说明缺口。
8. **按授权执行外部写入。** 代码改动只提交确切的新文件或修改；任务只提交一次。运行中任务的配置变更应生成可追溯的后继版本；若任务从远端 Git 拉取代码，先提交并推送，若从共享 vePFS 读代码，核对容器将读取的文件身份。后继任务先提交优先级 `6` 的新任务并核对其 ID/详情；只有已有授权明确覆盖停止操作时，才停止精确目标旧任务。用户指定保留旧任务时不停止；不要擅自克隆、重试或删除。
9. **提交后核验。** 从任务列表和详情页记录完整任务名、ID、状态、框架、队列、节点/GPU、镜像、入口、配置和 commit。`排队中` 只证明创建成功；`部署中`、`运行中` 只说明平台状态。必须从日志确认有效配置、拓扑、真实 batch 和有限 loss/grad，再从共享存储确认 checkpoint，才能判断训练链路启动并保存。
10. **控制失败传播。** 确定性失败时读取父任务的重试记录，记录所有 child task ID，准备独立输出和修复配置。按上述顺序先提交并核对唯一后继任务；若用户已授权清理旧重试任务，再停止精确旧任务并确认剩余错误重试为 0。用户指定保留旧任务时不执行停止。若新任务提交受阻，保留旧任务状态并报告，不能通过先停旧任务改变顺序。保留失败记录，除非用户明确要求，不删除任务历史。
11. **按证据诊断。** PSE 启动异常先核对有效 skip 配置和加载摘要，再解释 `sv_ce`、重建、ArcFace、AMP 或 NaN。VAD OOM 先读取每 rank 的真实分配/占用，再调整 micro-batch 和 accumulation。估算时长读取 rank-0 持久日志；VAD 的稀疏 batch 日志使用 50/100-batch 窗口，不用单个日志间隔或平台总运行时外推。

## VAD 批量 mask 推理补充

- 这是推理与数据交付任务，不套用训练 epoch、optimizer、accumulation 或 checkpoint-resume 语义。核对推理 TOML、固定 checkpoint、EMA/precision、阈值、输入树、输出树和逐 rank manifest。
- 云端提交前必须在目标代码版本完成真实单卡、多卡不重复分片和 `missing=0` 二次运行；没有完整通过前停在提交按钮之前。
- 平台任务使用精确 clean Git worktree、明确配置与 checkpoint。不要自行加入内容哈希或 `asset_contract`。提交后从日志确认所有 rank 加载各自唯一 shard，并从共享存储确认三类文件同步增长。
- 任务完成分为“推理完成”和“逐文件终验通过”两个阶段。后者必须以全量 validation attestation 为准；平台显示运行完成或只看文件存在都不够。
- GPU 显存占用低不等于任务异常。小模型纯推理常受音频解码、共享存储小文件读取和每样本多文件写入限制。调大 batch 前先用真实时长分布做吞吐、峰值显存、padding 浪费和输出一致性测试，不要直接修改在跑任务。
- 长尾录音必须单独审计：一条几十分钟音频可把整批 padding 放大数十倍，无下采样的 noncausal 全局 attention 还会随帧长近似平方增长。普通批次应剔除长样本并按真实长度重新裁剪；超长样本使用经过 full-vs-stitched 数值对照的单文件回拼路径，不能静默跳过或产出 `_segNNNN` 替代品。异常路径不得进入 NCCL barrier，以免 watchdog SIGABRT 掩盖原始 OOM。
- 失败后的 post-scan 已经原子生成下一轮精确 missing shards，应直接复用；不要在下一 pass 顶部重复一次千万级全量扫描。平台日志至少打印扫描开始/完成，区分共享存储元数据阶段与 GPU 卡死。
- 同一输出树的后继恢复任务可显式复用上次已发布的 `mode=missing` shards，避免再次遍历千万文件，但必须先匹配 base/output/manifest、输入与输出合同、limit、rank/world-size，逐个解析 shard，并核对路径、计数及 exact union/disjoint ownership。只允许把旧清单当作当前 missing 的超集：DataLoader 必须在解码前按 NPY+JSON+RTTM 三件套过滤，推理器继续逐样本防覆盖；任务结束后仍做新鲜 post-scan 和全量终验。禁止复用中断扫描的隐藏临时文件、新输出树、`mode=all`、合同/拓扑不一致或元数据损坏的清单。
- 从慢扫描任务切换到快速恢复任务时，先提交一个锁定明确代码版本且优先级为 `6` 的后继任务，取得 ID 并核对详情；仅在已有停止授权时停止旧任务并确认其写入者释放。若复用同一输出树，新入口必须在旧写入者退出前保持只读等待或由其他可验证的互斥机制阻止并发写入；做不到就停在提交前解决。运行摘要和任务描述须记录父任务、复用开关与 manifest 验证证据。GPU 碎片导致排队时不要重复提交规避队列。
- 可视化和配对音频必须写到生产输出树之外，避免被 no-orphan 终验判为污染；复制音频后解码两端并核对大小、采样率、采样点数和波形一致性，禁止移动源文件。

## PSE 完整预加载的最低要求

对架构和 speaker vocabulary 兼容的 PSE 权重完整预加载，应在后继 TOML 中显式设置：

```toml
preload_embedding_skip_front = false
preloaded_skip_prefixes = []
preloaded_embedding_skip_prefixes = []
preloaded_extraction_skip_prefixes = []
```

其中 `preloaded_embedding_skip_prefixes` 优先于通用 fallback，是 `sv_arc_margin` 被静默随机初始化的常见原因。最终还必须证明 checkpoint 含有 `embedding_model["sv_arc_margin.weight"]` 且形状、说话人数量和词表映射一致。

## 失败处理

- CloudFS 因服务关联角色或存储权限预检失败时，不要擅自开启账号级设置。先证明训练是否使用该挂载；仅在确认未使用时省略，并记录差异。
- 平台拒绝嵌套挂载时，改用中性目标和安全软链接，不要删除训练必需的共享文件系统。
- GPU 池不足时保留唯一任务处于排队状态，并报告容量约束；不要重复提交。
- 同配置、同阶段或同类输入上重复出现的 CUDA OOM 应按确定性资源/配置失败止损，不要等待平台按原配置耗尽自动重试。首次 OOM 先记录失败阶段、输入长度、其他进程、allocated/reserved/free/requested 和碎片化证据。需要后继实验时保留原 TOML 和失败任务证据，新建独立 `root_dir`，并按任务专项合同检查 micro-batch、累积、scheduler 和尾梯度语义。
- 如果保持 global batch，需要同时核对 scheduler 的 step 口径、checkpoint 保存间隔和每 epoch optimizer-step 数；不能只改 `batch_size` 后宣称训练等价。
- 已获授权的服务器推送或独立获授权的GitLab操作因仓库SSH端口超时时，可先验证同一Git服务的22端口，再使用一次性`GIT_SSH_COMMAND`；本机GitHub禁写不属于连接故障，禁止通过端口、URL、remote、代理或隧道解决。不得绕过未知主机或凭据问题。
- 高首批 `sv_ce` 不是整模型损坏的充分证据；先区分随机 ArcFace 头、词表不兼容、原始与加权损失以及 warmup。
- 不要在已运行进程中通过原地修改 TOML“修复”加载问题；创建后继配置并按授权重启或重提。
- 配置变更导致合同测试仅因旧字面值断言失败时，不得宣称测试全绿；确认它不是功能失败，并按用户授权范围选择同步测试或如实记录该项陈旧断言。

## 交付要求

对中文用户使用中文，先报告结果，再给证据和限制。至少说明：

- 检查或修改的文件、最终有效配置和 checkpoint/加载证据；
- 执行的测试、未执行的真实多机多卡预检及原因；
- commit/push 结果（若获授权执行）；
- 是否提交任务，以及任务名、完整 ID、当前状态和关键资源；
- 若发生失败与重提，列出父任务、自动重试 child、被停止的错误任务、后继任务及各自最终状态；
- 相比开发机环境采用的挂载、入口或 CloudFS 调整；
- 对批量推理说明源总数、每 rank 所有权、三类输出计数、抽查结果、最终全量 attestation，以及可视化/音频交付目录是否与生产树隔离；
- 诊断或 ETA 的数据来源、估算范围与不确定性。

## Artifact locations

本次材料存放遵循当前会话的明确选择。用户选择“仅保留本机”时，报告与已取到的日志快照保留在本机任务目录；历史服务器主存储约定不构成本次上传授权，也不将这次选择扩展为以后所有任务必须本地保存。需要服务器保存且已在授权范围内时，任务表单证据、日志截取、诊断 JSON 和 ETA 报告写 `/root/codex_artifacts/07_维护与迁移/06_环境模型及训练提交核验/火山平台声纹训练与评测的提交核验_volcengine/<task_id_or_run_id>/`；训练输出仍由实际 launcher/TOML 和共享挂载决定。

归拢旧产物或解释主机状态时读 [产物位置与统一根](references/artifact-locations.md)。

## 服务器产物的当前实体位置

2026-10-20 服务器目录已实际重命名和归位。访问 `dev_a800_2gpus:/root/codex_artifacts/` 的产物时，先读取根 `CURRENT_LAYOUT.json`；当前层级以真实目录和该指针为准。

历史日志、冻结清单和旧聊天可能仍写旧名。用指针中的 `resolver_script` 解析，或通过 `--records` 读取原清单的当前路径视图；不要据旧字符串重新创建英文旧根、恢复整批到 Mac，或改写冻结原件。当前数据入口为任务根的 `当前数据入口.json`。音频标签、模型身份、许可、来源分组及评测保护规则保持原有约束。
