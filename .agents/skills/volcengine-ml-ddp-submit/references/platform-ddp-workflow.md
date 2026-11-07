# 火山引擎 PyTorch DDP 自定义任务工作流

## 目录

1. 登录与浏览器控制
2. 提交前信息采集
3. 存储挂载转换
4. 任务表单
5. 提交前检查
6. 提交后核验
7. 已验证的 se_train 示例

## 1. 登录与浏览器控制

优先使用能够接入现有登录状态的 Edge/Chrome 会话。若常规浏览器控制无法访问该会话，使用当前环境提供的统一 Computer Use 接口；在 Codex Desktop 中使用 `cua_repl`，并复用同一持久会话及其可访问性状态。不要启用另一条会破坏现有会话的自动化路径。

当前组织的已验证 profile 在实时页面仍提供对应入口时，可按以下方式重新认证：

1. 选择 `企业联邦登录`；
2. 进入钉钉身份流程并选择 `钉钉登录`；
3. 等待用户完成必须由本人操作的授权；
4. 确认返回预期区域的机器学习平台 URL。

其他租户或账号遵循实时页面提供的认证方式，不要机械寻找钉钉入口。不要通过退出登录来重启流程。多窗口或多显示器环境中，先按标题和 URL 确认目标控制台。表格截断 ID 时打开详情或读取 tooltip/可访问文本；截图只能作为证据，不能替代完整字段。

**提交通道（2026-11-06 现状）**：服务器虽有 `/root/.volc/bin/volc` + `mlp`，但 `~/.volc/credentials` 不存在，CLI 报 `AccessKeyId or SecretAccessKey is empty`，因此**控制台"复制任务"是当前唯一可用通道**；`ml_task list` 只列非终态，查历史要用 `get -i`。逐字段操作与 AX 自动化要点见 [控制台复制任务提交](console-clone-submission-20261106.md)，不要在本文件里另立一套点法。

### 现有登录态的控制回退

`functions.exec` 的 `ALL_TOOLS` 没有列出浏览器工具，不证明独立命名空间的
`mcp__cua_repl` 不可用。优先调用当前已暴露的工具；首次初始化严格遵循其文档，
只执行一个入口API，并读完返回说明再继续。

若浏览器扩展的会话枚举/连接因 header policy 等报错，且环境提供 native App
控制，可使用同一个已登录浏览器的原生可访问性界面。2026-10-01 已验证入口：
`await cua.getApp("/Applications/Microsoft Edge.app")`。当 bundle ID
`com.microsoft.edgemac` 匹配多个 updater 副本时，使用已核实的主app绝对路径。
这保留已有账号；无需重新登录或启用另一套自动化。只使用工具返回文档中的API。

### 状态与页面新鲜度

从休眠的控制台tab读取前先刷新。运行时计数曾落后24小时，URL变化后主体也曾
短暂保留前一任务字段或加载占位符。待页面加载后联合核对最终URL、完整ID和名称，
再提取状态；不能仅凭地址栏断定当前展示的是目标任务。

同时核对持久日志最后时间/步数和checkpoint已保存步数，并标注时区。页面显示
运行中仅证明平台状态；持久日志有推进才证明训练推进。已停止任务读取时间线里的
停止原因、时间和操作账号。`ByUser` 是主动停止的证据，不当作训练崩溃；账号本身
不能区分用户手动操作与其他受授权流程。只读进度请求不授权重新启动它。

## 2. 提交前信息采集

填写表单前收集：

- 开发机 URL、区域、镜像和资源类型；
- 完整挂载表：存储类型/名称、源路径、容器目标和读写模式；
- 仓库绝对路径、远端分支、commit、工作区是否干净；
- launcher 和确切配置路径；
- 私有镜像仓库与 tag；
- 节点数、每节点 GPU 数与类型；
- 队列、可用区、优先级、运行时、重试策略和输出路径。

检查 launcher 与配置时确认：

- `torchrun` 或平台 DDP 正确提供 rank 信息；
- 单机框架自身会派生多 rank 时，确认平台入口每 worker 执行一次；不要再包一层会把 rank 数相乘的 `torchrun`。以实际容器进程树和日志验证，勿从框架名称推断。
- node rank、local rank、master address/port、world size 和 local world size 没有错误硬编码；
- 所有节点看到相同的代码、数据、输出和 checkpoint 路径；
- resume/lease 逻辑在 worker 重启时安全；
- 没有隐藏依赖引用看似可选的挂载。

使用 `git status`、`git rev-parse HEAD`、`git branch --show-current`，并用 `rg` 搜索每个存储目标。表单中从 Image Center 选择准确私有镜像，不要手填未经认证的镜像拉取 URL。

仓库改名或迁移后，还要核对 Conda/Pip 的 editable 安装映射、`PYTHONPATH`、已保存的 Hydra/训练配置、launcher 的 commit 锁和所有预训练权重绝对路径。分别在目标解释器里验证代码包和训练入口可导入，再做只解析配置的检查；旧路径软链接至多帮助仍存活的进程重新找到文件，不能恢复已退出的 DDP rank，也不能绕过启动脚本的版本检查。后继任务续训应从稳定的完整 checkpoint 恢复 optimizer/scheduler 与更新步数，并验证旧任务已释放全部写入者；具体 MMS-LLaMA 57,500 步案例见 `$avsr-reproduction-ops` 的 `references/mms-lrs3-training.md`。

### 长训的未来阶段依赖

查询ReDimNet2所有任务进度时，先用其项目技能的`references/task-status-and-readonly-progress.md`区分运行中的PTN／标注、受控停止旧任务、已结束有限验证与尚未启动LM／完整字实验。新标注薄索引进度用`bounded_metadata_progress.py`，不要调用会重读partial和旧commit的旧`--bounded`入口或因平台查询重新启动全库审查。平台、日志、检查点stat、恢复演练和质量分别说明；新进度用带日期证据，不把旧192worker或旧利用率称为本次实证。

除正在运行的 rank、日志和 checkpoint，还要沿实际 launcher 检查尚未执行的阶段。记录启动时固定的源码版本与摘要、当前共享 checkout、进程已加载代码及未来将重新打开的文件，四者不能互相代替。读取环境变量中的外置校验清单、smoke 凭证、数据审计文件、官方参考权重及后续输出路径，在目标训练容器逐一验证实际路径、可读性和预期身份；开发机可见不证明容器可见。

`/proc/<pid>/fd`、inode 与现有日志推进只能解释当前已打开对象或当前阶段。后续 `verify_source`、LM、导出和评测可能按旧绝对路径重新打开文件，即使当前训练不持有这些文件，目录改名仍会破坏接续。只读巡检应分别报告“当前阶段是否推进”与“后续阶段是否满足依赖”，并写清检查时刻和证据范围。

发现冻结源码漂移或旧路径消失时，保留原始校验清单、启动入口、源码差异和路径映射。不能重算冻结 SHA、取消校验或把当前源码摘要冒充开训摘要来消除报错；文档维护不授权修改活进程环境、覆盖运行源码、停止训练或重提任务。修复方案应基于原始身份和实际读取合同单独准备。存储重组的未来读取依赖核查用 `$artifact-layout-maintenance`；ReDimNetV2 的阶段合同用 `$redimnet2-reproduction-ops`。

### MMS 长训的冻结源码目录

MMS-LLaMA v6 的入口在训练开始与终验都核对固定 HEAD 和干净状态。2026-10-04 使用独立 detached worktree `/root/code/github_repos/MMS-LLaMA-fork-datarepair-v6-20261004`，代码修复commit `e3133df4d555a2b67d02f21b13ba4d8b40c5c0d2`；维护分支随后可更新文档，不会使正在长训的快照在终验时因HEAD改变失败。核对 worktree `.git` 所指 common gitdir 也位于容器挂载的 `/root` PFS；只挂worktree目录而遗漏gitcommon不能满足此合同。

两个实验可共用同一源码快照，但必须使用不同RUN与各自固定的数据合同；combined清单在修复完成前不生成正式提交入口。外置合同也用SHA固定。这个快照和commit是该次实验的历史证据，新提交仍需核验当前实际快照和合同，不机械复用日期路径。

MMS入口SHA前缀若包含Bash专属`set -o pipefail`，而平台外层默认shell没有证据，就用显式`/bin/bash -c`包住完整SHA检查与exec。开发机`/bin/sh`指向dash、旧wrapper自带bash，均不能证明平台外层的解释器。CPU可验证精确入口的SHA前缀由`/bin/sh -c`调用显式bash成功；训练执行仍以实际云日志为准。已提交任务只有出现真实Illegal option等启动失败证据才按后继流程修复，不因默认shell猜测重复提交。

## 3. 存储挂载转换

先建立映射表：

| 开发机源 | 自定义任务目标 | 模式 | 代码引用 | 处理方式 |
| --- | --- | --- | --- | --- |
| 实时填写 | 实时填写 | RO/RW | `rg` 证据 | 直接挂载/中性挂载+软链接/省略 |

基本规则：

- 保留数据集、checkpoint、输出或 rendezvous 所需的共享 vePFS，并保留读写模式。
- 容器目标不能存在包含关系，例如 `/root`、`/root/group-shared`、`/root/yaww-speech-private` 不能同时直接挂载。
- 只创建代码确实依赖的兼容软链接；验证源存在，接受已经正确的软链接，仅可移除空目标目录，拒绝覆盖非空目标。禁止宽泛的 `rm -rf`。

根据实时环境选择一种方案：

### 方案 A：用户 PFS 保持 `/root`

适用于仓库大量依赖 `/root/...` 路径的 PSE 任务：

```text
用户 PFS                → /root
group-shared PFS        → /shared/group-shared
yaww-speech-private PFS → /shared/yaww-speech-private
```

入口中安全创建：

```text
/root/group-shared       → /shared/group-shared
/root/yaww-speech-private → /shared/yaww-speech-private
```

### 方案 B：用户 PFS 改挂到中性目录

适用于共享文件系统必须保留 `/root/...` 目标、而用户文件系统可从中性目录运行的任务：

```text
用户 PFS → /mnt/<user>
```

仅为确认存在的用户目录建立兼容链接，例如 `/root/data` 或 `/root/workspace` 指向 `/mnt/<user>/...`，然后从核实过的中性路径进入仓库。

选择标准是：满足表单非重叠约束、保持训练代码所需路径一致，并让兼容改动最少。不要把任何一种历史方案机械套用到新的挂载表。

CloudFS 注意事项：服务关联角色可能要求账号级“存储权限预检”。未经明确授权不要开启全局设置。若所选训练流程经代码搜索证明未使用该挂载，可以省略并记录；否则停止并取得所需权限或配置。

## 4. 任务表单

建议顺序：

1. 确认区域、账号、自定义任务与 `PyTorch DDP`；
2. 从 Image Center 选择准确私有镜像和 tag；
3. 设置代码来源；共享 PFS 已提供仓库时，不再上传可能过期的源代码快照；
4. 入口命令执行安全兼容设置、进入已核验仓库并调用已入库 launcher；
5. 只添加 launcher/config 明确理解的环境变量；
6. 设置 worker 数及每个 worker 的 GPU 规格；
7. 选择用户要求的队列和兼容 AZ；
8. 添加已经验证、互不重叠的挂载；
9. 仅在能区分基础设施失败与确定性训练错误时使用有界自动重试；配置/OOM 诊断期间可关闭；除非有意保留，不保留失败资源；
10. 诊断、TensorBoard 等仅在训练或用户需要时开启；
11. 填写唯一任务名、准确描述、优先级 `6` 和可见性。

最长运行时间按用户明确给出的实验上限填写，并与 launcher 的 step/epoch 截止条件分别核对。不要因为推测云资源占用就自行缩短用户指定的时限；运行时间上限也不代表训练必然会运行到该时刻。

复制任务表单时，把代码编辑器视为专用编辑器而不是普通 textarea。自动化的
`fill` 可能把新入口追加到旧入口后面；提交前必须重新读取编辑器的实际完整值，
并要求它与预期入口逐字一致。任务创建成功后，再从详情页核对真实入口，不能只凭
“创建成功”提示。**具体点法（`AXPress` 点按钮、`set_value` 填字段、快照 60 秒过期、
数字框 `AXIncrementor` 不吃输入、必须改哪三处）统一见
[控制台复制任务提交](console-clone-submission-20261106.md)**；本节只保留"必须复核入口"这条纪律。

用户规定所有实验提交优先级为 `6`。必须从实时表单确认 `6` 可选且已选中；
提交后从任务详情再次核对实际优先级。若 `6` 不可选或无法保持选中，
停止在提交前并报告，不能自行改用其他值。

2026-09-28 的 MMS-LLaMA 后继任务遇到复制自优先级 4 的表单与空白表单均无法把
`6` 设为选中，点击后仍显示 `4`。当同账号已有优先级 6 的历史任务时，可复制该
任务作为新表单起点；**只有在最终表单及创建后的详情页都明确显示 `6` 时**，才能
认定满足用户要求。复制其他项目的任务必须逐项清除继承的入口、环境变量、额外
挂载、自动重试、规格、时限、名称、描述和可见范围，并重新验证全部字段。不要以
任务名或描述中的“p6”代替真实优先级，也不要提交已知为 4 的任务后假定能改为 6。
该例中的中间任务 `t-20260928040354-pfr99` 在详情页确认优先级 6，随后在仍排队时停止。
后继 MMS-LLaMA LRS3-only 任务 `t-20260928042338-mp2dh` 于 2026-09-28 创建，详情页确认
60,000 updates 的 launcher、336 小时上限和优先级 6；创建时还没开始跑。这些 ID 只是历史
证据，新请求必须重新读取实时状态，并以 `$avsr-reproduction-ops` 的实验合同为准。

## 5. 提交前检查

- [ ] 区域和账号正确；当前账号实际采用的认证会话有效，企业联邦/钉钉仅为历史已验证profile。
- [ ] 从 Image Center 选择了准确镜像 tag。
- [ ] 框架为 PyTorch DDP，节点数及每节点 GPU 数/类型正确。
- [ ] 队列、AZ、优先级实际选中 `6`、运行时和重试策略正确。
- [ ] 入口指向已核验 launcher，配置为预期后继配置而非已运行 TOML。
- [ ] 远端拉取代码时，所需 commit 已推送；共享 vePFS 提供代码时，容器将见到的 checkout、launcher 和已有数据摘要已核对，且不是陈旧快照。不为任务强加无关的 `asset_contract` 或通用 preflight。
- [ ] 所有节点可见共享输出、checkpoint 和数据路径。
- [ ] 必需挂载齐全、模式正确、容器目标互不重叠。
- [ ] CloudFS 前提已满足，或已证明该挂载未被使用。
- [ ] 环境变量（如 `CONFIG`、`RESUME`、secret 配置）符合目标实验。
- [ ] 新任务名以用户指定的 `zhounan_` 开头且唯一，描述如实记录路径适配和限制。
- [ ] 本次提交已获用户授权，包括当前会话此前给出的授权；不要重复索取确认。

用户规定创建后继实验始终先提交新任务并核对详情；旧任务仍排队时也一样。
顺序：创建优先级 `6` 的新任务并取得唯一 ID → 从新任务详情核对入口、
资源、优先级和挂载 → 若用户已授权停止旧任务，才停止精确匹配的任务并等待其进入已停止；
若用户要求保留旧任务，则保持其原状态 → 核验新任务调度与训练日志。若新旧任务共享输出，提交前必须让新入口在旧写入者退出前
无法写入（例如已有且已验证的 lease/等待闸门）；无法确保时先解决互斥，不能
靠先停旧任务规避。若新任务提交受阻，不要先停旧任务或重复提交。
状态标签 `排队中` 只证明创建成功；只有预期 worker 运行且日志出现真实 batch 才算跑起来。

## 6. 提交后核验

提交一次后：

1. 返回自定义任务列表并按唯一名称定位任务；
2. 从详情页记录完整 ID，不使用表格中被截断的值；
3. 记录 `排队中`、`部署中`、`运行中`、`失败` 或 `成功` 等状态；
4. 记录框架、队列、节点/GPU、镜像、入口、配置和代码 commit；
5. 有日志时检查分布式初始化；
6. 排队时检查容量/队列信息并报告，不创建重复任务；资源不足或 GPU 碎片不是失败，改为可抢占闲时任务需符合用户的训练目标；
7. 运行时确认全部预期 rank 出现、有效配置被读取、有真实 batch 和有限 loss/grad。checkpoint 在共享存储写入完整后，再报告保存成功。

千万级批量推理还要按资源阶段拆分核验。GPU rank 全部成功退出后，如果剩余工作
只是 missing manifest、输出树计数或逐文件语义验证，不要让昂贵 GPU 实例继续
空等数小时。先在真实共享盘验证 CPU-only 后继流程及等价性；取得停止授权后，
停止唯一 GPU 写入任务并确认资源/写入者消失，再启动一个 CPU 任务或现有 CPU
服务器后台进程。CPU 流程必须锁定明确 Git 版本、输入/输出路径、预期总数和 validator
分片数，只允许写原子 manifest、summary、failure log、attestation，禁止改动生产
NPY/JSON/RTTM。

missing 扫描应按镜像目录一次 `listdir` 后做三件套文件名集合判断，不能对每条源
音频串行发出三次远程 `stat`。这只优化恢复选择器，最终验收仍要打开并验证全部
产物。扫描和每个 validation shard 必须定期记录 processed/complete/missing、
elapsed、rate；无进度日志的数小时 `D` 状态不能仅凭“进程仍在”判断为合理。
切换前后不得有两个进程同时发布同名 manifest。

中断任务遗留的隐藏原子临时文件不得为了“过验收”而自动删除。只有在文件名严格
符合既有 writer 的临时命名、对应正式 sibling 已存在且逐条语义验证通过，并且
attestation 逐项记录临时路径、正式路径、大小和时间时，才可归为允许保留的中断
证据；孤立临时文件仍必须使终验失败。

## 7. 已验证的 se_train 示例

### 全量双伪标签 VAD 基模，2×8、无梯度累积（2026-09-08）

- 使用复制表单，但把实例数改为 2；框架为 PyTorchDDP，每个 worker 为 8 张
  A800 80GB。
- 每卡 batch 48、accumulation 1、global batch 768；专用 launcher 固定并检查
  2/8/16 拓扑，不经过旧通用 launcher。
- 入口只建立 `/root/group-shared` 与 `/root/yaww-speech-private` 的安全兼容链接，
  然后调用专用 launcher；表单删除冗余 `CONFIG` 和旧 lease 环境变量。
- 表单沿用 `dev_zhounan:4.0`、`aiplatform-SPEECH`、可用区B、优先级6和三个
  已验证 vePFS 挂载。
- 提交后同时核对平台任务、两个 worker、实际复制 TOML 和共享 rank-0 日志；观察
  到 Batch 0→30、loss 与 grad norm 有限、无致命错误后，才报告已启动。

这是特定环境的已验证模板。新任务必须重新检查实时资源、路径、分支和表单。

### 历史 PSE 示例（2026-07-31）

以下是 2026-07-31 的历史提交模板；新任务必须重新核对所有实时值：

- 仓库：`/root/code/gitlab_repos/se_train`
- commit：`16394474b526b5c28d74d0037e7a8a075092f5b6`
- launcher：`recipes/pse/merged_datasets_20260724/train_volcengine_2x8_globalutt_pairedrir.sh`
- 配置：`recipes/pse/merged_datasets_20260724/config/train/PSEUDiT_globalutt_sharedsv_pairedrir_identity_16k_8gpu_absent05_swap05_coldstart_fixed_fp16_samebase_primaryeval_volcengine_2x8.toml`
- 镜像：`prod-cn-beijing.cr.volces.com/aiplatform/dev_zhounan:4.0`
- 资源：PyTorch DDP，2 个 worker，每个 worker 8 张 A800 80GB
- 队列/AZ：`aiplatform-SPEECH`，可用区 B
- 任务名：`pse-globalutt-pairedrir-2x8-smoke-20260731`
- 初始状态：`排队中`

该历史任务使用方案 B：用户文件系统挂载到 `/mnt/zhounan`，保留必要共享 vePFS，并创建 `/root/data`、`/root/workspace` 兼容链接。仅在仓库搜索证明所选流程未引用 `/aispeech-cfs` 后省略 CloudFS，原因是平台在未开启账号级存储权限预检时拒绝服务关联角色配置。

<!-- MMS_FOURGPU_ACCUM2_20261004_LATEST -->
## MMS-LLaMA四卡累计两次的发布原则（2026-10-04）

本轮MMS采用1×4、2400帧/card、accum2，标称19200帧/optimizer update；60k/240k、priority6和336h保持。模型/数据仍由AVSR对应reference锁定；其他项目按其实际用户请求选择拓扑。平台资源、YAML、外置合同和真实Fairseq导入路径/SHA必须一致，不能用旧ASSET_ROOT导入掩盖worktree已写补丁。

累计组按全部real的有效token归一，最后一微批dummy不能清掉此前real梯度对应分母；dummy只用于必要同步。OOM清梯度必须同步处理已失效统计，本轮MMS累计组OOM即raise，避免带错误分母继续更新。CPU transport-shim测试可验证该逻辑，不证明完整LLM/GPU云训练已运行。

四卡accum2只保持标称预算，动态batch/RNG/FP16/尾组使其非逐位等价；optimizer/scheduler/save仍按optimizer updates，ETA读取真实稳定日志。先核实新priority6任务唯一ID，再按明确授权处理精确旧任务。Mac锁屏等提交阻碍期间，新ID为null且旧任务保留；准备/preflight通过不可写成已提交/排队。两地README分别只换latest标记块，保留各自不同历史body。
<!-- /MMS_FOURGPU_ACCUM2_20261004_LATEST -->

### 已运行父脚本与独立阶段接续

除环境和路径外，确认父 Bash 是否已解析函数、here-doc和条件块。改磁盘 wrapper 不能证明运行中的父脚本采用新逻辑；不要注入其内存、替换解释器或改旧日志。ReDimNet2 的旧check_stage会将已保留的捕获FLAC读取Traceback判为终验失败；新事件发生或独立接续版本更新都不会改变原父shell已缓存的函数。具体事件登记和新版本检查使用实验专用技能；修齐路径或独立源码不等于原同任务会自动接续。

用户要求保留当前训练时，可准备独立冻结源和明确的后继阶段入口，但“入口已准备”不等于“原任务将自动进入下一阶段”。实际提交或启动依用户授权和新鲜条件执行，不提前占用多卡等待，也不默认创建监控。确认父任务达到目标轮次、完整检查点和来源正确、旧任务终态及写入者退出、原输出锁可独占，并使用不重叠的新输出。开发机、CPU导入/配置检查、云容器路径和真实 LM 启动分别报告。

若获授权为后继阶段修复解码依赖，使用后继新进程专属库路径，在解释器启动前设置其环境，并核实际加载的原生库路径、摘要与版本及生产裁剪/压力读取；不替换活跃任务环境或全局库，不以整段读取成功替代随机裁剪核验。ReDimNet2的v2失败、v3私有库和事件登记详见实验专用阶段引用；开发机通过不等于真实worker或完整接续ready。

后继进程选私有解码库时，须保留已核平台的NCCL/NVIDIA基础库搜索路径，不能因CPU导入成功清空GPU基础设施路径。本次ReDimNet2最终显式入口v3/platform_entry.py固定私有库在首位，再加`/usr/local/nccl-rdma-sharp-plugins/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64`并移除外部PRELOAD/AUDIT，复用原门禁与互斥。当前PTN环境不动；新组合worker CPU检查及正式八rank GPU启动分别验收。


### ReDimNet2 接续入口的数据子进程语义（2026-10-24）

相同PyTorchDDP规格和状态恢复成功不能证明数据子进程语义一致。ReDimNet2的接续包装使用torchrun --run-path/显式spawn，改变了普通训练脚本下DataLoader的上下文；CPU双rank实际key对照证实冻结sampler丢失rank分片，legacy增强全局也依赖继承。该实验最终恢复普通worker.py入口，保留父epoch守卫、原配方和24workers，真实云端192个worker的rank/world/sampler/epoch/legacy及八rank首更新逐项验收。其他项目应核其具体采样器和进程合同，不一概强制fork。细节见redimnet2-reproduction-ops的references/ptn-resume-ordinary-entry-fix-20261024.md。

若用户已授权修复重启，安全后继可先创建新任务并读回唯一ID，再停止精确旧实例；新入口须有界等实际前驱终态与原子锁，未准入时不启动GPU。ReDim本轮外置小型终态receipt在新ID/实际停止后原子发布，固定前驱身份、新鲜度、权限、稳定SHA并在锁内复读，不改冻结包清单。发布报告或转移日志仍按用户原存储授权处理。
