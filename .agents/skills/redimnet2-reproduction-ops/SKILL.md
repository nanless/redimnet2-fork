---
name: redimnet2-reproduction-ops
description: Inspect and maintain the nanless/redimnet2-fork reproduction project, including B6 PTN/LM continuation, open-model sidecar annotation, complete-word variable-length fine-tuning, source and runtime identity, and protocol-matched evaluation. Use for this repository and its experiment records; use the matching task reference and distinguish historical results, CPU checks, real model execution and GPU acceptance.
---

# ReDimNet2 复现与实验维护

先确定用户要查进度、改代码、恢复训练、维护标注，还是做完整字微调：这几种任务的输入、版本和验收要求各不相同，写文档不会顺带把训练跑起来，也不会自动开始模型调用或后台巡检。本文用主训练（PTN）和大间隔微调（LM）两个阶段说明；配置键与文件名按磁盘上的原名写。

## 当前状态（2026-11-06 核查）

- **PTN 已完成 120/120 轮**。48—120 段自 `model_47` 受控接续：起 2026-10-26 07:51:27（北京，首个真实 SGD，绝对迭代 705,047），止 **2026-11-03 10:12:02（北京）**。`ptn_success.json` 记 `ptn_complete:true`、`epochs:[48..120]`、`LM_started:false`、`historical_numeric_root_cause_proven:false`、`three_segment_LM_contract_required:true`。`model_115.pt`/`model_120.pt` 各 650,778,662 B，sha256 `65497b6a…aad5c` / `2798fe23…b0cc2f`；`final_model.pt` 只是→`model_120.pt` 的 12 B 软链。末轮 `vox1-eer-8s` **0.399%**（该口径只读前 8 秒，与官方 0.26% 的整段协议不能直接比）。
- **LM 已完成 1 轮**：运行目录 `.../lm-from-ptn115-20261106-v1/lm` 于 2026-11-07 03:05–05:37 UTC 跑满 1 轮、13,075 步、`training_exit_code=0`，`vox1-eer-8s=0.293%`。**完成者的平台 ID 是 `t-20261106110454-vd8pq`，属平台侧记录；运行产物里没有 `task_id` 字段，本地也没有该 ID 的留档**——终态只能在控制台详情页看，“本地 grep 不到任务号”说明不了没跑。早期 `t-20261106195855-rq2b7`（入口误带 `--execute`，被 gate 的 argparse 拒绝，从未运行）与 `t-20261107053934-6lcmt`（冻结包内 `launch.sh` 缺执行位，`launcher_missing`）均已失败作废。`lm_receipt.json` 的 `ok:false` 是门禁只读 `train.log` 造成的假阴性（仓库源码已修，冻结包未重打）。单机 8×A800、优先级 6、自动重试关闭。受控包 `package-r3`：package sha256 `08a34b68…9aa003`、contract sha256 `bf437dc8…9230d0`；`model_1.pt`/`model_115`/`model_120` 的体积与完整 sha 现查（`sha256sum <artifact>`，或看仓库自动生成的 `docs/STATUS.md` 第 4 节），本页不抄写——重新冻结后这些值会变，抄一份就与包脱节。判据、命令与首屏自检见[LM 受控启动](references/lm-launch-20261106.md)。
- **自动标注已停摆**：2026-11-01 13:13:34（北京），平台任务 `t-20261028191840-s6429` 失败；累计 **6,064,755 / 11,521,178 ≈ 52.64%**（新完成 4,447,318 + 历史保留 1,617,437）。
- 以上都是**带日期的观测**，再查询前要取新快照。「已提交」只说明提交动作完成，不说明训练已开始，所以这一段不能当实时状态用。**易变事实（HEAD、权限位、pytest 摘要、产物 sha、最近几次提交）以仓库 `docs/STATUS.md` 为准**：它由 `tools/gen_status.py` 生成，`--check` 判是否过期、`--lint` 查正文硬编码。本技能正文不抄这些数字，抄了就会和生成结果对不上。

## 找到实际仓库

维护对象是 `nanless/redimnet2-fork`，地址 `https://github.com/nanless/redimnet2-fork`。已知服务器仓库为 `dev_a800_2gpus:/root/code/github_repos/redimnet2-fork`，每次仍按用户指定位置核对主机、仓库根、remote、branch、HEAD 和已有修改。开发机与实际八卡 worker 分开记录：一台上的进程或 GPU 说明不了另一台。

2026-11-05 只读核对：服务器该路径当时的 HEAD 是 `0a3f6dd`（2026-10-20）的 `docs/` 只有旧编号文件 <!-- volatile-ok: 2026-11-05 的历史快照，已标日期 -->，没有 `16-实验维护与验收边界.md`，也没有 `.agents/skills/`；带维护源和这份文档的是本机工作副本 `/Users/zhounan/Documents/声纹降噪/redimnet2-fork`（HEAD `605f560`）。读维护文档和技能源先看本机副本，服务器仓库按它实际有的内容读——技能里出现的文件名，服务器上不一定有。

本机“声纹降噪”资料根可能另有 `redimnet2-fork/` 副本。先跑只读的 `git rev-parse --show-toplevel` 并核对远端：不然 Git 会向父目录查找，写进另一个空仓库。历史临时工作树、别名或软链接不为迁就旧路径而恢复。

进入确认的仓库后，完整读根 `AGENTS.md` 和 `docs/16-实验维护与验收边界.md`。本技能的维护源是本机工作副本里的 `.agents/skills/redimnet2-reproduction-ops`，安装副本由维护者同步，安装目录里不单独发展唯一实现。2026-11-05 比对：两边各 28 个文件，27 个相同，只有 `references/model-storage-20261023.md` 安装副本多一段 2026-10-31 接续内容（安装副本较新）。改本技能时两处都要处理并复核差异，只改一边的话下次同步会把改动冲掉。仓库内文档和程序路径都以实际仓库根为基准，从安装技能目录用 `../` 猜回仓库会指错地方。

## 选择需要的参考

| 当前问题 | 先读 | 使用时注意 |
| --- | --- | --- |
| 所有任务进度、标注速度、PTN检查点或ETA | [任务状态与只读巡检](references/task-status-and-readonly-progress.md)、[训练巡检与恢复范围](references/monitoring-and-checkpoints.md) | 读有限的新快照；历史状态、更新尝试、真实优化步和质量分开 |
| B6结构、三语料配方、评测协议或环境父链 | [仓库与实验配方](references/repository-and-recipes.md) | 启动配置、已加载代码、磁盘HEAD和实际worker分别核对 |
| PTN恢复、model47受控接续、三段LM准备或故障取证 | [训练与接续](references/training-and-continuation.md)、[阶段接续与冻结身份](references/stage-gates-and-provenance.md) | 再读仓库 `tools/ptn_resume/README.md` 及 `INTERFACE.md`；旧单段入口不能改 parent 当跨段恢复用 |
| LM 从 115 冷启动、包与合同、提交与首屏自检 | [LM 受控启动](references/lm-launch-20261106.md) | 包/合同两个 sha 由外部注入；`new_run_root` 要指向不存在的目录；自动重试关掉 |
| FLAC定位、私有解码库或逐事件许可 | [FLAC定位回归](references/flac-seek-regressions.md) | 同一文件再次出错仍是独立事件；整段能读不证明随机seek可读 |
| 三训练域自动转写、字词时间、待核原因或结果发布 | [自动标注](references/open-annotation.md)、[三训练域Ultra实施记录](references/ultra-production-sidecar-20261025.md) | 当前范围仅实际VoxCeleb2、VoxBlink2、CN-Celeb2训练来源；旧混合批不作现行输入 |
| ASR计算提速、合批、保留旧结果或跨节点接续 | [转写提速与接续](references/asr-speed-and-continuation.md) | 先读末尾正式B8结果与查询口径，再查相关版本；旧坏候选、vLLM尝试和当前B8不是同一证据 |
| 官方骨干完整字自然变长微调、候选消费或长度交叉评测 | [完整字实验](references/complete-word-experiments.md)、[完整字词微调](references/word-complete-finetuning.md)、[双卡入口](references/word-gpu-admission.md) | 整数秒裁片仅是[旧诊断路线](references/duration-finetuning.md)，无法替代完整字输入 |
| 字界方法研究、18源实测或34源选型 | [方法研究](references/character-boundary-research.md)、[18源实测](references/multicase-boundary-audit.md)、[34源选型](references/boundary-selection-comparison-20261024.md) | ASR、VAD、强制对齐、原生/FIX时间及人工边界分别评价 |
| 修改、部署、测试、同步或Git交付 | [仓库与证据](references/repository-and-evidence.md) | 维护源、安装副本、冻结运行包和外部原件分别处理 |
| 模型迁移或原路径缺失 | [模型存储映射](references/model-storage-20261023.md) | 从当前 MODEL_LOCATIONS 和布局指针核实际状态，不据同名路径恢复大文件 |

各次旧任务的排队、失败和修复细节保留在[10月21巡检基线](references/20261021-audit-baseline.md)、[维护终止](references/ptn-maintenance-stop-and-resume-20261024.md)、[首次接续提交](references/ptn-resume-submission-20261024.md)、[spawn故障](references/ptn-resume-spawn-fix-20261024.md)、[普通脚本修复](references/ptn-resume-ordinary-entry-fix-20261024.md)和[10月25巡检](references/comprehensive-audit-20261025.md)。[早期全量计划](references/full-open-labeling-20261024.md)及[早期生产实施](references/production-sidecar-20261024.md)包含后来已收窄的混合范围。这几份说明当时发生了什么，不是今天的任务状态，也不是重跑许可。

## 运行和证据要求

1. 先核最新用户范围和真实任务 ID。下面这些各自单独报告，不互相顶替：平台 Running、日志推进、八个进程首次优化更新、完整轮次、检查点可恢复、模型质量。排队还没有实例时没有 run 目录，这不等于程序失败。
2. 用户要求不影响现训时，不停止、迁移、热改其进程、环境或冻结源码，也不在已有独占GPU上建立新的GPU运行上下文——这些都改了现训现场。0% 利用率、余量显存或已知 PID 都代替不了资源准入。另获授权的独立验证，仍要用自己的实际输入和资源证据。
3. 同阶段恢复要核完整模型、SGD、scaler、说话人映射和绝对时钟；缺随机数状态或轮内游标时，把要重做的范围写出来。PTN 转 LM、旧 35 轮接续及 47→48 是三条不同链路，按实际段记录处理；改旧SHA、删claim或清锁只是把失败盖住，不能算成功。
4. 自动标注保留来源、原答、原生与修补时间、未知和失败项。`candidate`、合法 JSON、模型一致或波形图都说明不了人工真值；VAD 有声段也不是完整字边界。未知身份不编 speaker，原训练、评测和诊断的保护不因此解除。
5. 正式全量附随标注生产只用已固定的本地开源模型，模型 API 调用为 0。有限正确性复核按当前或此前仍适用的明确授权，单列输入与模型观察；某个历史固定批的许可不能扩到新批，机器复核也不能写成人听。
6. CPU 合成、真实模型、GPU 工程、恢复、资源归还和人工审核分别留结果：这几个环节的通过标准不同，混在一起就分不清哪一步真的过了。带日期的 624、798、1077 等测试数属于对应源码；以后读本次实际汇总，不拿历史数量或预期用例数当通过证明。
7. 提交之后要跟到实际准入、worker、首更新和所需结果检查：平台收下任务只说明提交成功，说明不了它跑起来了。记录未知提交结果时先查重，不重复创建任务。文档维护、换工具或旧监控提示都不恢复已暂停的后台定时跟进；创建或恢复持久后台任务要有用户明确指示。

## 交付与 Git 约束

向读者先说明本次查到了什么、改了什么、证据支持到哪一步以及真实缺项，再给准确的任务、源码和文件入口。用入门研究生能理解的平实中文。公开仓库只放代码、可公开配置、合成夹具和汇总；业务录音、完整日志、诊断名单、模型及凭据留在本次已授权位置。用户要求报告与快照只留本机时，不因为服务器存储惯例另行上传。

新平台任务名用用户指定的 `zhounan_` 前缀，保存后回读实际名称。历史任务 ID、冻结包和提交时名称不随重命名改写。提交作者与提交者用该仓库身份 `nanless <francis7999@outlook.com>`。

自2026-10-28起，Mac 对 GitHub 只读：不推送，不经网页、API、gh 或自动同步上传任何文件、数据、提交对象或分支。这条限制的落地方式是保留禁推 `pushurl`；改 URL、remote、端口，或走代理、隧道换出口，都能把限制绕过去。已删除的远端 `codex/repository-audit-20261025` 不重建，也不为本任务重新扫描 26 个其他仓库的禁推设置。

用户明确允许的写入方式是把任务需要的仓库状态同步或拷贝至 `dev_a800_2gpus`，由服务器自己的仓库和进程连接 GitHub 执行已授权的提交、推送；Mac 可以通过 SSH 发起这些远程命令，但不能透传本机 git push 的数据流。Mac 本地编辑和 commit 仍可做。是否需要 Git 交付以当前任务授权为准，规则说明本身不表示已经部署或推送。
