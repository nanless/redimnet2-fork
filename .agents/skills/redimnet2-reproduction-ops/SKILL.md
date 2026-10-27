---
name: redimnet2-reproduction-ops
description: Inspect and maintain this ReDimNet2 repository, B6 PTN/LM continuity, open-model sidecar annotation, complete-word variable-length fine-tuning and protocol-matched evaluation. Use for this reproduction project; distinguish repository tests from external GPU evidence.
---

# ReDimNet2 复现维护

本技能的维护源在本仓库。已安装副本由维护者从仓库同步；不在安装目录单独发展唯一实现。先读仓库根 `AGENTS.md` 和 `docs/16-实验维护与验收边界.md`，按入门研究生能理解的中文汇报。

仓库身份为 `nanless/redimnet2-fork`，地址为 `https://github.com/nanless/redimnet2-fork`。先核 `git rev-parse --show-toplevel` 与远端身份；若当前位于“声纹降噪”资料目录，应检查其中的 `redimnet2-fork/` 副本。资料根目录未提交、没有远端的空Git不是本项目维护仓库，禁止让Git命令向父目录查找后误写另一暂存区。

## 先确定工作范围

- 询问实时状态时读取新鲜任务、进程、日志和检查点；带日期的旧报告只证明当时情况。平台排队且尚无实例时，没有运行目录不等于程序失败。
- 检查代码或文档时在仓库内维护，不修改正在运行任务的冻结源码或清单。新增提交作者与提交者使用仓库本地身份 `nanless <francis7999@outlook.com>`，中文说明问题、修复、验证及未完成项；提交与推送由本次请求的授权决定。
- 所有未来平台任务名必须以 `zhounan_` 开头，这是用户10月27日明确要求。标注准入/平台准备和PTN构建收据已用统一前缀检查；平台提交后回读实际名称。已有ReDimNet2任务可用“更多→编辑”原地修改名称，保留ID和运行配置。历史报告记录提交时名称与改名，不改冻结包、task ID或worker ID。
- CPU 合成检查、真实模型推理、GPU 恢复、资源归还和人工字界审核分别记录。旧 GPU 成功证据不能自动授予新提交相同结论。
- 历史审查见仓库`docs/reviews/2026-10-26-全仓二次审查与任务故障核验.md`，本次接续见`docs/reviews/2026-10-26-完整取证接续与标注对照执行记录.md`，实际状态以带源码身份的新记录为准。本轮619项检查曾因真实包导入错误未通过；修后624项已实际无跳过通过。上次新增退出信号、二进制检查和NFKC配组反例后的633项曾实际无跳过通过；此前新增终态刷新、排队等待与现场只读工具后651项实际无跳过通过；10月27日转写工具集成后713项实际无跳过通过（GPU按用户最新要求未测）；以后改动仍以最新实际汇总为准；不能删除旧失败或仅靠预期用例数宣布通过。model47受控接续、双卡工程成功和原故障根因定位是不同结论。
- 10月26日启动时效及接续巡检见仓库内 `docs/reviews/2026-10-26-PTN启动时效修复与八卡接续巡检.md`。jvhrd是在CUDA前等待新鲜终态材料超时，不是再次NaN；唯一后继ctjsk已实证八rank首次SGD，并在真实云容器核8个rank及192个worker存活；截至该次观测，完整一轮及新检查点仍待验；之后须重新查询实际状态。提交后必须继续核实际准入和首更新，不能停在排队状态直到材料过期。文档和工具维护不授权重新启用已暂停的后台巡检。
- 公开仓库不放业务音频、完整日志、真实诊断样本名单、模型和凭据。现场原件使用显式路径、摘要和用途绑定；缺失时保守拒绝或准确报告未验证。

## 按任务阅读

- 主训练、model47受控接续、完整故障取证、大间隔微调及三段准备：读 [训练与接续](references/training-and-continuation.md)；具体接口读取已定位仓库内的 `tools/ptn_resume/README.md`。
- 三训练域自动转写、字词时间、队列、发布/恢复、吞吐定位与独立提速验证、十进制时钟和撇号诊断：读 [自动标注](references/open-annotation.md)。
- 转写计算提速、阶段计时、有限对照、八卡显存预算与跨合同去重接续：读 [转写提速与接续](references/asr-speed-and-continuation.md)。
- 联合覆盖/唯一消费者、跨引擎审计、vLLM最小适配与整批恢复也由上述参考路由；实现见仓库`docs/reviews/2026-10-27-联合接续与跨引擎CPU实现验收.md`。本轮798项CPU单元测试实际无跳过通过，真实GPU、速度和人工字界未测。新数据门不授运行资格，不能修改原W2或固定总分母来接后继。
- 八卡 WebShell 实测失败与同合同恢复：先读仓库 `docs/reviews/2026-10-27-WebShell真实试验故障与标注恢复.md` 和转写提速参考。指定机器、已知进程和空余显存不覆盖旧父进程的持续独占监控；不得同卡另建 CUDA context。当前 `vllm_live_probe`真实构造已关闭，只保留检查及完整缓存恢复；生命周期清理未实现，不写成 GPU 验收通过。严格恢复保留旧 stat 的 device 字段，实际新节点不符即拒绝，不改旧冻结包或标签。
- 跨节点的新生产者接续：读仓库 `tools/open_annotation/跨节点原音内容证明与剩余输入.md`、`联合覆盖与唯一读取.md`和`跨节点标注接续运行.md`。显式新内容协议保留完整新旧stat，只允许device不同并实际核编码SHA；新旧生产者独立、旧已发布侧车全部排除。`freeze_legacy_successor.py`保留实际旧23文件/13计算模块，仅换合同version，不能用当前14模块冻结器悄悄升级旧计算方式。数据准备证明与当前云FD租约分别核，旧语义锁和旧队列父锁贯穿新GPU。最新真实准备、提交和运行结论以 `docs/reviews/2026-10-27-跨节点标注恢复实测.md` 为准；CPU通过不是八卡恢复成功。
- 剩余索引的角色计数与恢复挂载：另读仓库`tools/open_annotation/剩余角色元数据定稿.md`。`role_union_counts`须是当前剩余来源的真实统计，PTN与LM有重叠；父级总数单独保存。已完成的v1不能原位更正，使用`finalize_remaining_roles.py`发布同目录新索引、新计划和收据，分片原字节不改。原fresh-launch挂载脚本要求旧run不存在，不能直接拿来恢复；新入口保留旧W2命令与原挂载脚本身份，改由`verify_portable_successor_mounts.py`核新run和实际原RW/RO挂载。新queue、labels和执行证据不得写入旧冻结源码、输入、控制或锁声明目录。
- 长时间排队后的新鲜证明交接：读仓库`tools/open_annotation/排队后新鲜证明交接.md`。可选外层固定蓝图，只在实际新实例内建立自身PID/start/boot/nonce请求，由操作者据真实平台新观察发布唯一回复；仅补`current_platform_observation`，1800秒单次等待，原一小时时效与内层全部门不变。外层两源与原内层21源分别封存，不能改旧包。实例此时已分配并可能计费，CPU等待不等于免费或未占配额；请求、发布或intent不证明八卡推理成功。不自动查询平台、重启任务或恢复暂停的后台跟进。
- 官方模型完整字自然变长训练、候选消费与长度交叉评测：读 [完整字实验](references/complete-word-experiments.md)。
- 修改、部署、历史证据与仓库同步：读 [版本与证据](references/repository-and-evidence.md)。

不要把 VAD 有声段当成字界，不把 `candidate` 当人工真值，不自动改变训练或正式评测资格。用户禁止影响现训时，不停止或热改其进程、环境及冻结代码，不占用已有独占GPU。用户另行明确授权的独立验证或提交可在独立分配的资源执行，仍须当前数据和资源准入；不能仅因CPU准备通过就启动模型或训练。
