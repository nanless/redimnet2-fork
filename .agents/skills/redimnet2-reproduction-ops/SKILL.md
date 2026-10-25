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
- CPU 合成检查、真实模型推理、GPU 恢复、资源归还和人工字界审核分别记录。旧 GPU 成功证据不能自动授予新提交相同结论。
- 最新一次全仓审查记录在仓库`docs/reviews/2026-10-26-全仓二次审查与任务故障核验.md`：维护代码549项CPU回归通过，但PTN48轮真实前向NaN仍未定位首坏算子；不能续用旧“八卡正常”结论，也不能把model47阴性重放当修复。实时状态需再次读取。
- 公开仓库不放业务音频、完整日志、真实诊断样本名单、模型和凭据。现场原件使用显式路径、摘要和用途绑定；缺失时保守拒绝或准确报告未验证。

## 按任务阅读

- 主训练、同阶段恢复、大间隔微调及跨段准备：读 [训练与接续](references/training-and-continuation.md)。
- 三训练域自动转写、字词时间、队列、原子发布与恢复：读 [自动标注](references/open-annotation.md)。
- 官方模型完整字自然变长训练、候选消费与长度交叉评测：读 [完整字实验](references/complete-word-experiments.md)。
- 修改、部署、历史证据与仓库同步：读 [版本与证据](references/repository-and-evidence.md)。

不要把 VAD 有声段当成字界，不把 `candidate` 当人工真值，不自动改变训练或正式评测资格。用户禁止影响现训时，只做有界只读巡检与独立目录中的准备；不能因准备器通过就启动训练。
