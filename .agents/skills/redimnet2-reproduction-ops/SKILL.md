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
- 历史审查见仓库`docs/reviews/2026-10-26-全仓二次审查与任务故障核验.md`，本次接续见`docs/reviews/2026-10-26-完整取证接续与标注对照执行记录.md`，实际状态以带源码身份的新记录为准。本轮619项检查曾因真实包导入错误未通过；修后624项已实际无跳过通过。上次新增退出信号、二进制检查和NFKC配组反例后的633项曾实际无跳过通过；本轮新增终态刷新、排队等待与现场只读工具后651项实际无跳过通过；以后改动仍以最新实际汇总为准；不能删除旧失败或仅靠预期用例数宣布通过。model47受控接续、双卡工程成功和原故障根因定位是不同结论。
- 最新启动时效及接续巡检见仓库内 `docs/reviews/2026-10-26-PTN启动时效修复与八卡接续巡检.md`。jvhrd是在CUDA前等待新鲜终态材料超时，不是再次NaN；唯一后继ctjsk已实证八rank首次SGD，并在真实云容器核8个rank及192个worker存活；完整一轮及新检查点仍待验。提交后必须继续核实际准入和首更新，不能停在排队状态直到材料过期。文档和工具维护不授权重新启用已暂停的后台巡检。
- 公开仓库不放业务音频、完整日志、真实诊断样本名单、模型和凭据。现场原件使用显式路径、摘要和用途绑定；缺失时保守拒绝或准确报告未验证。

## 按任务阅读

- 主训练、model47受控接续、完整故障取证、大间隔微调及三段准备：读 [训练与接续](references/training-and-continuation.md)；具体接口读取已定位仓库内的 `tools/ptn_resume/README.md`。
- 三训练域自动转写、字词时间、队列、发布/恢复、十进制时钟和撇号诊断：读 [自动标注](references/open-annotation.md)。
- 官方模型完整字自然变长训练、候选消费与长度交叉评测：读 [完整字实验](references/complete-word-experiments.md)。
- 修改、部署、历史证据与仓库同步：读 [版本与证据](references/repository-and-evidence.md)。

不要把 VAD 有声段当成字界，不把 `candidate` 当人工真值，不自动改变训练或正式评测资格。用户禁止影响现训时，只做有界只读巡检与独立目录中的准备；不能因准备器通过就启动训练。
