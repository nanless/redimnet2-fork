# A800模型存储与本任务映射

本页维护更新于2026-10-24。下列业务映射来自2026-10-23的验证，不是今天重跑GPU基准得到的结果。

公共位置、原生缓存配置、hardlink写前物化、328项精确退役及同字节MFA重建规则，统一读 [A800模型存储规则](../../a800-storage-maintenance/references/model-storage.md)；已完成的批次和恢复记录读 [当前状态](../../a800-storage-maintenance/references/current-state.md)。整机盘点、下载模型共享及新批次精确缓存清理用 `$a800-storage-maintenance`；环境退役和业务推理/训练仍按各自的Skill。

本规则只适用于核实过的dev_a800_2gpus。canonical目标从服务器当前MODEL_LOCATIONS.json的逐项状态里选；旧的file alias和历史下载恢复项不能都当成必须存在的现用载荷，否则会把已退役的目标算成缺失。原合同/模型配置/源码SHA保留历史事实；不同来源或revision的同名文件不能互换。

## 本任务直接涉及的加载单元

以下是定位映射；可用状态以服务器 registry 的逐项状态为准。

| 原路径 | 共享模型路径（相对 namespace） |
|---|---|
| `/root/code/github_repos/redimnet2-fork/checkpoints/pretrained` | `code/github_repos/redimnet2-fork/checkpoints/pretrained` |

## 当前存储接续（2026-10-31）

本页的业务路径表及10月23的验证保持原事实。当前维护先读[A800当前状态](../../a800-storage-maintenance/references/current-state.md)与[共享/物化规则](../../a800-storage-maintenance/references/model-storage.md)。跨批只读合查用namespace的tools/model_storage_verify_all.py；--new-only只跳过legacy，全部新增登记和overlay仍在检查范围内。10月31的末次验证有179条共享路径通过，原26条legacy差异及其次数保持不变。

后续两个精确scope和73HF的正常发布/复制回退/alias保护见[三项执行](../../a800-storage-maintenance/references/selected-three-20261030.md)，原91组与69组/175路径计划不追加改写。维护用新的独立worker；已经导入的旧函数不热更新，热更会让运行中的worker读到未验收的代码。ModelScope及任意非HF的原位写，先显式物化或写新revision。

PEFrame535的分段、pip/npm缓存退役及单件4MiB恢复见[10月31批次](../../a800-storage-maintenance/references/safe-caches-20261031.md)。配置、来源/revision及现用安装保持；Qwen219和未知残留保留。缓存重建要花网络和时间，净量与恢复依据以各批的封存记录为准，旧的收益不再重报。
