# 产物位置与统一根

2026-11-09 起，服务器上保存的新材料一律落在 `/root/group-shared/voiceprint` 下，`/root` 只放代码仓库与临时日志。仓库 `redimnet2-fork` 的 `AGENTS.md` 顶部《大数据落点硬规则》写着原始规定与自检命令；本页只说产物位置怎么落。

## 三类落点

| 数据类型 | 落点 | 用途子目录示例 |
|---|---|---|
| 工具产物、准入材料、账本、评测原件 | `/root/group-shared/voiceprint/artifacts/redimnet2/<用途>/` | `volcengine-submission/<task_id_or_run_id>/`（本技能的表单证据、日志截取、诊断 JSON、ETA 报告）、`controls/`（准入与合同）、`retired-claims-<日期>/`（账本归档）、`device-locks/`（设备锁）、`repo-handover-<日期>/`（交接脚本）、`eval-official-<日期>/`（评测原件） |
| 实验产物：训练 run、检查点、日志、评测输入与导出、批量推理输出 | `/root/group-shared/voiceprint/experiments/redimnet2/<实验名>/` | 具体子目录由 launcher/TOML 的 `EXP_DIR` 或输出根决定 |
| 开源标注侧车：逐音频标签、分片、租约、队列回执 | `/root/group-shared/voiceprint/open_annotation/redimnet2-silver-<日期>/` | 按 handoff 与 run 分层，如 `handoff-v4r2-20261108-v6/run-v6/` |

材料放在哪里，也要看当前会话的明确选择：用户选择“仅保留本机”时，报告与日志快照留在本机任务目录，不上传；这次选择不变成以后所有任务都本地保存的规矩。训练输出始终由实际 launcher/TOML 和共享挂载决定，本页不改实验合同。

**不先写 `/root` 再 `mv`。** 训练容器里 `/root` 是容器本地盘，`/shared/group-shared`（容器内通常另挂成 `/root/group-shared`）是 vePFS，两者是不同挂载点。跨挂载点 `mv` 退化为“复制 + 删除”：几百 GB 的检查点先占双份空间，复制慢，中途失败还会留下半份文件，看起来像已经移动。容器里 group-shared 是只读挂载时，按平台工作流调成读写再提交，不把输出改回 `/root` 本地盘。

## 旧根与历史产物

`/root/codex_artifacts/` 与 `/root/experiments/` 是 2026-11-09 之前的旧根（历史路径），不再接收新产物。旧根里的 `CURRENT_LAYOUT.json` 仍可用于解析历史路径与旧迁移映射；读到过的 revision 不固定为未来值，用前重读。

主机侧的可连接性随时间变化，2026-11-09 复查：`dev_a800_2gpus` 可连，主机名仍是 `di-20260825114741-2bbpx`，`/root/codex_artifacts/CURRENT_LAYOUT.json` 存在（13,256 B，2026-10-29 修改，历史路径）；`dev_a800_4gpus`（`10.43.6.242:2222`）这次仍然 SSH 超时。一次超时推不出机器已关闭、工作已停止，也推不出两个别名共用磁盘。主机由用户指定时，先核验 hostname、挂载、realpath 和当前写入者；历史记录里的关闭说明替代不了这次请求的实测。

历史执行版本、协议哈希和记录中的原路径保留下来作为 provenance；当前执行、resume、报告导航和资源登记要用新根的真实路径，不靠旧路径别名维持运行。旧位置不留副本，也不留软链别名；复制或归档成功只说明文件抄过去了，写成“已完成迁移”会掩盖还指向旧位置的引用；两套执行树并存时分别启动任务，会得到重复运行。哪些已完成、哪些待处理，读最终迁移记录区分。迁移不包含重跑训练或推理，也不改变统计口径。

原始数据、第三方源码和模型 fork、权重、Conda 环境，以及作者归属尚未确认的正式训练输出，都不因为目录名相近就算作独立产物。

本轮的正式训练工作区和日志留在原位，通过 `/root/workspace` 和 `se_train_logs` 索引；有这两个入口不代表某个模型实验就在该目录。路径以实际任务/registry/TOML 为准；迁移共享输出时，动作范围只到文件位置；训练或推理是否重启是另一件事，需要另外的授权。
