# 2026-10-21 巡检基线（历史快照）

本页是一份历史快照，只保存已核实的定位信息和当时的关键结论，不是实时状态；要判断现在的情况，需重新取证。本机完整证据：

- `/Users/zhounan/Documents/声纹降噪/redimnet2八卡复现_20261019/训练进度检查_20261021_0123/训练情况与进度检查.md`
- 同目录 `最终进度摘要.json`、`云容器实时核查记录.txt`、`checkpoint_audit.json`、`progress_analysis.json`、`anomalies_summary.json`。
- 项目当前入口：`/Users/zhounan/Documents/声纹降噪/redimnet2八卡复现_20261019/执行记录.md`。
- 原聊天与34项技能理解：`/Users/zhounan/Documents/声纹降噪/理解笔记_20261021/RedimNetV2聊天与自创技能理解总览.md`；本轮新增本技能后，清单是35项，旧的34项扫描记录按历史含义保留。

任务 `t-20261019232824-8zqrl`，run root `/root/experiments/redimnet2/b6-mix-official-20261019/formal-main-7afb821-zero-20261019-2327`（历史路径），从零开始、没有恢复，启动commit7afb821。北京时间01:39:58，PTN跑到第10轮13,075/15,001批，累计148,084/1,800,120，占8.22634%。01:36三次实际worker的八卡占用在93%—100%，每卡约53.7/80GiB。开发机hostname `di-20260825114741-2bbpx`；实际worker `t-20261019232824-8zqrl-worker-0`。

第1—9轮用固定8秒Vox1-O测得的EER依次是4.208、2.661、2.624、2.416、2.179、2.093、2.213、2.280、2.596%。最低的是model6，最新完整点是model9；6→9回升0.503个百分点。9份检查点各650,775,840bytes，model/optimizer/scaler/epoch完整、浮点值有限。这批记录里没有批次级恢复。

cloud stdout八rank各5,859条冻结打印指标全部有限。epoch1读音频时出过一次异常，捕获后训练继续；已见的记录里没有OOM/NCCL等致命故障。9,600条short_audio是每worker/每原因限频前5例，只能说明这些例子，算不出整体drop率。

01:23:44的固定ETA基线给出PTN剩余283—286小时，预计11月1日晚间完成，不含LM、导出与最终评测。当时同时成立的是两件事：主训练正常推进，后续门禁仍不可用。

在云容器里实际核对，四个旧环境路径都不存在：`verification/volcengine/redimnet2-b6mix-20261019-v2/`下source manifest和smoke证明，`verification/volcengine/redimnet2-b6mix-20261019/`下data audit与官方参考权重。这些文件因历史目录迁移另有规范位置，但进程里的变量没有跟着动，路径照样指空。RUN_ROOT存在，因此不能写成训练数据源丢失。

冻结95文件manifest自身的SHA `5160f99e33caccf373563b6bc695d10f304044b780d10b5c40f6a4b0fab102d9`未变；92项匹配，3项差异，分别是train.py（恢复检查）、processor.py（旧字符串增强分支）、redimnet2.py（注释）。目前训练走PTN分支，没有确定计算变化的证据；下一段LM会重新verify_source。只修路径仍会撞上这3项SHA差异。这个问题本轮只记录，尚未修复。

历史artifact路径通过 `/root/codex_artifacts/CURRENT_LAYOUT.json`取维护入口和映射（历史根；2026-11-09 起新产物按仓库 `AGENTS.md` 顶部《大数据落点硬规则》写 `/root/group-shared/voiceprint/artifacts/redimnet2/<用途>/`），读到过的revision不固定为未来值。2026-10-21检查时的基目录是 `/root/codex_artifacts/07_维护与迁移/06_环境模型及训练提交核验/火山平台声纹训练与评测的提交核验_volcengine/`（历史路径）。使用前重新读取映射；已失效的旧路径不再依赖，也不擅自建兼容链接——按旧路径走会落到错误位置。
