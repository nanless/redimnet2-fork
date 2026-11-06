# 2026-10-21 巡检基线（历史快照）

本页只保存已核实的定位信息与关键结论，**不是实时状态**。本机完整证据：

- `/Users/zhounan/Documents/声纹降噪/redimnet2八卡复现_20261019/训练进度检查_20261021_0123/训练情况与进度检查.md`
- 同目录 `最终进度摘要.json`、`云容器实时核查记录.txt`、`checkpoint_audit.json`、`progress_analysis.json`、`anomalies_summary.json`。
- 项目当前入口：`/Users/zhounan/Documents/声纹降噪/redimnet2八卡复现_20261019/执行记录.md`。
- 原聊天与34项技能理解：`/Users/zhounan/Documents/声纹降噪/理解笔记_20261021/RedimNetV2聊天与自创技能理解总览.md`；本轮新增本技能后核心清单35项，旧34项扫描记录保留历史含义。

任务 `t-20261019232824-8zqrl`，run root `/root/experiments/redimnet2/b6-mix-official-20261019/formal-main-7afb821-zero-20261019-2327`，从零、无恢复，启动commit7afb821。01:39:58北京时间PTN第10轮13,075/15,001批，累计148,084/1,800,120=8.22634%。01:36三次实际worker八卡93%—100%、每卡约53.7/80GiB。开发机hostname `di-20260825114741-2bbpx`；实际worker `t-20261019232824-8zqrl-worker-0`。

第1—9轮固定8秒Vox1-O EER依次4.208、2.661、2.624、2.416、2.179、2.093、2.213、2.280、2.596%。最低model6，最新完整model9；6→9回升0.503个百分点。9份检查点各650,775,840bytes，model/optimizer/scaler/epoch完整、浮点有限。无批次级恢复。

cloud stdout八rank各5,859条冻结打印指标全部有限。一次epoch1读音频异常被捕获后继续；无已见OOM/NCCL等致命故障。9,600条short_audio是每worker/每原因限频前5例，不能计算全部drop率。

01:23:44固定ETA基线：PTN剩余283—286小时，预计11月1日晚间；不含LM、导出与最终评测。当前主训练正常推进与后续门禁不可用同时成立。

实际云容器核实四个旧环境路径均不存在：`verification/volcengine/redimnet2-b6mix-20261019-v2/`下source manifest和smoke证明、`verification/volcengine/redimnet2-b6mix-20261019/`下data audit与官方参考权重。文件因历史目录迁移另有规范位置，但进程变量未自动跟随。RUN_ROOT存在，不能写训练数据源丢失。

冻结95文件manifest自身SHA `5160f99e33caccf373563b6bc695d10f304044b780d10b5c40f6a4b0fab102d9` 未变；92项匹配、3项差异为train.py（恢复检查）、processor.py（旧字符串增强分支）、redimnet2.py（注释）。当前已走PTN分支无确定计算变化证据，下一LM会重新verify_source；仅修路径仍会遇3项SHA差异。此问题**本轮只记录，尚未修复**。

历史artifact路径通过 `/root/codex_artifacts/CURRENT_LAYOUT.json` 获取最新维护入口和映射，不把曾读revision固定为未来值。2026-10-21检查时新基目录为 `/root/codex_artifacts/07_维护与迁移/06_环境模型及训练提交核验/火山平台声纹训练与评测的提交核验_volcengine/`。使用前重新读取映射，不依赖已失效旧路径或擅自建立兼容链接。
