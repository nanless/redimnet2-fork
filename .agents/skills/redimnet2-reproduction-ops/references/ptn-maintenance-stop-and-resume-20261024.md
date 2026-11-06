# PTN维护终止与轮首恢复准备

本轮完整证据入口：`/Users/zhounan/Documents/声纹降噪/redimnet2八卡复现_20261019/PTN停更调查_20261024/PTN停更原因与安全恢复准备报告.md`。以下是2026-10-24带日期事实；请求实时情况时重新取证，不将此快照永久当当前。

## 已确认终止，不再沿旧Running/ETA

原task `t-20261019232824-8zqrl`经刷新平台详情确认已停止：worker于2026-10-23 19:42:59北京时间停止，任务时间线19:43:09“已终止：节点待下线，影响worker-0共1例实例”，操作人平台。资源组驱动更配目标535.261.03、进度5/15，旧Host9.2.24.83已承载其它8卡任务。维护申请发起、精确48小时政策起算、内核信号/退出码仍未知，不把缺日志当这些事实的证据。

完整共享日志最后指标为36/5525，原生CR进度为36/5535，stdout末mtime19:42:58.630。全stdout与01:31快照SHA相同；三份副本各七次FLAC不相加，未见第8次。检索无新增OOM/NCCL/非finite/fatal，也无PTN120或LM成功；原因依据平台时间线，不依据关键词没命中。

## 进程与锁的证据边界

旧worker WebShell仅Disconnect，未发送命令，未获得原容器逐PID/FD状态。开发机ps、原PID文件987、新容器同号PID均不能证明旧worker活性。空lock文件存在不表示持锁或已释放，不能删除文件过门。本轮未触原Host的新任务、未发送信号、未试持锁。实际恢复前仍要求控制面旧pod/writer终止证据、锁释放和逻辑实验互斥；平台终态不能自动替代全部writer核查。

## model35已CPU实装，恢复只能轮首

最新完整model35为650777251字节，SHA `fdaa88b963720ccdc35708c3ca9d18e56296edd8711eb9309772afacc92beb84`。受限CPU真实构建并strict装823 state，745模型浮点张量及584动量全有限；SGD动量同形且装后逐张量相等，AMP scale65536/growth_tracker1183等值。spk2id119274人/三倍速度扰动head357822×192、独立source95/189身份通过。未forward/backward/step/回写权重，CUDA未初始化。

对应下一轮36、零基绝对迭代525035、LR0.72、margin0.2。保持120总轮与15001批/轮，不写剩余85轮重置调度。缺完整RNG/loader/sampler/batch游标，只能重做36轮轮内约55—58分钟旧工作，不承诺逐批或逐位恢复。未来有更新checkpoint时重新审最新点，不机械固定35。

## 当前只准备，不能直接启动

用户本轮明确不重启、不迁移。`恢复准备_禁止执行/`仅本机候选YAML、精确diff、CPU语义验收及合同；无新task、远端run、launcher或部署。候选YAML仅exp_dir和checkpoint两项变化，原有效配置SHA `705100f8c5970cc7a22a9acb52b7d9d9de2c10aefc9f6d73000b4880d6b2d0b6`保留。独立审阅16项通过不是启动许可。

原平台fresh-zero入口不能重试、删run或尾加checkpoint。未来单独授权执行时，需要独立严格恢复入口，每rank完整模型+optimizer+scaler；原train的fresh fallback、head重建、丢动量均硬拒绝。新task/run/output与旧run不重叠，95/189与真实导入身份重验，新worker驱动/挂载/CUDA/NCCL及8rank、首个完整checkpoint/val真实验收。新进程使用私有1.2.2需显式记录及实worker生产裁剪验证，不热换旧环境。

跨段血缘为旧完整1…35→固定model35→新36…120，旧新CSV保留原件，外部按task/run/epoch汇总。现v4固定原run120及115/120，不能简单改parent；后续LM从逻辑115仅模型初始化、新优化器/scaler，跨段合同另审。七事件许可不延伸新PTN或LM。

## 同期无GPU准备

`/Users/zhounan/Documents/声纹降噪/ReDimNet2待办梳理_20261024/CPU准备_本轮/`保存最终v3源绑定真实输入预检：M1/M2 W2接受、W8因4family不足16拒绝；35项前后SHA/stat相同、四输出路径未创建。未调用launcher/probe/torchrun/CUDA，不将银标或结构预检升级正式数据。首轮过严审计挡住SoundFile的ldconfig查询和/dev/null，后继仅精确允许这两个只读运行依赖，保留失败原文。

独立收据比较器和采集合同只准备未来梯度/BN/恢复/峰值验收；synthetic测试不当真卡证据，也不表示v3已接入采集。正式一小时watchdog、验证/选模/LR策略缺口仍需独立实现验收。旧停止调查与新变长线准备分别保存，不以新线八卡权限代替本次旧PTN恢复执行权限。
