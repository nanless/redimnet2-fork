# PTN 普通脚本接续与数据 worker 语义（2026-10-24）

本页记录10月24日普通脚本的后继及当时的验收；文中的“当前有效任务”“model36待产”只适用于那一天。后续model47接续读[训练与接续](training-and-continuation.md)，最近带日期的进度读[任务状态](task-status-and-readonly-progress.md)；[10月25巡检](comprehensive-audit-20261025.md)也作为历史保留。

当前有效任务 `t-20261024154424-vggvm`（平台名v4；bundle/run v3），15:44:24北京创建。最终包17个文件，SHA `840235e7b5963ada274531a29401b9f2bd75fe92e985a8e8e560341ac48350d8`；两次独立部署回读和真实现场验收都已完成。P6/单worker八A800/dev_zhounan:4.0/720h/retryoff及原两挂载保持。后续先刷新真实证据，不机械沿用这份快照。

本轮用户明确要求检查失败的 PTN 接续并重新运行。首次的错误和后续的发现分别保留，不用重算旧冻结清单来消除问题。完整报告及原件在本机 `redimnet2八卡复现_20261019/PTN接续故障修复_20261024/`。

失败任务 `t-20261024115909-mhczc` 的八 rank 已经恢复 model35，但 Dataset 实例被附上了局部的 checked_epoch，DataLoader spawn 无法 pickle，在第0批退出。父 run_epoch 守卫修好之后，诊断后继 `t-20261024151842-wb58x` 于15:26:24八rank真实更新，25→150持续推进。这证明越过了崩溃，不证明配方采样等价。

进一步核验发现，新接续包装额外用了 torchrun `--run-path --start-method=spawn`，而旧正式训练链是普通的 train.py 脚本。冻结的 DistributedSampler.update 在 child dist 未初始化时置rank0/world1，并且不读环境变量；seed_worker虽按env设随机种子，却修不了源集合分片。CPU真实双rank/gloo/两worker、96唯一key、36/37两轮对照已经实证：普通脚本fork/dist true，各48、交集0/并集96；run-path spawn/dist false，均0/1、各96、交集96/并集96。这个CPU源key交集不能冒称云上每batch增强波形重复率或旧35轮实测。

最终的后继应恢复普通 worker.py 脚本入口，移除run-path/显式start-method；每轮的完整性守卫留在父run_epoch边界，不碰冻结Dataset/seed_worker/sampler/augs。worker底部的record(main)保留普通脚本异常的rank error.json。修好pickle能跑，不等于完整验收；还要检查启动方式如何影响真实数据子进程上下文和继承的legacy增强全局。

新增的模块级worker init先按原seed设恰好一次，再只读核get_worker_info、dist rank/world、fork、继承的sampler rank/world/partition/epoch和legacyTrue；不做worker collective/GPU操作，也不额外调用RNG。父AuditedLoader每轮收齐24本地原件，并跨8父rank共同核192份，才首次yield给原executor。原件带PID/ppid/start_ticks/nonce/bundleSHA；gate引用完整SHA，独立验收不能只数文件或看布尔。CPU2rank×2worker两轮已验key顺序与普通基线逐条一致，以及真实epoch/legacy状态。正式8×24收据仍须云端实验收。

后继bundle/run是独立v3，仍用原formal-main model35的SHA fdaa88b963720ccdc35708c3ca9d18e56296edd8711eb9309772afacc92beb84，不选待审的wb58x checkpoint。总120轮、起轮36/iter525035、15001批、8×96、24workers、LR.72/margin.2、全模型/SGD/AMP及私有decoder合同都不变；也不声称恢复了完整RNG/轮内游标。

切换门有三步：先提交正确的P6新任务并读回唯一ID/入口/资源，再只停止本轮自建的wb58x，等平台task与worker Terminated/active0之后发布一个小型外置的root-owned终态receipt。固定路径 `/root/experiments/redimnet2/b6-mix-official-20261019/ptn-resume-control-20261024/predecessor-wb58x-terminal.json`，schema ptn-predecessor-terminal/v1、完整task/run/唯一worker0/UTC新鲜≤1h；root700目录/600 regular，临时fsync后原子不覆盖发布。缺receipt最多等1800秒，其间不启动torchrun/GPU，也不伪造终态；在原子逻辑锁与旧锁EX准入下复读SHA/stat，活的孤儿SH仍然阻止启动。外置receipt的内容被采纳后，SHA存入parent_admission，不改冻结manifest。15项门禁的正负例已在CPU通过。

最新提交、终态和云端192worker/8restore/8update/持续批次验收，都在本机主报告与状态JSON里按实际时间记录；只有这些真实凭证通过，才称最终恢复。新model36/验证指标要等轮末真正完成后再另验，不把初始化或GPU吞吐当成新检查点。其他训练不干预，PAUSED heartbeat不恢复，报告和日志快照仍留在本机保存。

## 最终现场验收（15:51北京）

先核新task保存的入口逐字一致，再精确停止本轮诊断的前驱wb58x。15:45:35真实task/worker终止，实际凭证观测到的UTC是 `2026-10-24T07:46:59.351Z`，root600原子发布SHA `3b3a17d495ddd0d04cc578eec8f7e6adc7c07292cc8f9ce8dd424d7ba48fb193`。parent_admission实际采纳了该原文SHA，锁准入通过。发布器首次因默认python命令缺失而失败，原stderr保留；改成既有的绝对解释器后成功，包没有修改。

15:49:38，8restore、192个实际worker（唯一PID/8父分组/各rank nonce）和8gate的1536条原文SHA/PID/ppid/nonce引用，逐条通过独立复核；实际fork/dist rank-world8/sampler/epoch36/legacyTrue都正确。15:49:59.6676—.6689八rank完成首个真实优化步；15:50:57持久日志25→50→75、实时95/15001，八rank共24项loss均有限，LR.72/margin.2。15:51:23最终容器GPU93—100%，每卡约54GiB，host9.3.59.239/VPC10.43.6.66；完整UUID/PID/截图保存在本机。

普通script的torchrun各rank可能有独立session，entry自己的PG扫描不能冒称直接覆盖所有rank。正常退出/失败先由torchrun的SubprocessHandler对各rank组关闭；遗留的rank/数据worker持有的SH仍会阻止后继EX，最后才由平台整worker终态收束。拿掉锁或扩大kill范围都取代不了真实清理。这是边界说明，不是新增的停止授权。

最终状态是：恢复和当前轮的启动验收通过，训练在持续；还没有新的完整model36/本轮正式验证，不宣称已到120或性能有改善。下一轮worker重新创建仍由同一个门逐轮执行。完整本机报告的绝对路径：`/Users/zhounan/Documents/声纹降噪/redimnet2八卡复现_20261019/PTN接续故障修复_20261024/PTN接续故障修复报告.md`。
