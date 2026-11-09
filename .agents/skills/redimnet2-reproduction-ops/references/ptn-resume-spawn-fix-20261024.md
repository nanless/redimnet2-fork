# PTN model35 接续 spawn 故障与修复（2026-10-24）

> 最新更正：本页记录第一次闭包修复及wb58x启动历史。后续CPU双rank实证run-path/spawn使分片前提失效，wb58x仅作诊断证据且已停止；最终普通入口后继t-20261024154424-vggvm与真实192-worker验收见[普通脚本接续](ptn-resume-ordinary-entry-fix-20261024.md)。


本页适用于调查 t-20261024115909-mhczc 的首批前失败及其后继。用户明确要求“检查哪里有问题，把它重新跑起来”，授权了一次独立修复重提；旧的“只准备”限制不能沿用到这次已经授权的操作上。

## 已证实根因与测试缺口

失败任务12:43:33启动、12:45:10失败，新鲜平台worker终态是Failed。八rank都已严格恢复model35的823状态/完整分类头、SGD584动量与AMP，但停在0/15001，没有first_update，也没有新checkpoint。八个error.json都报 `Can't pickle local object main.<locals>.strict_loop.<locals>.checked_epoch`。

接续包v1把局部checked_epoch赋给dataset实例的set_epoch，DataLoader spawn在序列化对象图时失败。这是接续包装新引入的错误，不归咎冻结trainer、FLAC、NCCL或checkpoint。旧CPU模拟只走到训练循环，没有覆盖真实spawn取batch；以后同类修改应对真实Dataset/Processor对象图和实际spawn进程取批做有界回归，只mock DataLoader或只测pickle函数调用都不够。

## 独立修复及验收

新包 `/root/experiments/redimnet2/b6-mix-official-20261019/ptn-resume-code-model35-20261024-v2`（历史路径），15个文件、manifest锁14个成员，SHA `70fb366afb2535efdb3580e3bc8fed440b5e3ae094a6a33b884b7f4be144f364`。文件权限444/目录555。新run在同一父目录下 `ptn-resume-model35-20261024-v2`；旧v1包/失败run/原冻结源/清单都不改。

守卫放在父rank的train.run_epoch边界，用模块级guard+partial包装并在finally里还原，在set_epoch之后、取批/前向之前核bundle/source/model35；不给Dataset实例附局部函数。正式24workers、spawn、8×96、15001批/轮、120总轮不变。YAML只改exp_dir；model35原SHA fdaa88b963720ccdc35708c3ca9d18e56296edd8711eb9309772afacc92beb84保持。起轮36、绝对iter525035、LR.72/margin.2；没有完整RNG/轮内游标，不声称逐位连续。

已验证：冻结的真实raw_fast完整增强链与正式spk2id/训练表首96行，CPU spawn1worker在36/37轮各取96条有限音频，guard命中两次；独立2worker的真实spawn对象图保持基线顺序、旧闭包负例可复现、source/model35摘要拒绝在consumer之前。新包的CPU preflight在完整恢复/32解码窗/两终态门/原子旧锁通过，CUDA/DDP/torchrun/forward/update false。独立远端15文件逐SHA回读和只读权限通过。两次CPU夹具缺RANK或NUM_DATALOADER_WORKERS的异常保留，补齐正式seed_worker环境后再复验，不改生产源来掩盖。

新入口除了原task终态，还精确SHA绑定这次失败task/worker Failed/active0的新鲜凭证。实际启动仍以原子逻辑锁和原run锁准入、各rank独立SH持有作保护；只凭空的锁文件或开发机ps推断云writer退出不成立。终态凭证的24h门保留；长排队超过时效时，先核新鲜终态，再另审独立包，不现场改冻结manifest。

## 已提交后继与当前边界

唯一的新task是 `t-20261024151842-wb58x`，名 `redimnet2-b6mix-ptn-resume35-8a800-p6-20261024-v3`，15:18:42北京创建、15:19:48启动，host9.3.158.130/VPC10.43.6.64。P6、单worker8A800、PyTorchDDP、dev_zhounan:4.0、720h、重试off；共享RO→/shared/group-shared、个人RW→/root。创建前后的完整入口逐字一致，并含末行END_VERIFIED_ENTRY；提交结果未知时先查重，不重复提交。

15:21:23只读共享核验：8/8严格restore真实通过，还没有first_update/新checkpoint，也没有rank error。要等8个真实优化步和批次数递增都核过，才能称训练恢复；平台Running或CPU通过替代不了这些。首次完整model36/新验证要真正写完之后再报告；PTN120不自动调用固定旧run的LM入口。PAUSED的redimnet2 automation保持不变。

本机完整报告：`/Users/zhounan/Documents/声纹降噪/redimnet2八卡复现_20261019/PTN接续故障修复_20261024/PTN接续故障修复报告.md`；52份失败原件、CPU异常/测试/独立审阅、部署回读和平台证据都在同目录。报告和日志快照留在本机，不因为必要的代码部署而扩大同步范围。看最新状态先刷新实际平台和run证据，不机械沿用这个历史时间点。
