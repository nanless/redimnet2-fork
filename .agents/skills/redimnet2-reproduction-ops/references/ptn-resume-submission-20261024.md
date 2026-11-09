# model35 八卡 PTN 接续提交（2026-10-24）

> 最新更正：本页记录第一次闭包修复及wb58x启动历史。后续CPU双rank实证run-path/spawn使分片前提失效，wb58x仅作诊断证据且已停止；最终普通入口后继t-20261024154424-vggvm与真实192-worker验收见[普通脚本接续](ptn-resume-ordinary-entry-fix-20261024.md)。


> 2026-10-24后续更正：本页是首次提交的历史。t-20261024115909-mhczc于12:45:10失败，原因是接续包装里dataset的局部checked_epoch无法spawn序列化；8rank的恢复已经过了，但没有实际更新。用户已授权修复重提，新任务是t-20261024151842-wb58x，详见[spawn故障修复与新验收](ptn-resume-spawn-fix-20261024.md)。下列排队/未验陈述只对应最初采集时刻。

用户在旧维护调查后明确要求“那就提一个训练任务，继续训练的。”，这句话覆盖了一次独立的旧PTN接续。初次调查里“只准备”的记录仍是历史证据，不再当成当前禁止提交的依据。完整本机材料：/Users/zhounan/Documents/声纹降噪/redimnet2八卡复现_20261019/PTN接续提交_20261024/PTN接续提交报告.md；状态JSON及平台证据同目录。问实时情况先刷新，不机械沿用这份快照。

## 已提交及状态边界

有效 task 是t-20261024115909-mhczc，名称redimnet2-b6mix-ptn-resume35-8a800-p6-20261024-v2，2026-10-24 11:59:09北京创建，当时在排队。P6、单worker八A800、PyTorchDDP、720h、自动重试off；镜像dev_zhounan:4.0，可用区B aiplatform-SPEECH。共享只读源→/shared/group-shared，个人读写源→/root。原task t-20261019232824-8zqrl没有改；PAUSED的redimnet2 heartbeat没有恢复。

严格恢复 model_35.pt 的 SHA fdaa88b963720ccdc35708c3ca9d18e56296edd8711eb9309772afacc92beb84，650777251B。新run是 /root/experiments/redimnet2/b6-mix-official-20261019/ptn-resume-model35-20261024-v1（历史路径），源码包在同父目录 ptn-resume-code-model35-20261024-v1；run/package的v1名称是不可变合同，不因为平台叫v2就重复改写。只改exp_dir/checkpoint，保持总120轮、15001批/轮、8×96，第36轮轮首/绝对iter525035/LR.72/margin.2，完整分类头、SGD/AMP。缺RNG/loader游标，因此要重做轮内旧工作，不声称逐位连续。

13文件冻结包的manifest SHA ef0136503e3676f56a044237267824f2117c9c082eaf07f0c906203bd214fb55；平台从外部固定这个SHA，逐文件验证后才执行launch.sh --execute。代码包不用于同步报告，所有报告/截图/日志快照都留在本机。

## 已验与未验

CPU上实际严格装了823states/585有序参数/584动量/AMP，并核了真实scheduler；冻结train的主流程在CPU模拟到首loop通过，构造0forward；独立15正负例和5项Linux锁协议通过；私有libsndfile1.2.2的实际映射及三原件32生产窗通过；95/189源码与远端13文件SHA/mode444全回读一致。两卡GPU没有被占用。

父admissionEX保护三锁的EX准入，执行时降为SH，每个worker独立打开三锁的SH并持到退出；活孤儿仍会阻止新父EX和旧LM EX。清理只按自有的PG/SID/nonce做，不删除锁、也不广泛kill。八rank在实际首forward之前共同严格验恢复，fresh/head重建/丢momentum都拒绝。CPU预检不证明CUDA/NCCL/八rank已运行。后续查新run的checks/restore_verified_rank0..7.json、first_update_rank0..7.json、rank_logs及完整checkpoint；没有这些，不能说训练已恢复。

PTN120之后只做终验，LM不自动调用固定旧run的v4。跨段115/120与新优化器LM另审；七次历史FLAC事件的许可不扩张到新run，新异常不能靠放宽规则消除。

## 本次入口保存异常

首次提交的task t-20261024115627-gxhmb，提交后回读入口发现末尾有一段非预期中文“现在占着机器的那些”；提交前封存的AX里没有这段内容，来源未证实。只撤回了本轮这一条新排队项，刷新确认已停止，历史不删除；11:58:58新run仍不存在，也没有更新。原训练/其他任务都没有操作。修正版v2重填了入口，末行# END_VERIFIED_ENTRY，移出焦点，紧邻提交时逐字比较，并在提交后全文回读。不能因为任务创建成功就忽略入口差异；结果未知时先查重，不重复点提交。
