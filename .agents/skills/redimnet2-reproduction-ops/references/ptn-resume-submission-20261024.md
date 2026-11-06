# model35 八卡 PTN 接续提交（2026-10-24）

> 最新更正：本页记录第一次闭包修复及wb58x启动历史。后续CPU双rank实证run-path/spawn使分片前提失效，wb58x仅作诊断证据且已停止；最终普通入口后继t-20261024154424-vggvm与真实192-worker验收见[普通脚本接续](ptn-resume-ordinary-entry-fix-20261024.md)。


> 2026-10-24后续更正：本页是首次提交历史。t-20261024115909-mhczc于12:45:10因接续包装的dataset局部checked_epoch无法spawn序列化而失败；8rank恢复已过，但无实际更新。用户已授权修复重提，新任务t-20261024151842-wb58x，详见[spawn故障修复与新验收](ptn-resume-spawn-fix-20261024.md)。下列排队/未验陈述只对应最初采集时刻。

用户在旧维护调查后明确要求“那就提一个训练任务，继续训练的。”，覆盖一次独立旧PTN接续。初次调查“只准备”记录仍是历史证据，不再当成当前禁止提交。完整本机材料：/Users/zhounan/Documents/声纹降噪/redimnet2八卡复现_20261019/PTN接续提交_20261024/PTN接续提交报告.md；状态JSON及平台证据同目录。请求实时情况先刷新，不机械沿用此快照。

## 已提交及状态边界

有效 task t-20261024115909-mhczc，redimnet2-b6mix-ptn-resume35-8a800-p6-20261024-v2，2026-10-24 11:59:09北京创建，初期排队中。P6、单worker八A800、PyTorchDDP、720h、自动重试off；镜像dev_zhounan:4.0，可用区B aiplatform-SPEECH。共享只读源→/shared/group-shared，个人读写源→/root。未改原task t-20261019232824-8zqrl；未恢复PAUSED的redimnet2 heartbeat。

严格恢复 model_35.pt SHA fdaa88b963720ccdc35708c3ca9d18e56296edd8711eb9309772afacc92beb84，650777251B。新run /root/experiments/redimnet2/b6-mix-official-20261019/ptn-resume-model35-20261024-v1，源码包同父目录 ptn-resume-code-model35-20261024-v1；run/package v1名称为不可变合同，不因平台v2名称重复改写。只改exp_dir/checkpoint，保持总120轮、15001批/轮、8×96，第36轮轮首/绝对iter525035/LR.72/margin.2，完整分类头、SGD/AMP。缺RNG/loader游标，因此重做轮内旧工作，不声称逐位连续。

13文件冻结包manifest SHA ef0136503e3676f56a044237267824f2117c9c082eaf07f0c906203bd214fb55；平台从外部固定该SHA，逐文件验证后执行launch.sh --execute。代码包不是报告同步，所有报告/截图/日志快照保留本机。

## 已验与未验

CPU实际严格装823states/585有序参数/584动量/AMP，并核真实scheduler；冻结train主流程CPU模拟至首loop通过，构造0forward；独立15正负例和5项Linux锁协议通过；私有libsndfile1.2.2实际映射及三原件32生产窗通过；95/189源码与远端13文件SHA/mode444全回读一致。未占用两卡GPU。

父admissionEX保护三锁EX准入，执行时降SH，每worker独立打开三锁SH持到退出；活孤儿仍阻止新父EX和旧LM EX。只按自有PG/SID/nonce清理；不删除锁或广泛kill。八rank实际首forward前共同严格验恢复，拒绝fresh/head重建/丢momentum。CPU预检不证明CUDA/NCCL/八rank已运行。后续查新run checks/restore_verified_rank0..7.json、first_update_rank0..7.json、rank_logs及完整checkpoint。未见这些时不能说训练已恢复。

PTN120后仅终验，LM不自动调用固定旧run的v4。跨段115/120与新优化器LM另审；七次历史FLAC事件许可不扩张到新run，不以放宽规则消除新异常。

## 本次入口保存异常

首次task t-20261024115627-gxhmb在提交后回读入口发现末尾非预期中文“现在占着机器的那些”；提交前封存AX无该内容，来源未证实。仅撤回该本轮新排队项，刷新确认已停止，历史不删除；11:58:58新run仍不存在、无更新。原训练/其他任务均未操作。修正版v2重填入口，末行# END_VERIFIED_ENTRY、移出焦点、紧邻提交逐字比较并在提交后全文回读。不可凭任务创建成功忽略入口差异；遇未知结果先查重，不重复点提交。
