# 转写计算提速、容量估算与保留旧结果

维护实现位于仓库`tools/open_annotation/`。本参考对应转写阶段优化，不修改既有PTN训练或运行中冻结标注。先区分用户此刻要求“实际GPU验证”还是“仅依据已有机器信息估算”；后者不授权新增GPU进程或以未测倍数上线。2026年10月27日早期转写提速阶段，用户曾要求“GPU先不测了”，当时仅完成CPU实现和八卡信息测算；之后用户授权八卡WebShell实测，并在事故后明确要求恢复八卡标注。当前独立恢复须按本页末节及`docs/reviews/2026-10-27-跨节点标注恢复实测.md`的真实数据、互斥和运行条件执行，不能将早期限制误作当前禁止恢复，也不能把恢复授权扩展为再次同卡构造vLLM。不自动重开已暂停heartbeat。

## 选择转写计算方式

新合同可显式设置`runtime.asr_compute={dtype:float32|bfloat16, attention:eager|sdpa}`，`freeze_contract.py`对应成对CLI参数。旧缺省不得强改子模型配置，legacy基准沿其实际默认行为；显式FP32eager可能与旧默认不同，不能偷换成同一基线。Ultra保持原精度、单窗和原生/FIX观察，不通过换定位模型制造转写提速。

实际加载核参数dtype，显式profile核所有相关子配置。首个成功ASR调用观察实际attention forward、逐模块SDPA调用、Q/K/Vdtype、配置稳定及未执行模块；显式设置未兑现就停止试验，不伪装成单条音频unknown。上游存在顶层attention参数未传入子配置的问题报告，不能只看请求标签。该观察不是CUDA内核profiler，也不保证模型质量。

完整首窗观测只保留在终态，周期`chunk_committed.performance`使用轻量当前进程累计与增量，分开转写、定位、源审核、窗口、缓存和发布。首窗观察调用本身含真实推理时间，不得把整段都叫钩子额外耗时。新进程恢复重新计时，不能伪接上一次attempt的累计。

## 有限实际对照的准备与读取

`prepare_speed_validation.py`的schema为`annotation-speed-matrix-preparation/v1`；显式提供base合同、三域30/3000/512索引及SHA、两个目标UUID、协作锁根和解释器，生成legacy/FP32SDPA/BF16SDPA三组独立源码、合同和命令。它不执行命令，不授资源或八卡资格。输出不能进入仓库、旧合同/代码/索引树，输入与锁目录软链拒绝。先短小定位原因，再对选中候选做代表性完整流程，不能把十小时运行本身当验收。

`audit_speed_comparison.py`要求固定SHA的两份`redimnet2-annotation-speed-track/v1`描述和完整有限结果。speed模式仅ASR计算配置可变，模型/第三方代码/解码器/工程源码/质量规则及其他runtime相同；逐项核已保存原声身份字段及侧车/提交绑定（不重读或重哈希媒体）、窗口、文字、语言、token/EOS、原生与FIX字界及决策。目标文字不同的窗口不强配边界，缺失败样本不报完整总体。

全冷速度比须两轨真实新ASR执行、无完整侧车或阶段缓存复用、相同音频/窗口及已核实际计算观察。旧结果复用仍保留，但不能拿缓存消费当新模型速度。legacy可按真实子模块行为核对，不要求伪称全eager；缺实际观测就不授配置兑现结论。recovery模式同合同，逐来源及原生结果严格一致，只忽略两项缓存标记；不同仍保存差异，退出2。报告始终不授人工质量、训练/评测或自动八卡准入。

## 仅根据现有八卡信息估算

`estimate_speed_capacity.py`读取`annotation-existing-monitor-capacity/v1`监控摘要及SHA，输入观察吞吐、剩余来源和显式余量假设。NVML `total-used-free`可能非0，驱动保留和非实例占用须扣掉；每卡仅一已绑定计算进程时，才可据此估单实例。显存短窗峰值不是历史最坏情况，GPU利用率不是SM占用率，禁止用100/util推导必然倍数或理论上限。

10月27日11:34实据：八张80GiB A800各一个进程，最重14570MiB；不可用/非实例701MiB。静态5份/卡，另留8192MiB且每份×1.15时预算4份/卡。这些为该时刻信息及明确假设，不是将来启动许可。现维护启动器仍每UUID一个worker；未实现/验证多副本资源组，不能把32进程的显存表当实际支持32进程。若展示1.5/2/2.5倍ETA，必须标为假设并与进程数解绑。BF16/SDPA没有实际测速就不写提速成立。

## 去重接续

`prepare_speed_continuation.py`不构造旧WorkQueue/SidecarStore。默认provisional只读固定索引、一个完成槽快照及最多8个未封账片；已封账片保留旧producer引用，部分片已有完整侧车/commit也核身份后排除。孤儿保留并拒绝，不偷偷修复或重新推理。旧cache含合同身份，不能改合同SHA、重贴标签或拿旧ASR声称新profile已测。

`--execute-final`需显式真实旧writer终态观察及SHA/任务绑定、时间有效，取得既有`.parent.lock`的只读FD独占锁，并在持锁中核索引和broker稳定，才复制尚缺文本分片到新目录并生成remaining index。最终准备还会流式扫描未领取片中已有的旧侧车/commit，核身份后排除，并写新的`retained-unclaimed.jsonl`及SHA/count引用。孤儿或损坏保留并拒绝；provisional明确未扫描，不能当最终证据。终态JSON来自外部观察，不是工具认证平台；准备锁释放后也不是未来启动租约。旧结果+新剩余数量严格等于原11521178，保留完整片的固定索引/完成槽证据不等于重开百万侧车全审。

新八卡准入还须验证原任务真实终态、资源互斥和实际质量，不能直接放宽原固定全量分母。新`cloud_gate --union-mode`只做联合数据检查，和原运行门分别返回；不因剩余准备或数据检查成功便执行新任务。不允许新旧writer共写队列，不复制旧结果为新producer，不操作PTN。

本轮713项CPU实际通过、0skip，不是GPU速度或人工完整字证据。具体新鲜快照、代码、已修反例及尚缺条件见仓库`docs/reviews/2026-10-27-转写提速实现与八卡容量测算.md`；命令接口见模块README和`转写提速对照验收.md`。每次后续执行重新核用户指示、源码和现场资源，不把本参考数字写成实时状态。


## 方案选择与已经实现的范围

新规划在仓库`docs/plans/2026-10-27-转写提速方案反思与执行优先级.md`。长期优先研究Qwen1.7B BF16+官方vLLM的单卡共享请求引擎，八卡独立来源，Ultra先保留；后续已补最小adapter，但仍没有真实性能/质量证明。已有单窗BF16/SDPA是最少新增代码的实际对照候选，同参数双副本更少改变模型数值方式但需新资源组，不能无条件称其实施成本更低。4份/卡静态容量不决定最优方案，不能复制4份权重当默认终态，也不固定7ASR+1FA或给小模型直接TP8。

现有算子审计要求同工程源码、batch1及每次运行内各worker的UUID互不重复（基线和候选可顺序使用同两卡），不适合直接评价vLLM或同卡多副本；新架构已用独立协议，保留旧严格检查。恢复分开已发布字节不变、来源不重不漏、未发布重算差异及质量；固定cohort不自动解决半批发布，新ASR阶段库先持久化完整批次再逐源消费。不能直接改旧缓存键或producer；该阶段库不是ASR/FA完整异步调度器。

质量同时比较固定文字的FA和各自文字的端到端结果；只看共同文字子集有选择偏差。人工允许区间、削首/削尾/邻字及分层文字错误比表面小数位更重要。实际旧原生时间格点80ms，FIX小数不能称新增真实精度。未有人审金标时不报完整字准确率，候选吞吐与人工抽检质量分别统计。

vLLM批次不变仍beta，文档硬件CC>=8.0不等于Qwen3-ASR已验证，不能沿用旧CC>=9.0结论或只设seed就保证恢复逐位相同。vLLM内存有KV和工作区，不能套旧14.23GiB/实例容量。该次 CPU 实现时用户要求先不测 GPU；之后明确要求八卡 WebShell 实测。后续真实尝试与事故见本页末节，最小适配仍不等于实际资源组或 GPU 证明。

## 联合全库的数据检查与唯一读取

仓库`tools/open_annotation/联合覆盖与唯一读取.md`说明`annotation-union-coverage/v1`的实际格式。`union_coverage.validate_union(spec_path,spec_sha,mode='plan',output_index=...)`用SQLite逐源核原全集、旧完整片/部分片/未领取预置结果、新剩余输入及侧车提交。生产用途固定11521178及三域原计数；CPU小样本显式隔离。检查source完整记录、合同版本、runtime及模型来源，不只比两个数量。它不构造旧WorkQueue或SidecarStore，不修旧commit。

plan要求新鲜终态、旧父锁与broker稳定，核分工和旧保留结果；complete再核全部新旧结果及新队列无租约。complete可用已固定的历史真实终态，不强迫多天后伪造新旧观察，当前锁和broker仍必须核。`cloud_gate.check_union_data`使用独立`annotation-union-data-readiness/v1`与`data_checks_passed`，没有旧门的`passed`；GPU、云提交和训练/评测资格恒false。不能拿这个结果替代W2、实际模型、资源或质量条件。

显式新外部消费索引按不覆盖方式发布并给SHA。`UnionConsumer(spec_path,spec_sha,database_path,database_sha)`上下文只读数据库，核complete/spec绑定，按原分片与source摘要选定原producer并重开原侧车/commit。批量读取复用一个上下文，避免每源重新哈希整库；不按mtime猜版本，资格恒false。数据库或来源改变就拒绝或重新核验。全部旧保留时不造后继，走旧全量终验。

最终准备、联合plan与complete均非轻量进度巡检：会产生全库文本/元数据、旧结果读取和磁盘数据库开销。只在真实交接或终验边界执行；该工具最初的CPU实现记录没有扫描真实全库，不能外推真实音频SHA。10月27日后续现场准备与联合检查另见`docs/reviews/2026-10-27-跨节点标注恢复实测.md`，逐项区分来源文本、侧车内容和媒体内容证明。正在运行使用readonly_progress。

真实联合库构建中，逐片FULL提交、随机来源主键和小缓存可产生明显共享盘同步/读取等待；阶段抖动不等于进程卡死。当前没有安全的CPU审计断点续建协议，事务已提交或旧结果阶段计数到位都不能把隐藏临时SQLite当最终证明。先查实际PID/start、CPU/IO与日志时刻，再决定后续；不热改运行源码、不伪造完成报告、不盲迁仅剩少量空间的`/tmp`。实际云启动仍需独立新鲜平台观察；联合审计的启动观察只在其启动时核时效，不改写历史时间。

本次把恢复用SQLite中间索引放在共享盘，是维护执行时自行选择的架构，不是实验合同或模型标注的必要步骤。以后选择构建位置前，先实测代表性随机读写、事务同步成本和预计库大小，并核本地实际可用容量。旧结果核对及中间索引构建须单独计时，不能把其耗时称作模型标注慢。调整构建方案时，当前已验结果和唯一审计继续保留，不热补丁、不另启全库重扫、不降低验证，也不把尚未实现的内存或迁库方案写成已可用。

run/v1启动控制器对最终千万行计划库使用一次全表精确条件计数，避免`GROUP BY`的临时排序开销。须保留两个非空生产者、各自精确数量、总数守恒和`COLLATE BINARY`，拒绝未知/NULL/第三组及大小写或尾空格别名；不能只把两项SUM与报告比较后忽略额外来源。实际SQL计划和反例另验，不能据CPU检查写成八卡已启动。

恢复原标注可另用显式`annotation-portable-successor-run/v2`，以`preparation_attestation`替代v1的`union_attestation`。它复用真实准备遍历原11,521,178来源的分工，以及角色定稿已经完成的10,361,644剩余记录父片／互斥检查和816份内容证明，核固定完成收据、原／新计划与索引及必要摘要，不重开115万旧侧车或重新逐条扫描全库，也不读取联合计划库。`union_spec`只保留布局引用，不执行审计；原23文件引擎直接消费剩余索引。v1完整联合库要求保持，停止审计的原件和半成品保留但不冒充v1成功。v2实现另封新源码，仍须独立review、CPU测试和实际八卡验收；W2、挂载、旧锁、context守卫及最终联合消费／完整结果终验不变，机器边界不自动放行。

若可能长时间排队，可使用独立`queue_handoff.py`与`launch_queue_handoff.sh`，格式和接口见仓库`tools/open_annotation/排队后新鲜证明交接.md`。蓝图schema仍v1，内部runtime_base显式选run/v1或run/v2：前者须有最终联合报告和数据库，后者须有真实完成准备收据，两者都不接受隐藏半成品。默认check不写请求、不运行内层inputs或GPU；外层不全读v1计划库SHA，v2不使用联合库，两者结果须如实标为false，不能替代内层门。真实实例生成唯一请求后，操作者才根据实际平台提供新鲜观察；开发机publisher只核共享文件，不冒充查询云PID或认证平台。

内层21成员清单与外层2程序清单分别固定，拒额外导入、软硬链、FIFO和置换；读取先lstat拒非普通文件，再NONBLOCK防打开前置换造成永久阻塞。发布使用私有目录与无覆盖文件；既有请求、回复、声明和失败不删除。最终exec-intent保存后再完整复核源、蓝图、请求、回复、只补观察的运行描述、实际自身身份与截止时间。生产禁止假执行器/时钟/进程钩子，CPU双进程fixture成功不能写成真实云或GPU成功。

## 跨引擎审计与新阶段恢复

仓库`tools/open_annotation/跨引擎对照协议.md`对应`engine_protocol.py`与`audit_engine_comparison.py`。实际固定原声/PCM、模型、窗口、prompt/语言、generation、质量和定位，变化只限显式backend/code/precision/batching。vLLM轨须引用实际适配spec，核所有控制与源码/版本/模型文件；不能只写一个散列。每源完整侧车/commit和新vLLM native引用都核，request identity、原生ID、cohort SHA形状、EOS、终止、规范化、状态及冻结包绑定，失败不从分母移除。

固定文字定位与各自文字端到端分开；目标不同不强配时间、不报0误差。计时分别统计new/cache/reused/not_run；非合成完整新执行才计算声明墙钟比例，但报告仍`GPU_runtime_verified=false`、`validated_GPU_speedup=null`。已发布恢复以原字节为准，不给未提交跨引擎计算逐位保证。

`vllm_backend.py`普通导入及help不加载真实torch/qwen_asr/vllm。CPU入口`VLLMBackend.from_fake_engine`只接受cpu_fixture；真实`build_real_engine(...,execute=True)`低层构造需pilot、离线环境、实际完整权重/处理器输入、固定依赖及关键实际源码，并在构造前核enqueue/wait公开API，当前不得调用。按实际原生ID映射，不按位置或数字猜；EOS token观察与引擎EOS停止分别留证，矛盾和未知不升completed。

`ASRStageStore`是新producer命名空间。预约和整批commit分别事务；提交后消费异常可恢复零新调用。未提交重做需原cohort完整输入，异常和原生输出保留。旧数据库缺失就拒绝，不能静默重建。缓存恢复允许仅元数据，但若提供PCM仍核其实际字节。只验证本机SQLite和锁，不能外推PFS故障或八卡资源组。

本轮最终798项CPU单元测试实际0skip通过，独立审阅与文件级fake引擎→持久化→完整侧车→跨引擎审计→metadata恢复0新调用已通过；最终静态检查、安装同步和Git身份以私有收据为准。实际过程和未完成项见仓库`docs/reviews/2026-10-27-联合接续与跨引擎CPU实现验收.md`。不把这些结果当真实GPU、人工完整字或新任务提交许可。

## WebShell 真实失败与原标注严格接续

10月27日下午两次真实尝试均无可验收原生转写输出。第一次安装的 vLLM 0.19.1 不支持 UUID 形式的可见卡变量；第二次数字卡号进入 EngineCore 后，同卡第二个 context 触发原 `ParentChallenge.monitor`，原父进程中止八个自有标注 worker。完整事故和本机证据索引见仓库 `docs/reviews/2026-10-27-WebShell真实试验故障与标注恢复.md`。不能把本次写成无影响或速度已测得。

对该旧任务，新增 CUDA context 与原持续独占协议冲突，不论原进程是否已知或剩余显存多少。自制共享许可和新临时锁不能覆盖旧设备锁；以后先读实际冻结父进程的持续检查、worker attestation、设备锁及清理约束，再判断用户要求的测试是否能在其旁执行。不能放宽旧守卫来容纳新实验。

现维护 `vllm_live_probe.py`拒绝旧共享许可、任何已有 compute PID，以及替代原 inode 的新锁。但独立 CPU 子进程反例又发现 factory 异常时原锁可先释放，leader 退出也不证明 EngineCore 和 CUDA 排空。因此真实模型构造已经明确关闭；pilot 缺任一阶段结果就拒绝，自定义 factory 只允许 cpu_fixture。完整阶段缓存仍可零新调用恢复。完整 supervisor 与上下文归还尚未实现，不能以关闭入口声称清理已修好。

原标注恢复用 `resume_existing_annotation.py`及仓库 `tools/open_annotation/同合同标注任务接续.md`，保持原合同、全量索引、冻结包、queue 与 labels。原 fresh 启动脚本不可直接重复执行；原 claim 和语义锁保留，新 attempt2 永久 claim 不得删除。旧 v2 与现 v3 CLI 及 gate 名不同，按实际抽象语法树和字节固定。门后和 GPU 前重核全部可执行原件；外部终态观察和 other_successor_tasks 必须来自真实平台查询，不能手填冒充查询。

只读槽计账核到 1,158,718 已封账来源、4,624 片和 8 个 lease，不等于全量侧车和人工质量验收。旧 source stat 含 device；跨节点可能只有 device 变，但原 worker 仍拒绝。严格恢复继续核完整 stat，新节点不符须在 claim/GPU 前拒绝。不得删 device、伪造旧 stat 或改旧侧车；合法后继必须另有实际内容 SHA、前后稳定 stat、新旧 producer 和不重不漏联合账本。该次事故审查时尚未实现，后续实现见下节。后台巡检保持暂停。

## 显式内容身份与独立剩余生产者

新协议维护在仓库`tools/open_annotation/跨节点原音内容证明与剩余输入.md`、`跨节点标注接续运行.md`。`portable_source_identity.py`仅允许device不同，其余stat须相等，真实编码SHA及前后完整stat同时核；旧侧车不改。部分/未领取片内已发布结果逐条内容证明，已封账片在联合门逐侧车重开，但不伪称百万原音都在新节点重哈希。旧解码失败保留failed，不能计内容通过。

目录项先用scandir登记新SQLite，再遍历原全集；缺失来源只查库，避免千万次共享盘missing查询。旧stage cache明确不遍历/重贴，未完整发布的在途阶段可能重做；完整旧侧车不得重推理。清单、内容账本、全部原来源记录、旧保留及新剩余逐源互斥并完整，原分母11521178和三域不改。异常临时文件、孤儿和额外路径保留并拒绝，不自动删除。

旧实际生产代码是23文件、13计算模块，源快照在`legacy_snapshots/production_20261025`；新冻结器只改合同version，旧时钟v1和所有模型/源码/FP32单窗条件不变。当前一般冻结器是14模块和v2时钟，不能混用。继承的只是同一计算profile的历史证据，新八worker、设备和首片必须现场重新验收。

真实CPU准备/union在同节点核计划锁的device/inode。云运行器按显式版本消费固定证明：run/v1要求production联合plan和SQLite完整SHA；run/v2要求已完成准备／角色定稿收据，不消费联合库。两者另取得当前真实旧语义锁/旧queue锁；历史路径与inode保持，历史和当前device分别记录。原broker及完整completion map重核，FD交给原GPU父进程直至精确清理结束。新运行观察必须真实且一小时内，旧历史观察不改时间。已有永久lineage claim或失败证据不能删除重跑；先查实际任务。此前953项完整CPU回归只证明当时源码；v2新增分支须另记实际测试和部署，真实当前状态须读新实测报告，不以历史数字授GPU或人工质量结论。

后续复审发现，c4a时的剩余索引继承原全库角色数，实际PTN/LM用途逐源未变，但聚合统计错误。未来准备器直接数真实remaining roles；已有v1用`finalize_remaining_roles.py`另发`index-roles-v2.json`、`coverage-plan-roles-v2.json`及收据。只读原分片、账本和目录库，不读取媒体；所有旧SHA保留。PTN/LM角色可重叠，不按比例估计LM数量。原version、过滤器SHA和raw-line政策等继承项只是父级历史声明，元数据定稿不把它们改称新测结果；分片不变仅指相对v1。正式portable union还按实际remaining逐行重计角色一致。

原`verify_ultra_cloud_mounts_v2.py`是首次启动检查，旧run已存在时必然拒绝。恢复入口保留其SHA和原admission，不改原件；仍执行原W2历史质量门，实际挂载改用`verify_portable_successor_mounts.py`绑定新spec核精确GPFS source/target、用户RW、共享RO、明确别名、实际控制/输入/模型资产位置和全新独立run。默认CPU检查或模拟findmnt不证明云端挂载已通过。部署源码修改后必须新建控制源目录，不热改已部署版本；新run及证据应与旧源码、原输入、新输入、控制目录和旧声明目录互不包含。GPU前仍由原父进程核模型完整SHA、设备双快照和持续独占；全库complete终验仍在满足计划锁身份的准备节点执行。

旧侧车全量重开可显式`retained_read_backend=process`、`retained_read_workers=2..4`。默认thread/1不变；只并行旧完成片JSON校验，不读媒体、模型或重跑转写。spawn初始化固定producer/contract及11份校验源SHA；部署控制源须额外含`alignment_observer.py`。父进程不许已导入torch/qwen_asr/vLLM，spawn之前隐藏GPU环境并在所有退出路径还原；生产process仅Linux，PR_SET/GET不可忽略SIGKILL并在arm前后及每job核父PID/start。每批最多256个有序小tuple，SQLite仍main；正常、worker错误和父侧异常都先在旧父锁内排空，不能只为map异常处理。

`benchmark_retained_reads.py`需要实际.py入口、固定plan/SHA、显式少量positions/modes，最多8片/2048来源；按mode顺序只保留一池，完整有序证明相同且broker不变才出报告，不取旧执行锁、不授准入。`probe_retained_process_parent_death.py`仅测试自有合成CPU父子进程；实际LinuxTERM父、核精确子PID/start消失或Z，失败清理只走nonce/start和pidfd，不用裸PID信号。Conda缺标准pidfd API时必须核本机头文件434/424及实际fdinfo身份，不悄悄降级。所有旧审计半成品和原失败保留，只有实机更快且证明相同才替换精确自有CPU审计，不能操作PTN。该CPU机制不代表vLLM/EngineCore清理已实现；真实GPU构造仍关闭。

## 2026年10月28日：独立两卡实测与无数据库有界交接

用户明确要求真实验证并接入八卡，旧“GPU先不测”不再适用。当前八卡生产不可旁建context、两卡开发机仍有既有/未知PID时，优先申请独立云两卡pilot；沿实际镜像/挂载、zhounan_名称、P6与关闭重试，当前标注/PTN保持。prepare_bounded_speed_validation显式对比FP32-eager、FP32-SDPA、BF16-SDPA，同维护代码/v2源钟/Ultra与输入，不冒充旧23文件逐字节同软件。gpu_launch的auto2只允许pilot且实际恰两A800，原UUID锁/空闲双快照/context挑战/归还不变。

run_bounded_speed_pilot消费固定30/3000准备收据，核命令/源码与模型身份，逐轨归还并审阅后前进；512恢复用同合同labels和新有限queue，要求旧字节不变、零新ASR/FA。report的有限pilot≤4096 full读回使用set，不能为此创建临时SQLite；旧全量分支不执行。新代码与CPU不能写成真实GPU已通过，实际速度分加载、ASR/FA、源核验/窗口/发布与稳态，不据30小批总耗时外推17天。

实际4.57.6源码包含复合config递归attention setter/from_pretrained二次分派，不能把历史上游风险当本次已证瓶颈；仍须实际forward/SDPA/QKV观测。Qwen0.0.6声明vLLM extra0.14.0而现场0.19.1，接口存在不等于已兼容；真实vLLM构造及清理仍不因本次授权自动变成已验证。

bounded_speed_handoff/queue是显式新schema：已完成片核固定槽、仅最多8在途片逐源核既有结果，未领取片只引用完整父索引元数据，领取才读JSONL。reader必须核父cursor后描述完整闭包及partial保留/缺失守恒，拒绝同步缩小计数造成漏源。portable沿既有内容SHA和前后stat，仅device可变；旧producer/job/字节不重贴，孤儿保留拒绝。旧锁实际FD贯穿GPU和清理，新鲜终态只在启动核，不每领取一片重复要求一小时内观察。祖先语义/旧queue锁仍要在正式父子运行入口继承，不拿helper的最近旧writer锁代替全部祖先门。

禁止为本次接续重开联合全库审查、数据库或千万JSONL复制；旧历史章节仅作记录，不作为当前执行路线。详读仓库《有限同批转写提速准备》《独立云转写提速串行验证》《有界提速交接》。实际资源与实验结果以带日期报告和冻结源SHA为准；CPU通过、准备成功、平台Running、GPU真实执行、质量、恢复和正式八卡接续分别记账。

本日真实执行追加在仓库`docs/reviews/2026-10-28-转写提速真实两卡与八卡交接.md`。独立云两卡`t-20261028123255-dmsw5`的30源筛选已实际完成：三轨文字/token/原生及FIX字界一致，SDPA请求真实兑现；纯ASR约1.21/1.19倍，但首次加载不同，不能取30源总墙钟1.47/1.43倍外推千万来源。3000/512与正式接续必须继续读最新原件，不把本句历史快照当完成。

12:45真实3000基线前512源，两worker累计约87.6%耗于ASR，GPU均值约35%；源读写并非该段主要耗时。原生Transformers合批方案另在仓库`docs/plans/2026-10-28-原生转写批处理独立验证方案.md`：显式新policy，ASR固定cohort而Ultra保持单窗，不放松旧单窗合同；完整批次以文件原子保存后逐源消费，禁止调用依赖SQLite的旧ASRStageStore。实现、CPU测试不授真实批宽/速度或正式资格。不能把修改中的native源码混进已验单窗冻结；同一真实pilot与新正式包的计算语义源码仍须逐SHA匹配，仅worker/report控制差异可由明确独立integration_review准入。

正式薄索引运行接口读仓库`tools/open_annotation/有界提速正式运行.md`。当前运行器`--bounded-runtime-spec/--bounded-runtime-sha`强制绑定实际3000/512选型、逐差异文字字界复核、原始祖先锁及当前旧writer锁，永久唯一后继声明拒绝同旧快照换run重复处理。新queue和新producer独立，旧结果在读原音/模型之前复用原job；元数据终验明确不是全量侧车或人工质量证明。排队后专用材料交接必须用其对应新schema，旧queue_handoff不自动支持。正常先核实际新8任务与请求，再受控交接；若真实碎片使旧标注整八卡占用与新请求形成循环，先取得唯一正式新8任务ID并回读固定配置，凭实际碎片证据受控停止已授权的精确旧标注，再等新request并重新读取真实旧终态；详见仓库《标注八卡碎片排队时的受控接续顺序》。不能以pilot代替正式8卡ID，不能伪造平台观察或放松时效。主训练排除于停止或迁移操作之外。


## 10月28日真实合批差异与逐条声学特征修复

完整3000条实测：单窗FP32/SDPA相对普通注意力约1.138倍；原生B4相对原生B1约1.726倍、B8约1.987倍。后两者分别有110/122窗规范化文字变化，不能凭速度或机器candidate净增3授正式资格。实际五条state变化是4条unknown升candidate、1条candidate降unknown，含删尾词、换字及新零时长边界。原件、坏候选和所有异常保留，不改旧标签。

初步根因是官方Whisper特征提取在batch级先补零再STFT，原生同PCM并不保证同模型输入：非160整倍数的短音频可能有效mask多一帧，音频占位token实际增加；即使prompt相同，STFT尾反射上下文也可能不同。修复必须显式新预处理协议，逐条沿官方B1提特征后只在特征层合批，并保存真实generate输入特征/有效mask/非pad prompt的证明。不能仅凭PCM SHA宣布输入一致，不能修改旧冻结包补hash后把旧坏候选当通过。新计算源须重新冻结、CPU反例及真实30/3000/512验证；仍可能有decoder批形状数值变化，需要逐差异复核。

正式控制入口分别见仓库《原生合批正式准入与八卡接续》《原生八卡提速接续执行清单_20261028》，旧单窗v1不能给nativev2授资格。实际候选数学包原字节复制后，只有明确列出的4个CPU控制修复可由独立审阅放入正式包；改计算或资源源必须重新真实验证。

有限封存工具见《原生有限封存CPU诊断入口》：真实B1 3000条封存约62秒、119.206MiB、峰值1.284GiB；实际一次严格验证约166.6秒、峰值1.512GiB，侧车/commit/媒体重开0。三类有限证明及8个bootstrap重复解析不是零开销，须算实际预算，不删校验或无限抬大小上限。开发机300秒超时和本机CPU全集首次权限/包名误判日志保留。包名守卫修成精确包及点分子模块，真实torch/qwen_asr/vllm继续拒绝；不通过删除sys.modules让测试通过。修复后919项CPU普通模式实际通过；这不证明下一版新预处理或GPU质量已过。

真实中途恢复探针见《原生转写中途恢复探针》；512已完整侧车命中不能冒充首整批已保存、后续尚未保存的故障恢复。旧B4探针准备好但因其声学输入问题未启动；后续须按新实际候选重新绑定来源和计算源，不能直接复用旧坏profile授恢复资格。

修复后最新本机整套CPU为955项成功、1个真实Tensor类明确隔离跳过；该类6项真实Tensor在服务器CUDA不可见的受审计算源中已实际执行，连同合同反例为11项普通/-O零跳过。不要把本机跳过藏去，也不要将11项全写成11个Tensor反例。新输入协议详读仓库《单条特征保真批处理》，仍须真实GPU重新测。

新16源真实两卡已于15:25唯一提交xw6bs，完整名称和源码/计划在上述实测报告；实际source来自Git66b63f5、109文件，无overlay。运行根native-feature-asr-20261028-v1，入口只用plan-v2（fdab6044…）和pilot-evidence-v1；v1操作计划字段错误已在GPU前拒绝并保留。排队/Running/新30及3000质量、512恢复、八卡上线仍分开记录，不重复提交或将旧15源速度代用。

2026-10-28新16源xw6bs的3000条已全部实际完成：B1/B4/B8分别1748.644/1063.611/900.046秒，B8约1.943倍。3067窗口有效输入、文字、有效token、定位和decision零变化；119源原异常差别仅两处traceback内实际冻结入口根，精确替换后整份观察一致，原unknown和错误保留。三个512恢复均命中512，ASR/FA/新增侧车均0，不能替代中途未完成cohort恢复。独立B8中断恢复4hjb6已经实际完成并独审：8个已保存ASR窗口复用、26个窗口补算，30源34窗文字与字界不变，原文件SHA/inode/mtime保留；不能与512整侧车命中混淆。

新16源CPU封存v1实际300秒超时证据保留。仅CPU诊断可用显式v2，绑定原receipt+intent+真实child/固定argv/原冻结控制器与计划，原sealer/track/bundle不变，新目录单次301～900秒；不改变任何GPU/质量时限。v2不能递归，伪造未启动/返回0/耗时不足不接受。17项双模式及独审通过不等于实际封存或正式准入通过。

证明复用必须仍对全部已核输入逐次实际SHA。Linux曾实证同一tick改变文件且metadata六字段完全相同，原metadata候选不安全；修后61项普通/优化真实Linux测试无skip、原64反例全部拒绝。显式进程内scope和默认关闭的密封memfd只省JSON解析与映射重建，不省IO，不替代原全核；实际新3000/512整链启动成本仍须核。

当前119源的固定traceback入口差异使用专用`frozen_traceback_root_only_verified`，不放宽旧metadata规则。路径只从真实result→plan→main矩阵及实际bundle推导；只替换两处旧VAD错误中的实际冻结Python帧根，全文观察必须精确相同。61源含其他原异常、69源含额外原因，全部保留深比；unknown不可训练／评测仍严守。窗口presence/input、双方计数及所有非3项尾部padding的ASR相等旗标均严格校型／校值，文字、有效token和字界不得变化。114项双模式与独立14类反例、实际238对源观察通过；最终侧车原字节读回绑定仍保留。

新16源三份有限封存已有真实输出：基线125,428,063B/SHA44f2973173e9d83c64e4ea89d3035d77eee54e2a9b2789ea77f42c849bb91432，B8主122,738,770B/SHAe06e4585cfa19fe1865fcca261439e368a922770d469364a56bf6ada5393dc63，B8恢复18,528,803B/SHA12e8935592e75be48fcbec6c1d3f9dcbd89904d8f6aa6e38fb8d9e819534f126。首次主B8在60秒周期栈附近SIGSEGV，完整原件保留；改独立CPU诊断程序为Python线程持GIL采栈后，主134.762秒、恢复24.474秒完整完成，审计器／16数学源不改。10秒合成未复现旧SIGSEGV，根因未确定，不能称已证CPython竞态。新控制器18双模式独审与合成诊断3双模式均通过。

冻结父创建八个worker时逐次校验租约，子在baseline回复后仅等20秒父ACK。启动成本必须考虑这20秒，以及120秒首baseline和900秒ticket，不仅计总容量；不能放宽时限、热改GPU守卫或用CPUfixture冒充正式父进程。有限封存成功／真实合批速度和实际八卡启动分别验收。

真实首次完整选型在196.554秒严格拒绝decision_counts表达：原冻结Counter无failed零键，旧读回器补零。新N8169按实际合法records重建稀疏Counter，两项计数canonical精确比较，未知／额外零键和float/bool伪装拒绝；queue slot三键规范保持，原报告和失败材料不改。120双模式独审及完整1054项CPU回归通过（13项已说明环境跳过），后续另建formal-v2/selected-v2/outer-v2，不能热改763f-v1。首full约196秒多数为真实CPU工作，不是空闲等待；密封父证明和后续全SHA成本需分别实测，不以CPU成功冒充20秒ACK兼容。


正式v2已真实全核通过，但8次父证明验证的后7次累计29.830秒，超过冻结GPU父的20秒ACK等待，不能据此上线。M的独立新版本58823改为先后核完整祖先目录，并对每个文件使用不跟随链接、非阻塞FD核六字段、实际全量SHA及读后FD／路径身份；不改变16计算源或20／120／900／1800秒规则。真实Linux72项普通、72项优化模式零跳过，原64个改字节反例全部拒绝。正式材料另建v3（bundle37b1、selected3f12、outer26成员e628），v1／v2原件保留。完整父证明显式使用密封memfd以免每个子进程重复解析约300秒；每次完整内容SHA继续核验。v3实际成本、默认Q检查及正式八卡运行仍分别取新结果，不能将FD合成安全测试当真实性能或GPU上线。

v3真实成本已完成：full224.811秒，8次scope复用62.020秒，底层8次完整SHA16.106秒，关键后7次12.446秒低于20秒，峰值RSS约3.735GB；该代理量未包含真正云端spawn／锁链／ACK，不代替实际八卡启动。默认outer-v3检查又在大proof前发现缺少work_queue，原失败保留；新wrapper独立outer-v4增加GPU原字节解析模块，新特征外层27、旧协议26，GPU37b及selected3f保持。新增隔离-I-S导入整个准入审计链，而非仅合同与新核心。入口必须在首次import前拒额外文件／目录／pycache／软链，-B不能阻止旧pyc读取；实际源码SHA和任务入口一致性仍须逐项核。
