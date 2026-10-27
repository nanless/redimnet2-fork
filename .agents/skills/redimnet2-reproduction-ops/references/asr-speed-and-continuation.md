# 转写计算提速、容量估算与保留旧结果

维护实现位于仓库`tools/open_annotation/`。本参考对应转写阶段优化，不修改既有PTN训练或运行中冻结标注。先区分用户此刻要求“实际GPU验证”还是“仅依据已有机器信息估算”；后者不授权新增GPU进程或以未测倍数上线。2026年10月27日本会话最新明确要求“GPU先不测了”，已完成CPU实现和八卡信息测算，未切换任务，不自动重开已暂停heartbeat。

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

最终准备、联合plan与complete均非轻量进度巡检：会产生全库文本/元数据、旧结果读取和磁盘数据库开销。只在真实交接或终验边界执行；本轮未扫描真实全库，不宣称重新核音频SHA。正在运行使用readonly_progress。

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

真实CPU准备/union在同节点核计划锁的device/inode。新云运行器消费固定production plan证明和SQLite完整SHA（不是complete标签库），另取得当前真实旧语义锁/旧queue锁；历史路径与inode保持，历史和当前device分别记录。原broker及完整completion map重核，FD交给原GPU父进程直至精确清理结束。新运行观察必须真实且一小时内，旧历史观察不改时间。已有永久lineage claim或失败证据不能删除重跑；先查实际任务。相关代码通过953项完整CPU回归，真实当前状态须读新实测报告，不以该数字授GPU或人工质量结论。
