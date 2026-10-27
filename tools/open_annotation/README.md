# 开源逐音频标注与完整字消费工具

这里的代码可以维护、冻结和测试。它没有自动启动训练的入口，机器文字和边界也没有因此获得人工金标或训练资格。当前正式标注范围只有 VoxCeleb2、VoxBlink2、CN-Celeb2 的实际训练来源。诊断原件仍需要外部保护引用，不能因为它们原本来自训练集就自动用于微调。

这是从现行实验源码整理出的新维护版本。代码进入仓库后有新的摘要，不能拿旧云任务的合同或两卡结果证明本版本已经在图形处理器（GPU）上运行过。当前云任务的冻结目录、合同、队列和标签都不在本目录内，也没有被修改。

## 先做不需要模型的检查

在仓库根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tools/open_annotation/tests -v
```

测试使用临时合成数据，包括整数脉冲编码调制（PCM）WAV、伪模型文件和合成保护名单，不需要本机实验目录、真实音频、模型权重、网络或 GPU。没有 NumPy 时只跳过可选图形分析测试；没有 Matplotlib 时只跳过实际绘图测试，并明确记录缺少依赖。不能把这些跳过项写成渲染成功。

运行代码要求 Python 3.10 或以上。实际音频处理还需要 NumPy、SoundFile 及合同指定的解码库。实际推理需要 PyTorch、Transformers、Qwen ASR 与对应官方模型文件；它们的版本、路径、源码和文件摘要必须由外部合同逐项固定。绘图需要 Matplotlib 和可显示中文的字体。本工具不会修改已有环境、下载模型或安装依赖。

## 模块做什么

| 文件 | 负责内容 |
|---|---|
| `source_registry.py`、`manifest_builder.py` | 保留来源、原始划分、提供方说话人标签和旧语音活动检测（VAD）引用；把大清单分成小片 |
| `source_filter.py`、`audit_source_filter.py` | 读取原清单，筛出三个实际训练域；保留原行字节并核对输出，不读音频 |
| `audio_windows.py` | 完整源身份、原声采样点时钟、连续上下文窗口、严格旧 VAD 复用与异常留证 |
| `source_clock.py` | 根据已保存的十进制秒做精确有理数采样点覆盖；保留旧版本验证，不能用容差隐藏真实重叠 |
| `qwen_backend.py`、`alignment_observer.py` | 每次实际模型调用只有一个窗口；保存原始时间分类、库的 FIX 处理和官方返回，不额外裁剪或拉伸 |
| `normalization.py`、`quality_axes.py`、`schema.py` | 保存文字规范化中间步骤，分别记录文字、边界、语言、来源和身份的待审状态 |
| `work_queue.py`、`sidecar.py` | 单父进程锁、分片领取、并发代次、不可覆盖发布、缓存身份和中断恢复 |
| `readonly_progress.py` | 运行中无写锁读取完成分片下界；不扫描全库侧车，不代替文字与边界审核 |
| `union_coverage.py`、`union_consumer.py` | 核旧结果与真正剩余输入的精确联合覆盖，完整终验后按固定来源唯一读取；不授运行或训练资格 |
| `engine_protocol.py`、`audit_engine_comparison.py` | 固定输入、模型、窗口和质量规则，核跨引擎文字、字界、原生结束信息和恢复；区别新计算与缓存 |
| `vllm_backend.py`、`asr_stage_store.py` | 官方 Qwen 提示与解析的最小离线桥接、稳定请求映射和整批持久化；实际模型兼容及速度未测 |
| `report.py`、`audit_outputs.py` | 核对计账、侧车与提交记录；退出 GPU 后持同一父锁检查孤儿文件和软链接，不删证据 |
| `gpu_launch.py`、`gpu_bootstrap.py`、`gpu_challenge.py`、`gpu_common.py`、`gpu_vendor/` | 资源双快照、物理设备锁、七阶段显存挑战、精确子进程身份、归还与退出 |
| `freeze_contract.py` | 把维护源码复制到新目录，生成新的成对试验/全量合同和源码摘要清单 |
| `prepare_normalization_ab.py` | 最多16条已封存侧车的诊断对照准备：按新合同生成两组只含ASR的可追溯缓存种子，不读原声、不执行模型 |
| `audit_normalization_ab.py` | 只读核对两轨实际结果、原ASR固定、强制对齐重算及资源记录；按同一原文范围比较，输出含原文的私有详细报告 |
| `plot_normalization_ab.py` | 从已完成报告的固定计数生成新PNG；图只表达机器观察，不读原声/模型，不覆写已有图 |
| `word_adapter.py` | 保守完整字候选检查和工程接口转换，不释放训练资格 |
| `quality_figures.py` | 明确选单的 CPU 原声波形、频谱和原始/FIX/拥有区时间图；不把图当人工听审 |
| `validate_two_gpu.py` | 新冻结副本内的 30 条小批、3000 条基线（分别记录实际新推理与旧侧车复用）及 512 条部分缓存恢复验收；只能显式执行 |
| `cloud_gate.py`、`make_admission_spec.py`、`make_cloud_launcher.py`、`verify_cloud_mounts.py` | 重读两卡证据、准备单机八卡独立推理启动文件和核对共享盘挂载；不会提交平台任务 |

## 冻结新的实现

先准备完整的外部基础合同。它应包含模型文件、第三方推理源码、解码库、精度、单窗口执行策略和窗口长度等信息；仓库不附带真实环境合同。然后运行：

```bash
python3 tools/open_annotation/freeze_contract.py \
  --base-contract /srv/annotation-input/base-contract.json \
  --base-contract-sha '<该文件实际SHA256>' \
  --output /srv/annotation-runs/new-code \
  --version-prefix maintenance-20261025
```

输出目录必须不存在，并位于源码仓库之外。构建器复制维护源码，重算语义模块摘要，生成 `contracts/`、`bundle_files.json` 和 `freeze_receipt.json`。原合同不改；试验与全量合同除用途和版本名外保持相同语义。这里的成功只证明文件和合同可检查，不证明模型、GPU 或标签质量已经通过。

`gpu_launch.py` 现在必须显式传 `--run-root` 和 `--device-lock-root`。任务主目录可以包含冻结代码目录；队列、标签、资源证据必须各自独立，不能与代码等同、互为祖先或后代。设备锁目录必须在每次任务目录之外，并由同一机器上的协作任务共同使用，不能每个任务各造一套锁来规避互斥。

共享盘、库版本、权重与推理源码不完整时拒绝执行。启动器仍保留平台传入的 `RANK`、`WORLD_SIZE`，拒绝重复父进程；它不是八个独立父进程的分布式训练入口。资源快照里存在未知进程时不能共享其空余显存，也不会自动杀进程。

## 从侧车提出完整字候选

默认输出的 schema 是 `ultra-word-consumption-proposals/v1`，不是旧训练程序可直接读取的 `word-complete-units/v1`。命令必须显式指定生产源码副本、合同与摘要、原侧车和提交记录的小选单、仓库 `word_ft/schema.py` 及其摘要，以及完整诊断保护引用的外部路径和实际摘要：

```bash
python3 tools/open_annotation/word_adapter.py --help
```

保护引用 schema 为 `ultra-diagnostic-protection-reference/v1`。必需字段包括 `case_count`、`three_training_source_count`、`cases`；每个条目保留 `source_id`、`path`、已明确证明的 `normalized_alias_path`、`in_three_training_registry` 和 `benchmark_diagnostic_only_not_training_release=true`。摘要固定的是调用者给出的整份原始字节，不是仓库内一份替代名单。缺失、变更、计数不符或不明确保护策略都拒绝。公开测试中的合成引用不代表实际名单完整，不能拿来替代受控的正式引用。

命中保护源 ID 或已记录的任一路径表示时只记诊断。适配器不会用文件名或任意路径后缀猜测身份，也不声称做了全库内容或说话人去重。

只有机器候选的原生时间、规范化、语言和源身份都满足既有条件时，才可能提出同一原声、同一提供方标签、同一起点的 1、2、3、6 个单位前缀。粗多字项不能均分，重复字保出现序号，没有会话证据就保留空值。未知、零长、修补、截断、不支持或不足六个连续单位都保留原因，不能按秒凑档。

显式 `--emit-engineering-candidates` 还需要 `--audio-bindings` 及摘要，逐源核实际完整媒体编码字节、所有原声道 PCM、采样率、帧数和前后文件状态。工程输出仍固定 `engineering_candidate_pending`，`usable_for_training/evaluation=false`，人审为空。默认旧读取器和现行训练的划分检查会拒绝；显式工程读取通过不能改称训练准入。显式导出有拒绝项或输出零条时返回码为 2；默认检查全部计账且无完整性异常时可返回 0。

## 两卡与八卡入口的现场条件

`validate_two_gpu.py` 只允许从同一新冻结代码副本运行，需显式 `--execute --config --config-sha`。配置 schema 是 `annotation-three-training-validation/v1`，包含 `task_root`、`code_root`、`output_root`、`bundle_sha256`、`pilot_contract_name`、`pilot_contract_sha256`、`main_index`、`recovery_index`、`vertical_index_sha256`、`validation_UUIDs`、`device_lock_root`。索引引用均为 `{path, sha256}`，固定要求 3000 条主批、512 条恢复批和先已完成的 30 条小批。没有默认服务器、默认设备或默认旧合同。恢复测试只对自己创建且已核实身份的父进程发一次信号；它不能操作现有训练。

八卡准备器使用新 `ultra-eight-three-train-spec/v3`，显式记录任务根、运行解释器、两卡设备身份和全局设备锁目录。它会重读每个工作进程的资源、原生输出、恢复、归还与独立审阅证据，并核对实际全量三域数量和挂载。生成脚本不等于提交或运行。持久唯一任务声明出现后若入口失败，禁止删声明盲重跑；应先核实旧任务、父进程、队列与源文件身份，再另行审阅接续命令。本仓库没有自动取消、迁移或重复提交功能。

本次归仓只重新验证 CPU 软件行为。新版本的实际模型推理、两卡资源挑战、云八卡运行、全量输入状态、存储配额及人工完整字准确性仍需现场证据，不能继承历史试验数字。

## 第二轮软件检查补充

缓存、版本目录、逐源输出和队列元数据中的软链接现在会在读写前拒绝，包括指向同一输出树内部的别名，避免把两个逻辑路径写到同一位置。完成槽和日志文件还以禁止跟随软链接的方式打开；片编号与相对路径拒绝保留的当前/上级目录名和空字节。完整 CPU 核验会逐片把完成槽的三类状态计数与实际侧车重算数比较，并核对片级及逐源提交记录的完整身份字段；轻量完成槽检查仍不等于重新打开全部侧车。完整模式另返回 `full_sidecar_decision_counts`，不把未经核对的汇总当成质量结论。

Ultra 原生证据还必须有合法的时间槽张量摘要、与出现数量一致的张量形状、类别取值范围及输入序列内的位置。缺少或矛盾的证据只增加不确定原因，不修时间、不补字，也不放宽旧 VAD 区间检查。父启动器在读取模型文件、创建挂载别名或进行 GPU 准入前，先检查合同版本、用途、实际语义源码和运行参数。

这些修复只属于仓库新的维护身份。不能据此宣称正在运行的旧冻结部署已被修复、已重新验收，或认定现场已经发生过上述合成反例中的错误。重新部署仍须独立冻结与现场验证。

## 运行中的无锁只读进度

`readonly_progress.py` 用于正在运行的队列。它不构造 `WorkQueue`，不领取任务、不加锁、不创建队列目录，也不导入模型或GPU代码。读取一次原子替换的 `broker.json` 后，只对该快照中 `0..next-1` 且不在 `leases` 的分片，以 `pread` 读取各自1024字节完成槽；检查槽格式、摘要、索引/合同/分片身份和三类状态计数。不会扫描原声、分片JSONL、侧车结果或大日志。

```bash
python3 -B tools/open_annotation/readonly_progress.py \
  --index /absolute/input/index.json --index-sha '<索引完整SHA256>' \
  --queue-root /absolute/task/queue --contract-sha '<合同完整SHA256>' \
  --output /absolute/existing-audit-directory/progress-new.json
```

输出必须是队列树之外的新文件，父目录需已存在；已有文件或悬空输出软链均拒绝。所有输入以只读方式打开，完成槽不足、损坏、身份或分母矛盾时直接报错，不发布成功报告。该文件中的原始快照含运行标识，只应保存在任务证据目录，不要复制进公开仓库。

`committed_source_lower_bound` 是这一次快照已封账来源数的下界。活动租约即使稍后完成，也排除在本次统计之外。`candidate` 仍是机器候选，`unknown` 仍待复核；`failed=0` 不代表没有运行异常或边界错误。该检查不是逐侧车完整性核验，也不证明文字和完整字边界正确。单个快照不提供预计剩余时间；比较多个快照还必须保证索引和合同一致、说明观察窗口与分布。

## 源时钟和词内撇号的独立诊断版本

完整研究结论、可复用准备与执行命令维护在[源时钟与撇号对照说明](../../docs/reviews/2026-10-26-标注时钟及撇号对照.md)。真实转写和部署快照保留在仓库私有审计目录，不能复制进公开文档。

新生成的窗口声明 `source_offset_decimal_floor_start_ceil_end_v2`。它把作者最终返回、已经保存的十进制秒按有理数乘原采样率，再对起点向下取整、终点向上取整。例如16.24秒在16kHz中恰是259840采样点；不能让二进制浮点略小于整数而凭空制造相邻词重叠。没有增加容差，没有裁剪、平移或改写raw/FIX时间。原采样网格上真正存在的重叠，包括44.1kHz下非整数覆盖造成的重叠，仍待复核。旧VAD的v1时钟不变；历史侧车继续按自己声明的v1公式验证。新合同显式绑定各窗的v2声明，不能逐窗自选另一版本。

默认文字规范化仍为 `nfkc_punctuation_to_space_whitespace_collapse/v1`。诊断时可给冻结器显式传入 `--normalization-profile nfkc_preserve_ascii_word_apostrophe_punctuation_space/v2`，只保留两边都是字母的ASCII词内撇号。官方分词器明确保留这一字符，所以 `I'm` 和旧规则生成的 `I m` 是两个可测输入条件。弯撇号不自动改写，数字读法不扩展，原始文字及中间变换保留。新条件仍保留 `word_internal_apostrophe_requires_review`，不因为输入形式合法就自动判定转写或完整字正确。

`prepare_normalization_ab.py --help` 给出有界CPU准备入口。它要求明确的小选单及SHA、侧车目录、完成分片及SHA、原生产合同及SHA、两份新试验合同及SHA和全新的输出目录。小选单为 `{sample_count, rows}`，每项含 `source_id`、`sidecar_sha256`，可附 `sidecar_relative_path`；未附路径时依次读取 `00.json` 等。所有来源必须是三个实际训练域的原 `provider_train` 记录，保持来源字段原样，但本次用途只有诊断，不授训练资格。输入不完整、原任务/版本/来源或ASR观察矛盾、截断、重复、合同除显式profile外另有语义差异、输出已存在或为悬空软链，都拒绝。

准备器按每片最多8源生成同一索引，建立两组新版本的标签树身份及ASR-only缓存。缓存键来自新合同；ASR的原始生成token、EOS和观察字段原样保留，另外记录旧侧车和旧ASR摘要及“没有新执行ASR”。只有规范化结果按显式条件重新推导，不复制旧缓存目录、不预写强制对齐结果、不创建队列。真实GPU执行前必须重新取得空闲资源和公共设备锁。之后逐窗核ASR缓存确实命中、完整PCM和原生字段相同、强制对齐确实新执行，再比较raw非正长、FIX修改、最终零长和源时钟；种子已准备不能写成已消费或已通过边界测试。两个条件都使用相同v2时钟，因此该对照不能把时钟修复的收益混算到文字策略中。


## 吞吐调查与尚未执行的提速验证

见[标注吞吐瓶颈与提速验证安排](../../docs/reviews/2026-10-26-标注吞吐瓶颈与提速验证安排.md)。10月26日复核的三域3000条历史摘要中，转写阶段占两个进程累计运行时间87.81%，Ultra字词定位占4.20%。这批2635份新侧车、365份完整旧侧车复用，两阶段缓存命中为0，不是全冷启动测速，也不是今天的实时阶段计时。

`source_batch_rows=8`是CPU来源组织上限；正式单窗策略中，转写和定位实际调用均为1条。旧冻结生产包的`backend_metrics`在worker终态才保存；新维护版本每片另有轻量`performance`计数。10月27日维护版本已增加周期阶段记录和显式转写计算配置；后续有限vLLM批请求与整批持久化已通过CPU假引擎检查。CPU预取、正式转写/定位异步批调度及每卡多副本资源启动器仍未实现，真实批推理和模型速度未测。先短小测速定位原因，再有目的地扩大三域样本检查吞吐、文字、字界和恢复；不以十小时长跑替代阶段测量。

正式任务的单窗、FP32/eager配置不能热改。新批量或数值方式须绑定新合同、缓存和输出，保留完整身份及质量规则；先独立验证再准备运行切换，不能让新旧任务共写队列。当前CPU检查或历史GPU成功均不自动覆盖新方案。


## 10月27日转写候选、容量测算和去重接续

实际范围、713项CPU验证及八卡机器测算见[实现与容量报告](../../docs/reviews/2026-10-27-转写提速实现与八卡容量测算.md)。用户最新要求先不做GPU测试，因此下面的准备命令没有执行模型或新八卡提交；默认正式入口仍是每UUID一个worker。

| 新入口 | 功能和边界 |
|---|---|
| `prepare_speed_validation.py` | 从显式base合同、三域30/3000/512输入及两个UUID准备三组独立源码/配置/命令；不自动选择或启动 |
| `audit_speed_comparison.py` | 按完整侧车和终态核文字/字界/真实调用及同profile恢复，详见[对照说明](转写提速对照验收.md) |
| `prepare_speed_continuation.py` | 保留旧producer结果，默认只写provisional报告；最终剩余输入须旧writer终态、旧父锁及稳定快照，不授启动资格 |
| `estimate_speed_capacity.py` | 固定SHA的八卡既有监控到显存预算及明确假设的ETA，不运行GPU或推导必然提速 |

`freeze_contract.py`新增成对参数`--asr-dtype float32|bfloat16 --asr-attention eager|sdpa`，只改变转写；Ultra保持既有FP32和单窗配置。缺省不强改旧模型的子配置。实际加载与首窗会记录子模块attention forward、SDPA和Q/K/V精度；请求未兑现就拒绝，不能只看顶层配置。此观察不是CUDA内核分析。

准备器的输入描述schema为`annotation-speed-matrix-preparation/v1`，包含`base_contract:{path,sha256}`、`indices:{screen,main,recovery}`（各为path/SHA）、`target_uuids`两项、`device_lock_root`和目标`python`绝对路径。代码/输出放新外部任务目录，旧合同、旧索引树与锁目录软链均拒绝。运行`--help`查看参数，准备成功不表示机器空闲。

接续准备器通过`--help`说明旧index/contract/queue/sidecar-tree、输出及可选终态材料的显式参数。默认只读预估，不复制千万行文本；`--execute-final`才会复制剩余分片，须保留真实旧终态与父锁证明。最终准备还会扫描未领取片中已经存在的旧侧车，逐条核commit后写新的保留账本，只把真正缺失者写入剩余输入。孤儿不偷偷修复或重算。准备报告的完整片引用不等于重开全部旧结果；后续联合工具会做这项完整核验，不能直接改原固定总分母放行。

容量工具的输入是`annotation-existing-monitor-capacity/v1`已保存八卡监控摘要及SHA，并显式给观察吞吐与剩余来源；`--reserve-mib`默认8192、`--peak-factor`默认1.15都是可见假设。输出为`annotation-eight-GPU-capacity-estimate/v1`，`speedup_measured=false`。每卡静态可放几份与现有程序支持几份分别报告，不能把表格当多副本启动许可。


## 后续架构研究优先级

[方案反思与执行优先级](../../docs/plans/2026-10-27-转写提速方案反思与执行优先级.md)将官方vLLM共享请求引擎列为长期优先研究候选，已有单窗BF16实验为最少新增代码的验证路径、同参数双副本为另一过渡候选。后续本轮已实现联合覆盖、唯一消费者、独立跨引擎协议和vLLM最小适配；多副本资源组和真实模型验证仍未完成。现有`audit_speed_comparison.py`继续只覆盖同工程源码、单窗的算子对照，不删除旧限制来凑通过。

当前实现及检查见[联合接续与跨引擎CPU验收](../../docs/reviews/2026-10-27-联合接续与跨引擎CPU实现验收.md)，接口分别见[联合覆盖与唯一读取](联合覆盖与唯一读取.md)、[跨引擎对照协议](跨引擎对照协议.md)和[vLLM阶段恢复](vLLM最小适配与阶段恢复.md)。新阶段库在整批持久化后才返回单条结果；不能用固定批次承诺尚未提交计算跨恢复逐位相同。真实模型、资源控制器、质量分层及人工完整字检查仍待后续验证。用户“GPU先不测”的要求继续有效，本轮没有新模型运行或云任务提交。

## 联合数据检查与实际运行分别验收

下面只检查已保存的来源、合同及结果，不启动模型：

```bash
python3 -B tools/open_annotation/cloud_gate.py \
  --spec /absolute/union.json --spec-sha '<联合描述SHA256>' --union-mode plan
python3 -B tools/open_annotation/cloud_gate.py \
  --spec /absolute/union.json --spec-sha '<联合描述SHA256>' --union-mode complete \
  --consumer-index /absolute/new-audit/consumer.sqlite
python3 -B tools/open_annotation/audit_engine_comparison.py --help
python3 -B tools/open_annotation/vllm_backend.py --help
```

`plan`不要求新结果存在；`complete`逐条重开旧新结果并绑定消费索引。两者都不是运行中的轻量巡检：大库会产生文本、旧结果读取及数据库磁盘开销，仅在正式交接或终验边界执行。本轮未对真实千万条执行这些操作。报告使用独立的`data_checks_passed`，GPU运行、云任务提交及训练/评测资格始终未授予；旧资源挑战、W2和实际模型验收均不放宽。

vLLM命令默认只校验配置，`--execute`会构造真实引擎，当前禁止调用。CPU假引擎测试包括实际文件级“阶段库提交→原生输出→完整侧车与commit→跨引擎审计→元数据恢复且零新模型调用”，不证明真实CUDA/vLLM可运行、显存或提速成立。
