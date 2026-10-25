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
| `qwen_backend.py`、`alignment_observer.py` | 每次实际模型调用只有一个窗口；保存原始时间分类、库的 FIX 处理和官方返回，不额外裁剪或拉伸 |
| `normalization.py`、`quality_axes.py`、`schema.py` | 保存文字规范化中间步骤，分别记录文字、边界、语言、来源和身份的待审状态 |
| `work_queue.py`、`sidecar.py` | 单父进程锁、分片领取、并发代次、不可覆盖发布、缓存身份和中断恢复 |
| `report.py`、`audit_outputs.py` | 核对计账、侧车与提交记录；退出 GPU 后持同一父锁检查孤儿文件和软链接，不删证据 |
| `gpu_launch.py`、`gpu_bootstrap.py`、`gpu_challenge.py`、`gpu_common.py`、`gpu_vendor/` | 资源双快照、物理设备锁、七阶段显存挑战、精确子进程身份、归还与退出 |
| `freeze_contract.py` | 把维护源码复制到新目录，生成新的成对试验/全量合同和源码摘要清单 |
| `word_adapter.py` | 保守完整字候选检查和工程接口转换，不释放训练资格 |
| `quality_figures.py` | 明确选单的 CPU 原声波形、频谱和原始/FIX/拥有区时间图；不把图当人工听审 |
| `validate_two_gpu.py` | 新冻结副本内的 30 条小批、3000 条冷运行及 512 条部分缓存恢复验收；只能显式执行 |
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
