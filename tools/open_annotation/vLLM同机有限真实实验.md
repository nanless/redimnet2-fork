# 有限真实实验的空卡与原独占锁要求

**真实模型构造已在代码中关闭，当前只保留文件检查和完整阶段缓存读取。** 闲卡门和原独占锁检查虽已补充，完整CUDA生命周期仍未实现；不能把这些CPU检查写成真实GPU已经安全可用。

2026年10月27日发生了两次真实尝试，必须分开记录：

1. 第一次把UUID填入 `CUDA_VISIBLE_DEVICES`。现场vLLM 0.19.1把该值转换为整数，导致构造中的模型结构检查失败，没有生成原生转写结果。
2. 第二次改为已核数字卡号，模型结构检查已过，EngineCore进入GPU/NCCL初始化。旧标注父进程持续执行独占监控，检测到目标卡第二个计算进程后拒绝继续，原标注任务因此失败。这不是无影响的同机试验，也不是模型文字质量失败。

两次尝试都没有产出可用于审计的原生转写结果，**没有速度、峰值容量、文字或完整字边界的选型结论**。原方案允许用自建共享许可绕过既有独占协议，是错误的。用户指定机器不等于允许破坏该机器正在执行的独占条件。失败材料和后续恢复由现场分别保留；本助手CPU修复不是旧标注已恢复的证据。

事故前助手源码摘要为 `ed41480a08415608eccc995c8adade4019375a0b139db8db8598511b56341ed7`。原字节、原测试和原说明已封存于本机 `.audit-local/vllm-live-probe-20261027/shared-protocol-incident/`，不覆盖失败版本或日志。旧任务故障材料另行保留；本助手的CPU修复不是旧任务已恢复的证据。

`vllm_live_probe.py` 现已撤销同卡共享执行路径。任何既有计算PID，不论归属是否明确、显存是否充足、利用率是否为0，都拒绝构造引擎。旧的 `annotation-shared-GPU-explicit-admission/v1` 配置不再被接受。不能通过填入 `shared_gpu_test_authorized=true`、列齐已知PID、历史CUDA关联证明或另建一个自己的锁来绕开旧守卫。旧 `gpu_launch.py`、`gpu_challenge.py` 和设备锁协议保持原样。

## 默认检查与固定输入

默认 `--inspect-only` 核文件，不新建结果目录，也不导入torch、Qwen或vLLM。主配置使用 `annotation-bounded-vllm-probe/v1`，文件引用统一为 `{"path":"实际绝对路径","sha256":"文件实际SHA256"}`。必须固定以下内容：

- `adapter_spec`：现有 `annotation-vllm-asr-spec/v1` 配置文件。真实目的必须是 `pilot_validation`，模型权重、处理器文件、实际Qwen/vLLM源码、版本、语言、上下文、预处理与生成上限均固定。
- `inputs`：JSON数组文件，最多8窗。每窗有 `source_id`、`window_id`、`pcm_path`、`pcm_file_sha256`、`input_pcm_sha256`、`model_input_frames`、`model_sample_rate`。输入为无pickle的单声道float32 `.npy`，16000Hz，每窗最多30秒，合计最多120秒。
- `probe_source_files`：助手、`vllm_backend.py`、`asr_stage_store.py` 的实际源码摘要；未来申请独占执行时，还必须固定 `gpu_launch.py`、`gpu_challenge.py`、`gpu_vendor/resources.py`，核实自己遵循的是哪份真实独占协议。
- `output_root`：全新的独立实验目录。`protected_roots` 明确列出旧生产队列、旧结果及其他禁止写入的目录。输出不能与这些目录或模型树相交。恢复只能使用同一配置及输入，禁止删掉错误材料后冒充首次执行。
- `wall_seconds`：1至600秒；`model_API_calls_allowed` 为整数0，`release_training_labels` 为false。

单卡模型并行数固定1，batch最多2，显式生成上限最多512 tokens。与旧生产条件比较时可保持512，不能私自改短。`gpu_memory_utilization` 最多0.20，按总卡显存计算；这只是引擎预算，不是硬件隔离，也不能证明瞬时峰值绝对受限。

这里只核模型输入PCM。没有重新解码原媒体或核完整采样率、声道和裁片过程，所以报告同时保留 `input_model_PCM_verified=true` 和 `source_audio_identity_verified=false`。转写成功也不表示完整字边界或正式数据质量已通过。

## 闲卡与原独占锁只作为后续准备条件

以下是CPU协议检查和后续设计的必要条件，仍不足以重新开启真实构造。准备材料时，`resource_admission` 必须引用新的 `annotation-idle-GPU-exclusive-admission/v1` 文件，包含：

- `idle_gpu_test_authorized=true`、`shared_gpu_test_authorized=false`、`allowed_existing_processes=[]`。非空允许列表本身也被拒绝。
- 已核任务名称、实际 `hostname`、精确 `gpu_uuid` 和带时区的 `expires_utc`；助手核主机名，不代替平台认证。
- `producer_sha256` 为适配配置对象规范摘要，`inputs_sha256` 为输入文件实际摘要。
- `headroom_mib` 至少8192，`max_probe_memory_mib` 为明确实验预算，`driver_used_limit_mib` 为0至256之间的保守空卡显存上限。
- `existing_exclusive_ticket` 引用真实旧 `open-annotation-resource-ticket/v1` 文件，`exclusive_protocol_bundle_sha256` 固定其原程序包摘要。票据必须绑定相同UUID和原设备锁路径、设备号与inode；真实机器还核相同系统boot ID。

助手从原票据取锁路径，读取**已经存在的同一个锁文件**，核单链接普通文件、所有者、设备号和inode，再尝试非阻塞独占flock。不会创建、写入、删除、换名或重建该锁。原标注父进程还持锁时，即使某次NVML瞬间没有计算行，也立即拒绝。CPU假引擎测试能确认该锁在构造、推理和保存调用期间持有，但现有工作进程内的ExitStack会在返回或异常时释放它，尚不能证明EngineCore及全部自身CUDA上下文已经排空。缺少真实原票据、路径或inode变化、源码协议不明，都不能靠临时新建的JSON或 `/tmp` 锁代替。

原锁通过后，还要取得两次相隔5秒的新鲜NVML快照：目标卡**没有任何计算行**、驱动显存不超过保守上限、总量与预算满足条件；两次之间任何变化都拒绝。助手自己的新目录锁和UUID协作锁继续使用，但只用于防止本助手重复启动，不能代替原任务的设备锁。

这些限制只协调遵守相同flock的程序，不能阻止其他无关程序忽略锁新建CUDA上下文。实际调度仍须由现场维护者核实；更安全的下一次真实验证位置是专用空闲卡或独立验证任务，而不是旧标注正在执行的卡。

## 数字卡号兼容和证据

现场 vLLM 0.19.1 会对 `CUDA_VISIBLE_DEVICES` 做整数转换，直接使用UUID曾导致 `ValueError: invalid literal for int()`，使模型结构检查失败。这是已经留证的构造兼容错误，不是转写质量结论。

为处理该版本，助手实读 `nvidia-smi index,uuid,pci.bus_id` 全卡映射。数字编号必须恰为0至N−1，UUID和PCI地址唯一，编号顺序必须与PCI地址排序相同；目标映射及整表在双快照中保持一致。满足后，仅在自己的新进程中设置已核数字 `CUDA_VISIBLE_DEVICES` 和 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。许可、原锁和协作锁始终以UUID为主身份。回执保留选择证据，并明确 `CUDA_runtime_UUID_verified=false`，不声称已单独通过CUDA运行时重核UUID。

官方 [vLLM 0.19.1 API文档](https://docs.vllm.ai/en/v0.19.1/api/vllm/) 声明 `enqueue` 返回请求ID，`wait_for_completion` 返回完整输出并接受 `use_tqdm`。版本文档只能支持接口判断，现场安装源码、Qwen参数透传、真实语言集合和tokenizer EOS仍须实核。

## 保存、恢复与进程清理

检查命令：

```bash
python -B tools/open_annotation/vllm_live_probe.py \
  --spec /绝对路径/有限实验配置.json --spec-sha 实际文件摘要 --inspect-only
```

当前 `--execute` 只允许完整阶段缓存恢复。若存在缺失阶段，代码在资源申请及真实factory之前明确拒绝：`real GPU construction is disabled`。自定义factory只允许显式 `cpu_fixture`，不能对pilot配置注入真实factory绕过。

独立CPU进程反例表明：工作进程内锁在factory或推理异常后可能先释放，而EngineCore后代仍存活；现有父进程只等待leader也不能证明全部进程组和上下文排空。完整修复还需由supervisor跨工作进程持原独占FD，确认全部自身进程及上下文排空后才释放，并保留未知上下文证据，不能杀未知PID。本轮未实现这套生命周期，因此关闭真实构造；不能把发送清理信号或leader退出当成问题已解决。

每个完整批次先在独立ASR事务库持久化，再逐条导出；全部结果不可覆盖地发布到 `native-results.json`，每次恢复写独立回执。若所有阶段结果完整，再次运行只读取这些结果，**不导入或构造真实模型、不检查资源许可、不发新模型请求**。中途完整批次提交后退出，恢复只允许未持久化批次重做。若切换到新的安全协议或源码配置，应使用新的实验目录保留旧材料，不能改写旧实验身份。

## CPU反例覆盖

测试使用假权重、假引擎和合成快照，不运行GPU。除了PCM、源码、映射、不可覆盖发布及恢复外，重点覆盖：已知旧标注PID也拒绝；旧共享JSON不能放行；卡看起来空闲但原parent持锁仍拒绝；缺少真实原锁票据或inode变化拒绝；原协议锁在构造和推理时确实保持；完整缓存恢复零新构造。另有明确反例证明pilot缺缓存时真实构造被拒绝、自定义pilot factory不能绕过、完整pilot格式stage仅读零新调用。格式测试中的结果由CPU假引擎制作，不是实际权重或GPU证据。这些CPU结果不代表旧标注已恢复、完整CUDA生命周期已实现或真实GPU质量已通过。
