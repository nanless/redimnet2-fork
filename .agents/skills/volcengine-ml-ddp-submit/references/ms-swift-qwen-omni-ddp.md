# MS-SWIFT Qwen Omni 多机 Full SFT

只在火山引擎任务涉及 MS-SWIFT、Qwen Omni、DeepSpeed 或 model-only 续训时读取。

## 启动合同

- 从平台注入的 worker 数、每机 GPU 数、rank、master host 和 port 构造分布式环境；提交前断言真实拓扑符合用户要求。
- 全局 batch 为 `nodes × GPUs_per_node × per_device_batch × gradient_accumulation`。区分单步全局 batch 与一个 epoch 的样本数，不要仅凭显存余量扩大 batch。
- Full SFT 使用 `tuner_type/train_type=full`；核对最终生效参数，而不是只读启动脚本字面值。
- ZeRO-2 分片 optimizer state 和梯度，但参数仍在各 rank 保留副本。解释显存时分别估算参数、梯度、optimizer、激活和临时通信缓冲。
- 共享输出、数据和缓存路径必须在所有节点可见。兼容软链接应验证目标且不能覆盖已有非链接目录。

## 音频+LLM训练范围

- `tuner_type/train_type=full`只说明所选可训练参数采用全参更新，不能证明音频塔实际可训练。纯音频SFT通常应把音频塔、音频投影/aligner、Thinker LLM和`lm_head`纳入训练，并冻结视觉、`talker`和`token2wav`；最终范围以具体模型合同和用户要求为准。
- MS-SWIFT的开关可能先冻结一组参数，再通过`trainable_parameters`重新启用子模块；学习率字段名也可能沿用多模态抽象，例如`vit_lr`实际控制当前模型的音频塔参数组。不要从`freeze_aligner`、`freeze_vit`或字段名孤立推断训练范围。
- 启动前和线上都保存scope attestation，至少列出每个语义参数组的tensor数、参数量、`requires_grad`状态和必冻前缀。用日志中的总可训练参数做算术交叉验证。审计历史checkpoint时，再抽查base与checkpoint的音频塔、projector、LLM和`lm_head`代表张量；完全相同的音频张量可证明该历史run没有更新音频部分。
- ZeRO-2归约后的`parameter.grad`不是可靠的线上逐组合同。真实单卡smoke可检查逐组有限非零梯度；线上使用scope attestation、有限loss、Trainer归约后的grad norm和真实optimizer step共同证明训练已启动。

## 跨模型严格对齐

用户要求与TLive等模型“对齐”时，先区分两类配置并保存对照矩阵：

1. **必须逐项核对的实验控制量：** 数据文件及SHA-256、行数、Prompt、assistant-only loss边界、语义训练范围、epoch、world size、micro-batch、梯度累积、global batch、optimizer-step总数、基础LR和各模块LR、weight decay、Adam beta、梯度裁剪、scheduler、warmup step、保存step、保留策略、eval频率、seed和精度。
2. **允许按架构/运行时不同的实现量：** attention backend、模型专属最大长度、音频processor、packing/padding实现和DataLoader预取。必须列出并说明影响，不能把它们误报成完全相同。

若用户要求精确warmup对齐，优先传绝对`warmup_steps`；仅传比例可能因总step和框架取整产生偏差。改变正在运行任务的LR、warmup或训练范围时，创建新的launcher、输出目录和run ID，先记录旧任务最后step及checkpoint清单，提交优先级为 `6` 的后继任务并核对 ID/详情后，再按授权停止旧任务。

## Smoke与验收器

- 验收脚本的期望LR、warmup和训练范围必须来自命令行参数或manifest，不要硬编码某一历史实验。配置变更后同时复验新合同和旧默认的向后兼容性。
- 一步cosine smoke在step结束后可能记录`learning_rate=0`，带warmup的最初日志也可能为0；这不等价于“没有执行更新”。验证参数更新时应比较更新前后张量，或使用能产生非零LR的两步/constant smoke；验证训练范围时检查optimizer参数组、逐组梯度和scope attestation。
- 生产启动后从实际日志核对三类证据：解析后的各参数组LR、精确`warmup_steps`、以及第一批有限loss/grad norm。只看launcher字面值不算验收完成。

## Checkpoint 完整性

一个可评测或续训的 safetensors checkpoint 至少满足：

1. `model.safetensors.index.json` 可解析；
2. `weight_map` 引用的每个分片都存在且非空；
3. `config.json`、tokenizer/processor配置和必要的模型附属文件存在；
4. `trainer_state.json` 存在时能够解析；
5. 没有临时文件，且分片、索引和 trainer state 在保守稳定窗口内不再变化；
6. 同一路径没有正在写入的文件句柄或近期修改证据。

不要仅按目录名中的数字选择“最新”checkpoint。跨 run 会出现同名目录，必须同时记录物理路径、run ID、父checkpoint、本地step和累计step。

## Model-only 续训

`save_only_model` 通常没有 DeepSpeed optimizer/scheduler 分片。即使命令包含 `resume_from_checkpoint`，若使用 `resume_only_model`，也应按以下语义报告：

- 加载模型权重；
- optimizer、scheduler和数据游标不保证延续；
- 新 run 的 `global_step`/`epoch`属于本地坐标；
- 大实验累计step为 `run.step_offset + local_step`；
- 大实验累计epoch为 `run.epoch_offset + local_epoch`。

不要重命名正在训练的实际 `checkpoint-*` 目录。用实验注册表或评测目录表达累计step，避免破坏 Trainer 引用和后续保存。

## 状态与ETA

- 平台详情页证明任务调度状态；rank-0日志证明训练循环进度；共享目录证明checkpoint落盘状态。三者不能相互替代。
- 从至少多个日志间隔估算 `seconds_per_step`。距离下次 epoch 保存的ETA应包含剩余训练步、eval时间和保存时间，并明确误差。
- 用户要求测试“最新模型”时，只有发现比已评测checkpoint更新且完整的权重才重新推理；同一确定性评测不要重复运行并冒充新结果。
