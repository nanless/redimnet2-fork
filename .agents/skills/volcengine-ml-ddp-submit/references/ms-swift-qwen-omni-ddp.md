# MS-SWIFT Qwen Omni 多机 Full SFT

只在火山引擎任务涉及 MS-SWIFT、Qwen Omni、DeepSpeed 或 model-only 续训时读取。

## 启动合同

- 从平台注入的 worker 数、每机 GPU 数、rank、master host 和 port 构造分布式环境；提交前用断言核对真实拓扑是否符合用户要求。
- 全局 batch 为 `nodes × GPUs_per_node × per_device_batch × gradient_accumulation`。单步全局 batch 和一个 epoch 的样本数是两件事，分开写；显存还有余量，也不足以成为扩大 batch 的理由。
- Full SFT 使用 `tuner_type/train_type=full`；核对最终生效的参数，不只读启动脚本里的字面值。
- ZeRO-2 分片 optimizer state 和梯度，但参数仍在各 rank 保留副本。解释显存时分别估算参数、梯度、optimizer、激活和临时通信缓冲。
- 共享输出、数据和缓存路径要在所有节点可见；只有部分节点能看到时，其余节点上的进程读不到同一份文件。兼容软链接先验证目标；目标位置已经是非链接目录时保留它，覆盖会丢掉目录里的内容。

## 音频+LLM训练范围

- `tuner_type/train_type=full` 只说明所选可训练参数采用全参更新，不能证明音频塔实际可训练。纯音频 SFT 通常应把音频塔、音频投影/aligner、Thinker LLM 和 `lm_head` 纳入训练，并冻结视觉、`talker` 和 `token2wav`；最终范围以具体模型合同和用户要求为准。
- MS-SWIFT 的开关可能先冻结一组参数，再通过 `trainable_parameters` 重新启用子模块；学习率字段名也可能沿用多模态抽象，例如 `vit_lr` 实际控制当前模型的音频塔参数组。只看 `freeze_aligner`、`freeze_vit` 或字段名，推断不出训练范围。
- 启动前和线上都保存 scope attestation，至少列出每个语义参数组的 tensor 数、参数量、`requires_grad` 状态和必冻前缀，并用日志中的总可训练参数做算术交叉验证。审计历史 checkpoint 时，再抽查 base 与 checkpoint 的音频塔、projector、LLM 和 `lm_head` 代表张量；抽查张量相同只证明这些张量在两个快照中相同，不能证明整个音频塔从未更新。是否纳入优化器，要结合该 run 的 scope attestation、冻结配置与参数组证据。
- ZeRO-2 归约后的 `parameter.grad` 不能作为线上逐组核对的依据。真实单卡 smoke 可以检查逐组有限非零梯度；线上要同时看 scope attestation、有限 loss、Trainer 归约后的 grad norm 和真实 optimizer step，才能说明训练已经启动。

## 跨模型严格对齐

用户要求与 TLive 等模型“对齐”时，先区分两类配置并保存对照矩阵：

1. 逐项核对的实验控制量：数据文件及 SHA-256、行数、Prompt、assistant-only loss 边界、语义训练范围、epoch、world size、micro-batch、梯度累积、global batch、optimizer-step 总数、基础 LR 和各模块 LR、weight decay、Adam beta、梯度裁剪、scheduler、warmup step、保存 step、保留策略、eval 频率、seed 和精度。
2. 允许按架构和运行时不同的实现量：attention backend、模型专属最大长度、音频 processor、packing/padding 实现和 DataLoader 预取。这些差异要列出来并说明影响；把它们报成完全相同，比较就失去意义。

若用户要求精确的 warmup 对齐，优先传绝对 `warmup_steps`；只传比例，总 step 和框架取整会带来偏差。改变正在运行任务的 LR、warmup 或训练范围时，先创建新的 launcher、输出目录和 run ID，记下旧任务最后的 step 及 checkpoint 清单，再提交优先级为 `6` 的后继任务；核对 ID 和详情之后，才按授权停止旧任务。

## Smoke与验收器

- 验收脚本的期望 LR、warmup 和训练范围要来自命令行参数或 manifest；写死某一历史实验的值，换配置后验收就失效。配置变更后，新合同和旧默认的向后兼容性都要复验。
- 一步 cosine smoke 在 step 结束后可能记录 `learning_rate=0`，带 warmup 的最初日志也可能为 0；这不等于“没有执行更新”。验证参数更新时比较更新前后的张量，或者用能产生非零 LR 的两步/constant smoke；验证训练范围时检查 optimizer 参数组、逐组梯度和 scope attestation。
- 生产启动后从实际日志核对三类证据：解析后的各参数组 LR、精确的 `warmup_steps`、第一批的有限 loss/grad norm。只看 launcher 字面值不算验收完成。

## Checkpoint 完整性

一个可供评测或续训的 safetensors checkpoint 至少满足：

1. 分片模型的 `model.safetensors.index.json` 可解析；单文件模型则按实际加载器核对 `model.safetensors`，不必为它制造索引；
2. 分片模型的 `weight_map` 引用文件全部存在且非空；单文件模型可完整读取，并与所选加载配置匹配；
3. `config.json`、tokenizer/processor 配置和必要的模型附属文件存在；
4. `trainer_state.json` 存在时能够解析；
5. 没有临时文件，且分片、索引和 trainer state 在保守稳定窗口内不再变化；
6. 同一路径没有正在写入的文件句柄或近期修改证据。

目录名里的数字说明不了哪个 checkpoint 最新。跨 run 会出现同名目录，选的时候要同时记录物理路径、run ID、父 checkpoint、本地 step 和累计 step。

## Model-only 续训

`save_only_model` 通常没有 DeepSpeed optimizer/scheduler 分片。即使命令里带 `resume_from_checkpoint`，用了 `resume_only_model` 也按下面的语义报告：

- 加载模型权重；
- optimizer、scheduler 和数据游标不保证延续；
- 新 run 的 `global_step`/`epoch` 属于本地坐标；
- 大实验累计 step 为 `run.step_offset + local_step`；
- 大实验累计 epoch 为 `run.epoch_offset + local_epoch`。

正在训练的实际 `checkpoint-*` 目录留着原名：Trainer 按这个路径引用它，改名会破坏引用和后续保存。累计 step 用实验注册表或评测目录表达。

## 状态与ETA

- 平台详情页证明任务调度状态；rank-0 日志证明训练循环进度；共享目录证明 checkpoint 落盘状态。三者不能相互替代。
- 从多个日志间隔估算 `seconds_per_step`。距离下次 epoch 保存的 ETA 要包含剩余训练步、eval 时间和保存时间，并写明误差。
- 用户要求测试“最新模型”时，只有发现比已评测 checkpoint 更新且完整的权重，才重新推理；同一确定性评测重复跑一遍结果不变，容易把旧结果当新结果报。
