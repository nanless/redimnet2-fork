> 2026-10-23用户澄清后，本页仅为第一版任意秒裁片诊断。当前主实验保留完整字词、真实自然长度，先CPU后等待两卡空闲。请先读[word-complete-finetuning.md](word-complete-finetuning.md)，不要按本页秒数策略启动主微调。

# 官方骨干的声学VAD变长微调与时长评测

## 何时使用及边界

用户要求从已下载官方模型做变长微调、短语音泛化、不等长注册/测试或长度分布消融时读本页。它是独立研究线，不能修改或接续当前八卡PTN/LM任务，不能重写其冻结清单。先读相关源码目录AGENTS并核现场路径/HEAD/资源。

2026-10-23独立实现位于dev_a800_2gpus:/root/code/github_repos/redimnet2-duration-finetune-20261023，分支codex/duration-finetune-20261023，来自0a3f6dd501e224a2bed85b6207df7c99046b4272。输出/root/experiments/redimnet2/duration-finetune-20261023；本机完整研究、计划、代码和验收在/Users/zhounan/Documents/声纹降噪/ReDimNet2变长微调_20261023。详细使用说明在独立源码docs/19-声学VAD变长微调与时长矩阵.md，计划docs/plans/2026-10-23-duration-finetune.md。主仓库已有文档被保留复制到新工作树，原目录文件保持不变；新工作树是本次用户新功能的隔离实现，不是恢复旧临时别名。

## 起点和研究对照

首版从官方b6-vb2+vox2+cnc2_v0-lm.pt开始，51,152,991字节，SHA287365f6f485b19e65e5176554f8f7123bfa8d85185f3d2c040eab51acec9868。兼容入口/root/codex_artifacts/verification/volcengine/redimnet2-b6mix-20261019/reference/official-b6-vb2+vox2+cnc2_v0-lm.pt；重新核resolve及SHA，不把历史兼容路径当永久主目录。

公开载荷只有model_config/state_dict：821状态键、192维输出。严格按公开配置装骨干，新建与本次speaker_map绑定的SphereFace2头、冷SGD/AMP。不能说恢复了官方分类头或优化器；与八卡从零检查点和115轮LM接续完全区分。

M0官方零训练；M1同VAD固定5秒；M2同VAD变长。M1/M2同官方骨干、同新头seed、同样本/说话人顺序及五秒父窗、同更新预算/优化策略；首轮仅改长度计划。工程M2为1/2/3/5秒、概率1/6/1/6/1/6/1/2，非最佳配置结论。等更新时M2语音秒更少，必须分报样本、更新、语音秒和时间。

先用批内同长、批间变长；无波形wrap/repeat/pad、无lengths前向，两个rank同step同长度。现有逐条mask会改善补零，但不严格等价：TF有效帧边界/stride、归一化和普通训练BN仍有问题。不能把旧cos>=0.90测试或单样本mask近似当padding不变性证明。逐条真长度评测最直接；以后若混长成批需重新严测边界/梯度/BN，而不是只在pool加mask。

## 数据入口与解码器

新duration_ft/data.py及scripts/prepare_duration_finetune.py严格消费原VAD RTTM/旁侧JSON，核源stat、推理参数身份、有限合法时间、采样率/帧数、段数、重叠越界、标签SHA。缺/空/坏/过期/不足分别计，禁止静默回退。

VAD目录/root/group-shared/voiceprint/data/wespeaker_data/vox2_vb2_cnc2_v0/mix_join/rttm_firered；冻结inference_signature a56629f741671964f03e3b7ba3f2087d0b3ed76622b9d527e68b501eff325810。签名是推理身份摘要，不是数字签名或人工质量保证。语音时间轴可拼接不相邻核心段，保存原采样区间与拼接偏移；不冒称目标说话人纯度或完整字词边界。

当前工程清单576训练/72身份、96留出/12身份，三语料各24+4身份，每人8条，均够5秒预测语音。它是有界首先合格抽样，不是全量代表性质量集。当前训练源音频校验不含内容SHA，勿说逐字节复验；评测音频会hash原字节。正式分层/全量准备和跨语料实体/同源去重另做。

redimnet2解释器依赖audio-flamingo父环境，不能升级/替换共享环境。新进程局部LD_LIBRARY_PATH选择/root/experiments/redimnet2/duration-finetune-20261023/runtime_audio/lib，libsndfile1.2.2 SHA c872fb84af2afc0e96b603e0000f0f407af86c49ad1126f2785e5722c7bb27d7；训练会核真实/proc/self/maps路径。私有库从既有已审库字节复制，只供新任务；旧PTN仍原库、旧FLAC七事件许可不适用于新数据。

## 训练、恢复和输出保护

scripts/train_duration_finetune.py --config --dataset --official --output --device cpu|cuda，--stop-after N只截本次运行，不改total_steps。首次运行目录必须全新；完整train_step_N.pt、公开embedding_step_N.pt、各rank身份/曝光/RNG分开保存。BN运行统计冻结、affine可训练；工程版全骨干有梯度，不把head-only冒充全模型微调。

恢复同时核数据/配置/world_size/源码身份、global_step/采样位置和各rankRNG。新resume_guard在训练核心、CUDA和进程组初始化前运行，拒旧步覆盖、断尾/重复/缺rank日志、临时或失败残留、活跃写者及同一步已开始的恢复attempt。失败现场不删；要分支或审查后另定恢复方案。不要用改元数据或删日志伪造连续恢复。

CPU和当前CUDA设备随机状态分开设置；新头用CPU default_generator的局部fork，不能用torch.manual_seed悄悄manual_seed_all其他可见设备。完整RNG捕获/恢复仅本rank对应CUDA设备。实际GPU复现精度还需现场测试，不把CPU逐位一致外推到未跑GPU。

## 评测、阈值和统计

官方自然Vox1-O不是8秒等长：本次4708头范围3.9601–69.0401s、中位6.4801。快速验证首8秒wrap与论文自然整段不可混称。

scripts/eval_duration_matrix.py三个子命令prepare-calibration/calibrate/evaluate。主all-O轴0.5/1/2/3/full、全部37611对；扩展轴0.5/1/2/3/5/8/full固定4460共同对（2340正/2120负），1595实际端点。5秒可用20234对另表，不跨不同试验群体直接解释时长差。每格固定试验及中心嵌套裁片、精确采样数；不足长度拒绝，full自然整段；工程非标准协议显式标记，不当O成绩。

缓存绑模型/源代码/解码器/设备精度、原音频字节、真实区间和预处理；不能只按路径。GPU模型/设备异常立即停，单文件解码错有次数上限；原异常/完整缺测试验保留，缺测格metrics为空，不算成功子集成绩，不涂零。可续用已完成cache，但身份改变需新输出。

第一版测试是raw连续墙钟时长、无测试VAD。P1预测语音预算矩阵未因训练有VAD而自动实现，需同冻结模型/参数另准备标签并核真实资格；elapsed和predicted-speech要分表。

EER/minDCF描述测试区分度，其test扫阈值不作部署阈值。独立O开发集100身份×5跨video，1000正/123750负；每模型在dev自然full确定FAR1%/.1%阈值，冻结应用所有测试时长。Vox1-dev对O独立，对E/H不独立；官方预训练语义身份重叠未核。阈值没真实计算则unavailable，不从O补拟。

duration_ft/metrics.py支持ties、DCF、冻结阈值、single_speaker_bootstrap及paired_speaker_bootstrap；同人权m_s、异人m_s*m_t，默认500/seed20261023/95%条件CI。配对差必须同音频/裁片/预处理/试验摘要，不计训练seed/开发阈值不确定性。summarize_duration_results.py重算分数并作图；缺测不能伪造。CPU单格工具scripts/eval_duration_cpu_cell.py会隐藏CUDA，spawn单线程、按现场CPU资源限制worker、用共享严格cache；大真实提取是显式调用，非self-test。

## 资源与证据

2026-10-23初查两卡已有不可归属PID2323291，占约55,646/71,916MiB；GPU0/1显存余量不是独占授权，0%util也不等于空闲。scripts/launch_duration_two_gpu.py核新鲜显存/进程、预算和额外reserve，已占卡需明确共享授权。不得kill或清理原进程。

训练CUDA需DURATION_FT_GPU_AUTHORIZED=1和正gpu-memory-mib，评测需allow-gpu和预算；这些参数只是执行开关，不是用户授权替代品。PyTorch内存fraction不包括全部CUDA上下文/外部库，也不限制算力，不能承诺共享卡绝对无干扰。

现场是否完成GPU/完整矩阵/质量对照，读本次实施报告及对应command/summary/checkpoint，不凭本页历史状态推断。CPU4步恢复或32步工程微调不证明泛化提高。原仓库与原LM材料保留protection-before/after逐文件摘要；不写被冻结清单掩盖新实验差异。
