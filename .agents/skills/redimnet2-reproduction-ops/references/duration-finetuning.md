> 2026-10-23用户澄清后，本页只记录第一版任意秒裁片诊断。当前主实验保留完整字词、真实自然长度，先CPU后等待两卡空闲。读之前先看[word-complete-finetuning.md](word-complete-finetuning.md)；按本页的秒数策略启动主微调，走的是已被替代的口径。

# 官方骨干的声学VAD变长微调与时长评测

## 何时使用及边界

用户要求从已下载官方模型做变长微调、短语音泛化、不等长注册/测试或长度分布消融时读本页。它是独立研究线，与当前八卡PTN/LM任务分开：既不修改或接续它，也不重写它的冻结清单。动手前先读相关源码目录的AGENTS，再核现场路径/HEAD/资源。

2026-10-23独立实现位于dev_a800_2gpus:/root/code/github_repos/redimnet2-duration-finetune-20261023，分支codex/duration-finetune-20261023，来自0a3f6dd501e224a2bed85b6207df7c99046b4272。输出/root/experiments/redimnet2/duration-finetune-20261023（历史路径）；本机完整研究、计划、代码和验收在/Users/zhounan/Documents/声纹降噪/ReDimNet2变长微调_20261023。详细使用说明在独立源码docs/19-声学VAD变长微调与时长矩阵.md，计划docs/plans/2026-10-23-duration-finetune.md。主仓库已有文档复制到新工作树并保留原位，原目录文件不变；新工作树是本次用户新功能的隔离实现，不是恢复旧临时别名。

## 起点和研究对照

首版从官方b6-vb2+vox2+cnc2_v0-lm.pt开始，51,152,991字节，SHA287365f6f485b19e65e5176554f8f7123bfa8d85185f3d2c040eab51acec9868。兼容入口/root/codex_artifacts/verification/volcengine/redimnet2-b6mix-20261019/reference/official-b6-vb2+vox2+cnc2_v0-lm.pt（历史路径）；用前重新核resolve及SHA，历史兼容路径不能当永久主目录。

公开载荷只有model_config/state_dict：821状态键、192维输出。骨干严格按公开配置装，另新建与本次speaker_map绑定的SphereFace2头、冷SGD/AMP。官方分类头或优化器并没有被恢复；这条线也和八卡从零检查点、115轮LM接续完全分开。

M0是官方零训练，M1是同VAD固定5秒，M2是同VAD变长。M1/M2用同官方骨干、同新头seed、同样本/说话人顺序及五秒父窗、同更新预算/优化策略，首轮只改长度计划。工程M2为1/2/3/5秒、概率1/6/1/6/1/6/1/2，这不是最佳配置的结论。等更新时M2的语音秒更少，样本、更新、语音秒和时间四项要分开报。

先用批内同长、批间变长；不用波形wrap/repeat/pad，也不走lengths前向，两个rank同step同长度。现有逐条mask能改善补零，但不严格等价：TF有效帧边界/stride、归一化和普通训练BN仍有问题。旧cos>=0.90测试、单样本mask近似，都当不了padding不变性的证明。逐条真长度评测最直接；以后若混长成批，边界/梯度/BN要重新严测，只在pool上加mask不够。

## 数据入口与解码器

新duration_ft/data.py及scripts/prepare_duration_finetune.py严格消费原VAD RTTM/旁侧JSON，核源stat、推理参数身份、有限合法时间、采样率/帧数、段数、重叠越界、标签SHA。缺/空/坏/过期/不足分别计，静默回退会把缺的段当成正常数据。

VAD目录/root/group-shared/voiceprint/data/wespeaker_data/vox2_vb2_cnc2_v0/mix_join/rttm_firered；冻结inference_signature a56629f741671964f03e3b7ba3f2087d0b3ed76622b9d527e68b501eff325810。这个签名是推理身份摘要，既不是数字签名，也不保证人工质量。语音时间轴可以拼接不相邻核心段，原采样区间与拼接偏移要保存；目标说话人纯度、完整字词边界都不冒称。

当前工程清单是576训练/72身份、96留出/12身份，三语料各24+4身份，每人8条，都够5秒预测语音。这是有界的首先合格抽样，不是全量代表性质量集。当前训练源音频校验不含内容SHA，所以别说逐字节复验；评测音频会hash原字节。正式分层/全量准备和跨语料实体/同源去重另做。

redimnet2解释器依赖audio-flamingo父环境，因此不做升级/替换共享环境。新进程用局部LD_LIBRARY_PATH指向/root/experiments/redimnet2/duration-finetune-20261023/runtime_audio/lib（历史路径），libsndfile1.2.2 的 SHA 为 c872fb84af2afc0e96b603e0000f0f407af86c49ad1126f2785e5722c7bb27d7；训练会核真实的/proc/self/maps路径。私有库从既有已审库按字节复制，只供新任务；旧PTN仍用原库，旧FLAC七事件许可不适用于新数据。

## 训练、恢复和输出保护

scripts/train_duration_finetune.py --config --dataset --official --output --device cpu|cuda，--stop-after N只截本次运行，不改total_steps。首次运行要求目录全新；完整train_step_N.pt、公开embedding_step_N.pt、各rank身份/曝光/RNG分开保存。BN运行统计冻结、affine可训练；工程版全骨干有梯度，head-only不能冒充全模型微调。

恢复同时核数据/配置/world_size/源码身份、global_step/采样位置和各rankRNG。新resume_guard在训练核心、CUDA和进程组初始化前运行，拒绝这些情况：旧步覆盖、断尾/重复/缺rank日志、临时或失败残留、活跃写者、同一步已开始的恢复attempt。失败现场不删；要分支或审查之后再定恢复方案。改元数据或删日志伪造出来的连续恢复，下一次恢复就对不上。

CPU和当前CUDA设备的随机状态分开设置；新头用CPU default_generator的局部fork。torch.manual_seed会悄悄对全部可见设备做manual_seed_all，这里不用。完整RNG捕获/恢复仅本rank对应CUDA设备。实际GPU复现精度还要现场测试；CPU逐位一致推不到没跑过的GPU。

## 评测、阈值和统计

官方自然Vox1-O不是8秒等长：本次4708个头的范围是3.9601–69.0401s，中位6.4801。快速验证用的首8秒wrap和论文的自然整段不能混称。

scripts/eval_duration_matrix.py有三个子命令：prepare-calibration/calibrate/evaluate。主all-O轴是0.5/1/2/3/full、全部37611对；扩展轴0.5/1/2/3/5/8/full固定4460共同对（2340正/2120负），1595个实际端点。5秒可用的20234对另表列出；跨不同试验群体直接解释时长差，结论站不住。每格固定试验及中心嵌套裁片、精确采样数；不足长度拒绝，full用自然整段；工程上的非标准协议显式标记，不当O成绩。

缓存绑模型/源代码/解码器/设备精度、原音频字节、真实区间和预处理；只按路径绑不够。GPU模型/设备异常立即停，单文件解码错有次数上限；原异常/完整缺测试验保留，缺测格的metrics留空，不算成功子集的成绩，也不涂零。已完成的cache可以续用，身份变了就要新输出。

第一版的测试用raw连续墙钟时长，没有测试VAD。P1的预测语音预算矩阵不会因为训练用了VAD就自动成立，要同一冻结模型/参数另准备标签并核真实资格；elapsed和predicted-speech分表。

EER/minDCF描述的是测试区分度，它扫出来的test阈值不当部署阈值。独立O开发集是100身份×5跨video，1000正/123750负；每个模型在dev自然full上确定FAR1%/.1%阈值，冻结后应用到所有测试时长。Vox1-dev对O独立，对E/H不独立；官方预训练的语义身份重叠未核。阈值没有真实计算就写unavailable，不从O补拟。

duration_ft/metrics.py支持ties、DCF、冻结阈值、single_speaker_bootstrap及paired_speaker_bootstrap；同人权m_s、异人m_s*m_t，默认500/seed20261023/95%条件CI。配对差取同音频/裁片/预处理/试验摘要，不计训练seed/开发阈值不确定性。summarize_duration_results.py重算分数并作图；缺测不伪造。CPU单格工具scripts/eval_duration_cpu_cell.py会隐藏CUDA，spawn单线程、按现场CPU资源限制worker、用共享严格cache；大真实提取是显式调用，不是self-test。

## 资源与证据

2026-10-23初查时，两卡已有不可归属的PID2323291，占约55,646/71,916MiB；GPU0/1显存余量不是独占授权，0%util也不等于空闲。scripts/launch_duration_two_gpu.py核新鲜显存/进程、预算和额外reserve；已占卡要有明确的共享授权。原进程不kill，也不清理。

训练CUDA要DURATION_FT_GPU_AUTHORIZED=1和正的gpu-memory-mib，评测要allow-gpu和预算；这些参数只是执行开关，代替不了用户授权。PyTorch内存fraction不包括全部CUDA上下文/外部库，也不限制算力，所以共享卡上的绝对无干扰承诺不了。

现场是否完成GPU/完整矩阵/质量对照，要读本次实施报告和对应的command/summary/checkpoint，不能凭本页的历史状态推断。CPU4步恢复或32步工程微调证明不了泛化提高。原仓库与原LM材料保留protection-before/after逐文件摘要；用被冻结的清单掩盖新实验差异，掩盖不住。
