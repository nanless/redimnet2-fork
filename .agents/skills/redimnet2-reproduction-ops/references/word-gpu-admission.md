# 完整字词双卡操作与CPU检查

2026-10-23。本轮用户明确先完成CPU、等两卡空闲再运行。当前开发机两卡仍有既有进程，因此没有运行GPU训练、CUDA前向或NCCL实算，也没有留后台等待程序。本页写的是新增独立GPU实现的实际边界；原CPU版本和检查点不修改，八卡PTN/LM目录与冻结材料不修改。

## 新入口

独立树`/root/code/github_repos/redimnet2-duration-finetune-20261023`新增`word_ft/gpu_train.py`、`word_ft/gpu_resources.py`、`scripts/launch_word_two_gpu.py`和两份`configs/word_ft/*gpu_float32.json`。GPU入口是完整训练循环，不是绕过CPU入口限制的捷径。受支持的配置只有一种：两进程、每rank一个GPU、NCCL、float32、不用AMP/TF32。真实长度逐条前向、等权累积、固定BN统计、同一前缀组/头初始化与完整步保存都保留。

GPU检查点用独立的`word-training-gpu-fp32/v1`，配置要求`runtime_policy=cuda-float32-nccl/v1`。CPU断点在CUDA初始化之前就拒绝。GPU恢复绑定torch/CUDA/NCCL/cuDNN、驱动、两卡UUID/顺序、预算、Python/numpy/scipy、soundfile及重采样源码等身份；启动时固定源码摘要，之后每步、保存前及父监控都要检查漂移。一边执行旧导入代码、一边把新磁盘摘要写进断点，恢复出来的就不是同一套代码。

## 资源检查与进程归属

先在CPU上核数据、配置、官方权重与私有解码库，再取两UUID的固定目录合作锁。读取两次真实nvidia-smi快照，至少相隔5秒，检查真实采集时钟窗口、UUID/驱动/内存一致性、有无既有compute PID、保守driver-used上限及显式预算＋预留。容器看不见的PID或未知计数按占用/不明拒绝；利用率0%不能当作空闲的证据。

固定锁目录为`/root/experiments/redimnet2/.word-gpu-device-locks`，锁覆盖本次torchrun的整个生命周期。票据遇到符号链接直接拒绝，且只单次读取；它绑定字节摘要、父进程启动时刻/祖先关系、持锁FD及inode、采集新鲜度、配置/数据/输出/恢复请求。检查点有两个：CPU准备之后一次，打开CUDA之前再一次。大数据核验超过票据时限时，在开CUDA之前就拒绝；放宽旧票据，绑定的就是过期状态。

合作锁不等于平台排他预留，阻止不了外部调度。启动后要继续检查外部PID；本实验出异常或发现外部进程时，只退出自身实验，NVML里任意PID都不发信号。torchrun用新session及`--run-path --start-method=spawn`，工作进程继承同一进程组；清理时还核进程组、session、启动时刻和一次性nonce。父进程退出后也要查残留工作进程；父进程结束不等于全部结束。只读nonce，不保存完整的进程环境。

PyTorch allocator预算不等于驱动/NCCL/上下文的全量硬上限；首次真卡要测峰值并核预留。容器里NVML hostPID如果不能可靠映射，按拒绝处理；“恰好新出现两PID”不能作为自动认领的依据。NVML total与CUDA totalGlobalMem的严格检查也还没在真卡上确认；若因此拒绝，保存两原值再核原因，不直接放宽。两个接口的定义分别见[NVIDIA NVML文档](https://docs.nvidia.com/deploy/pdf/NVML_API_Reference_Guide.pdf)和[CUDA12.4运行时参考](https://docs.nvidia.com/cuda/archive/12.4.0/pdf/CUDA_Runtime_API.pdf)。

## 已完成的检查

- 61项服务器CPU mock/负例全部通过，0跳过；覆盖旧/假快照、忙卡/未知PID、锁/票据/源码/运行身份与CPU断点拒绝。这些用例没有执行GPU算子。
- 真实CPU进程测试确认：leader先退出后，残留工作进程被自身组清理，无关的测试进程保持存活。
- 真实torchrun/spawn CPU测试确认：两rank的进程组/session属于本次新torchrun，各一个CPU线程，CUDA未初始化，无残留。
- 北京时间15:14:45.649与15:14:50.738做了唯一一轮真实忙卡双快照，间隔5.089秒；既有PID2323291仍占两卡，used55,657/71,925MiB，free25,571/9,303MiB。入口以exit2正确拒绝。探测进程没有import torch，也没有创建torchrun/ticket/训练输出。

失败首轮测试的原文保留：其中一次只是新异常类型与测试期望不一致，修正精确断言后复验，生产检查没有放宽。真实CPU训练的原`word_ft/train.py/schema.py`摘要保持不变；新GPU文件另作快照，新版摘要覆盖不了旧CPU训练身份。

## 真卡空闲后的执行顺序

先检查参考/待审单位政策与共同前缀组，再做双卡四步工程验证、连续与2+2恢复、公开导出回读、最短单位的前反向、梯度与BN、进程清理、显存峰值及吞吐。候选只走显式银标工程配置，边界真值不因此升级。硬件通过之后才安排有界的U1/U2质量对照，用同单位表、冻结的独立开发阈值，以及未参与选模的测试集。

`probe-only`要显式指定：它只读查询设备，不运行模型或GPU计算，也不创建torchrun训练进程：

```bash
cd /root/code/github_repos/redimnet2-duration-finetune-20261023
/root/miniforge3/envs/redimnet2/bin/python scripts/launch_word_two_gpu.py \
  --probe-only --gpu-indices 0 1 --memory-mib 4096 \
  --report /root/experiments/redimnet2/duration-finetune-20261023/<全新检查文件>.json
```

实际执行还要显式给`--execute`及config/dataset/official/output；命令用最终交付配置和全新输出，本页的占位路径不原样运行。当前忙卡条件下不执行。所有CPU/GPU科学状态、原始日志及新源码摘要见本机`ReDimNet2变长微调_20261023/验证结果/双卡入口CPU验收/`，独立审阅见`验证/GPU资源与进程保护独立审阅.md`。这套代码存在、CPU检查通过，既不等于GPU硬件可运行，也不等于短语音泛化改善。
