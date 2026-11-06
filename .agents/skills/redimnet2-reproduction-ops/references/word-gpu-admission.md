# 完整字词双卡操作与CPU检查

2026-10-23。本轮用户明确先完成CPU、等两卡空闲再运行。当前开发机两卡仍有既有进程，因此没有运行GPU训练、CUDA前向或NCCL实算，也没有后台等待程序。本页是新增独立GPU实现的实际边界；原CPU版本和检查点不修改，八卡PTN/LM目录与冻结材料不修改。

## 新入口

独立树`/root/code/github_repos/redimnet2-duration-finetune-20261023`新增`word_ft/gpu_train.py`、`word_ft/gpu_resources.py`、`scripts/launch_word_two_gpu.py`和两份`configs/word_ft/*gpu_float32.json`。GPU入口是完整训练循环，不是绕过CPU入口限制；仅支持两进程、每rank一个GPU、NCCL、float32、无AMP/TF32。保留真实长度逐条前向、等权累积、固定BN统计、同一前缀组/头初始化与完整步保存。

GPU检查点采用独立`word-training-gpu-fp32/v1`，配置要求`runtime_policy=cuda-float32-nccl/v1`。CPU断点在CUDA初始化之前拒绝。GPU恢复绑定torch/CUDA/NCCL/cuDNN、驱动、两卡UUID/顺序、预算、Python/numpy/scipy、soundfile及重采样源码等身份；启动时固定源码摘要，后续每步/保存前及父监控检查漂移，不能一边执行旧导入代码一边把新磁盘摘要写进断点。

## 资源检查与进程归属

先CPU核数据、配置、官方权重与私有解码库，再获取两UUID的固定目录合作锁。读取两次真实nvidia-smi快照，至少相隔5秒，检查真实采集时钟窗口、UUID/驱动/内存一致性、无既有compute PID、保守driver-used上限及显式预算＋预留。容器看不见的PID或未知计数仍按占用/不明拒绝，不能因利用率0%认空闲。

固定锁目录为`/root/experiments/redimnet2/.word-gpu-device-locks`，锁持续覆盖本次torchrun全生命周期。票据拒绝符号链接并单次读取，绑定字节摘要、父进程启动时刻/祖先关系、持锁FD及inode、采集新鲜度、配置/数据/输出/恢复请求；在CPU准备之后、打开CUDA之前再次检查。大数据核验若超过票据时限，应在开CUDA之前拒绝，不放宽旧票据。

合作锁不等于平台排他预留，无法阻止外部调度。启动后继续检查外部PID；本实验发生异常或发现外部进程时只退出自身实验，绝不对NVML任意PID发信号。torchrun使用新session及`--run-path --start-method=spawn`，工作进程继承同一进程组；清理还核进程组、session、启动时刻及一次性nonce。父进程退出后也查残留工作进程，不把父进程结束冒充全部结束。只读取nonce，不保存完整进程环境。

PyTorch allocator预算不等于驱动/NCCL/上下文的全量硬上限；首次真卡需测峰值并核预留。容器中NVML hostPID若不能可靠映射，仍拒绝，不能用“恰好新出现两PID”自动认领。NVML total与CUDA totalGlobalMem的严格检查也尚未真卡确认；若因此拒绝，保存两原值核原因，不能直接放宽。两接口的定义应分别看[NVIDIA NVML文档](https://docs.nvidia.com/deploy/pdf/NVML_API_Reference_Guide.pdf)和[CUDA12.4运行时参考](https://docs.nvidia.com/cuda/archive/12.4.0/pdf/CUDA_Runtime_API.pdf)。

## 已完成的检查

- 61项服务器CPU mock/负例全部通过，0跳过；覆盖旧/假快照、忙卡/未知PID、锁/票据/源码/运行身份与CPU断点拒绝。它们没有执行GPU算子。
- 真实CPU进程测试确认leader先退出后的残留工作进程被自身组清理，无关测试进程保持存活。
- 真实torchrun/spawn CPU测试确认两rank的进程组/session属于本次新torchrun、各一CPU线程、CUDA未初始化、无残留。
- 北京时间15:14:45.649与15:14:50.738唯一一轮真实忙卡双快照相隔5.089秒，既有PID2323291仍占两卡，used55,657/71,925MiB，free25,571/9,303MiB；入口以exit2正确拒绝。探测进程未import torch、未创建torchrun/ticket/训练输出。

失败首轮测试原文保留：其中一次只是新异常类型与测试期望不一致，修正精确断言后复验，没有放宽生产检查。真实CPU训练的原`word_ft/train.py/schema.py`摘要保持不变；新GPU文件另作快照，不能用新版摘要覆盖旧CPU训练身份。

## 真卡空闲后的执行顺序

先检查参考/待审单位政策与共同前缀组，再做双卡四步工程验证、连续与2+2恢复、公开导出回读、最短单位前反向、梯度与BN、进程清理、显存峰值及吞吐。候选只能显式银标工程配置，不能升级边界真值。硬件通过后才安排有界U1/U2质量对照，使用同单位表、冻结独立开发阈值和未参与选模的测试集。

`probe-only`必须显式指定：它只读查询设备，不运行模型或GPU计算，不创建torchrun训练进程：

```bash
cd /root/code/github_repos/redimnet2-duration-finetune-20261023
/root/miniforge3/envs/redimnet2/bin/python scripts/launch_word_two_gpu.py \
  --probe-only --gpu-indices 0 1 --memory-mib 4096 \
  --report /root/experiments/redimnet2/duration-finetune-20261023/<全新检查文件>.json
```

实际执行另需显式`--execute`及config/dataset/official/output，命令必须使用最终交付配置和全新输出，不将本页占位路径原样运行；当前忙卡条件下不执行。所有CPU/GPU科学状态、原始日志及新源码摘要见本机`ReDimNet2变长微调_20261023/验证结果/双卡入口CPU验收/`，独立审阅见`验证/GPU资源与进程保护独立审阅.md`。这套代码存在及CPU检查通过不等于GPU硬件可运行，也不等于短语音泛化改善。
