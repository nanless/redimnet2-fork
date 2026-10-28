# TLive-Omni 火山引擎 DDP

只在任务涉及 TLive-Omni Full SFT 时读取。模型、数据和梯度合同由 `$tlive-omni-speaker-count-sft` 维护；本页只保留平台提交的不变量。

## 共享路径

- 用户 vePFS 挂到 `/root`，用于独立 Conda 环境。
- group-shared vePFS 挂到 `/shared/group-shared`，用于部署 workspace、模型、数据、索引、日志和输出。
- 若 JSONL 音频是 `/root/group-shared/...`，launcher 必须安全验证或创建 `/root/group-shared -> /shared/group-shared`，不得覆盖非软链目录。
- Torch extensions、Triton 和临时文件使用每节点独立的 `/tmp`；不要让 16 rank 共用一个编译缓存。

## 2×8 启动合同

从平台变量映射：

```text
MLP_WORKER_NUM    -> NNODES
MLP_WORKER_GPU    -> NPROC_PER_NODE
MLP_ROLE_INDEX    -> NODE_RANK
MLP_WORKER_0_HOST -> MASTER_ADDR
MLP_WORKER_0_PORT -> MASTER_PORT
```

断言 2 workers、每 worker 8 GPU、node rank 为 0/1、每节点 8 个可见 GPU，训练进程内 `WORLD_SIZE=16`。调用 `torchrun --nnodes --nproc-per-node --node-rank --master-addr --master-port`，禁止 `--standalone`。

全局 batch 是 `16 × per_device_batch × accumulation`。以 micro-batch 1、accumulation 16 为例，global batch 为 256；3,757,173 条样本的一轮是 14,677 optimizer steps。

## 复制参考任务

复制前先只读核实参考任务。复制后逐项复核镜像、PyTorchDDP、队列/AZ、优先级 `6`、运行时、2×8 A800 规格、两项 vePFS、入口、名称、描述、输出和自动重试；创建后从详情页再次核对实际优先级。初次配置验证保持自动重试关闭；路径、OOM 或依赖错误属于确定性失败，先修配置再创建独立后继任务。替换任务须先提交并核对新任务 ID/详情，再停止旧任务。

共享 vePFS 代码不使用网页源代码上传。默认使用 clean pushed commit；若用户没有授权 commit/push，可建立专用部署 snapshot 和哈希 manifest，并在交付中明确未 commit/push。

提交只执行一次。平台显示运行中后，还要从两个 worker 的持久日志证明 rendezvous/collective 初始化、world size 16、模型权重加载和至少一个有限 loss batch，才能宣称训练启动。
