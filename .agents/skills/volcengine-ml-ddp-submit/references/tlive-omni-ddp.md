# TLive-Omni 火山引擎 DDP

只在任务涉及 TLive-Omni Full SFT 时读取。模型、数据和梯度合同由 `$tlive-omni-speaker-count-sft` 维护；本页只保留平台提交的不变量。

## 共享路径

- 用户 vePFS 挂到 `/root`，独立 Conda 环境放这里。
- group-shared vePFS 挂到 `/shared/group-shared`，部署 workspace、模型、数据、索引、日志和输出放这里。
- JSONL 音频路径是 `/root/group-shared/...` 时，launcher 要安全地验证或创建软链 `/root/group-shared -> /shared/group-shared`；目标已经是非软链目录时就保留原样，覆盖会丢掉里面原有的内容。
- Torch extensions、Triton 和临时文件按节点各用独立的 `/tmp`；16 个 rank 共用一个编译缓存会争用同一份文件。

## 2×8 启动合同

从平台变量映射：

```text
MLP_WORKER_NUM    -> NNODES
MLP_WORKER_GPU    -> NPROC_PER_NODE
MLP_ROLE_INDEX    -> NODE_RANK
MLP_WORKER_0_HOST -> MASTER_ADDR
MLP_WORKER_0_PORT -> MASTER_PORT
```

启动前断言 2 个 worker、每 worker 8 GPU、node rank 为 0/1、每节点 8 个可见 GPU，训练进程内 `WORLD_SIZE=16`。启动命令用 `torchrun --nnodes --nproc-per-node --node-rank --master-addr --master-port`，不用 `--standalone`：它按单机本地 rendezvous 处理，多机场景下各节点接不到同一个 master。

全局 batch 是 `16 × per_device_batch × accumulation`。以 micro-batch 1、accumulation 16 为例，global batch 为 256；3,757,173 条样本按这个标称 batch 向上取整，约为 14,677 个 optimizer steps。这是标称值，实际还要核对 DDP 分片、drop_last 与尾累积的处理。

## 复制参考任务

复制前先只读核实参考任务。复制后逐项复核镜像、PyTorchDDP、队列/AZ、优先级 `6`、运行时、2×8 A800 规格、两项 vePFS、入口、名称、描述、输出和自动重试；创建完成后再从详情页核对实际优先级。当前已验证可复制的 8 卡 P6 表单是 `t-20261026074833-ctjsk`，它的字段基线与操作步骤见[控制台复制任务提交](console-clone-submission.md)，本文不重复点法。初次配置验证时自动重试保持关闭：路径、OOM 和依赖错误都是确定性失败，重试只会重复同一个错误，所以先修配置，再创建独立后继任务。替换任务时先提交并核对新任务 ID 和详情，再按已有停止授权处理这个具体旧任务；没有停止授权就保留旧任务。

共享 vePFS 上的代码不走网页源代码上传。默认用 clean 的 pushed commit；用户没有授权 commit/push 时，可以另建专用部署 snapshot 和哈希 manifest，并在交付里写明未 commit/push。

提交只做一次。平台显示运行中之后，还要从两个 worker 的持久日志看到 rendezvous/collective 初始化、world size 16、模型权重加载，以及至少一个有限 loss 的 batch，才能说训练启动了。
