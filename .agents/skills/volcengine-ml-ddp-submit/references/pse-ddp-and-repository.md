# PSE 2×8 DDP 与远程仓库操作

## launcher 契约

PSE 静态 launcher 从平台读下面五个变量：

| 平台变量 | 含义 |
| --- | --- |
| `MLP_WORKER_NUM` | 节点数，即 `nnodes` |
| `MLP_WORKER_GPU` | 每节点 GPU 数，即 `nproc_per_node` |
| `MLP_ROLE_INDEX` | 当前节点 rank |
| `MLP_WORKER_0_HOST` | master address |
| `MLP_WORKER_0_PORT` | master port |

两个 8-GPU worker 的配置拓扑是 `2/8/16`：平台表单选 PyTorch DDP、worker 数填 2，规格选 `ml.pni2l.28xlarge`（GPU 型 112 vC，每 worker 8 张 A800-SXM4-80GB）。规格每次都从实时控制台重新核实；资源目录调整后，历史名称会失效。历史文档写过的 `ml.hpcpni2l.28xlarge` 在 2026-11-09 的实际任务详情里查不到，以详情页显示为准。

代码开发机的 GPU 数不改变平台任务规格，只限制本地预检能做到什么程度。一张 GPU 的开发机无法验证 2×8 launcher 的启动行为，本地结果也就不能写成已验证。

## 入口命令不变量

平台入口是一行 `/bin/bash <脚本绝对路径>`（网页编辑器对长命令会丢首字符，写法见[控制台复制任务提交](console-clone-submission.md) 第 4.1 节）。下面五件事由那个脚本承担，不在入口里展开：

1. 使用 `set -Eeuo pipefail`；
2. 验证仓库和所有中性存储源路径存在；
3. 用安全且幂等的函数创建兼容软链接：已正确链接的直接接受，只移除空目标目录，非空目标不覆盖；所有 worker 同时启动时也要得到一致结果，优先用已入库并通过检查的 wrapper；
4. 进入 `/root/code/gitlab_repos/se_train`；
5. 调用已核验的 Volcengine 2×8 launcher，例如 `recipes/pse/merged_datasets_20260724/train_volcengine_2x8_globalutt_pairedrir.sh`。

挂载策略以 [平台 DDP 任务工作流](platform-ddp-workflow.md) 为准。PSE 常用方案是用户 PFS → `/root`，共享 PFS → `/shared/...`，再建立 `/root/group-shared` 和 `/root/yaww-speech-private` 的安全兼容链接；实际选哪一种以实时挂载表为准，避免覆盖实际目录。

## PSE 表单专项检查

- `CONFIG` 指向新的后继 TOML，不指向已经跑过的配置；
- `RESUME=auto` 只用于全新实验目录或已知兼容 checkpoint；
- `PSE_WANDB` 与预期 secret 配置一致；secret 值不写进日志和报告；
- stale lease takeover 和 distributed timeout 有明确设置；
- 只保留训练实际使用的 vePFS；未使用的 CloudFS/TOS 移除，省掉无关的 IAM 检查；
- 仓库由 `/root` PFS 提供时，重新选择代码依赖并清空旧的本地 `se_train` 上传快照，否则代码来源不唯一；
- 运行时和重试策略有意设置；
- 提交只在当前会话授权包含实际提交、且完整表单已核对后执行。已有授权持续有效，同一动作不重复索取确认。

## 修改配置前检查远程仓库

```bash
ssh <dev-host> 'cd /root/code/gitlab_repos/se_train && \
  git status --short --branch && git log -3 --oneline'
```

先确认预期分支、工作区状态、目标 TOML、launcher 和 checkpoint，再用仓库的 `load_config` 读取完整父链。历史 TOML 已被任务用过时，新建后继 TOML，不在原文件上改：改完之后就无法证明旧任务读的是哪份配置。

## 按授权提交和推送

先分清 Git 服务和实际执行主机。本机 Mac 对 GitHub 始终只读，下面任何 push 示例都不授权在 Mac 上执行。本次明确允许的 GitHub 写入只在 `dev_a800_2gpus` 自身仓库和进程里执行；SSH 发起远程命令可以，本机 push 透传不可以。原 `se_train` 的 GitLab 命令只作用于已授权的 GitLab 仓库，改 remote 也不会把它变成 GitHub 上传通道。本机上的 disabled 配置与已删除分支仍受原有约束；提交作者按该项目的用户要求设置。

核对服务器原始 remote 配置和实际展开 URL：既有的 `insteadOf` 会把指定的 GitHub 目标替换成第三方代理，只看目标名会以为推到了 GitHub。认证用服务器上已有的方式，不搬运 Mac 私钥，也不搬本机 push 流程。用 bundle 同步浅历史时，先核原真实浅边界和服务器可用对象；本机 bundle verify 成功，证明不了空服务器已经能导入。缺祖先时可以由服务器从指定仓库只读取得固定提交，导入后再核 HEAD、作者和工作区。认证失败保留原件；远端 SHA 未回读一致前不报告 push 完成。

任务授权确实包含所需 commit/push 时才执行；纯检查或只准备表单不包含这些写入。暂存确切的新文件或修改，不用宽泛路径：

```bash
ssh <dev-host> 'cd /root/code/gitlab_repos/se_train && \
  git diff --check && git status --short && \
  git add <exact-file> && \
  git commit -m "PSE：<中文原因>" && \
  GIT_SSH_COMMAND="ssh -p 22 -o ConnectTimeout=15" \
    git push origin <branch>'
```

上述已授权 GitLab 服务器仓库可能默认走 12222 端口，同一 Git 服务也能通过 22 端口访问。服务器 push 超时时，确认主机和仓库身份一致后才用一次性 `GIT_SSH_COMMAND`；它不构成本机 GitHub 禁写的例外。服务器 fetch/push 用同一 override，并核验：

```text
HEAD == origin/<branch>
worktree clean
```

交付时报告确切文件、commit hash、远端分支、push 结果和任何未提交的用户改动。
