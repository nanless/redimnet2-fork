# PSE 2×8 DDP 与远程仓库操作

## launcher 契约

PSE 静态 launcher 使用以下平台变量：

| 平台变量 | 含义 |
| --- | --- |
| `MLP_WORKER_NUM` | 节点数，即 `nnodes` |
| `MLP_WORKER_GPU` | 每节点 GPU 数，即 `nproc_per_node` |
| `MLP_ROLE_INDEX` | 当前节点 rank |
| `MLP_WORKER_0_HOST` | master address |
| `MLP_WORKER_0_PORT` | master port |

两个 8-GPU worker 要求配置拓扑为 `2/8/16`，平台表单选择 PyTorch DDP、2 个 worker，并在当前资源命名仍有效时选择 `ml.hpcpni2l.28xlarge`（每 worker 8 张 A800）。每次都从实时控制台重新核实规格，不把历史名称当成永久值。

代码开发机的 GPU 数不会改变平台任务规格，但会限制本地预检能力。一张 GPU 的开发机不能验证 2×8 launcher；不得伪造该预检结果。

## 入口命令不变量

入口命令必须：

1. 使用 `set -Eeuo pipefail`；
2. 验证仓库和所有中性存储源路径存在；
3. 用安全且幂等的函数创建兼容软链接：接受已正确链接，只移除空目标目录，拒绝覆盖非空目标；所有 worker 同时启动时也必须得到一致结果，优先使用已入库并通过检查的 wrapper；
4. 进入 `/root/code/gitlab_repos/se_train`；
5. 调用已核验的 Volcengine 2×8 launcher，例如 `recipes/pse/merged_datasets_20260724/train_volcengine_2x8_globalutt_pairedrir.sh`。

挂载策略以 [平台 DDP 任务工作流](platform-ddp-workflow.md) 为准。PSE 常用方案是用户 PFS → `/root`，共享 PFS → `/shared/...`，再建立 `/root/group-shared` 和 `/root/yaww-speech-private` 的安全兼容链接；但必须根据实时挂载表选择，不能覆盖实际目录。

## PSE 表单专项检查

- `CONFIG` 指向新的后继 TOML；
- `RESUME=auto` 只用于全新实验目录或已知兼容 checkpoint；
- `PSE_WANDB` 与预期 secret 配置一致，但不得输出 secret 值；
- stale lease takeover 和 distributed timeout 有明确设置；
- 只保留训练实际使用的 vePFS；移除未使用的 CloudFS/TOS，避免无关 IAM 检查；
- 仓库由 `/root` PFS 提供时，重新选择代码依赖并清空旧的本地 `se_train` 上传快照；
- 运行时和重试策略有意设置；
- 只有当前会话授权包含实际提交且完整表单已核对，才执行提交。已有授权持续有效，不再为同一动作重复索取确认。

## 修改配置前检查远程仓库

```bash
ssh <dev-host> 'cd /root/code/gitlab_repos/se_train && \
  git status --short --branch && git log -3 --oneline'
```

确认预期分支、工作区状态、目标 TOML、launcher 和 checkpoint。用仓库的 `load_config` 读取完整父链。若历史 TOML 已被任务使用，新建后继 TOML，不在原文件上修改。

## 按授权提交和推送

先区分Git服务和实际执行主机。本机Mac对GitHub永远只读，任何下面的push示例都不授权在Mac上执行。本次明确允许的GitHub写入只在`dev_a800_2gpus`自身仓库和进程执行；SSH发起远程命令可以，本机push透传不可以。原`se_train`的GitLab命令仍仅作用于已授权的GitLab仓库，不得靠改remote把它当GitHub上传通道。本机disabled配置和已删除分支均保留原限制，提交作者按该项目的用户要求设置。

核服务器原始remote配置与实际展开URL，防止既有`insteadOf`把指定GitHub目标替换为第三方代理。认证用服务器已有方式，不搬运Mac私钥或本机push流。若用bundle同步浅历史，先核原真实浅边界和服务器可用对象；本机bundle verify成功不是空服务器已能导入的证明。缺祖先可由服务器从指定仓库只读取得固定提交，导入后再核HEAD、作者和工作区。认证失败保留原件，远端SHA未回读一致前不报push完成。

只有任务授权确实包含所需 commit/push 时才执行；纯检查或仅准备表单不包含这些写入。暂存确切的新文件或修改，不使用宽泛路径：

```bash
ssh <dev-host> 'cd /root/code/gitlab_repos/se_train && \
  git diff --check && git status --short && \
  git add <exact-file> && \
  git commit -m "PSE：<中文原因>" && \
  GIT_SSH_COMMAND="ssh -p 22 -o ConnectTimeout=15" \
    git push origin <branch>'
```

上述已授权GitLab服务器仓库可能默认使用12222端口，而同一Git服务也可通过22端口访问。服务器push超时时，只有确认主机和仓库身份一致后才使用一次性`GIT_SSH_COMMAND`；它不是本机GitHub禁写的例外。服务器fetch/push使用同一override，并核验：

```text
HEAD == origin/<branch>
worktree clean
```

交付时报告确切文件、commit hash、远端分支、push 结果和任何未提交的用户改动。
