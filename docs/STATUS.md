# 仓库状态（自动生成，不手改）

> 本文件由 `tools/gen_status.py` 生成。正文不复制这里的数字：正文写“见
> [STATUS.md](STATUS.md) 第 N 节”或复核命令，由读者自己查。
> 重新生成：`python3 tools/gen_status.py`；校验是否过期：`python3 tools/gen_status.py --check`；
> 检查文档有没有又把易变事实写死：`python3 tools/gen_status.py --lint`。
> 为保证可 diff，本文件不含生成时间戳；“时间”只来自 git 提交与实验产物的文件时间。

---

## 1. 仓库

- 完整 HEAD：`d28b06dcb3a929967d425904534f7554a1f4b4d3`
- 当前分支：`main`
- 顶端提交：`d28b06d 2026-11-08 01:12:40 +0000 docs/13：补与官方结果表的完整对比（12 格实测，含校准证据与 SITW 缺列说明）`
- 未提交改动：138 项（`git status --porcelain | wc -l`；若本文件刚被改，会把自己算进去）

```
 M .agents/skills/redimnet2-reproduction-ops/SKILL.md
 M .agents/skills/redimnet2-reproduction-ops/references/20261021-audit-baseline.md
 M .agents/skills/redimnet2-reproduction-ops/references/asr-speed-and-continuation.md
 M .agents/skills/redimnet2-reproduction-ops/references/boundary-selection-comparison-20261024.md
 M .agents/skills/redimnet2-reproduction-ops/references/character-boundary-research.md
 M .agents/skills/redimnet2-reproduction-ops/references/complete-word-experiments.md
 M .agents/skills/redimnet2-reproduction-ops/references/comprehensive-audit-20261025.md
 M .agents/skills/redimnet2-reproduction-ops/references/duration-finetuning.md
 M .agents/skills/redimnet2-reproduction-ops/references/flac-seek-regressions.md
 M .agents/skills/redimnet2-reproduction-ops/references/full-open-labeling-20261024.md
 M .agents/skills/redimnet2-reproduction-ops/references/lm-launch-20261106.md
 M .agents/skills/redimnet2-reproduction-ops/references/model-storage-20261023.md
 M .agents/skills/redimnet2-reproduction-ops/references/monitoring-and-checkpoints.md
 M .agents/skills/redimnet2-reproduction-ops/references/multicase-boundary-audit.md
 M .agents/skills/redimnet2-reproduction-ops/references/open-annotation.md
 M .agents/skills/redimnet2-reproduction-ops/references/production-sidecar-20261024.md
 M .agents/skills/redimnet2-reproduction-ops/references/ptn-maintenance-stop-and-resume-20261024.md
 M .agents/skills/redimnet2-reproduction-ops/references/ptn-resume-ordinary-entry-fix-20261024.md
 M .agents/skills/redimnet2-reproduction-ops/references/ptn-resume-spawn-fix-20261024.md
 M .agents/skills/redimnet2-reproduction-ops/references/ptn-resume-submission-20261024.md
... 其余 118 项省略
```

复核：`git rev-parse HEAD`、`git log -1 --format='%h %ad %s'`、`git status --porcelain | wc -l`

## 2. 关键脚本可执行位

| 脚本 | 权限 | `test -x` | 能否 `./xxx.sh` |
|---|---|---|---|
| `tools/ptn_resume/runtime/launch.sh` | 755 | ok | 能 |
| `tools/ptn_resume/lm_launch/package_launch.sh` | 755 | ok | 能 |
| `tools/open_annotation/launch_queue_handoff.sh` | 755 | ok | 能 |
| `tools/open_annotation/launch_portable_successor.sh` | 755 | ok | 能 |
| `tools/ptn_reference/archive/launch.sh` | 755 | ok | 能 |

复核：`test -x <file> && echo ok`、`stat -c '%a %n' <file>`、`git ls-files -s <file>`（`100755` 即带可执行位）

## 3. 测试

命令：`/root/miniforge3/envs/USE/bin/python -m pytest tools/open_annotation/tests -q -p no:cacheprovider`

- 退出码：0
- 摘要行：`1094 passed, 18 skipped, 6 warnings`

注意：这是本次生成时的观测。用例数、失败项都会变，引用前先重跑。

## 4. 已知产物

sha256 只给前 16 位，够 diff；要完整值用下面第 6 节的命令。

| 产物 | 路径 | 大小 | 摘要 | 文件时间 |
|---|---|---|---|---|
| LM 最终权重 | `$EXP/b6-mix-official-20261019/lm-from-ptn115-20261106-v1/lm/models/model_1.pt` | 265.5 MiB | `9858cf629eeb6eea` | 2026-11-07 05:37:32Z |
| PTN 第 115 轮 | `$EXP/b6-mix-official-20261019/ptn-resume-model47-capture-20261026-v1/ptn/models/model_115.pt` | 620.6 MiB | `65497b6a00efac41` | 2026-11-02 12:49:21Z |
| PTN 第 120 轮 | `$EXP/b6-mix-official-20261019/ptn-resume-model47-capture-20261026-v1/ptn/models/model_120.pt` | 620.6 MiB | `2798fe2379c44116` | 2026-11-03 02:12:02Z |
| v4-r2 冻结包清单 | `$EXP/open-annotation-ultra-20261025/native-speed-successor-20261028-v4-r2/frozen-native-reviewed-v4-r2/bundle_files.json` | 10.6 KiB | `e47d88f51834605f` | 2026-11-06 10:45:53Z |
| v4-r2 准入副本清单 | `$EXP/open-annotation-ultra-20261025/native-speed-successor-20261028-v4-r2/admission-20261108/formal-code/bundle_files.json` | 11.1 KiB | `342fdb9b90912608` | 2026-11-07 14:49:02Z |
| v4-r2 v4 pilot 副本清单 | `$EXP/open-annotation-ultra-20261025/native-speed-successor-20261028-v4-r2/pilot-20261108/prepared-v6/native-fp32-sdpa-b8/code/bundle_files.json` | 11.1 KiB | `914a1da53c7811b6` | 2026-11-07 17:59:19Z |
| LM 受控包 r3 | `$EXP/b6-mix-official-20261019/lm-from-ptn115-20261106-v1/package-r3` | 208 members | package `08a34b68ec26515b` / contract `bf437dc8fffe6982` | 2026-11-06 10:46:12Z |
| v4-r3 冻结包 | 无此目录 | missing | — | — |

复核：`sha256sum <artifact>`、`stat -c '%s %y' <artifact>`；包身份用其中的 `bundle_files.json`（pilot 与正式包是否同代码，用 `diff -rq` 比两个 code 目录，不要比目录名）

## 5. 最近三次标注提交

任务号从目录名与 `publish-work/*.json` 的文件名取；平台侧终态本地**没有**留档，`terminal-*.json` 里的 `task_id` 字段已知会错绑旧任务，不能当新任务的终态用。

| 代次 | 交接目录 | 任务号（取文件名） | 失败原因一句话 | 侧车条数 |
|---|---|---|---|---|
| v2 | `handoff-v4r2-20261107-v2` | `t-20261107191026-bwfgm` | 无本地留档（run 目录为空） | 0 |
| v3 | `handoff-v4r2-20261108-v3` | `t-20261108000732-jrrg2` | 无本地留档（run 目录为空） | 0 |
| v4 | `handoff-v4r2-20261108-v4` | `t-20261108005758-s9w6f` | `gpu_common.Refused: worker initialization or CUDA allocation phase timed out`（resources-attempt1/failure.json） | 0 |

复核：`ls /root/experiments/redimnet2/open-annotation-ultra-*/handoff-*`；`find <run> -path '*labels*' -type f | wc -l`；`cat <run>/resources-attempt1/failure.json`

## 6. 说明

- 本文件只用于承载易变事实。`docs/*.md` 正文不写死 HEAD、权限位、测试计数、产物 sha、“已修/已验证”、目录里有什么。
- 复核全部产物 sha：

  ```bash
  sha256sum /root/experiments/redimnet2/b6-mix-official-20261019/lm-from-ptn115-20261106-v1/lm/models/model_1.pt \
            /root/experiments/redimnet2/b6-mix-official-20261019/ptn-resume-model47-capture-20261026-v1/ptn/models/model_{115,120}.pt
  ```

- 本文件由 `python3 tools/gen_status.py` 生成；`--check` 判过期，`--lint` 查正文硬编码。两者都不需要 GPU、不联网、不写仓库外路径。
