# 仓库状态（自动生成，不手改）

> 本文件由 `tools/gen_status.py` 生成。正文不复制这里的数字：正文写“见
> [STATUS.md](STATUS.md) 第 N 节”或复核命令，由读者自己查。
> 重新生成：`python3 tools/gen_status.py`；校验是否过期：`python3 tools/gen_status.py --check`；
> 检查文档有没有又把易变事实写死：`python3 tools/gen_status.py --lint`。
> 为保证可 diff，本文件不含生成时间戳；“时间”只来自 git 提交与实验产物的文件时间。

---

## 1. 仓库

- 完整 HEAD：`d83b68b889874fe7698e1031a33e6ed4bc60160c`
- 当前分支：`main`
- 顶端提交：`d83b68b 2026-11-09 08:10:59 +0000 大数据落点改为 group-shared：准入放宽 + 跳过已有标注 + 四代格式兼容`
- 未提交改动：3 项（`git status --porcelain | wc -l`；若本文件刚被改，会把自己算进去）

```
 M docs/README.md
 M docs/STATUS.md
?? docs/21-标注结果归集验收报告.md
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
- 摘要行：`1132 passed, 18 skipped, 6 warnings`

注意：这是本次生成时的观测。用例数、失败项都会变，引用前先重跑。

## 4. 已知产物

sha256 只给前 16 位，够 diff；要完整值用下面第 6 节的命令。

路径里的 `$EXP` = `/root/experiments/redimnet2`，是 2026-11-09 之前的实验产物根（**历史路径**，只读）。2026-11-09 起新产物写 `/root/group-shared/voiceprint/experiments/redimnet2/<实验名>/`，准入材料、账本、评测写 `.../artifacts/redimnet2/<用途>/`。

| 产物 | 路径 | 大小 | 摘要 | 文件时间 |
|---|---|---|---|---|
| LM 最终权重 | `$EXP/b6-mix-official-20261019/lm-from-ptn115-20261106-v1/lm/models/model_1.pt` | 265.5 MiB | `9858cf629eeb6eea` | 2026-11-07 05:37:32Z |
| PTN 第 115 轮 | `$EXP/b6-mix-official-20261019/ptn-resume-model47-capture-20261026-v1/ptn/models/model_115.pt` | 620.6 MiB | `65497b6a00efac41` | 2026-11-02 12:49:21Z |
| PTN 第 120 轮 | `$EXP/b6-mix-official-20261019/ptn-resume-model47-capture-20261026-v1/ptn/models/model_120.pt` | 620.6 MiB | `2798fe2379c44116` | 2026-11-03 02:12:02Z |
| v4-r2 冻结包清单 | `$EXP/open-annotation-ultra-20261025/native-speed-successor-20261028-v4-r2/frozen-native-reviewed-v4-r2/bundle_files.json` | 10.6 KiB | `e47d88f51834605f` | 2026-11-06 10:45:53Z |
| v4-r2 准入副本清单 | `$EXP/open-annotation-ultra-20261025/native-speed-successor-20261028-v4-r2/admission-20261108/formal-code/bundle_files.json` | 11.1 KiB | `342fdb9b90912608` | 2026-11-07 14:49:02Z |
| v4-r2 v4 pilot 副本清单 | `$EXP/open-annotation-ultra-20261025/native-speed-successor-20261028-v4-r2/pilot-20261108/prepared-v6/native-fp32-sdpa-b8/code/bundle_files.json` | 11.1 KiB | `914a1da53c7811b6` | 2026-11-07 17:59:19Z |
| v4-r6 生产包清单（20261109 准入） | `$EXP/open-annotation-ultra-20261025/native-speed-successor-20261028-v4-r2/admission-20261109/formal-code/bundle_files.json` | 11.1 KiB | `69f009e255bd9248` | 2026-11-08 06:03:52Z |
| LM 受控包 r3 | `$EXP/b6-mix-official-20261019/lm-from-ptn115-20261106-v1/package-r3` | 208 members | package `08a34b68ec26515b` / contract `bf437dc8fffe6982` | 2026-11-06 10:46:12Z |
| v4-r3 冻结包 | 无此目录 | missing | — | — |

复核：`sha256sum <artifact>`、`stat -c '%s %y' <artifact>`；包身份用其中的 `bundle_files.json`（pilot 与正式包是否同代码，用 `diff -rq` 比两个 code 目录，不要比目录名）

## 5. 最近三次标注提交

任务号从 `publish-work/*.json` 的文件名取；平台侧终态本地**没有**留档，`terminal-*.json` 里的 `task_id` 字段已知会错绑旧任务，不能当新任务的终态用。侧车只判有无、不逐个数：labels 树现在有上百万个文件，`find` 一次要几十分钟，而失败代次的 0 侧车才是有用的证据。

状态按 `queue/` 与 `failure.json` 判，不探测进程；生产已停但队列文件还在时，这里仍可能显示“运行中”，实时状态以任务页与 `docs/19` 第 2 节为准。

2026-11-09 起交接目录按存储规范写在 `/root/group-shared/voiceprint/open_annotation/redimnet2-silver-<日期>/` 下；`20261109` 已就位，更早三代仍在 `/root/experiments/redimnet2/open-annotation-ultra-*/`（历史路径）、迁移中；旧基址只读。归档与跳过规则见 `docs/20-标注结果归档与跳过规则.md`。

| 代次 | 交接目录 | 任务号（取文件名） | 状态 / 失败原因一句话 | 侧车条数 |
|---|---|---|---|---|
| v4 | `handoff-v4r2-20261108-v4` | `t-20261108005758-s9w6f` | `gpu_common.Refused: worker initialization or CUDA allocation phase timed out`（resources-attempt1/failure.json） | 0 |
| v5 | `handoff-v4r2-20261108-v5` | `t-20261108141622-9fkcm` | 无本地留档（run 目录为空） | — |
| v6 | `handoff-v4r2-20261108-v6` | `t-20261108144507-mrvsh` | `KeyboardInterrupt: parent terminated`（resources-attempt1/failure.json） | 有侧车（不逐个数） |

复核：`ls /root/group-shared/voiceprint/open_annotation/redimnet2-silver-*/handoff-*`（新基址）；旧代次用 `ls /root/experiments/redimnet2/open-annotation-ultra-*/handoff-*`（历史路径）；`find <run> -path '*labels*' -type f | wc -l`；`cat <run>/resources-attempt1/failure.json`

## 6. 说明

- 本文件只用于承载易变事实。`docs/*.md` 正文不写死 HEAD、权限位、测试计数、产物 sha、“已修/已验证”、目录里有什么。
- **存储规范（2026-11-09 起，用户硬性要求）**：大数据默认位置一律在 `/root/group-shared/voiceprint` 下——实验产物 `experiments/redimnet2/<实验名>/`，准入材料、账本、评测 `artifacts/redimnet2/<用途>/`，开源标注侧车 `open_annotation/redimnet2-silver-<日期>/`；`/root` 只放代码仓库与临时日志。`/root` 与 `/root/group-shared` 是两个不同挂载点，跨挂载点 `mv` 会退化成“复制 + 删除”（几十万小文件按小时计），所以新数据一开始就写最终位置。详见 `docs/19-今日工作与后续入口.md` 第 6 节；标注按代次归档、建索引与跳过规则见 `docs/20-标注结果归档与跳过规则.md`。
- 复核全部产物 sha（下面三条是 2026-11-09 之前的旧基址，**历史路径**；新产物按上面的规范写新基址）：

  ```bash
  # 历史路径（2026-11-09 前的产物仍在这里）
  sha256sum /root/experiments/redimnet2/b6-mix-official-20261019/lm-from-ptn115-20261106-v1/lm/models/model_1.pt \
            /root/experiments/redimnet2/b6-mix-official-20261019/ptn-resume-model47-capture-20261026-v1/ptn/models/model_{115,120}.pt
  ```

- 本文件由 `python3 tools/gen_status.py` 生成；`--check` 判过期，`--lint` 查正文硬编码。两者都不需要 GPU、不联网、不写仓库外路径。
