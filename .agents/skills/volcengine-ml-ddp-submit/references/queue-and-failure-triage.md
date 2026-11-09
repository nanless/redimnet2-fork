# 排队、失败定位与交接窗口

提交之后的三件事：排队等多久、失败了先看哪里、交接任务要在什么时间窗内接上。这三件事都不看训练质量，只看“有没有真的在跑”。

实测基准 2026-11-08 至 2026-11-09（平台控制台 + `/root/group-shared/voiceprint/experiments/redimnet2/**` 运行目录 + `docs/STATUS.md` 第 5 节；2026-11-09 前的旧目录在 `/root/experiments/redimnet2/**`，历史路径）。排队位次、任务终态这类数字每轮都变，用的时候重读，不要照抄本文。

---

## 1. 排队：时间不可控，八卡要同一节点

2026-11-09 观测到的排队位次在 112/120 与 107/120 之间来回变动，过了几个小时又退回原来的量级。这说明排的不是“前面还剩多少人”，而是“什么时候有一台机器空出满足规格的资源”。

- 八卡任务要求**同一节点**空出 8 张卡。两台各空 4 张不能拼起来，所以队列里常有“总数够、位置没动”的情况。
- 位次回退不是自己插队失败，也不是任务被重置。它是别人释放或占用资源的结果。
- 排队期间平台状态只到 `排队中`，没有容器、没有日志、没有 GPU 分配。这时能核的只有：任务存在、字段正确、优先级 6。
- 排队不是失败，也不是“卡住”。重复提交只会多出几个抢同一份资源的副本，并且让后来判断“哪个才是目标任务”变难。

报告口径：写“排队中，位次 107/120（2026-11-09 某时观测）”。位次是瞬时值，隔一段时间会变，不要把一次读数当成预计时间。给出 ETA 需要历史同规格任务的等待分布，单次位次推不出等待时长。

GPU 碎片或池子不足属于容量约束，写进报告就够；绕过队列的做法（换队列、换 AZ、降规格再改回来）都会改变实验合同，需要单独确认。

---

## 2. 失败定位：按 平台日志 → 落盘 → 侧车 的顺序

任务显示失败以后，先花一分钟按下面三步走。顺序反了容易把“平台没留档”当成“程序没跑”，或者把“没产出”当成“代码挂了”。

### 第 1 步：平台日志最后几行

控制台实例日志的末尾通常直接给出失败原因。这一步最快，也最容易被跳过。

2026-11-08 的 v4 就是这样定位的：`resources-attempt1/failure.json` 里写着

```
gpu_common.Refused: worker initialization or CUDA allocation phase timed out
```

含义是 worker 初始化或 CUDA 分配阶段超时，不是训练逻辑错误，也不是数据问题。看到这类信息就不用再翻代码。

日志末尾为空的两种常见情况：容器根本没起来（镜像拉取失败、入口命令不存在），或者任务在排队阶段被停。这两种都不该往训练代码方向查。

### 第 2 步：`run-*` 目录有没有落盘

去共享盘的运行目录看 `run-*` 目录里有什么：

```bash
R=/root/group-shared/voiceprint/experiments/redimnet2/<实验>/<手数目录>/run-v6
ls -la "$R"
```

- 目录存在且有 `resources-attempt*`、`queue/`、`labels/` → 容器起来了，失败发生在运行期。
- 目录存在但基本是空的（只有一两个占位文件）→ 容器启动过，但没走到写产物的阶段。
- 整个 `run-*` 不存在 → 任务没进入执行阶段。2026-11-07 的 v2（`t-20261107191026-bwfgm`）与 2026-11-08 的 v3（`t-20261108000732-jrrg2`）属于这一类：本地无留档，run 目录为空，平台侧记录也补不出更多。

这一步的价值是把“程序跑到哪”与“平台说了什么”分开。两边不一致时，以目录里能 `stat` 到的东西为准。

### 第 3 步：侧车是不是 0

侧车指运行目录里按样本写出的标注结果（`labels/<contract>/<corpus>/*.jsonl` 一类）。计数：

```bash
find "$R" -path '*labels*' -type f | wc -l
```

`docs/STATUS.md` 第 5 节把最近三代标注提交整理成表，v2、v3、v4 的侧车条数都是 0。0 侧车的含义很具体：**没有任何一条样本产出**，所以问题在“开始写之前”，不在“写错”。

run-v6 的 `labels/` 下只有 `identity.json` 与三个语料目录（`cnceleb2`、`voxblink2`、`voxceleb2`），同样没有结果文件。这种形态说明目录结构建好了、写入没发生。

### 三步合起来怎么用

| 第 1 步 | 第 2 步 | 第 3 步 | 指向 |
|---|---|---|---|
| 有明确错误行（如 CUDA 超时） | `run-*` 有 `resources-attempt*` | 0 侧车 | 运行期基础设施失败，配置与数据没被消费 |
| 日志为空 | `run-*` 不存在 | 无侧车可数 | 没进入执行阶段，先查镜像、入口、挂载 |
| 日志正常退出 | `run-*` 完整 | 0 侧车 | 程序跑完了但没写结果，查写入路径与数量合同 |
| 日志正常退出 | `run-*` 完整 | 侧车数与预期不符 | 部分失败，按 manifest 逐条比 |

`docs/STATUS.md` 是本仓库唯一的易变事实页（自动生成），表格里的任务号、失败原因、侧车数都以它为准，并且它明确写了一条：`terminal-*.json` 里的 `task_id` 字段已知会错绑旧任务，不能当新任务的终态用。这一条要在报告里照抄，不要把本地终态文件的 task_id 当成平台任务号。

---

## 3. 交接窗口：任务创建后 5 分钟内起 `auto_publish`

采用队列交接（handoff）方式提交时，任务容器里的 waiter 与开发机上的 `auto_publish.py` 是一对：waiter 写出 `request.json`，`auto_publish` 读到后生成观测并 `publish`，waiter 才继续往下走。

时间窗的实测值：

| 环节 | 数值 | 证据 |
|---|---|---|
| waiter 等待上限 | `wait_seconds` 被断言在 1..1800 秒 | `native_batch_queue_handoff.py`：`1<=wait<=1800` |
| `auto_publish` 自身轮询 | 360 次 × 5 秒 = 1800 秒 | `auto_publish.py` 主循环 |
| v6 实际节奏 | `request.json` 06:47:58 UTC → `publication-v1` 06:50:28 UTC，约 2.5 分钟 | handoff-v6 目录时间戳 |

所以做法是：任务创建成功后**立刻**在开发机上起 `auto_publish`（5 分钟内）。它自己会等 `request.json` 出现，起早了没有副作用；起晚了只要还在 1800 秒里也能接上，但接不上的后果是双向的——waiter 侧到点退出，容器白起，这一代任务只能作废重来。

`auto_publish` 的调用形态：

```bash
AUTO_PUBLISH=/root/group-shared/voiceprint/artifacts/redimnet2/repo-handover-<日期>/auto_publish.py
# 迁移前这批脚本在 /root/codex_artifacts/repo-handover-20261107/（历史路径），还没搬时把上面一行换回旧路径
/root/miniforge3/envs/qwen3-asr/bin/python3.10 "$AUTO_PUBLISH" \
  --ctrl <outer-controller 目录> \
  --blueprint <blueprint.json> --blueprint-sha <sha> \
  --handoff-root <handoff 根> --workdir <publish-work> \
  --purpose-task-name <平台任务名>
```

三个易错点：

1. `--blueprint-sha` 要与文件内容一致，publish 会再核一次；sha 对不上直接拒。
2. `--workdir` 里的 `terminal-*.json`、`observation-*.json` 是**不可覆盖**的（`w()` 里 `assert not p.exists()`），同一目录重跑会立刻失败。换一代就换一个 workdir。
3. `--old-task-id` 有默认值（`t-20261025054739-4kzhp`）。新一批交接要显式传当前前驱任务号，否则终态观测会把旧任务写进新证据里。

另外，`auto_publish` 生成的观测里 `competing_successor_tasks` 恒为空、`remote_proc_enumeration_performed` 恒为 false，这两项是脚本写死的，不能当“已确认没有竞争任务”的证据。

---

## 4. 报告口径

- 排队：写位次与观测时刻，不写 ETA，除非有同规格历史等待分布。
- 失败：三步各给一条证据（日志末行、`run-*` 的 `ls`、侧车计数），再给结论。只有其中一条时不写“根因”。
- 交接：写任务 ID、创建时刻、`request.json` 与 `publication-v1` 的时刻、差多少秒；超过 1800 秒窗口的写“错过窗口”，不写“稍后补发”。
- 平台终态与本地留档分开写；本地留档缺 `task_id` 时如实写“本地无该 ID 留档”，不从文件名反推。
