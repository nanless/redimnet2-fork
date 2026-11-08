# 标注生产现状与看板（2026-11-09）

这页回答三件事：现在跑的是哪个任务、进度怎么算、看板怎么开。历史三代失败（v2/v3/v4）的边界见仓库 `docs/16-实验维护与验收边界.md`；本页只写 11-08 之后在产线跑的这一代。

## 现行任务

| 项 | 值（2026-11-09 00:23 北京实测） |
|---|---|
| 平台任务 | `t-20261108144507-mrvsh`，名称 `zhounan_redimnet2_ultra_native_feature_b8_handoff_8a800_p6_20261107_v1` |
| 创建时间 | 2026-11-08 14:45:07 北京（06:45 UTC） |
| 交接目录 | `/root/experiments/redimnet2/open-annotation-ultra-20261025/handoff-v4r2-20261108-v6/` |
| 运行目录 | 同目录下 `run-v6`（`queue/`、`labels/`、`resources-attempt1/`） |
| 引擎 | `admission-20261109/formal-code`，bundle sha256 `69f009e2…4a2a`；与 pilot 同一份字节 |
| 合同 | `contracts/zhounan-speed-native-fp32-sdpa-b8-silver_annotation-20261109.json`，sha256 `d6431fec…727d` |

任务号来自 `publish-work/observation-t-20261108144507-mrvsh.json`（2026-11-08 06:48 UTC 记 `state:Running`、`active_workers:1`）。同目录 `terminal-t-20261108144507-mrvsh.json` 的文件名虽然写着新任务号，内容里的 `task_id` 却是旧任务 `t-20261025054739-4kzhp`，拿它当新任务终态会引错（见仓库 `docs/16` 的“任务 ID 归属”）。

## 进度：按分片收据求和，不数文件

```bash
cd /root/code/github_repos/redimnet2-fork
python3 tools/annotation_progress.py                 # 打印 markdown 看板
python3 tools/annotation_progress.py --run <run目录> --window 30
```

工具读两处：`queue/broker.json`（`next`、`completed_chunks`、`leases`）和 `queue/completed/*.json` 的 `source_count` 求和。分片收据只有一千多个，秒级出结果；`find labels | wc -l` 在 GPFS 上要十几分钟，而且会把正在写的半成品算进去，两个口径对不上。仓库 `docs/ANNOTATION-PROGRESS.md` 是同一工具生成的快照，看它之前先看末尾采样时刻。

实测读法（2026-11-08 16:23 UTC）：

| 项 | 值 |
|---|---|
| 本轮已完成 | **453,985 源 / 1,817 分片** |
| 折算总进度 | 2,071,422 / 11,521,178 = **17.98%** |
| 最近 60 分钟 | 48,895 源/小时（16:02 与 16:17 两个时刻分别是 48,443 与 48,464，量级稳定在 4.7—4.9 万） |
| 按此速率跑完本轮 | 约 **8.0 天** |
| 队列 | 8 个 lease、8 个 worker 目录、无 `failure.json`、无 `quarantine.json` |

“17.98%”与 11-01 停摆时的 52.64% 不是同一分母口径：本轮 lineage 的起算基数是保留的 1,617,437 源，1028-v1 完成过的 4,447,318 源不在本 lineage 内，会被重算。报进度时把基数和分母一起写，否则两个百分数没法比。

判断“还在跑”看三样：`completed/` 里最新收据的 mtime 持续推进、`resources-attempt1/GPU_*.jsonl` 的 mtime 在更新、`broker.json` 的 `next` 在涨。单看 `labels/` 目录大小或文件数不行，GPFS 的元数据会有延迟。

## 看板

```bash
cd /root/code/github_repos/redimnet2-fork
/root/miniforge3/envs/USE/bin/python3 tools/annotation_dashboard_server.py --port 8899 --window 60
```

- 本机浏览器 `http://127.0.0.1:8899/`，内网 `http://10.43.6.199:8899/`；`/data.json` 给同一份数字的 JSON。
- 页面每 30 秒自刷新，每次请求现算，不缓存、不落库；关掉进程就没了，需要常驻就放 tmux 或后台任务。
- 省略 `--run` 时按 mtime 取 `open-annotation-ultra-20261025/*/run-*` 里最新一个；换任务后要确认它选中的是本轮目录，选错了页面会安静地显示旧 run 的数。
- 2026-11-08 16:08 UTC 起已有实例在跑（`--port 8899 --window 60`，解释器 `/root/miniforge3/envs/USE/bin/python3.10`）。重启前先看端口是否被占：`ss -lntp | grep 8899`。

## 这一代为什么能跑起来

pilot 9 个步骤全部通过（screen/main/recovery 各 b1、b4、b8）：3000 源的主阶段 b4、b8 相对 b1 的冷启动端到端比分别是 **1.7427×** 与 **1.99397×**（`cold_end_to_end_speed_ratio`，实测）；512 源恢复三轨都是 0 新增 ASR/FA 调用。代价是 b8 与 b1 的生成 token 槽不同：`ASR_generated_token_ids_changed_windows=2623`、`post_eos_token_ids_changed_windows=2623`，但有效 token、规范化文字、语言、字界和 decision 的差异窗口都是 0。`changed_rows_are_accuracy_errors:false` 记的就是这件事：逐位不同来自批内 padding 槽，不是文字变了。

质量复核仍是缺项：`selected-admission-attempt-v6.json` 里 `quality_review_status` 写的是 `ABSENT — see report; cannot be derived from the 16-source A/B`，`formal_admission:false`。速度和 9/9 通过不等于文字/字界已有人审。

## 与旧路线的关系

- 旧的 `bounded_metadata_progress.py` / `readonly_progress.py` 仍可用于严格终验（它们核来源闭包、槽身份和生产者归属）；日常看进度用本页的收据求和，轻得多。
- 旧 `portable-successor-20261027-v1` 的队列是前一代，保留只读；新任务不共写旧队列。
- 六列评测与标注是两条线：标注产出的是待核候选，`usable_for_training/evaluation` 不因任务跑完而变成 true。
