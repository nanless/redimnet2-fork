# 标注进度看板（自动生成，不手改）

> 生成：`python3 tools/annotation_progress.py --out docs/ANNOTATION-PROGRESS.md`。
> 数据来源：`queue/broker.json` 与 `queue/completed/*.json` 的 `source_count` 求和；
> 不用文件计数——GPFS 上几十万文件要十几分钟，而收据只有几千个。
> 存储规范（2026-11-09 起）：新 run 根写 `/root/group-shared/voiceprint/experiments/redimnet2/<实验名>/`，侧车写 `.../open_annotation/redimnet2-silver-<日期>/`；更早的标注代次也迁往该归档目录，迁移期间旧基址可能还在（历史路径），见 `docs/20-标注结果归档与跳过规则.md`。

## 1. 总量

| 项 | 值 |
|---|---|
| 任务 | `t-20261108144507-mrvsh` |
| 本轮目录 | `/root/group-shared/voiceprint/open_annotation/redimnet2-silver-20261109/handoff-v4r2-20261108-v6/run-v6` |
| 分母（三域全量） | 11,521,178 |
| 本 lineage 起算基数 | 1,617,437（1027 保留） |
| 本轮已完成 | **966,520** 源 / 3,863 分片 |
| 折算总进度 | 2,583,957 / 11,521,178 = **22.43%** |
| 本轮还剩 | 8,937,221 源 |

## 2. 速率与预计

| 项 | 值 |
|---|---|
| 最近 60 分钟 | 0 源 → **0 源/小时** |
| 按此速率跑完本轮 | 未能计算（窗口内无新增） |

## 3. 队列与进程

| 项 | 值 |
|---|---|
| broker next / completed_chunks | 3871 / 3863 |
| 正在处理的分片（leases） | 8 |
| worker 目录数 | 8 |
| 失败记录 `failure.json` | **存在** |
| 隔离片 `quarantine.json` | 无 |

## 4. 怎么读这张表

- **折算总进度**用 1,617,437 作基数：新生产者的 lineage 从 1027 保留量起算，
  1028-v1 的 4,447,318 源不在本 lineage 内、会被重算。所以 52.64% 与本表的百分比
  不是同一分母口径，不能直接相减。
- **重启的代次**：11-08 的 v2–v5 四代都没产出侧车（v5 的 `run-v5` 是空目录），
  v6 才重新开始计数；本表只算当前 run 目录里的分片收据。
- **速率**只看最近一个窗口：整体平均会把早期的低速和现在的稳态混在一起，
  判断“还要多久”用窗口速率更准。
- 出现 `failure.json` 或 `quarantine.json` 时不等于停跑，先看内容：
  隔离片是失败超阈值的分片被移出候选，其他片照常推进。
- 任务号默认从 `publish-work/observation-t-*.json` 的文件名取；
  `terminal-*.json` 里的 `task_id` 字段会错绑旧任务，不读它。

---

采样时刻：2026-11-09 06:05:02 +0000
