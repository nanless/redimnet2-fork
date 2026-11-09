# 冻结包纪律与提交链（2026-11-08）

这页把“什么算同一份冻结包”“平台入口怎么填”写在一起。五条纪律来自 11-07—11-08 四次真实失败和一次成功交接；细节原件在仓库 `docs/16-实验维护与验收边界.md`，本页给可直接照做的形状。

## 五条纪律

### 1. bundle == pilot，按字节比，不按目录名

正式包的 `bundle_files.json` 要和 pilot 那份逐字节相同。2026-11-08 这代：`formal-code/bundle_files.json` 与 `pilot-20261108/prepared-v11/native-fp32-sdpa-b8/code/bundle_files.json` 的 sha256 都是 `69f009e2…4a2a`；两份 code 目录 `diff -rq` 只剩 `freeze_receipt.json` 与 `native-formal-code-preparation.json` 两行差异，都是回执文件。

目录名相似说明不了什么。11-08 之前 `frozen-native-reviewed-v4-r2`、`admission-20261108/formal-code`、`pilot-20261108/prepared-v6/.../code` 三份清单 sha 互不相同，其中只有一份和 pilot 同字节。判断方法固定成两条：`sha256sum <两份>/bundle_files.json`，再 `diff -rq --exclude=__pycache__ <pilot code> <formal code>`。

### 2. `REVIEWABLE_CONTROL` 只有四个文件，且要引擎源先全等

`native_batch_admission.py` 第 207 行：

```python
REVIEWABLE_CONTROL={'native_batch_admission.py','native_batch_readback.py',
                    'native_batch_queue_handoff.py','bounded_speed_runtime.py'}
```

`check_formal_bundle()` 对合同里 16 项 `engine_source_files`（含 `worker.py`）做的是无条件等式比较。意思是：改 `worker.py` 一类引擎源，就得打新包、用新包重跑 native pilot、拿新 admission；改那 4 个 CPU helper，也要在引擎源全等之后才轮到“可审阅替换”，并且附独立审阅原件。两者都不是“改完直接续跑”。

### 3. check 通过 ≠ 能跑

admission 的 `passed:true` 只证明合同、路径、哈希和 pilot 证据自洽。11-07 14:55 的 admission v5 是 `passed:true`；一小时后 v4 生产在 worker 初始化/CUDA 分配阶段超时，`labels` 0 条、`completed_chunks:0`。同一代里 pilot 还出现过打包副本缺 `import device_preflight` 的 `NameError`。

所以提交后要另取首片证据：`labels/` 下出现已提交侧车、`completed/` 出现收据、`broker.next` 上涨。平台 Running、CPU 检查、pilot 通过都不能代替这三样。

### 4. device 是挂载命名空间的标签

同一个文件在开发节点读到 `st_dev=57`，在集群 GPFS 命名空间记 `2097190`，换一套容器挂载又可能是 `75`，而 inode/bytes/mtime/ctime 完全相同。把 device 放进“内容是否变了”的等式，会把没变的文件判成漂移。

现行语义：内容身份用 `inode / bytes / mtime_ns / ctime_ns`（以及已存的 `file_sha256` / `pcm_sha256`），device 不同只打印 `PREFLIGHT_DEVICE_MISMATCH`、回执记 `device_relaxed:true` 后继续；内容不符才 `PREFLIGHT_IDENTITY_MISMATCH` 并 `exit 21`。放宽 device 和放宽内容身份是两件事。

### 5. claim 写在门后，退役要留证据

`.formal-parent-locks/bounded-speed-<sha>.claim.json` 是“一链一生产者”的硬门，绑定 `new_contract/new_bundle/new_run_root/index` 等核心。claim 一旦写入，换引擎就会被 `another bounded producer was permanently claimed` 挡住。

写法：门禁、pilot、admission 都过了再写 claim。反过来先写 claim 再补门，失败现场会被锁成“已认领”，下次只能走退役流程。退役按顺序做：按字节归档 `claim.json` 和同名 `.sha256`（2026-11-09 起归档到 `/root/group-shared/voiceprint/artifacts/redimnet2/ledger/retired-claims-<日期>/`；2026-11-08 那次在 `/root/codex_artifacts/retired-claims-20261108/`，历史路径）→ 写清原声明时间、声称的 run、任务终态与崩溃形态、为什么旧引擎不可能继续、为什么新引擎算不同生产者 → 写回滚方法（复制回 `.formal-parent-locks/` 并删新声明）→ 最后才从原位置删除。先删再补理由，等于把“这个生产者真的跑过”的现场抹掉。

## 提交前检查清单

**C1 清残留 claim。** 查 `.formal-parent-locks/` 里还有没有指向旧 run 的 claim。有就按上一节退役，不要直接 `rm`；也不要因为“再提一次试试”绕过它。顺带确认旧 handoff 目录没有被新 blueprint 复用（`new_run_root` 指向不存在的目录，gate 会拒绝写已有输出）。

**C2 失败现场 0 侧车。** 复用一个失败目录之前，先数它的侧车：`find <run>/labels -type f | wc -l` 和 `broker.json` 的 `completed_chunks`。v4 生产失败时两项都是 0，所以那个目录只是失败现场，没有脏数据；如果非 0，要先按内容证明处理已写侧车，不能直接在原目录续跑。

两项都过了再提交，提交后按第 3 条取首片证据。

## 平台入口：用短脚本，不用长命令

入口字段填：

```bash
# 新交接脚本按硬规则落 /root/group-shared/voiceprint/artifacts/redimnet2/repo-handover-<日期>/
# 2026-11-07 起的现存副本仍在 /root/codex_artifacts/repo-handover-20261107/（历史路径），迁移前照旧调用
/bin/bash /root/codex_artifacts/repo-handover-20261107/run_handoff_v7.sh   # 历史路径（迁移前调用这里）
```

脚本内容（v6 的实例，换任务时改 `CTRL`/`BP`/sha）：

```bash
#!/bin/bash
set -euo pipefail
# 新一批把 admission 材料放 /root/group-shared/voiceprint/artifacts/redimnet2/<用途>/admission-<日期>/
CTRL=/root/experiments/redimnet2/open-annotation-ultra-20261025/native-speed-successor-20261028-v4-r2/admission-20261109/outer-controller-v4r4   # 历史路径（2026-11-09 版）
BP=.../handoff-20261109b/blueprint.json
cd "$CTRL"
export PYTHONPATH="$CTRL"
exec /root/miniforge3/envs/qwen3-asr/bin/python3.10 -B "$CTRL/native_batch_queue_handoff.py" \
  wait --blueprint "$BP" --blueprint-sha <sha> --execute
```

写进标注 spec、blueprint 或 handoff request 的绝对路径要用 `/shared/group-shared/...` 的物理拼法：现有校验器要求路径本身和祖先都不是符号链接，`/root/group-shared` 是软链会被拒；文档里给人看的路径仍写 `/root/group-shared/...`。

原因：控制台入口是多行文本框，长命令直接粘进去会丢首字符（2026-11-08 实测，重填一次才对上），而 `set -euo pipefail` 少一个字符可能整段不执行。短脚本只有一行，能在提交后全文回读比对。

## 三段式交接与 auto_publish

1. **wait**：`run_handoff_v7.sh` 在控制器目录里等，直到 GPU 父进程写出 `handoff-v6/request-v1/request.json`。request 带 `created_utc`、`wait_seconds:1800` 和身份。
2. **publish**：`auto_publish.py` 每 5 秒轮询一次 `request.json`，一出现就生成 `terminal-<task>.json`（旧任务终态）和 `observation-<task>.json`（新任务 Running），再调 `native_batch_queue_handoff.py publish`。这个动作要快：request 只在 `wait_seconds` 窗口内有效，轮询本身又是 5 秒一格，拖过窗口就要重新 wait、重新取 request。
3. **GPU 父进程**：publish 写出 `reply.json` 后，父进程才创建 `run-v6` 并开始第一批。

`auto_publish.py` 的终态观测是“旧任务 Failed/active_workers 0”，新任务是“Running/1 worker”；它不做远端进程枚举（`remote_proc_enumeration_performed:false`），所以这些字段是共享文件观测，不是平台认证。

## outer controller 按实际 bundle 重建

控制器目录 `outer-controller-v4r4/` 不是通用工具：`controller-files.json` 逐个列出 27 个文件的 sha，`build-receipt.json` 绑定 `GPU_bundle` 的 sha（本次 `69f009e2…`）和 `GPU_executed:false`。换 formal-code 的 bundle 后，拿旧 controller 配新包会在 `check` 阶段报哈希不符。

重建顺序：`build-controller` 从实际 bundle 生成新目录 → 写新 `build-receipt.json` → `build_handoff_v6.py` 把 controller 的 root 与 manifest 指进 blueprint → 再 `check`。本次 controller 就是按 `admission-20261109/formal-code` 建的，pilot 与正式包同字节，所以 controller 不需要第二份。
