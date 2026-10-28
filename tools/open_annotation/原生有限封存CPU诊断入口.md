# 原生有限试验封存器的单CPU诊断

`run_native_cpu_sealer.py`用于定位已结束的有限native试验封存为何耗时。它只运行固定的`native_batch_readback.py`，不接受任意命令、shell、模型入口或GPU操作。默认只核对计划，只有`--execute`才创建全新的证据目录并启动CPU子进程。CPU完成不等于文字/字界质量或正式八卡准入通过。

本入口不替换四个正式准入控制文件，不修改15个计算源或GPU资源管理源。原试验的冻结审计包逐成员核SHA，封存器也使用明确文件SHA。新控制脚本部署到独立目录；旧试验包和已有输出保持原字节。

## 固定计划

[计划样例](templates/native-cpu-sealer-plan.example.json)包含以下七个字段：

| 字段 | 含义 |
|---|---|
| `schema` | 固定为`annotation-native-cpu-sealer-run/v1` |
| `sealer` | 实际`native_batch_readback.py`绝对路径及SHA |
| `track` | 已完成的有限native track绝对路径及SHA |
| `audit_code_root` | 真实试验的原冻结审计代码目录 |
| `audit_bundle` | 该目录`bundle_files.json`的路径及SHA，所有成员重新核SHA |
| `execution_root` | 全新、不与输入/源码/producer重叠的证据目录 |
| `deadline_seconds` | 正数且不超过300秒，包含写启动意图后的检查和执行等待 |

模板SHA占位符故意不可执行。计划不能临时指向其他Python脚本。执行目录不能已存在，也不能放进sidecar、queue、resources或冻结代码树；旧结果不得覆盖或删除。

## 检查与执行

在有权限读取同一份共享文件的实际云节点上，使用与原试验一致的Python解释器：

```sh
/root/miniforge3/envs/qwen3-asr/bin/python3.10 -B /独立CPU控制目录/run_native_cpu_sealer.py \
  --plan /固定计划.json --plan-sha 实际计划SHA
```

此时不创建目录、不启动子进程。检查通过后，对同一固定计划加`--execute`：

```sh
/root/miniforge3/envs/qwen3-asr/bin/python3.10 -B /独立CPU控制目录/run_native_cpu_sealer.py \
  --plan /固定计划.json --plan-sha 实际计划SHA --execute
```

父进程使用自身`sys.executable`，构造唯一固定参数数组，启用Python隔离模式和无缓冲输出；不通过shell解释。子进程用`runpy`运行原sealer，保留其`__file__`并明确补原入口目录，避免另造一个数学实现。`CUDA_VISIBLE_DEVICES`设为空，模型离线，线程数设为1；Linux还将子进程限制到它原先允许的一个CPU。macOS不提供Linux affinity时，记录未实施硬亲和性，不能声称已经硬限单核。

每60秒通过Python标准库`faulthandler`向`stderr.log`写一次当时调用栈。若300秒内未结束，控制器只向自己启动且尚未收割的直接子进程发送TERM，一秒后仍未退出则KILL，随后`wait4`收割；清理和写收据期间忽略重复SIGTERM/SIGINT，结束后恢复原handler；不查询、不发送信号给PTN、旧标注或其他既有进程。支持pidfd时使用实际该child的pidfd，其他系统在未收割该PID前操作，避免PID复用。

## 留下什么证据

`intent.json`在子进程启动前写入并刷盘，记录主机、父PID、固定argv、解释器/控制源/封存源/track/audit包SHA、截止时间和每60秒栈设置。之后的正常完成、Python异常、启动异常、超时和可捕获的SIGTERM/SIGINT都进入`finally`，保存`receipt.json`，不能在timeout前抛错而漏记结果。

| 文件 | 内容 |
|---|---|
| `stdout.log` / `stderr.log` | 原stdout、异常及定时调用栈，不删减 |
| `child-control.json` | 子进程实际PID、CUDA可见性、是否实施单CPU亲和性及CPU编号 |
| `receipt.json` | 开始/结束、耗时、timeout、退出码、是否收割子进程、原异常、日志SHA |
| `sealed-readback.json` | 原sealer成功写出的封存文件，不由控制器伪造或补齐 |

峰值RSS来自`wait4`对**本次直接子进程**返回的`ru_maxrss`，包括正常、报错及被终止的进程；Linux按KiB转字节，macOS原值按字节处理。它不冒充整个任务树或GPU显存。若子进程根本未启动，峰值明确为null。实际封存输出若存在，记录文件大小与SHA；输出缺失时，即便进程返回0也不记为完成。封存文件仍受256MiB硬上限约束，超过则保留异常，不截断、删字段或无限抬高上限。

默认check失败或执行目录已存在时，不修改那个目录；没有启动child，错误由CLI返回。执行目录创建后发生的异常都会尽力留下intent/receipt和实际可得文件；磁盘写入本身失败时，原错误不能凭空变成完整证据。

## 读这些计时前先确认范围

2026-10-28曾在dev上封存真实3000条时300秒超时，原异常已保留。随后只读固定4条侧车和4个首窗口批次，15个JSON读取合计约0.516秒、祖先lstat约0.0206秒，审计器未导入torch或模型库。这些文件可能已被此前读取预热，不能用4条结果外推整个3000条耗时。冻结审计器逐批次重复检查同group计划，上限256请求；这是组内重复工作，不是全局3000条平方。

本入口用于在实际云节点保留完整阶段调用栈、耗时、峰值RSS和输出大小，判断慢在路径/文件读取、整批校验还是JSON复制/序列化。诊断完成后仍须用原生v2读回和质量协议验收；不会因为换节点、加定时栈或CPU返回0而放松原检查。

实际云单CPU B1 3000封存后续已完成：62.077秒、124,996,371字节、峰值RSS约1.284GiB；60秒栈在末尾JSON序列化。随后仅一次dev封存校验严格核原完整批次和提交字节，3000来源/3067窗口通过，耗时166.621秒、峰值RSS约1.512GiB，未重开原侧车/commit或媒体。两段口径不同，不能把封存成功写成候选质量或八卡正式准入。
