# 原生有限试验封存器的单CPU诊断

`run_native_cpu_sealer.py`用于定位已结束的有限native试验封存为何耗时。它只运行固定的`native_batch_readback.py`，不接受任意命令、shell、模型入口或GPU操作。默认只核对计划，只有`--execute`才创建全新的证据目录并启动CPU子进程。CPU完成不等于文字/字界质量或正式八卡准入通过。

本入口不替换四个正式准入控制文件，不修改原协议15个／逐条特征协议16个计算源或GPU资源管理源。原试验的冻结审计包逐成员核SHA，封存器也使用明确文件SHA。新控制脚本部署到独立目录；旧试验包和已有输出保持原字节。

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

当前维护版每60秒由Python线程等待后，持有全局解释器锁（GIL）调用标准库`faulthandler.dump_traceback`，向`stderr.log`写一次调用栈；退出时取消并收尾该日志线程。旧原件曾用C watchdog的`dump_traceback_later`，历史字节保留。若300秒内未结束，控制器只向自己启动且尚未收割的直接子进程发送TERM，一秒后仍未退出则KILL，随后`wait4`收割；清理和写收据期间忽略重复SIGTERM/SIGINT，结束后恢复原handler；不查询、不发送信号给PTN、旧标注或其他既有进程。支持pidfd时使用实际该child的pidfd，其他系统在未收割该PID前操作，避免PID复用。

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

## 实际超时后的单次CPU诊断预算（v2）

v1仍严格限300秒，不修改历史计划或超时收据。若真实v1子进程已启动、实际达到原预算后由控制器终止并收割，可用显式`annotation-native-cpu-sealer-run/v2`计划，在**新目录**只延长一次CPU诊断到301～900秒。额外必须提供原`prior_timeout_receipt`及`prior_timeout_intent`的绝对路径和SHA；其余sealer、track、审计根和bundle必须与原v1逐字相同。v2不能递归引用v2，不能把普通失败、未启动、返回0、伪造耗时或复制到别处的JSON当作实际超时。

检查会回读原计划、原root内receipt与intent、已核冻结bundle内原控制器SHA、固定child代码和argv、解释器实际SHA、真实child-control的PID和CUDA空值，以及UTC起止和单调耗时的对应关系。原日志、异常及未完成输出全部保留；新输出目录必须与旧目录和所有生产树隔离。v2默认check依然不运行子进程。

这只是有限已完成试验的CPU封存诊断预算，不延长GPU父进程120秒首基线、900秒ticket、正式外层1800秒等待，不改变4096有限分母、256MiB容量或任何文字／字界／恢复条件。诊断运行使用CUDA空、单CPU、线程1与nice15，不重新推理音频。17项普通及优化模式用例和独立审阅已通过，实际现场结果仍按独立receipt记录。

新16源B1第一次v1在300.346秒超时、退出-15且已收割，峰值RSS736,059,392字节，无封存输出，60秒周期栈仍在`_load_track`。原异常保留。不能从事后0.7秒CPU空闲快照断定前300秒的瓶颈；v2独立运行用于取得完整耗时与输出，结果未回读前不记成功。

新16源同一B1 track的v2实际已完成：132.522秒、返回0、未超时，子进程已收割；原输出125,428,063字节（约119.62MiB），SHA为`44f2973173e9d83c64e4ea89d3035d77eee54e2a9b2789ea77f42c849bb91432`，峰值RSS1,569,714,176字节（约1.462GiB）。这次实际未用满300秒；不能仅据两次耗时推断缓存或某个共享盘原因已经证实。旧v1超时和周期栈保留。实际v2控制器SHA为`7a951fd55026af31088bf908f06c2bc925f2f1584220c69b8d74ee891cf7b7b5`；这里只证明基线封存成功，B8主运行、512和完整正式准入需各自核验。

## B8封存段错误与周期调用栈的修正

真实B8主运行CPU封存v1在60.174秒以SIGSEGV（-11）退出，并已收割；这不是300秒超时，不适用v2延长预算。原stderr 3531字节，先出现60秒定时栈（`schema._finite_tree`→`validate_record`→有限审计器），随后记录段错误，峰值RSS726,089,728字节，没有封存输出。原件、原计划和7a951控制器保持不变。现场约444GiB可用内存及这份Python栈不能单独证明段错误的C层根因。

[CPython官方问题116008](https://github.com/python/cpython/issues/116008)记录了活动帧与异步栈转储的竞态，并给出从Python持GIL调用转储的规避方式；[官方文档](https://docs.python.org/3.10/library/faulthandler.html)说明旧`dump_traceback_later`由watchdog线程实现。它们支持当前排查方向，不能冒充本次core/C栈。维护版因此只把周期诊断改为Python线程内调用，不关闭周期栈或致命错误记录，不动原审计器、16计算源、数据、结果和质量规则。`intent`及`receipt`明确写`stack_dump_mode=python_thread_GIL/v1`。

修后原退出、超时、信号收割、固定来源和预算用例保留；另实跑短周期转储后正常退出，18项普通／优化测试通过。实际B8须在独立新root用相同track、原sealer和审计包重新完成；没有成功原件前不据测试宣称正式接续材料已齐。

同环境的有界合成对照也已实际执行：固定Python3.10.20、nice15、单CPU、CUDA空，从已核冻结schema只抽取纯`_finite_tree`，两种转储方式各运行10秒、周期10毫秒。两端返回0，旧方式2743次循环，Python线程方式2805次循环／586次转储，完整stderr保留；**本轮没有复现旧SIGSEGV**，因此不能把上述竞态确定为现场根因。合成诊断程序`diagnose_cpu_stack_capture.py`默认只check，3项普通／优化测试通过，不加载原音、模型或GPU。新封存控制器403626的18项双模式独立复验通过；实际主B8新root检查通过后重新执行相同完整有限校验。

使用新403626诊断控制器，相同原B8主track、原3412封存入口和6279审计包的v1／300秒完整复验已经实际成功：134.762秒、返回0、未超时、子进程已收割；输出122,738,770字节，SHA为`e06e4585cfa19fe1865fcca261439e368a922770d469364a56bf6ada5393dc63`，峰值RSS1,676,390,400字节。60和120秒持GIL周期栈均保留。旧SIGSEGV与新成功分别记账，这证明独立诊断程序下该完整校验可完成，不等于确定了原SIGSEGV的C层根因。
