# 主训练数值失败的 CPU 取证

本目录只读取已保存的波形张量和训练检查点，核文件摘要、实际形状、逐样本幅度和模型/优化器状态量级。它不构造模型、不运行前向或训练、不请求 CUDA，也不改变 AMP、学习率、数值守卫或现有冻结任务。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
nice -n 10 python3 -B tools/ptn_diagnostics/inspect_numeric_cpu.py \
  --wav /absolute/evidence/saved_wav.pt --wav-sha '<实际完整SHA256>' \
  --output /absolute/new-audit/wave-stats.json
```

检查点量级另运行一次，避免把波形检查拖入较长的模型扫描：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
nice -n 10 python3 -B tools/ptn_diagnostics/inspect_numeric_cpu.py \
  --checkpoint /absolute/evidence/model_N.pt --checkpoint-sha '<实际完整SHA256>' \
  --output /absolute/new-audit/checkpoint-stats.json --max-seconds 900 --rate-mib 16
```

每个输入都必须有自己的摘要；输出必须不存在，而且位于各输入目录之外。解释器需支持 PyTorch 的 `weights_only=True` 和 `mmap=True`；不支持或内容不属于已声明格式时会保留错误并退出，不自动改为任意对象反序列化。文件摘要和全部扫描前后状态必须一致。预算在文件块和张量块之间检查，不能保证中断底层阻塞的文件系统调用；需要硬时限时由调用者只限制本次新建的自有 CPU 进程。

波形检查保留实际 shape、stride、storage 与逻辑字节数，每条记录统计有限值、NaN、Inf、均值、总体标准差、RMS、零值及幅度。样本标识只输出摘要；不重采样、裁剪、归一化或再做增强。没有显式 `--sample-rate` 时不从张量长度猜时长。

检查点同时支持训练 dict 的 `model` 字段和旧裸参数字典。模型参数、整数计数、优化器张量分开统计；逐块转 CPU float64 计算量级，避免整份模型复制。RMS使用缩放累计，避免大有限参数在计算平方时又溢出。BN参数通过 `running_var` 前缀定位，其他Norm只按名字辅助列出；工具没有重建模型，不能声称识别全部归一化层。模型状态有限不等于失败时模型有限，更不等于GPU前向安全。

测试仅使用小型合成张量：

```bash
python3 -m unittest discover -s tools/ptn_diagnostics/tests -v
```

本工具的成功退出只表示读取和数值统计完成。它不授予恢复训练资格，也不把稍早检查点加保存波形的结果冒充失败时权重、随机状态和内核环境的逐位重现。

## 独立前向对照

仓库`tools/ptn_forward_probe.py`是现场48轮故障的专用诊断入口，当前只接受已核的47轮823项模型状态和96×1×32240原批。详细设计见[隔离诊断方案](../../docs/plans/2026-10-26-PTN前向非有限值隔离诊断.md)。它不是通用训练入口，不能用于恢复或覆盖旧任务。

参数要求原冻结源码树及189文件清单、清单/配置/检查点/波形各自完整64位SHA256、全新输出、共享设备锁目录、两张指定GPU的UUID，以及`--case baseline-amp`、`hooked-amp`或`hooked-fp32`。先用`python -B tools/ptn_forward_probe.py --help`查看完整参数。每个case用独立新进程，只在两卡均无计算进程且双快照一致后启动；CUDA可见设备显式绑定这两UUID。不能用“显存还够”或“未知host PID可能是自己”放行。

每次加载相同完整输入与新模型，保持train模式并记录梯度图设置；不再增强、不裁音频、不反向或更新。默认保留梯度图；全FP32大批的显存不能从半精度峰值保证，若显式`--no-grad`，结果只能作为无梯度诊断，不是原训练验收。父调用应只给本次自有进程有界硬时限，保留标准输出、错误、退出码、工具及依赖摘要和退出后资源归还证据。

模块探针观察484个边界并不等于捕获每一个原始算子。模型47重放正常也不能排除故障时模型漂移，不能直接批准原样重启或修改全局精度。原48/1773模型、此前迭代形成状态和RNG没有保存，是明确的取证缺口。
