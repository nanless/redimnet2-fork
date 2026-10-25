# 主训练跨段接大间隔微调的CPU准备

当前工具只生成和检查证据合同，没有训练执行入口，不取现训锁。`cpu_evidence_complete`表示一次CPU证据检查是否齐全，`ready_for_lm_execution`始终为false；不能拿它当云端启动成功。

`readiness.py`固定本次原任务1—35轮与普通接续36—120轮的身份，核八份model35完整恢复证明、两份连续验证记录、新段无未知异常、两任务真实终态与无写入者、同说话人映射、接续run的115/120检查点，以及整条谱系最佳检查点的身份。LM只从115初始化模型，优化器与梯度缩放器重新开始；同阶段PTN恢复则恢复全部训练状态。

## 使用方式

在仓库根目录运行。输入引用表由现场持有人生成，每项写服务器逻辑路径、实际只读路径和真实SHA256；私有表和日志不要入Git。

```bash
python -m tools.lm_continuation.readiness build \
  --references /受控证据目录/references.json \
  --output "$PWD/tools/lm_continuation/results/prepared.json"
python -m tools.lm_continuation.readiness check \
  --contract "$PWD/tools/lm_continuation/results/prepared.json" \
  --output "$PWD/tools/lm_continuation/results/check.json"
```

输出必须位于本准备包内，且不能覆盖已有文件；`results/`已忽略。缺115/120、缺轮、错父任务、新异常、未退出写入者等返回非零。不能以更改旧校验清单或把缺失字段填为成功来通过。

## 固定依赖与新维护身份

`legacy_checks/`复用原source95/完整source189、模块导入、私有解码库、原七事件和CPU检查点的函数，不调用旧单run完整120轮入口。历史源字节摘要与新维护摘要记录在[维护身份与历史差异](../../docs/reviews/LM归仓_维护身份与历史差异.json)。

为不公开诊断样本名单，音频探测改为对所有合法源都覆盖固定回归点和定位表邻点。原合法24593、159744邻点、319488邻点仍受检；319487只在PTN裁剪合法时出现，不伪称合法LM起点。新维护版摘要已固定，不能继承旧GPU或现场验收结论。

新段日志使用固定旧纯字节异常识别器，裸LibsndfileError/OSError/ValueError/TypeError也拒绝；原七事件许可只属于原run。真实证据齐全后，默认分支会CPU导入PyTorch并mmap读取检查点、执行source/decoder检查。本机合成测试替代这些尚未发生的现场输入，不代表真实115/120已存在。

未来现场应使用原redimnet2解释器，并在启动解释器前选择已审私有解码库，清LD_PRELOAD/LD_AUDIT并保留平台基础库路径。不能在已经加载soundfile后热换库。真正执行还须在原段、续段和逻辑链原子锁范围重新检查全部旧依赖、终态、写入者与新输出唯一认领，随后验证八rank冷初始化和LM1保存点；本工具不实现该执行器。

## CPU测试

```bash
PYTHONPATH=. python -B -m unittest discover -s tests -p 'test_lm_continuation.py'
```

全部夹具在临时目录合成，不读取Mac实验目录、真实音频、私有名单或模型，不使用GPU或外网。
