# 第47轮接续包的固定接口

本包只处理同阶段PTN的47→48，总目标120。构建器不提交、不启动、不拿训练锁。真正执行器由维护者另行提供并审阅；缺少执行器或真实CPU恢复时不能运行。

构建命令采用：

```bash
python -B -m tools.ptn_resume.build --spec /absolute/private/build-spec.json --output-dir /absolute/new-package
```

spec格式 `ptn-resume-build-spec/v1`，字段固定如下（真实路径、摘要由原件取得，不能使用占位字符串）：

- `legacy_bundle_root`：原16成员与`bundle_files.json`所在绝对目录；整份索引必须是旧840235…，其16成员逐项按索引核。
- `source_root`：原不可变189文件树；`source95`为 `{path,sha256}`，必须是原5160…清单；189索引取旧bundle中`source_189.json`（b342…），不重写旧索引。
- `checkpoint`：`{path,sha256}`，固定失败vggvm目录下`ptn/models/model_47.pt`，固定完整SHA；`speaker_map`同样为`{path,sha256}`，要求原固定说话人表摘要。
- `original_terminal`、`predecessor_terminal`：两个新鲜控制面观测JSON引用`{path,sha256}`；分别绑定原1—35任务与失败36—47任务，实际任务和实例终态、active_workers为0，有带时区UTC。不能拿旧Succeeded或旧35轮终态冒充最新失败前驱。
- `new_run_root`：新训练输出的绝对路径，当前必须不存在。构建器不创建它。
- `patches`：精确两个`{relative,path,sha256}`：替换`applications/speaker_rec/wespeaker_lite/utils/executor.py`；新增`applications/speaker_rec/wespeaker_lite/utils/numerical_failure.py`。不允许其他原189字节、权限或成员变化。
- `runtime_files`：显式提供新执行器文件`{name,path,sha256}`，至少包含`entry.py`、`worker.py`；其余按真实依赖从`strict_restore.py`、`cpu_check.py`、`epoch_guard.py`、`loader_worker_audit.py`、`predecessor_gate.py`、`launch.sh`、`decoder_runtime.py`、`audio_review.py`选择。构建器仅核摘要、编译Python语法并原字节复制，不导入或执行。新代码的47/48/705047接线由独立审阅和真实CPU检查证明。

构建器自动把本仓维护的`state_gate.py`与`contracts.py`原字节复制为`runtime/state_gate.py`与`runtime/ptn_contracts.py`；两份都写入新合同与bundle摘要，不依赖可变仓库的PYTHONPATH。

输出目录包括`source/`（独立190成员新树）、`runtime/`（显式新包装文件）、`resume.yaml`（旧有效配置只改checkpoint与exp_dir）、`resume-contract.json`、`source_delta.json`、`source_manifest.json`、`bundle_files.json`。旧16成员只作为只读来源，不能把其中model35启动代码原样当作本次运行入口。

合同schema固定`ptn-resume-model47/v1`，真实执行器应从显式`--contract`和`--contract-sha`读取，不能默认猜旧目录。合同记录source189历史身份、新190身份、两个源码差异、新运行目录、检查点/映射、配置和包装文件身份；`runtime_admission`永为false，表示构建完成没有授予执行资格。

实际恢复比较入口由`state_gate.verify_restored`提供：调用者传入实际构造并load_state_dict后的model/SGD/scaler/scheduler/margin与配置，函数重新核可信model47文件并读取，逐状态与参数顺序精确比较，再实际调用冻结scheduler/margin的705047时钟并验证0.72/0.2。不能用调用者自供`restored:true`替代。结果绑定合同及模块摘要，实际CPU与八rank分别调用；锁、资源、模块来源、数据worker及前向前all-gather由执行器额外保障。

原189树新增helper造成新成员总数190；替换的executor旧摘要与新摘要都保存。构建失败的独立目录保留，不补写成功或覆盖后重封。

排队后可用`state_gate.verify_restored(..., terminal_evidence={"original":{path,sha256},"predecessor":{path,sha256}})`提供新鲜控制面原件。函数会按固定任务/run/终态及1小时时效重新读取核验，实际ref/SHA写入返回证明，不接受布尔值，不改静态构建合同或旧pin。缺省仍核合同原引用，因此排队导致陈旧时准确拒绝。
