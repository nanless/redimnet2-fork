# 从47轮恢复到48轮：新包与真实状态检查

本工具只准备本次同阶段主训练恢复，目标仍是120轮。它不恢复已丢失的48轮部分训练，也不启动大间隔微调。构建器不提交平台、不访问GPU、不拿训练锁、不创建训练输出；只有输入、源码和独立新包的文件操作。真正执行器必须显式提供并接受独立审阅。

先读[固定接口](INTERFACE.md)，按实际原件写私有spec，再运行：

```bash
python -B -m tools.ptn_resume.build \
  --spec /absolute/private/build-spec.json --output-dir /absolute/new-package
```

`--output-dir`的父目录必须已经存在，目标目录必须全新，并与旧源码、旧任务、新训练输出均不重叠。失败后的部分目录保留，不覆盖重试。原件引用一律要求绝对路径和完整SHA；不接受软链或缺省猜测。输入中会出现私有位置和控制面原件，实际spec、包、日志和结果只放私有审计目录，不能提交公开仓库。

## 源码身份不能混淆

原始95份清单、189成员清单和16成员恢复包分别核原摘要。构建器先逐份验证原189文件的字节、大小、权限和成员集合，再复制成独立新树，只允许两处变化：替换`executor.py`，新增`numerical_failure.py`。原树和原清单不写入，源成员由189变为190。新`source_delta.json`同时列旧/新摘要；新`source_manifest.json`和新`bundle_files.json`有独立身份。两处新源码的实际语义和故障取证能力仍须独立审阅，摘要及语法正确不能替代该审阅。

旧16成员的运行代码固定35→36，不能直接启动。新执行器文件由spec逐份提供，至少有新`entry.py`和`worker.py`；沿用名字不表示可沿用旧硬编码。四个核心旧启动文件若摘要完全未变，会在构建时拒绝。其他确实未变的decoder等辅助文件可保原身份，公开派生版本必须用其新SHA，不得改旧索引掩盖差异。

新包自动包含本仓`state_gate.py`与`contracts.py`，复制为`runtime/state_gate.py`与`runtime/ptn_contracts.py`，并写入新合同和包摘要。部署入口可以直接`import state_gate`，不依赖可变仓库PYTHONPATH。实际运行仍必须使用`python -B`，禁止向已冻结包写`__pycache__`。新增未列文件会使包检查失败。

## 配置必须保留数值类型

新`resume.yaml`使用`yaml.safe_dump`写出，并在写入前用`yaml.safe_load`回读，按完整对象逐类型和值精确比较。不能因为JSON在语法上可作YAML，就直接把JSON序列化结果当作此训练器的YAML：PyYAML的YAML 1.1解析会把JSON风格`2e-05`读为字符串，而原YAML的`2.0e-05`是浮点数。首个实际CPU候选因此在SGD构造时失败，尚未进入前向/GPU；原候选和错误保留。修复发生在构建器，不在优化器中加`float()`掩盖合同类型变化。

测试包含实际`weight_decay=2e-5`、其他科学数值、整数、布尔、null和嵌套结构，并用回读的参数构造小型SGD。新包必须重新生成，不能改旧候选配置后继续沿用旧摘要。

## 完整状态门实际检查什么

执行器从`--contract`与`--contract-sha`读取合同，构造实际模型、分类头、SGD和AMP缩放器，恢复状态后调用：

```python
proof = state_gate.verify_restored(
    contract_path, contract_sha,
    model=model, optimizer=optimizer, scaler=scaler,
    scheduler=scheduler, margin=margin_scheduler,
    config=effective_config, speaker_map_path=actual_map,
    terminal_evidence=fresh_terminal_refs,  # 缺省使用静态合同原引用
)
```

函数自行核model47完整SHA、字节数和扫描前后状态，再用CPU、`weights_only=True`、`mmap=True`读取；不接收调用者替换的payload或`restored:true`。不支持安全读取时直接拒绝，不降级到任意pickle。它精确比较823个模型状态、分类头形状、585个参数的原顺序与583/2分组、584份SGD动量、完整scaler，核旧映射和实际新映射。真正调用所传冻结调度器705047，核学习率0.72、间隔0.2；不会自己再实现一套调度公式。

该函数不构造模型、不做前向、反向或优化器更新。它会推进调度器和margin至已声明的起始时钟，这是本次状态核验的一部分。返回证明始终保留`runtime_admission:false`：单份精确状态不证明控制面权限、稳定锁、资源归属、decoder、全部八rank或192个数据子进程。执行器仍必须在任何前向/更新之前完成全部检查和八rank一致确认。

控制面终态引用要求固定任务ID、run、真实worker终态、active_workers为整数0、带时区时间且不超过1小时。排队后可提供两份新的具体`{path,sha256}`，内部重新读取核验并写入实际证明；不能传布尔值，不能改旧合同pin。文件校验只证明给定原件的完整性与语义，不声称它具备平台认证签名。材料必须由现场实际控制面观察取得，不能凭需求预写。

锁和去重由实际父/worker执行器负责：原1—35及失败36—47两个run的各两把原稳定锁、逻辑链锁、准入锁，以及唯一新输出，必须在同一准入过程持有。旧锁不能删建，检查与启动之间不能释放锁；worker应独立持有同一稳定inode的共享锁，避免父丢失后第二训练进入。此构建器不拿锁，因此不把构建成功写成独占执行成功。

## 需要读取的原16成员

除下列16份外，还要原`bundle_files.json`本体（840235…）、原95清单（5160…）、model47原件和实际说话人映射、新鲜终态，以及新的两份源码/包装文件。

| 原成员 | 需保留并核对的字段或内容 | 本次使用限制 |
|---|---|---|
| `strict_restore.py` | SOURCE/OLD/RUN、model35、配置与映射pin、823/584/585、35/36/525035 | 只读还原旧语义，新版本改为明确47/48/705047 |
| `cpu_check.py` | 原实际构造/完整状态比较顺序、decoder和音频检查 | 不能复制旧cpu_admission当新通过 |
| `entry.py` | 稳定锁、独占去重、环境、单父八rank、旧postflight | 新前驱、新输出及48—120必须重新实现和测 |
| `worker.py` | 前向前恢复门、八rank一致、原参数顺序、首次真实更新 | 新八份恢复/首更新证明不能沿用旧文件 |
| `epoch_guard.py` | rank内守卫、源码/检查点身份、24worker | 新轮48的实际分片必须验证 |
| `loader_worker_audit.py` | worker来源、rank/world/epoch/seed/legacy增强 | 不减少worker数绕过检查 |
| `predecessor_gate.py` | 稳定终态读取、时效、原task/run身份 | 旧wb58x不是本次直接前驱 |
| `decoder_runtime.py` | 实际导入路径、soundfile及libsndfile映射 | 新进程实际核验，不靠旧maps或布尔声明 |
| `audio_review.py` | 读取窗口、实际decoder、原证据规则 | 旧许可不自动覆盖本次新异常 |
| `launch.sh` | 单父入口、解释器、环境、参数 | 不执行原35→36脚本 |
| `source_189.json` | relative/sha256/bytes/mode完整189集合 | 原b342…永不修改；新190另表 |
| `dependencies.json` | small_full_sha_files、tables的size/mtime/首尾/历史全SHA | 小文件实哈希；大表首尾不能称新全量哈希 |
| `audio_selfcheck.json` | 原样本和窗口、来源及decoder检查输入 | 私有名单不入公开仓库 |
| `resume.yaml` | 全部有效配置、增强、调度、验证与原35检查点 | 新配置仅改checkpoint和exp_dir；源码变化另列 |
| `parent_terminal_attestation.json` | 原task/state/worker/time | 历史证据，不能当本次fresh |
| `failed_task_terminal.json` | 更早失败任务/state/worker/time | 历史证据，不能冒充vggvm新鲜终态 |

模型47只有epoch/model/optimizer/scaler，不含原每rank RNG或48轮数据游标。本次只能恢复已保存完整状态及绝对时钟，从48轮开头重做；不承诺与不中断未来逐位相同。已有失败输入、故障时扩展取证包和model47须分别保留，不把扩展取证包误命名成完整model48。

## CPU测试与未完成项

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tools/ptn_resume/tests -v
```

测试使用合成189树、合成控制面记录和真实小型CPU模型/SGD/scaler。为了使用小文件，测试中显式替换固定生产身份和状态数量；这些替换没有CLI入口。测试不能宣称实际B6、原189运行环境、八卡恢复或本次故障已修复。实际CPU完整构造、两卡工程验收、锁内八rank恢复及新轮48完整检查点仍由现场新证据证明。

后续大间隔微调的三段约定见[三段准备协议](三段LM准备协议.md)。原两段`tools/lm_continuation/readiness.py`保持原判据并应继续拒绝失败vggvm，不能就地放宽。
