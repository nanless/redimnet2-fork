# 阶段接续与冻结身份

本页保存10月23—24日的阶段合同和当时准备状态。10月24日原任务因节点待下线终止、model35完成CPU恢复时，尚未启动后继；该句不描述今天的状态。之后的model35修复、model47接续和ctjsk进度，先读[训练与接续](training-and-continuation.md)及[任务状态](task-status-and-readonly-progress.md)。旧v4不能简单换parent用于跨段，原[维护终止记录](ptn-maintenance-stop-and-resume-20261024.md)保留。

2026-10-23当前版本指向`event-review-20261023-v4/platform_entry.py`：本轮用户明确授权逐条审查第4—7次，固定七次发生/三份音频，第8次仍拒绝。具体定位、用例及证据边界见[FLAC定位回归](flac-seek-regressions.md)，仓库当前说明为docs/18-FLAC定位异常与接续检查.md。v3三事件/两音频及10月22日验收是历史，不等于v4已安装或通过；下文带日期的旧说明保留。

## 先定位真实依赖

目录映射只帮助查找；训练进程保存的绝对路径不会因映射更新而自动改写。迁移前不能只看当前open FD/inode：启动后暂不读取的校验清单、smoke证明、data audit、官方参考权重，仍是LM/导出/评测的活跃依赖。还需检查wrapper、launcher、父进程白名单环境、阶段分支和下一次读取时机。

对现行 `applications/speaker_rec/scripts/cloud/run_b6_mix_reproduction.sh` 及其任务入口核对：

- `SOURCE_CHECKSUM_FILE`、`SMOKE_SUCCESS_FILE`、`DATA_AUDIT_FILE`、`OFFICIAL_REFERENCE`、`RUN_ROOT` 实际值与容器内存在性。
- 启动时 manifest自身 SHA 与run_identity；manifest每个文件的SHA。路径存在、manifest身份一致、被列源码一致是三层检查。
- `verify_source` 在每次 `run_stage` 前重验manifest本身和各文件。PTN 已于 2026-11-03 完成、LM 已于 11-06 提交，但这道门只证明当次 manifest 与被列源码一致，不证明新阶段的分类头、冷启动或输出唯一性——那些另由 LM 合同与 gate 核（见下节）。
- 重启平台入口可能要求原固定HEAD、干净工作区和旧依赖；checkpoint可读不能替代这些检查。
- PTN115 checkpoint是否属于同一从零run、speaker map匹配、LM新optimizer/scaler、最终官方权重路径和输出目标。

进程已加载代码、磁盘当前HEAD、源码文件SHA分别记录。后续更改是否影响当前运行需追实际调用分支与动态worker导入，不能用“Python已启动”一概排除影响。

## LM 阶段门（2026-11-06 实现并用于 r3）

LM 不是"PTN 之后再跑一次同样的训练"，它的门和 PTN 恢复门方向相反：PTN 要**恢复**优化器与 scaler，LM 要**丢掉**它们。当前实现见仓库 `tools/ptn_resume/lm_launch/lm_stage_gate.py`（包内 `runtime/`）与 `tools/ptn_resume/three_segment_lm_check.py`（三段证据检查器，单文件纯标准库）。

- **`spk2id.json` 是执行门，不是参考文件。** 映射缺失或不可读时，`train.py` 会从"按名字重组分类头"退回"按索引切片"，357,822 → 前 115,362 行，**形状刚好合法**且 `strict=False`，于是静默训错而不报错。gate 的顺序是：文件存在 → sha256 等于合同 `speaker_map.sha256` → 非空 → 编号为整数 → 人数等于合同 `ptn_speakers`（119,274）→ 读 LM 标签重数 115,362 → 每个 LM 说话人都在映射里。任一步不符即 `LM_GATE_REFUSED`。
- **`RECIPE` 必须显式绑定到包内 `resume.yaml`。** `train_b6_mix_cloud_8gpu.sh` 里 `RECIPE` 优先于 `REPRO=1` 的默认配方；不显式传就会出现"gate 验包内配方、训练跑仓库配方"。gate 传 `RECIPE=<包>/resume.yaml` 并断言 `env["RECIPE"]==recipe`。
- **`ready_for_lm_execution` 恒为 false。** 三段检查器的 `build`/`check` 输出里这个字段是硬编码的 false，`check` 还会拒绝任何自称 true 的合同。2026-11-06 真实数据：`build` 17/17、`check` 18/18（含 `contract.no_drift`），均 exit 0；退出码语义 0=全 pass、2=有 fail、3=有 unsupported。
- 检查器对"拿不到原件"的处置是 `unsupported` 而不是 pass：原 95 清单、平台终态这类证据缺失时退出码 3、`checks_passed:false`。不要为了让它 exit 0 去放宽判据或补造条目。
- 包与合同的身份由外部注入（`LM_PACKAGE_SHA`/`LM_CONTRACT_SHA`），验的是"这份包就是被审过的那份"，不构成执行许可；`runtime_admission` 始终 false。

## 发现漂移后的处理

巡检/文档任务记录影响范围与证据，不擅自改源码、恢复旧软链、重写冻结manifest或重启训练。修复获授权后，先拟定可审查的接续方案：恢复经过核实的原始只读依赖，或使用独立后继身份和明确的新基线；保护本次已有检查点和原始冻结材料。不能仅为了通过门禁把三处变化写成新SHA并冒称原实验未漂移。

修复完成需要验证容器实际路径、源码内容、门禁/初始化兼容性和运行身份，不能用文档已更新或开发机存在文件作为完成凭据。修改训练相关路径/源码前阅读仓库AGENTS与平台技能；用户的文档维护请求不自行扩张成训练修复。

## 材料与复现记录

历史task、smoke、从零启动、当前run、后继run分开记。保留原始校验及日期，入口链接最新报告。遵守本次材料位置选择；服务器默认存放惯例不能覆盖用户“本次报告和日志只留本机”的决定。服务器仓库维护文档与复制整个本次巡检证据包是不同操作，后者不能由一般文档维护请求推定授权。

## 不热改当前训练的接续修复

已运行 Bash 的函数体、here-doc、已解析条件块和捕获环境不会因磁盘脚本更新而自动替换。检查未来每道判定，包括训练结束后的 check_stage，不能只检查 LM 前的 verify_source。ReDimNet2 原 wrapper 会因历史捕获 FLAC 读取的 Traceback 拒绝主训练终验；保留旧日志，不删文本、不包装解释器来掩盖。

用户要求不影响当前训练时，将精确原件路径恢复与独立后继执行身份分别处理：

- 仅在明确授权修复失效依赖后，按现行布局验证原件和摘要，创建精确文件级兼容链接，不恢复整棵旧树、不覆盖冲突，保留可逆记录。四依赖恢复不意味着三份活跃源码无漂移。
- 独立源从开训提交重建；保留原冻结清单字节，将每项原绝对路径安全映射到独立源，逐项核原 SHA。独立源 95/95 与活跃树仍 92/3 分别报告，不能把新摘要冒充原冻结身份。
- 在新解释器核实模型、训练、数据、检查点模块的实际导入位置，防止 editable 安装指回活跃树；新源不加入当前主训练环境。
- 同 run 主训练120轮完成、115/120权重完整有限、旧平台任务终态、写入者退出和原锁释放后，才允许一次性 LM。证据带任务、run 和新鲜时刻；新输出与原 run、源码均不重叠。首次 LM 从本 run115 冷初始化优化器和混合精度状态。
- 已知捕获异常按独立发生逐条匹配原任务/run/attempt、已核日志前缀、原始偏移和字节、音频、异常、DROP及精确轮次和前后批号。同样本或同归一化块SHA在另一轮再次出现仍是新事件，不能合并或自动放行；只允许当前经授权审阅的记录。额外Traceback/DROP、非有限值和致命标记仍拒绝。原始字节及SHA保留，用于比较的换行规范化和原始SHA分别记录；原PTN历史事件许可不延伸到新LM日志。
- CPU 预检、真实八卡 LM、轮末质量与最终评测分别验收。返回零不替代八 rank 冷启动状态、LM1完整权重和验证结果。

2026-10-22 的实施目录为 /root/experiments/redimnet2/b6-mix-official-20261019/lm-continuation-20261022。主仓库 docs/17-本次主训练的大间隔微调接续修复.md 解释身份和边界。是否已安装、容器已验收、父任务已终态或 LM 已启动均取新鲜证据，不能凭目录存在推断。

2026-10-22上午第23轮6223→6224的第三次捕获错误另行审阅后，新操作版本为`/root/experiments/redimnet2/b6-mix-official-20261019/lm-continuation-20261022/event-review-20261022-v3`；保留凌晨continuation旧版及近文件末端检查真实失败、正确拒绝的v2版本，沿用同一source_root、一次性输出与全局互斥。已审三次的位置为1/8740→8741、10/13729→13730、23/6223→6224；第四次即使同音频同块也仍拒绝，不能据这项修复声称120轮、终态、无writer或LM已经通过。安装摘要、CPU子项与完整门禁、开发机音频复查和真实容器验收分别记录。


解码读取的整段及首中末检查成功，不能排除生产随机裁剪的seek/read失败。本次v3仅在未来LM新解释器启动前固定其私有`runtime_audio/lib`的LD_LIBRARY_PATH；核实际加载库路径、SHA、版本及soundfile.py身份，再核实际PTN/LM裁剪长度与近文件末端压力读取，任一失败拒绝。不要修改活跃PTN环境、系统库、音频或冻结源码。旧1.0.28与私有1.2.2的开发机19例对照为5失败/0失败，两音频整段PCM相同；历史三次随机起点未记录，不能据此称全部历史调用已重放、所有裁剪或训练逐位等价。v2失败记录保留，新版测试、安装、worker验收和完整ready分别取证；不由开发机通过推断容器或未来LM通过。

2026-10-22的历史平台入口使用v3/platform_entry.py，复用原门禁、输出及全局guard。未来LM的LD_LIBRARY_PATH固定为私有runtime_audio/lib在首位，随后保留实际worker的`/usr/local/nccl-rdma-sharp-plugins/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64`，清外部LD_PRELOAD/LD_AUDIT；不能因CPU通过删掉GPU基础库。CPU三命令也使用此固定组合，不修改当前PTN原环境。新增组合的实际worker CPU检查另验，未来八rank GPU通信、真实库身份和冷启动需启动后验收，不由CPU结果推断。

新v4仍共用原source-7afb821、lm-execution-once和guard，原95/189、PTN120、115/120、平台终态/无writer/两原锁与8rank冷初始化要求不变。PTN历史许可不延伸到LM，LM裸read_failed（无Traceback也包括）与新Traceback均拒绝。新增版本测试、实际安装、worker CPU子项、完整ready和GPU启动分别取证；准备说明不能代替执行。
