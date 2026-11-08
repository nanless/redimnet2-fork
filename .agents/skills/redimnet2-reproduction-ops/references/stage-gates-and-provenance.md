# 阶段接续与冻结身份

本页保存10月23—24日的阶段合同和当时准备的接续状态。10月24日，原任务因节点待下线而终止，model35完成CPU恢复时后继还没有启动；这一句只描述当时，不描述今天。model35的修复、model47的接续和ctjsk进度，先读[训练与接续](training-and-continuation.md)及[任务状态](task-status-and-readonly-progress.md)。旧v4换一个parent并不能跨段使用，原[维护终止记录](ptn-maintenance-stop-and-resume-20261024.md)保留。

2026-10-23当前版本指向`event-review-20261023-v4/platform_entry.py`：本轮用户明确授权逐条审查第4—7次，固定七次发生/三份音频，第8次仍拒绝。具体定位、用例和证据边界见[FLAC定位回归](flac-seek-regressions.md)，仓库里的当前说明是docs/18-FLAC定位异常与接续检查.md。v3三事件/两音频和10月22日的验收都属历史，不等于v4已经安装或已经通过；下文带日期的旧说明保留原样。

## 先定位真实依赖

目录映射只方便查找；训练进程保存的绝对路径不会因为映射更新而自动改写。迁移前只看当前open FD/inode不够：启动后暂不读取的校验清单、smoke证明、data audit、官方参考权重，仍是LM/导出/评测的活跃依赖。这些依赖什么时候被读，还取决于wrapper、launcher、父进程白名单环境、阶段分支和下一次读取的时机。

对现行 `applications/speaker_rec/scripts/cloud/run_b6_mix_reproduction.sh` 及其任务入口核对：

- `SOURCE_CHECKSUM_FILE`、`SMOKE_SUCCESS_FILE`、`DATA_AUDIT_FILE`、`OFFICIAL_REFERENCE`、`RUN_ROOT` 的实际值，以及这些文件在容器里是否存在。
- 启动时的 manifest 自身 SHA 和 run_identity，以及 manifest 里每个文件的 SHA。路径存在、manifest 身份一致、被列源码一致，是三层不同的检查，前一层通过不代表后一层也通过。
- `verify_source` 在每次 `run_stage` 前重新核对 manifest 本身和其中每个文件。PTN 已在 2026-11-03 完成、LM 已在 11-06 提交，但这道门只证明当次 manifest 与被列源码一致；新阶段的分类头、冷启动、输出唯一性不在它的证明范围内，由 LM 合同与 gate 另行核对（见下节）。
- 重启平台入口可能要求原来的固定 HEAD、干净工作区和旧依赖；checkpoint 能读，不代表这些条件已经满足。
- PTN115 checkpoint 是否属于同一个从零 run、speaker map 是否匹配、LM 是否新建 optimizer/scaler，以及最终官方权重路径和输出目标。

进程已加载的代码、磁盘当前 HEAD、源码文件 SHA 分别记录，三者不是一回事。后续改动会不会影响当前运行，要看实际调用的分支和动态 worker 导入，不能用“Python已启动”一概排除。

## LM 阶段门（2026-11-06 实现并用于 r3）

LM 不是“PTN 之后再跑一次同样的训练”。它的门和 PTN 恢复门方向相反：PTN 要**恢复**优化器与 scaler，LM 要**丢掉**它们。当前实现见仓库 `tools/ptn_resume/lm_launch/lm_stage_gate.py`（包内 `runtime/`）与 `tools/ptn_resume/three_segment_lm_check.py`（三段证据检查器，单文件纯标准库）。

- **`spk2id.json` 是执行门，不是参考文件。** 映射缺失或不可读时，`train.py` 会从“按名字重组分类头”退回“按索引切片”：357,822 → 前 115,362 行，形状刚好合法，又带着 `strict=False`，于是静默训错而不报错。gate 的判定顺序是：文件存在 → sha256 等于合同 `speaker_map.sha256` → 非空 → 编号为整数 → 人数等于合同 `ptn_speakers`（119,274）→ 读 LM 标签重数 115,362 → 每个 LM 说话人都在映射里。任一步不符的结果是 `LM_GATE_REFUSED`，任务不会以这份映射启动。
- **`RECIPE` 要显式绑定到包内 `resume.yaml`。** `train_b6_mix_cloud_8gpu.sh` 里 `RECIPE` 的优先级高于 `REPRO=1` 的默认配方；不显式传，就出现“gate 验包内配方、训练跑仓库配方”的分叉。gate 的做法是传 `RECIPE=<包>/resume.yaml`，并断言 `env["RECIPE"]==recipe`。
- **`ready_for_lm_execution` 恒为 false。** 三段检查器的 `build`/`check` 输出里，这个字段是硬编码的 false，`check` 还会拒绝任何自称 true 的合同。2026-11-06 的真实数据：`build` 17/17、`check` 18/18（含 `contract.no_drift`），都 exit 0。退出码语义：0=全 pass、2=有 fail、3=有 unsupported。
- 检查器对“拿不到原件”的处置是 `unsupported` 而不是 pass：原 95 清单、平台终态这类证据缺失时，退出码 3、`checks_passed:false`。用放宽判据或补造条目的办法换 exit 0，会把“没查到”改写成“通过”，缺口从此看不出来。
- 包与合同的身份由外部注入（`LM_PACKAGE_SHA`/`LM_CONTRACT_SHA`），验的是“这份包就是被审过的那份”。它不构成执行许可，`runtime_admission` 始终为 false。

## 发现漂移后的处理

巡检或文档任务的范围是记录影响范围和证据；改源码、恢复旧软链、重写冻结manifest、重启训练都不在这个范围里。修复拿到授权后，先写一份可审查的接续方案，二选一：恢复经过核实的原始只读依赖，或改用独立的后继身份和明确的新基线；同时保住本次已有的检查点和原始冻结材料。三处变化如果被写成新SHA并冒称原实验未漂移，来源记录就断了，门禁也失去了原本要比对的对象。

修复完成要验证容器里的实际路径、源码内容、门禁/初始化兼容性和运行身份；文档已更新、开发机上存在文件，都不算完成凭据。改训练相关路径/源码前阅读仓库AGENTS与平台技能：用户提出的是文档维护请求，不会因此扩张成训练修复。

## 材料与复现记录

历史task、smoke、从零启动、当前run、后继run分开记。保留原始校验值和日期，入口指向最新报告。材料放在哪里按本次的选择来；服务器默认的存放惯例不能覆盖用户“本次报告和日志只留本机”的决定。在服务器仓库里维护文档，和把整份本次巡检证据包复制过去，是两件不同的事，后者不能从一般的文档维护请求里推定授权。

## 不热改当前训练的接续修复

Bash 已经跑过的函数体、here-doc、已解析的条件块和捕获的环境，不会因为磁盘上的脚本更新而自动替换。所以未来每一道判定都要单独检查，包括训练结束后的 check_stage；只看 LM 前的 verify_source 不够。ReDimNet2 原 wrapper 会拒绝主训练终验，原因是它历史上捕获过 FLAC 读取的 Traceback；旧日志保留原样，删文本或用包装解释器掩盖，都会让这次拒绝失去凭据。

用户要求不影响当前训练时，精确原件路径的恢复和独立后继执行身份分开处理：

- 只有在明确授权修复失效依赖之后，才按现行布局核对原件和摘要、创建精确到文件的兼容链接；不恢复整棵旧树，不覆盖冲突，并留下可逆记录。四个依赖恢复，不等于三份活跃源码没有漂移。
- 独立源从开训那个提交重建：保留原冻结清单的字节，把每项原绝对路径安全映射到独立源，再逐项核对原 SHA。独立源 95/95 与活跃树仍 92/3，两个结果分别报告；新摘要不能冒充原冻结身份。
- 在新解释器里核实模型、训练、数据、检查点这几个模块实际从哪里导入，避免 editable 安装指回活跃树；新源不加入当前的主训练环境。
- 同一个 run 的主训练120轮完成、115/120权重完整有限、旧平台任务进入终态、写入者退出和原锁释放，这些条件齐了才允许做一次性 LM。证据带任务、run 和新鲜时刻；新输出与原 run、源码均不重叠。首次 LM 从本 run115 冷初始化优化器和混合精度状态。
- 已知捕获异常按独立发生逐条匹配：原任务/run/attempt、已核日志前缀、原始偏移和字节、音频、异常、DROP及精确轮次和前后批号。同样本或同归一化块SHA在另一轮再次出现，仍是新事件，不能合并，也不能自动放行；只允许当前经授权审阅的记录。额外Traceback/DROP、非有限值和致命标记仍拒绝。原始字节及SHA保留；用于比较的换行规范化和原始SHA分别记录。原PTN历史事件许可不延伸到新LM日志。
- CPU 预检、真实八卡 LM、轮末质量与最终评测分别验收。返回零不替代八 rank 冷启动状态、LM1完整权重和验证结果。

2026-10-22 的实施目录为 /root/experiments/redimnet2/b6-mix-official-20261019/lm-continuation-20261022。主仓库 docs/17-本次主训练的大间隔微调接续修复.md 说明了身份和边界。是否已安装、容器是否已验收、父任务是否已终态、LM 是否已启动，都取新鲜证据；目录存在推不出这些结论。

2026-10-22上午第23轮6223→6224的第三次捕获错误经另行审阅后，新操作版本为`/root/experiments/redimnet2/b6-mix-official-20261019/lm-continuation-20261022/event-review-20261022-v3`。凌晨的 continuation 旧版和“近文件末端检查真实失败、正确拒绝”的 v2 版本都保留；新旧共用同一source_root、一次性输出与全局互斥。已审三次的位置为1/8740→8741、10/13729→13730、23/6223→6224；第四次即使同音频同块，也仍然拒绝。这项修复不足以支持“120轮、终态、无writer或LM已经通过”的说法。安装摘要、CPU子项与完整门禁、开发机音频复查和真实容器验收，分别记录。


整段解码读取和首中末检查通过，仍不能排除生产随机裁剪的seek/read失败。本次v3只做一件事：在未来LM新解释器启动前，固定它私有`runtime_audio/lib`的LD_LIBRARY_PATH。之后核实际加载的库路径、SHA、版本和soundfile.py身份，再核实际PTN/LM裁剪长度与近文件末端压力读取，任一失败就拒绝。活跃PTN环境、系统库、音频和冻结源码都不改动。旧1.0.28与私有1.2.2的开发机19例对照为5失败/0失败，两音频整段PCM相同；历史三次随机起点没有记录，所以不能说全部历史调用已重放，也不能说所有裁剪或训练逐位等价。v2的失败记录保留；新版测试、安装、worker验收和完整ready分别取证——开发机通过推不出容器通过，更推不出未来 LM 通过。

2026-10-22的历史平台入口使用v3/platform_entry.py，复用原门禁、输出及全局guard。未来LM的LD_LIBRARY_PATH固定为私有runtime_audio/lib在首位，其后保留实际worker的`/usr/local/nccl-rdma-sharp-plugins/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64`，并清掉外部LD_PRELOAD/LD_AUDIT。CPU 通过不能成为删掉 GPU 基础库的理由，那些库在八卡启动时才用得上。CPU三命令也使用这个固定组合，当前PTN原环境不动。新增组合在真实 worker 上的 CPU 检查另做；未来八rank的 GPU 通信、真实库身份和冷启动要等启动后验收，CPU 结果推不出来。

新v4仍共用原source-7afb821、lm-execution-once和guard，原95/189、PTN120、115/120、平台终态/无writer/两原锁与8rank冷初始化要求不变。PTN历史许可不延伸到LM：LM裸read_failed（无Traceback也包括）与新Traceback均拒绝。新增版本测试、实际安装、worker CPU子项、完整ready和GPU启动分别取证；准备说明代替不了执行。
