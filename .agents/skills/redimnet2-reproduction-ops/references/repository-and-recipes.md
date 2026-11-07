# 仓库与实验配方

## 身份与历史边界

开发仓库在 `dev_a800_2gpus:/root/code/github_repos/redimnet2-fork`。SSH alias、hostname、仓库路径需现场核对；开发机不是实际八卡训练 worker。阅读真实仓库 AGENTS，保留用户未跟踪文档和已有改动。此仓库用户要求后续提交说明用详细中文；是否提交、推送仍按当前任务授权。

原聊天标题 `Redimnetv2复现实验`，ID `01a1501b-b327-7312-9123-c2e34e57c78a`。用户声明 `zhounan` 提交均为个人修改。上游基线 `c5bbe0b76e37df698c403f8844e41304ceab6307`，早期个人分析点 `ff67b8d3ef824368edaff7ee5f23f9b35b12b182`；`9d384869` 不是官方版本。八卡修复点 `7afb821ac7b0062c7d6edda0317d5dc67bbb15f8`；2026-10-21巡检时磁盘 HEAD 为 `0a3f6dd`（历史快照，当前值用 `git rev-parse HEAD` 查）<!-- volatile-ok: 2026-10-21 的历史快照，已标日期 -->区分启动身份、磁盘当前源码和进程实际已加载代码，不把这些历史值当未来默认值。

历史临时源码工作树已按用户要求归档清理，正式任务使用主目录。不要为兼容旧叙述擅自恢复别名或软链。本机“源码”“第二版源码”是历史快照，不是现行源码。

## 已核实的解释器父链

2026-10-21 的共享文件视图审计及后续清理保护记录显示，`/root/miniforge3/envs/redimnet2` 是 venv：`pyvenv.cfg` 的 home/executable 和 Python 软链接指向 `/root/miniforge3/envs/audio-flamingo/bin/`。环境位于 Conda 的 envs 目录或 conda-meta 为空，都不能据此认定它是残缺 Conda 环境。`audio-flamingo` 与已退役的 `audio_flamingo_3` 是不同精确前缀；保留前者是 redimnet2 的基础解释器依赖。

这是带日期的开发机/共享文件系统依赖证据，不等于实际八卡 worker 此刻的解释器或训练状态证明。启动、接续、导出和清理时仍核实相应 launcher 与 worker 的实际路径；无当前进程不能消除下一阶段依赖。环境维护及完整退出/恢复记录用 `$server-conda-environment-maintenance` 的 `references/dev-a800-retirement-20261022.md`，不据相似模型名合并或删除父环境。

## 当前三语料 B6 设计

当前选择是 VoxCeleb2、VoxBlink2、CN-Celeb2 三语料，B6 ReDimNet2Wrap，16kHz 波形输入、192维 embedding。PTN 120轮；LM 1轮在 PTN 120轮完成后从**该次运行第115轮**初始化，非第120轮或最后若干轮平均。公开配置文件名100ep、实际轮数120和LM初始化115之间有不一致，不能推断论文作者真实最终训练轮数。

| 项目 | PTN | LM |
|---|---:|---:|
| 样本数 | 11,521,178 | 5,021,010 |
| 原始说话人/类别 | 119,274 / 357,822 | 115,362 |
| 每卡/8卡总batch | 96 / 768 | 48 / 384 |
| 帧数/采样点 | 200 / 32,240 | 500 / 80,240 |
| 每轮批数 | 15,001 | 13,075 |
| speed | 三语速扩类 | 关闭 |
| margin | 第15轮开始，第30轮达0.2 | 0.3 |

PTN 样本组成：Vox2 1,092,009、VB2 9,904,382、CNC2 524,787。LM 去除3,912名不符合条件者，无新增说话人；分类头按名字映射，禁止随机切片替代映射。

SGD 配置基准LR0.06，legacy缩放 `8*96/64=12` 后峰值0.72，8轮warmup；momentum0.9、Nesterov，骨干WD2e-5、投影头1e-5，AMP、seed42。当前 `compile_backbone:null`，`legacy_aug:true` 走原始增强目录，训练实际 loss 来自 SphereFace2 投影头返回值，不能仅按残留 CrossEntropyLoss 字段命名。

当前官方复现配方没有变长/VAD裁片；这些是个人改进/独立消融，多个设置变化不能称单因素对比。首次32批NaN短测已确认调度器epoch长度错位15001/32；没有首次异常算子证据，不能断言它是唯一根因。第二版 PTN64步、LM32步调度对齐仅证明操作路径，不证明最终精度。

## 评测与导出语义

轮末快速验证：Vox1-O 37,611对、4,708条音频，固定前8秒，短音频wrap，embedding归一化后cosine。最终质量必须同一整段语音协议比较 Vox O/E/H/B、VOICES、CN，eager FP32，CN使用embedding平均。旧拼接说法已过时。

公开三语料权重整段 O/E/H EER 0.255%/0.409%/0.790% 是公开参考结果，不能写为本次自训已达到。导出加载需训练配置、严格backbone、规范化 `module`/`_orig_mod` 前缀，重复键拒绝；只丢弃已识别投影头。当前修复已有真实PTN/LM四条加载路径192维且maxdiff0的证据，旧“检查点格式必不兼容”是已解决的历史问题。最终自训导出与六路双权重评测须重新验证完成状态。

## 独立的官方骨干变长微调

2026-10-23新增的独立变长微调和注册×测试时长矩阵用[duration-finetuning.md](duration-finetuning.md)。它从已下载官方三语料LM骨干、新头与冷优化器开始，不接当前八卡PTN轮号。首版批内同长、批间变长，评测逐条真长度；现有mask的宽门槛测试不能证明补零不变性。当前工程配置与小清单不是已选定的正式全量质量配方，实际GPU/质量完成状态须读对应本次证据。

## 完整字词主实验（2026-10-23用户澄清）

当前主线为[word-complete-finetuning.md](word-complete-finetuning.md)：保留完整一个或几个字的自然长度，先CPU、两卡空闲后再验收GPU。整数秒duration_ft保留诊断。自动ASR/对齐候选保留待审身份，完整字词主评测不能用任意时间裁片填格。
