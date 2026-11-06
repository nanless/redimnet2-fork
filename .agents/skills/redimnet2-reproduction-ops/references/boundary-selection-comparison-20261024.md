# 34源四法边界选型实测（2026-10-24）

当前入口：`/Users/zhounan/Documents/声纹降噪/ReDimNet2边界选型对照_20261024/实测选型报告.md`；机读`选型策略_v1.json`，真实远端新根`/root/experiments/redimnet2/boundary-selection-20261024-v1`。先读当前状态及最终原始结果，不把GPU前准备报告当当前状态。

## 已证范围与当前选择

34原件216.9008125秒：旧6困难+12业务、新16开发。主manifest SHA9c7becfb18c6d29df989eb4da79863ed6b855721632cf934e4bd1e4195999948，同原音/NPY/PCM/时钟/文字/语言。482是接口单位，含英语撇号归一化碎片，不是482正确完整字词。34×4完整计账，117适用组合完成、19NA；provider4另轨50汉字，不增独立来源。

当前开发选型优先Qwen3-ASR忠实转写＋Ultra主体定位，FireRed仍是活动提议和上下文线索；Head作可审细化，BFA作独立phone/词组复核，官方Qwen保基线。此选择基于真实错位/削尾/邻句案例，不是人工精度榜；human_review仍null，训练/正式评测资格未释放，也未改全量v5冻结合同或自动提交八卡。

主结果用各作者最终返回层公平对照：官方29源426单位20零；BFA30源381词组/1351phone几何无非法；Ultra无头29源426单位10非正+1越界；Ultra加头29源426单位几何无非法。BFA48粗组覆盖97汉字但无内部char owner，不能均分。2空target不等无人声，2Hindi未准备，Danish仅BFA已备路由。

Hello：官方.48–.56、BFA.4675–4.6975，Ultra主体4.08–4.56；匿名观察BFA含中文邻句而Ultra只Hello。公园：官方园零长、组合.72–1.12只得到公，BFA/Ultra组合得到公园。呃的主体区间4.96–5.60覆盖持续有声，但呃/嗯声类文字仍未决。重复Bag及“人”继续不自动放行，不以模型听辨投票作金标。

## Ultra实际版本与修补层

作者repo17slever17/Qwen3-ForcedAligner-Ultra，rev765dc1d714e4c80196782a5df5bfbcb3e1416736；主SHAe6fcf5be2c11b265f7217ba6102917a7fe64438506357be6999f6ad5a707cb5f，head SHA50a70d6ed969446770fb9dc1063f5f4e6e4c86eac05fd3efb637492407d0e2ac。新canonical `/root/group-shared/voiceprint/models/a800_2gpus/research/qwen-ultra-20261024/Qwen3-ForcedAligner-Ultra_765dc1d7`，实际resolve在/shared。25资产逐SHA验证；运输镜像不是身份权威。模型注册表是否含新research条目另核，不自动宣称全局published。

最终collector `Ultra/ultra_runner_v3_timestamp_numeric.py` SHAcd7bfbfddbd1cc5b8b2d5bb9693f8b819ed4d6dcf24d1b9a2a944fa81b2bf4cf，支持shared_qwen_fixed_text/provider_fixed_text。初次采集器把真实float80.0拒成缺属性，保完整失败；新版本从实际属性取有限正整数量值，拒bool/字符串/非有限/分数，无151705/80猜默认，不改作者/模型。

无头/加头29源实际thinker输入与rawlogit NPY SHA全同。raw5非正+1越界；FIX反增5非正；Head后规则改91单位（其中6由非法变合法，4个duration_limited仍false），最终5单位恰10ms。raw head产生81对重叠，inline后84，再由overlap清0。不要以合法化证明完整词，也不能用官方raw对Ultra最终层单独排赢家。

保存raw coarse→每次FIX→duration mask→raw head logits/offset→inline逐行→overlap→duration bound。posterior同式重算标diagnostic；没有live module.training字段，不补写成已观测（实际源码两次eval另列）。官方本轮只保存pre-FIX时间向量及返回单位，不能泛称所有方法都有thinker logits。

## BFA完整样本与原生限制

最终runner BFA/delivery-v3/bfa_runner_v3.py SHAe78f5f785e965fde336baf8e07632e0fc1ea3e2dc4653f7645a2a38b3bbde6b2；contract同目录runtime_contract-v2.json SHA1794c2bdafe3f94f41042375110d9ac92fb33d6498ab404bad703cb994e37368。226包/模型/新隔离依赖pins固定；pycantonese3.4.0/pylangacq0.16.0/wordseg0.0.2/tabulate0.9.0/wcwidth0.2.13只装新target，不改旧env。

原生int(16.24*16000)少1样本；新v3仅用最小nextafter确保SRT秒表示int往返N，原钟和原音不变。实际30输入NPY的全N前缀/零pad及独立CPU从原NPY重算RMS逐byte均通过。全部word/phone时间与首轮相同；仅一源SRT表示和466confidence轻微变化，原467字段差异保留。

仍有24源有效DP网格未延伸至物理尾，最大74.5ms；不等于24源已证削音。完整encoder前缀不等于有效DP覆盖到尾，不能补零假恢复或改原证据。

已核默认自动首尾SIL可跳、并非机械强迫首尾phone；bonus30/class_prior_norm=true/punctuation_sil=false。business10原SIL后验低与business07 SIL top1强却被内部路径分配给v是不同机制。4源×基线/关闭bonus/关闭prior，12支精确cached-logit CPU解码，原路径/时间基线复现，微confidence差保留；没有统一更优开关，不擅自全局改配置。

## API、图与来源资格

逐源原增益波形/RMS/25ms窗5ms步频谱、最终四法及官方raw诊断层都保存。root与子代理实际看图属于AI图像审阅，不是人听；机器听辨也不写human。

用户本轮直接批准固定34及必要裁片到内部new-api…/v1的qwen3-8-omni-flash。新16整源16/17请求；匿名主126视图来自29parent，109方法core关联去重97core+29context，100usable/7flagged/19HTTP400；Danish另2视图2usable，总145本轮请求尝试，不把旧18历史65请求重算成新请求或独立原件。

7flags包含[听不清]/声音描述/嗯的字段语义，不能统称真实词句矛盾，原flags和独立内容分层分别保留。19短片10.0625–90.4375ms的generic400不证明明确100ms平台阈值；不补零绕过、不当无语音。API生成削音判断可能过度断言、猜字；通过结构不等内容真值。

PRIVATE_join中的legacy status `audible_native_outer_envelope`仅表示几何可导出，并未证明可辨；内部零长/文字未决照留。粗组共同外包络可做几个字研究，不能伪造内部单字。

## 资源与接续

本轮只在两卡机空闲GPU1串行，各阶段canonical UUID flock+新鲜双快照+单可见设备；GPU0曾忙时不触碰，末provider Head归还15:27:17/22Z两卡均2MiB且compute空。新末次SSH回读超时保留，不据此断言机器下线/PTN停止。PTN不重启/迁移/改环境或冻结清单，活动run t-20261024154424-vggvm只读。

此单卡推理门禁不代替B6真实双卡训练或八卡rank验收。八卡全量接入仍需新代码/模型/策略合同和代表批真实恢复/吞吐/数据质量，不能用这34开发源或旧v5官方Qwen凭证替代。后台PAUSED不自行恢复。
