# 34源四法边界选型实测（2026-10-24）

当前入口：`/Users/zhounan/Documents/声纹降噪/ReDimNet2边界选型对照_20261024/实测选型报告.md`；机读`选型策略_v1.json`，当时远端新根`/root/experiments/redimnet2/boundary-selection-20261024-v1`（历史路径）。先读当前状态及最终原始结果，不把GPU前的准备报告当成当前状态。

## 已证范围与当前选择

34个原件共216.9008125秒：旧的6困难+12业务、新的16开发。主manifest SHA9c7becfb18c6d29df989eb4da79863ed6b855721632cf934e4bd1e4195999948，同原音/NPY/PCM/时钟/文字/语言。482是接口单位，含英语撇号的归一化碎片，不是482个正确完整字词。34×4全部计账，117个适用组合完成、19个NA；provider4另有一条50汉字的轨，不增加独立来源。

当前开发选型优先Qwen3-ASR的忠实转写＋Ultra主体定位，FireRed仍是活动提议和上下文线索；Head作可审细化，BFA作独立phone/词组复核，官方Qwen保基线。这个选择基于真实错位/削尾/邻句案例，不是人工精度榜；human_review仍是null，训练和正式评测资格未释放，也没有改全量v5的冻结合同或自动提交八卡。

主结果用各作者最终返回层做公平对照：官方29源426单位，20个零长；BFA30源381词组/1351phone几何无非法；Ultra无头29源426单位，10个非正+1个越界；Ultra加头29源426单位，几何无非法。BFA的48个粗组覆盖97个汉字，但没有内部char owner，不能均分。2个空target不等没有语音，2个Hindi未准备，Danish只有BFA备了路由。

Hello：官方.48–.56、BFA.4675–4.6975，Ultra主体4.08–4.56；匿名观察里BFA含中文邻句，Ultra只有Hello。公园：官方园零长、组合.72–1.12只得到公，BFA/Ultra组合得到公园。呃的主体区间4.96–5.60覆盖持续有声，但呃/嗯声类的文字仍未决。重复的Bag及“人”继续不自动放行，模型听辨投票不作金标。

## Ultra实际版本与修补层

作者repo是17slever17/Qwen3-ForcedAligner-Ultra，rev765dc1d714e4c80196782a5df5bfbcb3e1416736；主SHAe6fcf5be2c11b265f7217ba6102917a7fe64438506357be6999f6ad5a707cb5f，head SHA50a70d6ed969446770fb9dc1063f5f4e6e4c86eac05fd3efb637492407d0e2ac。新canonical `/root/group-shared/voiceprint/models/a800_2gpus/research/qwen-ultra-20261024/Qwen3-ForcedAligner-Ultra_765dc1d7`，实际resolve在/shared。25个资产逐SHA验证过；运输镜像不是身份权威。模型注册表是否含新的research条目要另核，不自动宣称全局published。

最终collector `Ultra/ultra_runner_v3_timestamp_numeric.py` SHAcd7bfbfddbd1cc5b8b2d5bb9693f8b819ed4d6dcf24d1b9a2a944fa81b2bf4cf，支持shared_qwen_fixed_text/provider_fixed_text。初次采集器把真实的float80.0拒成缺属性，完整失败原样保留；新版本从实际属性里取有限正整数量值，拒bool/字符串/非有限/分数，没有151705/80的猜默认，也不改作者/模型。

无头/加头的29源，实际thinker输入与rawlogit的NPY SHA完全相同。raw有5个非正+1个越界；FIX反而多了5个非正；Head后的规则改了91个单位（其中6个由非法变合法，4个duration_limited仍是false），最终5个单位恰好10ms。raw head产生81对重叠，inline后84，再由overlap清0。合法化不能证明完整词，也不能拿官方raw对Ultra最终层单独排赢家。

保存顺序是raw coarse→每次FIX→duration mask→raw head logits/offset→inline逐行→overlap→duration bound。posterior用同一式子重算，标diagnostic；没有live module.training字段，就不补写成已观测（实际源码的两次eval另列）。官方本轮只保存pre-FIX时间向量及返回单位，不能泛称所有方法都有thinker logits。

## BFA完整样本与原生限制

最终runner是 BFA/delivery-v3/bfa_runner_v3.py SHAe78f5f785e965fde336baf8e07632e0fc1ea3e2dc4653f7645a2a38b3bbde6b2；contract是同目录的runtime_contract-v2.json SHA1794c2bdafe3f94f41042375110d9ac92fb33d6498ab404bad703cb994e37368。226包/模型/新隔离依赖的pins固定；pycantonese3.4.0/pylangacq0.16.0/wordseg0.0.2/tabulate0.9.0/wcwidth0.2.13只装进新target，不改旧env。

原生的int(16.24*16000)少1个样本；新v3只用最小的nextafter确保SRT秒表示int往返N，原钟和原音都不变。实际30个输入NPY的全N前缀/零pad，以及独立CPU从原NPY重算RMS逐byte，都通过。全部的word/phone时间与首轮相同；只有一个源的SRT表示和466confidence轻微变化，原467字段的差异保留。

仍有24源的有效DP网格没有延伸到物理尾，最大74.5ms；这不等于24源已经证实削音。完整encoder前缀不等于有效DP覆盖到尾部，补零假恢复或改原证据都不行。

已核：默认的自动首尾SIL可以跳过，并非机械强迫首尾phone；bonus30/class_prior_norm=true/punctuation_sil=false。business10原SIL后验低，与business07 SIL top1很强却被内部路径分给v，是两种不同机制。4源×基线/关闭bonus/关闭prior，12支精确cached-logit CPU解码，原路径/时间基线复现，微小的confidence差保留；没有统一更优的开关，不擅自全局改配置。

## API、图与来源资格

逐源的原增益波形/RMS/25ms窗5ms步频谱、最终四法及官方raw诊断层都保存。root与子代理实际看图属于AI图像审阅，不是人听；机器听辨也不写成human。

用户本轮直接批准固定34及必要裁片到内部new-api…/v1的qwen3-8-omni-flash。新16整源16/17请求；匿名主126个视图来自29parent，109方法core关联去重97core+29context，100usable/7flagged/19HTTP400；Danish另2视图2usable。本轮总共145次请求尝试，不把旧18历史的65请求重算成新请求或独立原件。

7个flags包含[听不清]/声音描述/嗯的字段语义，不能统称成真实词句矛盾；原flags和独立内容的分层分别保留。19个短片10.0625–90.4375ms的generic400，不证明平台有明确的100ms阈值；不补零绕过，也不当无语音。API生成的削音判断可能过度断言、猜字；结构通过不等于内容真值。

PRIVATE_join里的legacy status `audible_native_outer_envelope`只表示几何上可导出，并没有证明可辨；内部零长/文字未决的照旧保留。粗组的共同外包络可以做几个字的研究，不能伪造出内部单字。

## 资源与接续

本轮只在两卡机的空闲GPU1上串行，各阶段用canonical UUID flock、新鲜双快照和单可见设备；GPU0忙的时候不碰，最后provider Head归还时15:27:17/22Z两卡都只有2MiB且compute为空。新末次SSH回读超时原样保留，不据此断言机器下线/PTN停止。PTN不重启/迁移/改环境或冻结清单，活动run t-20261024154424-vggvm只读。

这个单卡推理门禁不代替B6真实双卡训练或八卡rank验收。八卡全量接入仍需要新代码/模型/策略合同和代表批的真实恢复/吞吐/数据质量，不能用这34个开发源或旧v5官方Qwen的凭证替代。后台PAUSED不自行恢复。
