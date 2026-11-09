# 全量生产sidecar实施（2026-10-24）

本页是早期混合范围的实施历史。用户在10月25日已经把本轮正式标注限定为VoxCeleb2、VoxBlink2、CN-Celeb2的实际训练来源；MUSAN、Child、RIR和额外评测域不再按下文的旧计划继续。现行三域链路先读[Ultra实施记录](ultra-production-sidecar-20261025.md)，后续的正式B8和查询口径读[转写提速与接续](asr-speed-and-continuation.md)。本文里各个“当前”任务、合同、P0/P1数字，都对应原来的观察时刻。

主根 `/Users/zhounan/Documents/声纹降噪/ReDimNet2全量开源标注_20261024`；详细计划 `docs/plans/2026-10-24-production-sidecar-execution.md`，实际报告 `全量实施_20261024/实施与验收报告.md`，实时状态 `当前状态.json`。服务器新根 `/root/experiments/redimnet2/open-annotation-20261024`（历史路径）。本页是带日期的实施记录；继续之前要核实际的sequence/parent_result及唯一平台ID，不能从静态文字推断提交已完成。

## 范围与来源

PTN/LM同源角色合并，六测试协议展开端点、Child正式清单及保护18另计；source_id是dataset+NUL+保留扩展名的相对路径，不是按PCM内容去重。完整的16,943,440文本引用已经读过；最终唯一分母要等catalog完整合并和固定index，不能把PTN/LM的行数直接相加。正式300/1000/4382批来自全声明文本的域内minhash，分层不代表自然占比；prefix批只作工程热身。完整注册按低优先级限流；v1与4CPU v2逐桶字节等价才交接。只清理本次精确自有的CPU构建，不碰GPU/训练。

MUSAN实际2016源，共109.2896064原生头小时（660music/930noise/426speech）；独立的source index进入ASR，增强资产不能当SV身份用，空ASR也不等非人声。RIR60000物理滤波器是ASR-NA，资产源身份的CPU审计另计。62016 metadata sidecar不代表2016内容标注或RIR完整身份已验。12源MUSAN先导含26ms/406s/980s长尾，不截短掩盖。

旧FireRed `root_depth=2`映射保留源扩展名；隐藏meta是 `.filename.wav.rttm.json`。旧header无原音SHA，只有source路径/size/mtime/原钟/模型与签名。A_域可写绝对/shared/...，B/C为相对；只核已知/root/group-shared↔/shared/group-shared确切别名，不按basename/suffix放宽。legacy stat/clock重用不冒称历史SHA绑定。5条CN-Celeb2旧RTTM三位小数端点较精确原钟超1/4/7采样点已留原文，保持unknown——提高epsilon或clip只是把这个偏差抹平，不是修正。

## 生产执行合同

production-code-v1/v2/v3都是独立只读新目录。模型/实际14 Qwen源码、11语义模块、decoder与策略固定；代码一变就是新合同新版本，不能改旧冻结SHA。v2先保raw后调用fix_timestamp，失败保raw/traceback且failed；v3审计时把单字母spoken-form分开，不把全大写自然句一概当缩写。实际Qwen env是soundfile0.13.1，映射libsndfile1.2.2 SHA c872fb84af2afc0e96b603e0000f0f407af86c49ad1126f2785e5722c7bb27d7，与PTN私有修复同bytes；照抄PTN的Python版本，或只设LD_LIBRARY_PATH就说换了库，都不算。

每GPU一个离线ASR+FA副本，独立进程非DDP；FP32/eager/TF32false，核心24s加两侧2s，原音按完整连续区间拥有。VAD只提议切点，VAD之外的内容不丢、不拼接。source文件/原生PCM流式hash、原钟和模型输入PCM分别保存。每个sidecar一个mirrored版本目录；journal/atomic commit/稳定父锁与chunk锁/generation fencing/completion map，卡数不进jobkey。恢复时读回同合同已完成的结果，不每次扫千万个侧车；轻map终验不声称逐文件全验。

GPU准入持UUID合作锁后≥5s双快照，任何既有/未知compute拒绝；占用0%不是空闲。每worker单UUID三轮原生随机分配、释放7相位，两新鲜NVML样本对应唯一opaque row；这不是host/container PID翻译。保留实际pidfd/start/session/nonce；流程是summary SHA→ready→父ACK→drain→归还双快照。只清理本次已证自有的session，不杀NVML宿主PID。

实际最新代码bundle v3 f48dea5ac4921fc40961e74ecd51be11e8b5dd0388749a61ff7b0cd18aa5b2ea。B4 pilot合同2ccb30caedff5998cc7e456fccef886c00ab1c2ec471e4f0aedc8f028877fc31，B1 86642801920a7b462a07304e396067adcbfabbdbff140755e9ab319cbf3ac717；这些不是银标正式合同。当前16:52左右的前台v3 sequence实际执行300b4/300b1/1000中断恢复/MUSAN12/10h4382；最终结果要重读，不预写通过。1000测试要真的留下missing，再证明恢复后新增了结果、旧SHA/mtime不变；P3新输出处理满10h，不用P1/P2缓存冒充吞吐。

## 结果解释与八卡

CPU最新56+5+2测试与5源真解码垂直链、实际2→8进程257fixture唯一调度都实过，但这些不算模型正确性。v1 reg18真实18/18全读回，10candidate8unknown0failed；只有一个worker处理源、另一个空着，不称双卡吞吐。v2全库300实读300/300、连续覆盖全、67candidate233unknown0failed；原向量zero314+negative41，与库fixed/items非正长554分开，另有147修补窗、22源越界。5个VAD量化异常不是FLAC新解码错误。不要放宽规则去提高candidate；候选率不是精度，人审仍null、自动训练放行0。

开发16源独立CPU轨固定源/窗口/PCM；FireRed盲ASR与MMS给定文本定位是两条不同的轨。MMS的文字不能当独立ASR投票，模型一致/波形图也不能当真人已经听过。原生与修补时间分别绘图，重复表面用保序occurrence匹配；差异不是误差真值。权重加载失败原样保留，CPU torch2.1的mmap需要str路径，旧失败不覆盖。

八卡只在完整真实的W2、全量索引、异常计账/独立审阅都过了之后，才提交唯一的P6任务。平台上是单worker实例8A800，一父带8个独立推理；不能清平台rank变量绕过只一父guard，模型/代码/decoder/持久挂载都要在云端实际验证。/root用户PFS RW、/shared/group-shared RO；720h上限不是预测工期，实际要按P3分领域加权估，2→8线性只能算情景。正式轻核map释放GPU，完整逐文件CPU审计另记。当时（10-24）在跑的PTN t-20261024154424-vggvm不可停止/改动，后台PAUSED不自行恢复；该任务后续已失败（10-25 终态 Failed），PTN 现由 ctjsk 跑完 120/120（2026-11-03），这句约束不再适用于今天的操作。


## Ultra后端后继（2026-10-25）

本页保留官方FA/v5历史。用户已授权按实测结论接入Ultra主体；后继代码/合同/4000代表批和恢复证据见[Ultra生产接入](ultra-production-sidecar-20261025.md)。v1派生heldout轴的问题保留原件，v2独立修复；4000是否完成、真实恢复或八卡提交，要读后继的当前状态和实际凭证，不能拿本页旧的v5通过记录替代。


2026-10-25用户后续明确纠正：本轮仅三个实际训练数据集VoxCeleb2/VoxBlink2/CN-Celeb2，不包含MUSAN、Child或额外评测域；旧混合索引仅历史。[最新三域边界](ultra-production-sidecar-20261025.md)优先，新全量索引与真实3000/512凭证分别验收。
