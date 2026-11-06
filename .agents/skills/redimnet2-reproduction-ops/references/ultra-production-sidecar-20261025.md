# Ultra主体新版sidecar与真实代表批（2026-10-25）

本页记录10月24—25日Ultra接入和三训练域范围纠正。下文Queue、index待完成和首轮待调度均为当时快照，后文随后完成记录不覆盖旧证据。后续跨节点恢复及10月28正式B8先读[转写提速与接续](asr-speed-and-continuation.md)，最新带日期任务表读[任务状态](task-status-and-readonly-progress.md)；[10月25巡检](comprehensive-audit-20261025.md)继续作历史。

## 当前用户范围纠正：仅三个实际训练数据集（优先于下文历史混合批）

用户明确拒绝将MUSAN、Child和额外评测域纳入本轮验收；根已承认扩大范围并停止自有混合测试，历史原件保留。**当前仅VoxCeleb2、VoxBlink2、CN-Celeb2中实际train_ptn/train_lm来源**，后续八卡也只标这三个训练域；不得沿用旧混合全量35fca3或11740953作为当前提交输入。旧研究/原配方增强资产不是本轮扩充标注范围的授权。

新代表3000各域1000，主index13c47a9537307f9cbc1276efe311c9283354b0d897da3d6f6956f20df6bd34dc；原音21330.103375秒。新vertical30 index9495db78aebe9d05b6666dce58faa38e8cbdc1897e80af295b5ac70c5034a356，实际全读通过；新512 indexf809112a4e7936f8442a9a7a6905109fdab56f8efbd909aac4998ab0ac8ef638，仅1676原音秒。源包在远端data-three-training-v1，实际run在validation-three-training-v1；生产runtime仍冻结v2、不为改变清单重改模型/原合同。

新3000是baseline：365已完成三训练域结果经source原对象/cSHA/commit/jobkey/新鲜encodedSHA与stable stat核后按原bytes复用，未复制其它域、stagecache或旧queue，不冒称新模型执行；其余2635实际GPU补算。完整复用证据run/verified-three-training-reuse.json；保原execution producer，不能把该阶段叫全冷跑。后续512仍要真实新GPU执行/TERM/恢复/原生对比；ASR-only种子必须来自baseline中确实存在且等于record.asr的真实cache包，复用sidecar无cache者不假装可用种子。

全量三训练域独立索引在catalog-three-training-only-v1下CPU过滤/逐片计账；原始PTN/LM同源合并、实际最终计数和SHA等待新index完成后读回。云工具仅用三域新版gate/spec，排除旧RIR/Child/MUSAN门槛；用户既有全量提交授权保留，仍须新真实验收及P6单机八卡实际回读。PAUSED后台不自行恢复。本文以下4000/8域/旧全量数字全部是已停止历史，不作当前验收条件。


## 三训练域八卡已提交：t-20261025054739-4kzhp

北京时间2026-10-25 05:47:39创建唯一三train标注任务t-20261025054739-4kzhp，名称redimnet2-ultra-3train-sidecar-8a800-p6-20261025-v1。保存后详情/YAML核实际P6、PyTorchDDP、1×ml.pni2l.28xlarge（8A800）、720h/2592000秒、重试Never/off、环境变量0、镜像dev_zhounan:4.0，aiplatform-SPEECH/AZB。/root用户PFS RW与/shared/group-shared组PFS RO两非嵌套挂载。空白表单P6点击仍4，改复制已核P6旧PTN表单并完整替换入口/名称/描述，保存后再核6；旧PTN未改。

唯一launcher cloud-three-train-v2/frozen-launch-v2/launch-ultra-w8.sh SHA24570b4a221c85197eaaba05f90f77e1562cc5a1e01345ae14447b341b089271，spec SHA19e2880a5070fa267a72196190e8c132d4c6f5d87bad68694e27f2df603182c1，C最终收据97315379814e969809f79216bf763c7e40e7709dabbae09cab4b8d88d4570840，19实际远端证据引用同SHA。正式RUN formal-three-training-ultra-v1与语义claim唯一，勿再fresh提交/执行，先查本ID。实际完整入口和挂载收据在本机云端准备/实际提交_t-20261025054739-4kzhp.json。

最新UTC22:03:57平台Queue且正式RUN尚未创建；八worker/实际UUID/容器GPFS/首批sidecar仍待调度，不称已运行或全量完成。详情仅提示有配额、可能GPU碎片；平台AI“排队诊断”把本ML任务误判为Seedance视频生成，回答不能作根因证据，不采纳其停止/重提/权限建议。PTN仍epoch41/batch7525。既有PAUSED后台未恢复，不能从任务已创建推断后台已启用。两卡/C已通过仅第一阶段silver生成，训练和正式评测自动释放仍false。

## 三训练域实际3000与512已完成（UTC 2026-10-24晚）

仅本轮三个实际训练域：30真实验收、3000 baseline于18:46:31Z完成，3000/24片/874 candidate/2126 unknown/软件failed0，父1684.5647秒，资源双归还；365旧同源原bytes严格复用、2635补算。512实际恢复已exit0：120已发布后精确自有父TERM，392补齐，旧SHA/bytes/远端mtime与chunk marker不变；8真ASR-only包精确消费、缺失FA8窗真实补算。512与baseline同源原生观测完全一致；独立审计217源跨UUID、208新ASR/193新FA窗一致。不重复启动已完成sequence。现行监督器tools/validate_three_training_sequence_v1.py SHA8c5398af37388d0d26ac8bb66092725636a7227e021a527151f3a1641344478f，旧ab158监督器仅混合历史。

新全量catalog-three-training-only-v1/index.json SHA b9912a2fbfd6642c8e8f6384d29768f22aab563c06bf00907a54bb27242fac2d，11,521,178源：Vox2 1,092,009、VoxBlink2 9,904,382、CN2 524,787。45,975片全部20,026,899,415B独立逐源/SHA回读通过，catalog-three-training-readback-v1/summary.json SHA37fb322d25b89893b187a87d96139875239eea6fbed1c8a088ef2e38c66a869e。PTN/LM重合只计一次，不把LM重复加总。

随机48源96PNG已实际两代理查看并保逐源观察，原声SHA/原钟/波形频谱分层可回溯；这是机器视觉，人工听审0。另三域固定18源FireRed真实CPU盲ASR18成功/0失败/API0/GPU0；18与主模型连续全源输入身份相同，3表面一致，15差异含语种覆盖不匹配和同音/缩合/重复。不能将一致当真值、差异当Qwen必错。尤其0.836秒短CN2主预测Vietnamese/独立中文、Bye14/bae16，以及raw/FIX零长、源首尾活动未覆盖，都保为未决。candidate29.13%不是完整字准确率；silver可保全机器候选及unknown，训练/正式评测释放仍false。

当前云工具仅cloud-three-train-v2；最终C实际收据、原字节远端证据、live builder和唯一封存launcher/真实平台任务ID分别读取最新本机当前状态。没有ID不得称已提交。Mac会再次锁定；收到已解锁后也以当前工具实测为准，不绕锁，先完成全部服务器可做的准备再报告具体UI阻断。既有PAUSED后台不自行恢复。

本地入口 `/Users/zhounan/Documents/声纹降噪/ReDimNet2_Ultra全量Sidecar_20261025/README.md`、`当前状态.json`、`执行记录.md`。远端专属根 `/root/experiments/redimnet2/open-annotation-ultra-20261025`。继续前先读当前状态、最新真实parent_result、独立审阅与唯一平台ID；本引用不预写4000/恢复/八卡已经通过。

用户已授权按34源实测选型接入Ultra主体，在指定两卡机测几千真实条，检查文字、完整字边界、吞吐及实际断点恢复，通过后提交唯一P6八卡全量银标任务。无需重复确认意愿。PTN t-20261024154424-vggvm不能停止/改配方/迁移；本任务自己的新诊断进程确需修复时可核身份安全停止，不扩大为杀未知compute PID。既有后台PAUSED保持。

## 模型与生产语义

Ultra是第三方17slever17/Qwen3-ForcedAligner-Ultra，来源官方Qwen3-ForcedAligner-0.6B；不是ReDimNet2声纹骨干。模型revision765dc1d714e4c80196782a5df5bfbcb3e1416736、主体SHAe6fcf5be2c11b265f7217ba6102917a7fe64438506357be6999f6ad5a707cb5f继承前轮原件。当前no-head与前轮实际no-head一样，经官方Qwen3ForcedAligner API加载Ultra权重；只有head分支才引入作者LocalBoundaryAligner/mask/局部偏移/投影，不能把两分支混读。

保Qwen3-ASR-1.7B、原FireRed来源绑定提议、完整连续24s拥有区+两侧2s。单次模型call宽度1；source_batch仅CPU组织。原音全读/原钟/输入PCM、target encoding与occurrence、实际thinker输入张量SHA、FP32 slot logit SHA/shape/argmax、raw→全部FIX→返回items逐层保留。正常窗不落巨型logits全数组，软件异常保存完整已观测slot bytes/base64与trace；不猜时间token默认值。

正式批标API0。旧34原件和必要裁片的内部API许可只对应旧固定批，不自动扩大到新4000。机器candidate、几何正长、模型一致都不等完整发音或人工金标；quality_axes分文字忠实、字界、语言、重复、来源/split，自动训练/正式评测释放始终false。空ASR不当无语音，unsupported逐窗NA，不硬塞English；旧VAD量化异常保原文unknown，连续ASR不丢其外内容。

## 已用版本与真实缺陷

从最终v5副本继承原子sidecar/commit、journal、completion map、generation fencing及GPU资源证明，旧代码不改。

- production-code-v1 bundle b151139e121ac6b2ee9cdf3661446e8a4c1c2b5649fe811dcfd7d6db9b610b33；实际32通过，但派生heldout轴漏provider_evaluation/evaluation:*，原source和禁训保护正确。自有4000在152部分源后被精确停止，保完整原件，不能称完成。
- production-code-v2 bundle **54cd49b77ce3c82bbd61bc4ca3cc9389016e908e976ef2574fb953b3249554b0**；仅quality_axes.py变，按canonical split/角色判断heldout，不匹配任意子串。pilot **785d7a9621992bb4d0f5234b21aefd42ed96eda776516700e6026f1eec6474df**；silver **92042533743eef0b68b74fc187f3c0269618c5331297ad789835f61b6af36234**；文件名pilot/silver-ultra-no-head-fp32-b1-de9c0ad55e5e-20261025.json。
- v2实际32：两个worker各有数据，32/8片全读、14candidate18unknown软件failed0，heldout14正确；与v1同33窗全部原生模型观测一致，2旧VAD异常的trace仅代码版本绝对路径不同，原差异保留。首次严格全字段检查失败与后续按字段解释都保留，不改旧原件。

初次部署自行筛文件名前缀漏小批JSONL，源处理前失败。后继用完整包部署`data-complete-v2`，先逐77文件SHA及全部4index实际引用shard/计数/嵌套全验再上GPU。打包本身SHA通过不证明依赖闭包完整。

解释器可能无os.pidfd_open；停止自有测试须复用冻结gpu_common.open_pidfd已核Linux syscall兼容实现，并用admission、argv、start ticks、输出/cSHA/index身份核定，不临时用裸PID kill。首个无API失败发生在发信号前的事实与成功一次TERM另记。停止后新鲜双空闲和PTN持续日志分别核验。

## 4000、恢复与云端

代表4000 index **8691d8bffaeeb304e23eb487b73ce6f72071178979d05dddebfb91bb0c801019**，8数据域，原34全部排除。68165.8345625原音秒，不是要求运行18.9h/10h。3812历史钟+188本轮fresh MUSAN头，缺失语言保unknown；抽样分层偏置不代表全库比例。Child600有提供方文本参考可算CER，仍非人工时间gold。

compact32 index86345306f7b983e225d93360614ea1458d2f2bdcd4d80bbb39a66d8e3da7c1f2，146.04原音秒。恢复512 indexc8fb8b95b5c7fbe5803adc6af75a68f8fd975fd6be1dcea20f8fb8846fcc4788严格为4000子集。

新v2监督器tools/validate_ultra_sequence_v2.py SHAab15819d69387bcc3b05cbd6a525a9dd913aec3060d75f211f70802d646066ef；先接受真实已完整cold4000的精确bundle/cSHA/index/全读/归还凭证，再8个真实ASR-only包种子、精确自有父TERM、missing FA真实补算、512全读和旧SHA/bytes/mtime不变，和cold4000中的同512源原生比较。13项CPU夹具不是GPU恢复通过。producer-row归属变化可被严格比较报差，须保同PCM/cache输入/原producer区别，不自动等同模型变差或放宽检查。

云工具在本地`云端准备/`，只在真实新W2与C具体审阅SHA齐后建admission spec，不预造通过票据。时间下界用真实UTC；北京时间10月25凌晨对应UTC10月24，不照目录日期填未来时间。正式全量index仍旧冻结 **35fca3e58b745752cede268aa10205e55a64500cde3314f8d519e55c40c161b9**，11740953源/45983片，RIR60000另CPU身份NA。P6/单实例8A800/8独立worker/唯一父，不能清RANK绕parent guard；实际挂载/root RW、group共享RO、源码/模型和全部worker必须云端回读。未知提交结果先查重，不重复任务、不碰PTN。

外置图审输出在生产标签树之外，固定选单/source/sidecar SHA、原增益原钟、全声道波形/ch0频谱、raw/FIX/owned分层，复制原音再核SHA。已有中文字体需显式配置，字体/工具/选单身份留证；实际PNG查看、模型听辨、浏览器可播放、人类听审分别记账。图只能提示削音/邻字/尾残余，不能当人工完整字金标。
