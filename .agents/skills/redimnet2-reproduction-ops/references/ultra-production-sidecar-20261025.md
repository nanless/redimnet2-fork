# Ultra主体新版sidecar与真实代表批（2026-10-25）

本页记录10月24—25日的Ultra接入和三训练域范围纠正。下文里的Queue、index待完成和首轮待调度都是当时的快照；后面“随后完成”的记录不覆盖旧证据。后续的跨节点恢复及10月28的正式B8先读[转写提速与接续](asr-speed-and-continuation.md)，最新带日期的任务表读[任务状态](task-status-and-readonly-progress.md)；[10月25巡检](comprehensive-audit-20261025.md)继续作历史。

## 当前用户范围纠正：仅三个实际训练数据集（优先于下文历史混合批）

用户明确拒绝把MUSAN、Child和额外评测域纳入本轮验收；根已经承认扩大了范围，并停止了自有的混合测试，历史原件保留。当前只标VoxCeleb2、VoxBlink2、CN-Celeb2中实际train_ptn/train_lm的来源，后续八卡也只标这三个训练域；旧混合全量的35fca3或11740953不能当当前提交输入。旧研究里的原配方增强资产，也不构成本轮扩充标注范围的授权。

新代表批3000，各域1000，主index是13c47a9537307f9cbc1276efe311c9283354b0d897da3d6f6956f20df6bd34dc；原音21330.103375秒。新vertical30的index是9495db78aebe9d05b6666dce58faa38e8cbdc1897e80af295b5ac70c5034a356，实际全读通过；新512的index是f809112a4e7936f8442a9a7a6905109fdab56f8efbd909aac4998ab0ac8ef638，只有1676原音秒。源包在远端的data-three-training-v1，实际run在validation-three-training-v1；生产runtime仍冻结在v2，不为了改清单去动模型或原合同。

新3000是baseline：365个已完成的三训练域结果，经source原对象/cSHA/commit/jobkey/新鲜encodedSHA与stable stat核对后按原bytes复用，没有复制其它域、stagecache或旧queue，也不冒称是新模型执行；其余2635个由真实GPU补算。完整复用证据在run/verified-three-training-reuse.json；原execution producer保留，这个阶段不能叫全冷跑。后续512仍要真实的新GPU执行/TERM/恢复/原生对比；ASR-only种子要取自baseline里确实存在、且等于record.asr的真实cache包，复用了sidecar但没有cache的，不能假装可用种子。

全量三训练域的独立索引在catalog-three-training-only-v1下做CPU过滤/逐片计账；原始PTN/LM同源合并后的实际最终计数和SHA，等新index完成后读回。云工具只用三域的新版gate/spec，排除旧RIR/Child/MUSAN门槛；用户既有全量提交授权保留，但仍要新的真实验收及P6单机八卡的实际回读。PAUSED后台不自行恢复。本文以下4000/8域/旧全量的数字全部是已停止的历史，不作当前验收条件。


## 三训练域八卡已提交：t-20261025054739-4kzhp

北京时间2026-10-25 05:47:39创建唯一的三train标注任务t-20261025054739-4kzhp，名称redimnet2-ultra-3train-sidecar-8a800-p6-20261025-v1。保存后核详情/YAML，实际是P6、PyTorchDDP、1×ml.pni2l.28xlarge（8A800）、720h/2592000秒、重试Never/off、环境变量0、镜像dev_zhounan:4.0，aiplatform-SPEECH/AZB。/root用户PFS RW与/shared/group-shared组PFS RO，两个非嵌套挂载。空白表单点P6仍是4，改成复制已核的P6旧PTN表单，并完整替换入口/名称/描述，保存后再核是6；旧PTN没有改。

唯一的launcher是cloud-three-train-v2/frozen-launch-v2/launch-ultra-w8.sh SHA24570b4a221c85197eaaba05f90f77e1562cc5a1e01345ae14447b341b089271，spec SHA19e2880a5070fa267a72196190e8c132d4c6f5d87bad68694e27f2df603182c1，C最终收据97315379814e969809f79216bf763c7e40e7709dabbae09cab4b8d88d4570840，19个实际远端证据引用同一个SHA。正式RUN formal-three-training-ultra-v1与语义claim唯一，不要再fresh提交/执行，先查这个ID。实际完整入口和挂载收据在本机云端准备/实际提交_t-20261025054739-4kzhp.json。

到UTC22:03:57，平台仍是Queue，正式RUN尚未创建；八worker/实际UUID/容器GPFS/首批sidecar仍待调度，不能说已经运行或全量完成。详情只提示有配额、可能有GPU碎片；平台AI的“排队诊断”把这个ML任务误判成Seedance视频生成，它的回答不能作根因证据，其停止/重提/权限建议不采纳。PTN仍是epoch41/batch7525。既有PAUSED后台没有恢复，不能从任务已创建推断后台已启用。两卡/C只通过了第一阶段的silver生成，训练和正式评测的自动释放仍是false。

## 三训练域实际3000与512已完成（UTC 2026-10-24晚）

只看本轮三个实际训练域：30的真实验收、3000 baseline在18:46:31Z完成，3000/24片/874 candidate/2126 unknown/软件failed0，父1684.5647秒，资源已双归还；365个旧同源按原bytes严格复用，2635个补算。512的实际恢复已经exit0：120发布之后精确TERM自有的父进程，392个补齐，旧SHA/bytes/远端mtime与chunk marker不变；8个真ASR-only包被精确消费，缺失的FA8窗真实补算。512与baseline的同源原生观测完全一致；独立审计217源跨UUID、208新ASR/193新FA窗一致。已完成的sequence不重复启动。现行监督器是tools/validate_three_training_sequence_v1.py SHA8c5398af37388d0d26ac8bb66092725636a7227e021a527151f3a1641344478f，旧的ab158监督器只对应混合历史。

新的全量索引是catalog-three-training-only-v1/index.json SHA b9912a2fbfd6642c8e8f6384d29768f22aab563c06bf00907a54bb27242fac2d，共11,521,178源：Vox2 1,092,009、VoxBlink2 9,904,382、CN2 524,787。45,975片共20,026,899,415B，独立逐源/SHA回读通过，汇总在catalog-three-training-readback-v1/summary.json SHA37fb322d25b89893b187a87d96139875239eea6fbed1c8a088ef2e38c66a869e。PTN/LM重合的只计一次，不把LM重复加总。

随机48源96PNG已由两个代理实际查看，并保留逐源观察；原声SHA/原钟/波形频谱分层都可回溯。这是机器视觉，人工听审是0。另三域固定18源的FireRed真实CPU盲ASR为18成功/0失败/API0/GPU0；18与主模型的连续全源输入身份相同，3个表面一致，15个差异包含语种覆盖不匹配和同音/缩合/重复。一致不能当真值，差异也不能当Qwen必错。尤其是0.836秒的短CN2主预测Vietnamese/独立中文、Bye14/bae16，以及raw/FIX零长、源首尾活动未覆盖，都保为未决。candidate29.13%不是完整字准确率；silver可以保全机器候选及unknown，训练和正式评测的释放仍是false。

当前云工具只有cloud-three-train-v2；最终的C实际收据、原字节远端证据、live builder和唯一封存launcher/真实平台任务ID，分别读最新的本机当前状态。没有ID就不能称已提交。Mac还会再锁；收到已解锁后也以当前工具的实测为准，不绕锁，先把所有服务器上能做的准备做完，再报告具体的UI阻断。既有PAUSED后台不自行恢复。

本地入口 `/Users/zhounan/Documents/声纹降噪/ReDimNet2_Ultra全量Sidecar_20261025/README.md`、`当前状态.json`、`执行记录.md`。远端专属根 `/root/experiments/redimnet2/open-annotation-ultra-20261025`。继续之前先读当前状态、最新的真实parent_result、独立审阅和唯一平台ID；本页不预写4000/恢复/八卡的通过结论。

用户已授权按34源实测的选型接入Ultra主体：在指定的两卡机上测几千条真实数据，检查文字、完整字边界、吞吐及实际断点恢复，通过后提交唯一的P6八卡全量银标任务。这个意愿不必重复确认。PTN t-20261024154424-vggvm不能停止/改配方/迁移；本任务自己的新诊断进程确需修复时，可以核过身份再安全停止，但不扩大成杀未知的compute PID。既有后台PAUSED保持。

## 模型与生产语义

Ultra是第三方的17slever17/Qwen3-ForcedAligner-Ultra，来源是官方Qwen3-ForcedAligner-0.6B；它不是ReDimNet2的声纹骨干。模型revision是765dc1d714e4c80196782a5df5bfbcb3e1416736，主体SHAe6fcf5be2c11b265f7217ba6102917a7fe64438506357be6999f6ad5a707cb5f继承前轮原件。当前no-head与前轮实际的no-head一样，经官方Qwen3ForcedAligner API加载Ultra权重；只有head分支才引入作者的LocalBoundaryAligner/mask/局部偏移/投影，两个分支不能混读。

保留Qwen3-ASR-1.7B、原FireRed来源绑定的提议、完整连续的24s拥有区加两侧2s。单次模型call宽度1；source_batch只在CPU上组织。原音全读/原钟/输入PCM、target encoding与occurrence、实际thinker输入张量SHA、FP32 slot logit SHA/shape/argmax、raw→全部FIX→返回items，逐层保留。正常窗不落巨型logits全数组；软件异常时保存完整的已观测slot bytes/base64与trace，不猜时间token的默认值。

正式批标API0。旧34原件和必要裁片的内部API许可只对应旧的固定批，不自动扩到新的4000。机器candidate、几何正长、模型一致，都不等于完整发音或人工金标；quality_axes分文字忠实、字界、语言、重复、来源/split，自动训练/正式评测的释放始终是false。空ASR不当成无语音，unsupported逐窗标NA，不硬塞English；旧VAD量化异常保留原文的unknown，连续ASR之外的内容不丢。

## 已用版本与真实缺陷

从最终v5副本继承原子sidecar/commit、journal、completion map、generation fencing及GPU资源证明；旧代码不改。

- production-code-v1 的bundle b151139e121ac6b2ee9cdf3661446e8a4c1c2b5649fe811dcfd7d6db9b610b33；实际32通过，但派生的heldout轴漏了provider_evaluation/evaluation:*，原source和禁训保护是对的。自有的4000在152部分源之后被精确停止，完整原件保留，不能称完成。
- production-code-v2 的bundle是**54cd49b77ce3c82bbd61bc4ca3cc9389016e908e976ef2574fb953b3249554b0**；只有quality_axes.py变了，按canonical split/角色判断heldout，不匹配任意子串。pilot是**785d7a9621992bb4d0f5234b21aefd42ed96eda776516700e6026f1eec6474df**；silver是**92042533743eef0b68b74fc187f3c0269618c5331297ad789835f61b6af36234**；文件名pilot/silver-ultra-no-head-fp32-b1-de9c0ad55e5e-20261025.json。
- v2的实际32：两个worker各有数据，32/8片全读、14candidate18unknown软件failed0，heldout14正确；与v1相同的33窗全部原生模型观测一致，2个旧VAD异常的trace只有代码版本绝对路径不同，原差异保留。首次严格全字段检查失败，与后续按字段的解释都保留，不改旧原件。

初次部署时自己按文件名前缀筛，漏掉了小批JSONL，在源处理之前就失败了。后继改用完整包部署`data-complete-v2`，先逐77文件核SHA、把全部4index实际引用的shard/计数/嵌套全验一遍，再上GPU。打包本身的SHA通过，不证明依赖闭包完整。

解释器可能没有os.pidfd_open；停止自有的测试要复用冻结的gpu_common.open_pidfd，这是已核过的Linux syscall兼容实现，并用admission、argv、start ticks、输出/cSHA/index身份核定，不临时用裸PID kill。首个无API失败发生在发信号之前，这个事实与成功的一次TERM分别记录。停止之后，新鲜双空闲和PTN持续日志分别核验。

## 4000、恢复与云端

代表批4000的index是**8691d8bffaeeb304e23eb487b73ce6f72071178979d05dddebfb91bb0c801019**，8个数据域，原34全部排除。68165.8345625原音秒，并不是要跑18.9h/10h。3812历史钟+188本轮fresh MUSAN头，缺失语言保unknown；抽样分层有偏，不代表全库比例。Child600有提供方的文本参考，可以算CER，但仍不是人工时间gold。

compact32的index是86345306f7b983e225d93360614ea1458d2f2bdcd4d80bbb39a66d8e3da7c1f2，146.04原音秒。恢复512的index是c8fb8b95b5c7fbe5803adc6af75a68f8fd975fd6be1dcea20f8fb8846fcc4788，严格是4000的子集。

新v2监督器是tools/validate_ultra_sequence_v2.py SHAab15819d69387bcc3b05cbd6a525a9dd913aec3060d75f211f70802d646066ef；它先接受真实已完整cold4000的精确bundle/cSHA/index/全读/归还凭证，再要8个真实ASR-only包种子、精确TERM自有父进程、missing FA真实补算、512全读和旧SHA/bytes/mtime不变，并与cold4000里同512源的原生结果比较。13项CPU夹具不是GPU恢复通过。producer-row归属变化可能被严格比较报成差异，这里要保同PCM/cache输入/原producer区别，不自动等同模型变差，也不放宽检查。

云工具在本地`云端准备/`，只在真实的新W2与C具体审阅SHA齐了之后才建admission spec，不预造通过票据。时间下界用真实UTC；北京时间10月25凌晨对应UTC10月24，不照目录日期填未来时间。正式全量index仍冻结在**35fca3e58b745752cede268aa10205e55a64500cde3314f8d519e55c40c161b9**，11740953源/45983片，RIR60000另做CPU身份NA。P6/单实例8A800/8独立worker/唯一父，不能清RANK绕parent guard；实际挂载/root RW、group共享RO、源码/模型和全部 worker 都要云端回读：挂载点和worker只有在云端读回才算数。提交结果未知时先查重，不重复任务，也不碰PTN。

外置图审的输出在生产标签树之外，固定选单/source/sidecar SHA、原增益原钟、全声道波形/ch0频谱、raw/FIX/owned分层，复制原音之后再核SHA。已有中文字体要显式配置；字体/工具/选单身份留证。实际PNG查看、模型听辨、浏览器可播放、人类听审分别记账。图只能提示削音/邻字/尾残余，不能当人工的完整字金标。
