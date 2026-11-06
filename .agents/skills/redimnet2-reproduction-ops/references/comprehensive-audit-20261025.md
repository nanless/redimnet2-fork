# 2026-10-25 全面巡检、跨段LM CPU准备与完整字消费

最新本机入口：`/Users/zhounan/Documents/声纹降噪/ReDimNet2_全面巡检修复_20261025/全面巡检与修复报告.md`、`本聊天继续执行的范围与要求.md`、`当前状态.json`。每个状态读其证据时刻，不把12点快照冒充之后实时状态。原始日志与报告留本机；不干预现PTN、不改冻结摘要。三训练域范围以Ultra20261025引用为准。

PTN t-20261024154424-vggvm于北京12:01—12:05共享实读43/12225→12425，8rank推进；model36—42与验证均产生。最新42 CPU mmap检查model/epoch/optimizer/scaler及浮点有限性通过；最新SHA e7db3cd511effdb6d745d3af1a6971940ba4b39d360117890b63acc140a3e97d。新段最佳41/1.324%，整谱系最佳原27/1.159%，42/1.547%，协议为Vox1 quick8秒，不与官方LM整段混比。固定源189/189、原95/95、bundle16+manifest、本轮第43轮192数据worker/1536SHA引用逐项复验通过；36—42轮仅本次读门禁原件，不冒称各轮192份全复验。新段FLAC/Traceback/OOM/NCCL等0；原7历史事件保留。活跃共享主仓库现在88/95一致、7差异，新增4为模型路径迁移；旧92/3保为历史。四精确文件兼容链接4/4摘要匹配。此为共享证据，实际云八卡和节点必须现场另读。

跨段LM CPU准备已在独立目录完成v4，SHA5d7fde3e404cbd304204b354c9960253cf5317d9bc5cfbdff195ecc384372283，入口PTN_LM/跨段LM准备_v1/最终交付说明.md。固定原1—35与普通接续36—120、model35 SHA、speaker-map/source、CSV、后继ptn_success.json和新run115/120；旧v4不改。C实际发现v3裸异常漏检，v4复用固定旧_FATAL/_OTHER_EXCEPTION后四反例拒绝；实现及独审分别43测试全过，B另6有限接口探针。真实43快照exit2，115/120/终态缺失属于准确拒绝。当前43拒绝分支及受控fixture未import torch；未来证据齐备的默认real_checkpoint会CPU import torch/mmap并运行旧source/decoder子检查。ready_for_lm_execution永远false，不取训练锁/执行LM，当前未部署。完整历史及未修反例保留；最小包b45b0d9c0399a647e6aa9930df66aa7784225ff3622cca6a061f53a09a8c0c17。CPU验收不证明未来现场ready；两段终态/writer/锁内实验、私有decoder、八rank冷optimizer/scaler与LM1后验仍须真实完成，旧7事件不扩许可。

八卡标注唯一t-20261025054739-4kzhp已P6/1x8A800/720h提交，最近根解锁后平台仍Queue无实例，开发机UTC04:10:45无正式RUN/claim/lock。原3000和512真实验收已完成，不重复跑；当前剩余是实际调度/八worker/首批及持续输出/全库审计。未发现确定入口错误；排队不靠fresh重提/删claim/清RANK修，原AI排队助手把ML误读视频任务的回答不采。SSH超时和Mac锁屏是观测通道问题，不能推断云训练挂掉。旧PAUSED后台不恢复。

完整字自然变长仍未有真实GPU训练结果，4step silver工程配置不当正式实验；生产sidecar usable_for_training/evaluation=false。新Ultra完整字消费适配在本机独立目录、默认新proposal schema，未知/NA/修复或原生几何不合格拒绝。只有额外真实media encoded/nativePCM/clock绑定才能出engineering_candidate_pending，正式loader/train门必须拒绝，不翻训练资格。最终具体测试和独审以该目录交付收据为准，不从工具存在推断验收。继续需要人工完整字听审、足量不同合法family、B6双卡前反向/BN/完整保存/恢复/吞吐，随后唯一八卡独立训练；不复制4family凑8rank，不把机器视觉或ASR一致当人工gold。


完整字消费修复的当前软件版本是ultra_word_adapter_v3.py，SHA780745cabfaa0f06d32f3a96b95e31529f0992a409d128a87a7bf4c5f410c922，在本机八卡标注/Ultra完整字消费适配_v1。有限差异代码审阅未见新确定问题，23项CPU由实现与独立A分别实际重跑通过；原18源sidecar/commit重放18/0完整性错误、3family/12待审模板、15诊断。不是新GPU/听审；真实media绑定0、真实旧格式dataset0。默认schema及正式loader/train split拒绝保持，synthetic transport仅工程接口。最终详单读该目录README、当前实现与验收状态及独审收据。

v3修复显式导出refusal/零条仍返回成功，真实CLI缺媒体/unknown零导出/默认inspect实际2/2/0。另绑定固定外部diagnostic_protection_reference.json SHA6a4b44c3c4a2b8c3ee5d71239d686690ca905c4c6da40b935c037b7186c4f24e，34诊断中12三训练域原件虽source内标志未改，按精确ID/记录原路径或alias拒proposal与工程导出，缺/变保护引用拒绝。不改原source行、生产冻结合同或全库index。当前18无诊断保护命中，未发现实际泄漏。旧v2错误返回码复现、各版本与反例保留。无新GPU/API/全库扫描/训练/部署，候选不释放正式训练评测。
