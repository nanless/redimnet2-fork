# vLLM最小离线适配与ASR阶段恢复

维护入口为`vllm_backend.py`和`asr_stage_store.py`。本轮只使用本机CPU假引擎验证软件行为，没有安装或导入真实推理库、加载模型、启动GPU或访问训练服务器；不能据此宣布vLLM权重兼容、实际速度、GPU资源归还或文字/字界质量通过。

## 输入、实际请求和原生结束信息

适配描述使用`annotation-vllm-asr-spec/v1`，显式提供模型目录、引擎版本、精度、单卡TP=1、整卡显存预算、模型上下文上限、温度0、输出token上限、seed及有限批量。真实执行时还必须固定完整权重/处理器文件摘要，由真实工厂在构造前核覆盖范围和实际字节；默认CPU结构校验不读取模型，不能证明这些文件齐全或摘要相符。不会继承示例4096输出上限或显存默认值；不识别的引擎设置会拒绝，避免配置写了却被静默忽略。

每个输入包含`source_id`、`window_id`、`input_pcm_sha256`、`model_input_frames`和16000Hz采样率。实际新计算还需有限、单声道float32的`audio`数组，其字节摘要必须与声明相同。稳定外部request ID由输入身份和producer摘要生成，原生vLLM request ID单独保存。

真实桥接采用公开`enqueue()`先取得每个输入的原生ID，再用`wait_for_completion()`取得完整输出。按原生ID映射，允许返回乱序，但缺失、重复或陌生ID会拒绝并保存失败证据。不根据数字ID猜来源，也不依赖多个相同音频提示文字区分请求。安装版本缺少该公开API时直接拒绝，不偷偷升级或改为不可靠的位置匹配。

原生`RequestOutput`保留prompt token、completion token、原始解码回答、finished、finish_reason及stop_reason。`eos_observed`只表示实际token中观察到已声明EOS；当vLLM不把EOS放进输出token而明确以该EOS停止时，`termination_verified`可为真，`eos_observed`仍为假。只有`finish_reason=stop`不够，未知结束信息保留unknown；达到输出上限记truncated，abort/error记failed。非法语言、空转写、解析失败不从分母删除。

输出使用`annotation-asr-native-result/v1`，包含兼容字段`status/raw_text/normalized_text/language/normalization_audit/native`及完整请求、producer和实际批次身份，始终保留`usable_for_training/evaluation=false`。这只是转写阶段结果，没有字界或人工准确性结论。

## 整批持久化后再逐来源消费

新`ASRStageStore`只接受自己的`annotation-asr-stage-store/v1`身份，不读取旧StageCache、不删除旧合同字段，也不改旧producer。每个明确工作分片使用独立store，同一store串行调用，模型内可以批处理。

SQLite使用DELETE日志与FULL同步。一次事务预约完整cohort，一次事务保存该批所有原生结果及摘要，然后才返回逐request结果。没有先提交前半批、再重新推理后半批的步骤。若在整批已持久化后、逐来源消费前中断，恢复会读取整批已有结果，实际新模型调用为0；已完成请求的原始批次身份不因后续新请求而改变。

尚未成功提交的cohort可以有界重做，但需要原批完整输入。只提供其中一行或在交叉批次中偷偷更换组合会拒绝；原生调用异常、漏输出等完整证据写入新store的`attempts/`，不删除失败再伪称首次成功。整批已持久化后发生的消费异常会明确记`stage_committed_before_error=true`。

全缓存恢复可只提供已声明的PCM身份元数据，不重新读取原音，报告`provided_PCM_verified_rows=0`。若调用者提供了数组，即使缓存命中也核实际PCM摘要，不能拿旧hash配新的音频。完整结果复用、实际新模型行数和新执行批次数分别计账，不用缓存命中宣布新引擎提速。

这里验证的是本机SQLite事务和本机文件锁，远端PFS的锁、同步及故障语义未实测。单store锁并非跨八卡的生产调度器，也不授旧writer交接或八卡全量准入；这些仍由联合覆盖及资源控制器负责。

## CPU构造与真实工厂

CPU入口为`VLLMBackend.from_fake_engine(spec, engine, store, parse_answer=..., prompt_builder=...)`，要求`purpose=cpu_fixture`并强制synthetic标记。`run_batch(rows)`或别名`transcribe_requests(rows)`返回结果、阶段缓存命中、实际新计算数量和时间，但始终不授GPU或正式标签资格。`tests/test_vllm_backend.py`提供完整可执行的合成示例。

真实低层工厂为`build_real_engine(spec, store_root, execute=True)`。缺少显式execute、pilot用途、离线环境、固定依赖版本、实际权重及处理器运行文件摘要、关键实际导入源码摘要时拒绝；README和下载缓存不作为运行输入。它在构造前核公开ID映射API，延迟构造官方`Qwen3ASRModel.LLM`，继续使用官方提示模板和转写解析，核实际tokenizer EOS和支持语言。模型文件按块哈希，避免一次读入多GB权重。该函数只构造引擎，没有推理或资源准入承诺；本轮没有调用。普通注入构造不会记录为已运行真实工厂。

CLI缺省仅校验描述文件：

```bash
python3 -B tools/open_annotation/vllm_backend.py \
  --spec /absolute/new/spec.json --spec-sha '<实际SHA256>'
```

`--execute --store-root ...`才会触发真实构造，未来只能由已经核实的资源组控制器使用，当前“GPU先不测”的要求下禁止调用。旧运行源码、原清单和PTN均不涉及本模块。

官方API依据：[Qwen vLLM入口](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py)、[vLLM enqueue及wait_for_completion](https://docs.vllm.ai/en/stable/api/vllm/index.html)。接口可用性和依赖身份以未来实际固定环境为准，不用网页示例冒充本机兼容证据。
