# 有限原音、字界裁片和内部API复核

这些工具只做用户指定的小样本诊断，不建设数据库，不发现新样本，不遍历音频目录，不修改标签或队列，不调用GPU。真实音频、完整文字、样本清单和原始回答保存在仓库已忽略的`.audit-local/`，不要提交公开Git。API结果不是人工听审、字界金标或训练／评测资格。

## 先固定来源

`collect_sample_audio.py`读取显式JSON列表，最多24个唯一来源。每项含`review_id`（依次q01…q24）、已保存的`row.source_id`、`row.audio_path`和`row.audio_recorded_file_sha256`。通过已有SSH配置短时只读指定服务器，以低优先级逐个读取这些明确路径；打开前后核stat和实际编码SHA，总编码数据最多256MiB，不写服务器，不递归遍历。输出新绝对目录，保存SSH错误和只包含明确成员的tar，核本地摘要后展开，保留失败证据。

```sh
python -B tools/open_annotation/collect_sample_audio.py \
  --selection /ABS/selected-final-24.json \
  --output /ABS/NEW-audio-selected --host dev_a800_2gpus
```

显式守卫不是Python的assert，优化模式`-O`也不能移除。只有已证明来源用途及已有传输授权的样本才可发API；工具不会替操作者取得授权。

## 原窗或必要字词裁片

`prepare_sample_audio_review.py`先核编码、完整原采样率多通道float32 PCM、采样率、声道、帧数与侧车SHA，再复用原标注context样本范围。长源仍使用原context，不拼接重叠文本；输出16kHz单声道float32 WAV，其PCM SHA必须与原模型输入一致。原声不做VAD删除、增益、补零或字界修补。libsndfile写FLOAT WAV会带PEAK时间戳，重新写出的WAV编码SHA可能不同；不能因此声称编码逐字节一致，也不能用PCM相同取代已发送文件的真实SHA。分别保存每次编码和PCM摘要。

```sh
python -B tools/open_annotation/prepare_sample_audio_review.py \
  --selection /ABS/selected-final-24.json --audio-root /ABS/audio-selected \
  --output /ABS/NEW-api-views
```

必要裁片通过另一个明确小列表选择`review_id`、`edge`（first／last）和`count`（1—6）。选的是已保存units的首／尾若干项，连续保留从第一项起点到最后一项终点之间**全部原声**。若中间零长词未进入units，它的声音仍可能在裁片里；工具保存对应窗口的完整native/FIX/items及所选native_index，不能把“2个保留unit”称作“只含2个完整字”。

```sh
python -B tools/open_annotation/prepare_sample_audio_review.py \
  --selection /ABS/selected-final-24.json --audio-root /ABS/audio-selected \
  --crop-requests /ABS/crop-requests.json --output /ABS/NEW-api-crops
```

裁片清单保存来源与请求表同份已解析字节的SHA、原编码和PCM SHA、源采样范围、侧车SHA、原source decision、实际输入WAV/PCM SHA和未验收标记。裁片只供诊断，`complete_word_boundary_verified`、训练和评测可用性均为false。输入表在执行期间被替换不能改变已解析输入所声明的摘要。

## 先盲审，再核具体冲突

`sample_audio_review.py`固定内部逻辑目的地、请求模型及既有source-sn，默认只预检，不读取凭据、不发请求。生产清单要求显式`source_ids`为1—24个唯一来源，全部case.source_id与它精确相等；同源必要裁片可有多个case，视图数与原件数分别汇报。

```sh
python -B tools/open_annotation/sample_audio_review.py \
  --manifest /ABS/manifest.json --manifest-sha SHA256 \
  --output /ABS/NEW-api-blind --profile blind --workers 2
```

真实请求在已授权同一批原件、裁片和内部目的地的前提下加`--execute`。凭据只从已配置的0600私有文件运行时读取，不保存或打印Authorization。不要将密钥写入参数、日志或仓库。重定向和环境代理禁用；原始SSE保留但凭据回显会脱敏并触发异常，不当音频结论。

第一个正常原窗是独立canary，结构及协议通过后才滚动派发，最多两个请求并发；canary不重复请求。确定性网关、结构或协议错误停止后续派发，保留已在途请求及全部异常。仅有限暂时性传输错误允许最多两次尝试，不用失败回答判定音频质量。

SSE和HTTP错误体有实际读块deadline、体积和行长限制，读取前后及DONE/EOF均检查。连接及HTTP头仍使用urllib的socket超时，这不是整个网络栈的硬墙钟保证。不会把“没有换行但持续滴字节”误称永远受readline总时限约束。合成HTTP测试覆盖迟到DONE、滴流、错误体和凭据转义／跨帧回显。

`--profile reference`只用于盲审后仍有具体冲突的少量输入，必须另建输出目录。参考是可错数据，模型不能据此覆盖自己的听音；保留盲审与参考审阅冲突，不投票伪造真值。returned model如为受支持别名也保存实际值。

## 如何读结果

- `usable_model_observation`只表示协议、结构及既有内容检查未触发；**不证明heard_text真实或边界正确**。
- API记录的input／owned范围是本次WAV局部0…frames，原context、owned及原源采样范围看SHA绑定的输入清单。不能把局部owned直接当原标注owned。
- 人工context切口与真实原件起止分开。句意没说完、音乐突然停、没有淡出，都不单独证明语音音素削首削尾。
- API时间只是粗略描述，不用于纠正毫秒字界。盲审与参考可能改变首字、词形或人名；冲突保留待人工复核。
- 几何合法、无告警候选也可能错词；unknown也可能文字正确而因语言覆盖、FIX、旧VAD或时钟误差被拒。

绘图入口和输出解释另见[已保存原音与字界图审](已保存原音与字界图审.md)。本次具体实测见[10月28日有限抽样报告](../../docs/reviews/2026-10-28-待审原因有限抽样与PTN实测.md)。
