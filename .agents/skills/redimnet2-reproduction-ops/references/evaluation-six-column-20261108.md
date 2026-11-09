# 六列官方协议评测（2026-11-08）

这页记录 PTN-115、PTN-120、LM-1 与官方权重在同一套官方协议下的实测，以及三个容易把结果读错的点：协议 sha、截断清单、CN-Celeb 与 SITW。

## 六列与附加一列

六列指 Vox1-O、Vox1-E、Vox1-H、Vox1-B、VOICES、CN-Celeb。前四列是 VoxCeleb1 的官方清洗协议，后两列是域外。另有一列 `Vox1-O-8s` 只读前 8 秒、短音频 tail-wrap，是训练期验证口径，**不能**和整段 O 的 0.26% 对减。

协议文件与实测规模（`/root/codex_artifacts/eval-official-20261108/provenance.json`，历史路径；新评测按仓库 `AGENTS.md` 的硬规则写 `/root/group-shared/voiceprint/artifacts/redimnet2/eval-official-<日期>/`）：

| 列 | 协议 | 对数 | 不同音频 |
|---|---|---|---|
| Vox1-O | `voxceleb1/protocols/veri_test2.txt`，sha256 `0bc0a0fe…36ea` | 37,611 | 4,708 |
| Vox1-E | `list_test_all2.txt`，`e4bbe7a6…4aa3` | 579,818 | 145,160 |
| Vox1-H | `list_test_hard2.txt`，`006b4882…55f4` | 550,894 | 137,924 |
| Vox1-B | `list_test_bilingual.txt`，`c744d327…5fcc` | 808,574 | 42,561 |
| VOICES | `VOiCES_challenge_2019_eval.SID.trial-keys.lst`，`407a7974…be0a` | 3,607,516 | 11,392 |
| CN-Celeb | `cnceleb/CN-Celeb_flac/eval/lists/trials.lst`，`3e0df69a…2338` | 3,484,292 | 17,973 |

## protocol_sha 是必填项

评测身份里写 `protocol_sha256`，不是可选项。`duration_ft/evaluate.py` 的 `strict_standard` 分支要求它等于 `STANDARD_O_SHA256 = 0bc0a0fe…36ea`，不等就 `EvaluationRefused`。目的是防止“用了一份改过的或截断的试验表，却把结果标成官方 O”。

本次六列评测把每份协议的实际路径、sha256、对数、不同音频数都写进 `provenance.json`，`cells.csv` 每格再带 `trials_file / trials_sha256 / trials_count / unique_utts`。写报告时连同这些字段一起引，别只写一个 EER。

## 截断清单会造出假差

11-07 的等价性闸门为了压时间，把 E 截成前 5,000 对（`head -5000 list_test_all2.txt`），跑出 0.28%，再拿官方 README 的 0.41 去减，得到 `delta-E=-0.13`。这个 -0.13 不是模型差，是切片差：样本换了，难度分布也换了。

正式跑用全表后，官方权重在 E 上是 0.409，与 README 的 0.41 只差 0.001。教训固定成一句：**不同清单的对数不能对减**；要报差，先确认两份结果的 `trials_sha256` 相同。

## CN-Celeb 用注册向量平均

CN-Celeb 的注册段有多条语音，评测用 `--cnceleb-mode avg`，即多段注册的 embedding 取平均后再算余弦。这个口径和官方结果表一致；换成分段各自打分再平均，数值会变，不能和 3.59 比。

## SITW 数据缺失

SITW 在 `cells.csv` 里标的是 `DATA_MISSING`（`n/a (dataset absent)`），不是 0，也不是“跑失败”。本机没有这份数据，六列里缺这一列。汇总时保留缺失标记；用别的集合顶替，或把缺失写成 0 参与平均，都会让对比失真。

## 实测结果

2026-11-07 18:58—22:52 UTC，两张卡按 12 个分片跑完（`ledger.csv` 每行有 start/end/seconds/exit_code）。单位 %，越低越好；括号内是与官方 README 的差（百分点）。

| 列 | 官方 README | 官方权重实测 | PTN-115 | PTN-120 | LM-1 |
|---|---:|---:|---:|---:|---:|
| Vox1-O | 0.26 | 0.255 | 0.420 | 0.399 | **0.287** (+0.027) |
| Vox1-E | 0.41 | 0.409 | 0.530 | 0.529 | **0.391** (−0.019) |
| Vox1-H | 0.79 | 0.790 | 0.983 | 0.974 | **0.735** (−0.055) |
| Vox1-B | 1.15 | 1.153 | 1.405 | 1.510 | 1.235 (+0.085) |
| VOICES | 3.03 | 3.029 | 4.731 | 4.755 | 3.147 (+0.117) |
| CN-Celeb | 3.59 | 3.605 | 3.717 | 3.768 | **3.378** (−0.212) |
| SITW | 0.87 | 缺数据 | — | — | — |
| Vox1-O-8s | — | 0.239 | 0.404 | 0.399 | 0.293 |

读法：

- 官方权重四舍五入后与 README 基本重合（O 0.255 对 0.26、E 0.409 对 0.41、H 0.790 对 0.79、CN 3.605 对 3.59），说明这套入口和协议对齐得住。
- LM-1 在 E/H/CN-Celeb 三列低于官方，在 O/B/VOICES 三列高于官方；PTN-120 在六列上都明显差于官方。单点差 0.02—0.21 个百分点不构成“复现成功”或“复现失败”的判决，尤其是 O 只有 37,611 对。
- 8 秒列和整段列的方向并不一致：官方权重 0.239 好于整段 0.255，LM 0.293 略差于 0.287，PTN-120 0.399 持平。截短不一定更差，取决于模型和协议；两列不能互相替代，也不能拿 8 秒列去对官方 0.26。

## 汇总文件会过期

`summary.csv`/`cells.csv` 是 `build_summary2.py` 的生成物。11-08 那份 `summary.csv` 生成于 18:45，而分片从 18:58 才开始跑，所以它整张表停在 `NOT_RUN`。看结果以 `ledger.csv` + `csv/<tag>__<part>.csv` 为准；要更新汇总就重跑 `build_summary2.py`，不要手改生成物。

## 复现入口

```bash
# 新评测按仓库 AGENTS.md 的硬规则建 /root/group-shared/voiceprint/artifacts/redimnet2/eval-official-<日期>/，
# 把 run_eval.sh、build_summary2.py 与该轮分片脚本复制过去后，在这个新目录里运行。
cd /root/group-shared/voiceprint/artifacts/redimnet2/eval-official-<日期>
bash run_eval.sh          # 两卡、按 part 串行，写 ledger.csv 与 csv/
/root/miniforge3/envs/redimnet2/bin/python build_summary2.py
```

2026-11-08 那轮的原件在 `/root/codex_artifacts/eval-official-20261108/`（历史路径），只读复核旧结果时仍去那里。

导出与评测的格式语义见仓库 `docs/13-训练权重导出与评测.md`；本轮四个权重的 sha256 在 `provenance.json` 的 `weights` 里，引用时连 sha 一起写。
