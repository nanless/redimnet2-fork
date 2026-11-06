# LM 启动包（只准备，不提交）

三语料 B6 的大间隔微调（LM）只有一个合法起点：第三段 PTN 的第 115 轮权重
`<PTN_EXP_DIR>/models/model_115.pt`。这个目录把"启动前必须成立的事"写成了可执行文件，
本身**不提交任务、不触碰服务器**。

| 文件 | 作用 |
|---|---|
| `env.template` | 启动环境模板（数据根、PTN 输出、新 EXP_DIR、显式 model_115、解释器） |
| `run_lm.sh` | 唯一入口：7 项前置检查 → 真跑 LM 初始化门 → 冻结启动身份 → 交给既有云端脚本 |
| `submit.conf.yaml` | `volc ml_task submit -c` 的任务配置草案（镜像/队列/规格/挂载留 TODO） |

## 为什么是这层 wrapper，而不是别的入口

| 入口 | 能不能用 | 原因 |
|---|---|---|
| `run_b6_mix_reproduction.sh MODE=formal` | **不能** | 它的 formal 分支先 `run_stage ptn` 重训 120 轮，并要求同源码八卡 smoke 成功件；本段 PTN 已由冻结的 190 成员包跑完 |
| `train_b6_mix_cloud_8gpu_lm.sh` | 能用 | 只是 `STAGE=lm` 的包装；但它默认按 mtime 取最新权重（会拿到 120），且没有"目录必须为空"的检查 |
| **`run_lm.sh`** | **推荐** | 在既有脚本外面补齐这两件事和另外 5 项前置；不复制训练逻辑 |

`run_lm.sh` 只做三件新增的事，训练逻辑全部交给
`applications/speaker_rec/scripts/cloud/train_b6_mix_cloud_8gpu.sh`：

1. **拒绝覆盖**：`EXP_DIR` 已存在就退出 3（失败后换 `…-v2/lm`，不删不改名）；
   同时拒绝把 `EXP_DIR` 放进代码目录或 PTN 段目录。
2. **钉住起点**：文件名必须是 `model_115.pt`，且 sha256 必须是
   `65497b6a00efac412a82d563aebc1e767056fef23e7f4234f48f89cc8b3aad5c`；
   `PTN_EXP_DIR/spk2id.json` 必须存在。
3. **先过初始化门**：真跑 `check_b6_lm_init.py --expected-epoch 115`，把 JSON 落到
   `$EXP_DIR/lm_initialization.json`，失败即 `code=2` 退出，不进入训练。

另外它拒绝 `NUM_EPOCHS/BATCH_SIZE/SAMPLE_NUM_PER_EPOCH/SAVE_EPOCH_INTERVAL/NUM_AVG/RECIPE`
被外部覆盖，也拒绝任何 `SKIP_*=1`。

## 怎么用

```bash
# 1) 准备环境（在私有审计目录里改动，不要提交到仓库）
cp tools/ptn_resume/lm_launch/env.template /受控目录/lm.env
$EDITOR /受控目录/lm.env
set -a; . /受控目录/lm.env; set +a

# 2) 先确认输出目录是新的（wrapper 也会查，这里先看一眼）
test ! -e "$EXP_DIR" && echo NEW || echo "EXISTS -> 换 -v2"

# 3) 提交前最后一次只读体检：三段证据 + LM 冷启动合同
python3 -B tools/ptn_resume/three_segment_lm_check.py build \
  --references /受控目录/three-segment-references.json --output /受控目录/contract.json
python3 -B tools/ptn_resume/three_segment_lm_check.py check \
  --contract /受控目录/contract.json --output /受控目录/report.json

# 4) 真正提交（需要另行授权；本包不执行）
volc ml_task submit -c tools/ptn_resume/lm_launch/submit.conf.yaml
```

容器内由平台执行的入口就是 `run_lm.sh` 本身（见 `submit.conf.yaml` 的 `Entrypoint`）。

## 提交前还差哪些输入

| 缺项 | 从哪来 | 为什么不能猜 |
|---|---|---|
| `ImageUrl` | `volc ml_task export --task t-20261026074833-ctjsk --config`（只读）或控制台 | 镜像决定 torch/CUDA/私有解码库；换镜像等于换环境 |
| `ResourceQueueName` | 同上 | 队列决定调度与配额 |
| `TaskRoleSpecs.Flavor` | 同上 | 必须是 8×A800-80GB 同型 |
| `Storages`（VepfsId/SubPath） | 同上 | 挂错就看不到 `/shared/group-shared` 与 `/root/experiments`，训练会当场失败 |
| `tools/ptn_resume/lm_launch/` 已部署到该镜像的仓库路径 | 一次显式部署（写服务器，需授权） | 该 commit（`0a3f6dd`）没有 `tools/` 目录 |
| 三段证据里的两项 unsupported | 见 `redimnet-review/launch/impl-lm.md` | 原 95 份清单无原件；第三段平台终态未观测 |

## 入口参数必须和 gate 的 argparse 对齐

平台的"入口命令"最终会走 `lm_launch.sh` → `runtime/lm_stage_gate.py`，`"$@"` 原样透传。gate 只认
`--package-root / --contract / --contract-sha / --package-sha / --verify-only`（以及兼容用的
`--execute`）。**提交前用 `python -B runtime/lm_stage_gate.py --help` 本地比对一遍**：
多一个 gate 不认识的参数，argparse 会以 `unrecognized arguments` 直接退出(2)，
排到卡上时一条 gate 日志都不会有，看起来像"包坏了"。

`--execute` 只为兼容从 PTN `entry.py` 抄来的入口命令而保留，它是 **no-op**：执行本来就是默认行为，
带不带它判据完全一致（`tests/test_lm_stage_gate.py` 有对应用例）。
2026-11-06 的真实事故：入口命令末尾多了 `--execute`，而当时 r3 包内的 gate 还不认它——
r3 包保持原字节未改，修正做在提交入口（去掉该参数）与仓库源码（改成兼容）两侧。

## 不变量（改这个包时不许破坏）

* `EXP_DIR` 只能新建；失败后版本号 +1，不覆盖、不"修好再更"。
* `MODEL_CKPT` 只能是第 115 轮，且必须显式传（缺省按 mtime 会取 120）。
* LM 是冷启动：只载模型参数，SGD 动量与 GradScaler 全新；出现
  `resume: restored optimizer state from` 即为违反合同。
* 自动重试关闭；同一目录二次冷启动会让"唯一新输出"失效。
* 训练超参只来自 `official_repro_lm.yaml`，不用环境变量绕。
