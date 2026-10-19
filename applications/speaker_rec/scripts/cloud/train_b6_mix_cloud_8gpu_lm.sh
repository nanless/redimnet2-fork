#!/usr/bin/env bash
# =============================================================================
# 火山引擎 8 卡：b6 混合数据线的第二阶段（大间隔微调，STAGE=lm）启动脚本
# =============================================================================
#
# 这只是 train_b6_mix_cloud_8gpu.sh 的一层包装：把 STAGE 固定成 lm 再转发过去。
# 两个阶段的全部逻辑（路径改写、开训前检查、断点续训、机器可读输出行）
# 都在 train_b6_mix_cloud_8gpu.sh 里，这里不重复实现。
#
# 大间隔微调要用主训练的结果当起点，所以必须先跑完主训练。默认从
#   $REPO_ROOT/exp/vox2_vb2_cnc2/b6_cloud/ptn
# 里按修改时间取最新的 model_<N>.pt；用 PTN_EXP_DIR 可以换成别的目录，
# 用 MODEL_CKPT=<权重文件> 可以指定某一个具体文件。
#
# 用法（8 卡，数据与结果都在持久化存储上）：
#   NUM_GPUS=8 \
#   DATA_ROOT=/mnt/vepfs/wespeaker_data \
#   EXP_DIR=/mnt/vepfs/exp/b6_cloud/lm \
#   PTN_EXP_DIR=/mnt/vepfs/exp/b6_cloud/ptn \
#   applications/speaker_rec/scripts/cloud/train_b6_mix_cloud_8gpu_lm.sh
#
# 本机 2 卡小样验证：先跑 ptn 小样，再把 PTN_EXP_DIR 指过去跑这个脚本。
#
# 注意：不要同时显式设置 STAGE=ptn，本包装脚本会把 STAGE 覆盖成 lm。
# =============================================================================

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STAGE=lm exec "$SCRIPT_DIR/train_b6_mix_cloud_8gpu.sh" "$@"
