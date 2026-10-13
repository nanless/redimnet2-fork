# 官方预训练模型复现评测报告

面向刚接触说话人验证的研究生。这里汇总本仓库对 **ReDimNet2 官方预训练权重** 的复现结果：我们测到的等错误率（EER），以及与论文 / README 表格的差异。

## 先看这里

在约定范围内（不含 SITW），**复现结果与官方数字一致**。绝大多数协议上，实测减去官方的差值在 **0.01～0.06** 个 EER 百分点以内；CN-Celeb 在改用正确的「多段注册取平均」协议后，最大偏差约 **0.09**。

## 建议阅读顺序

| 顺序 | 文档 | 内容 |
|------|------|------|
| 1 | [01-背景与评测说明.md](01-背景与评测说明.md) | EER 是什么、评了哪些集合、「实测减去官方」怎么读 |
| 2 | [02-VoxCeleb1结果.md](02-VoxCeleb1结果.md) | Vox1-O / E / H 全表（20 个权重文件） |
| 3 | [03-域外试验结果.md](03-域外试验结果.md) | Vox1-B、VOICES、CN-Celeb（6 个混合数据训练的型号） |
| 4 | [04-数据与复现步骤.md](04-数据与复现步骤.md) | 评测数据路径、命令、目录里有什么文件 |
| 5 | [05-训练数据检查.md](05-训练数据检查.md) | 训练语料是否齐全（Vox2 / MUSAN / RIRS / CNC2 等） |

可交互汇总页（Cursor Canvas，表 + 图）：  
`~/.cursor/projects/.../canvases/redimnet2-eval-alignment.canvas.tsx`  
（文件名里的英文是编辑器工具生成的，学生不必理解；在 Cursor 里点开该文件即可并排查看。）

## 范围说明

| 项目 | 是否完成 |
|------|----------|
| 20 个官方 `.pt` 权重下载 | 是 |
| VoxCeleb1-O / E / H | 是 |
| VoxCeleb1-B、VOICES SID、CN-Celeb Avg | 是（混合数据训练表里的 6 个型号） |
| SITW | **不做**（数据需申请，本次跳过） |

## 原始数字文件

合并后的 CSV 仍放在 `checkpoints/` 根目录，便于脚本读取：

- `eval_vox1_O_all.csv` — Vox1-O
- `eval_vox1_EH_all.csv` — Vox1-E / H
- `eval_ood_all.csv` — Vox1-B、VOICES、CN-Celeb（Avg）
- `eval_cnc_avg_all.csv` — CN-Celeb Avg 单独汇总
- `eval_per_ckpt*/` — 分型号中间结果（可选查阅）
- `pretrained/` — 权重文件本身

## 术语速查

- **EER（%）**：等错误率，越低越好。本报告单位一律为百分点。
- **Δ（实测减去官方）**：`实测 EER − 官方 EER`。接近 0 表示数字与官方表一致；略负表示我们测得稍低一点（不一定等于「更强」）。
- **lm / ptn**：大间隔微调后的权重 / 主训练阶段权重。
- **混合数据训练的型号**：在 VoxCeleb2 之外混了 VoxBlink2、CN-Celeb2 等数据训练的型号（文件名含 `vb2`、`cnc2`）。
