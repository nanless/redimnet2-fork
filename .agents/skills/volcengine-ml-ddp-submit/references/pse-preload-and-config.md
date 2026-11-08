# PSE 配置继承与 checkpoint 预加载

## 有效配置审计

在仓库根目录、使用训练所用的同一 Python 环境读取配置：

```bash
PYTHONPATH=. /root/miniforge3/bin/python - <<'PY'
from speech_enhancement.runners.pse.utils.config_loader import load_config

path = "recipes/pse/merged_datasets_20260724/config/train/<successor>.toml"
cfg = load_config(path)
for key in (
    "root_dir", "preload_model_path", "preload_embedding_skip_front",
    "preloaded_skip_prefixes", "preloaded_embedding_skip_prefixes",
    "preloaded_extraction_skip_prefixes", "amp_dtype",
):
    print(key, repr(cfg["meta"].get(key)))
print("distributed", cfg.get("distributed"))
PY
```

同时打印 trainer 的梯度累积和 checkpoint 保存间隔。子 TOML 单独看不够：`extends` 链会递归合并，父配置里的键也会进来。子配置里的 `preloaded_skip_prefixes` 覆盖不了从父配置继承的另一个键 `preloaded_embedding_skip_prefixes`，两个键名字不同，各读各的值。

## 关键加载语义

`trainer.py::_preload_model` 的策略顺序为：

1. 加载 extraction checkpoint，并应用 extraction skip allowlist；
2. 根据 `preload_embedding_skip_front` 决定是否移除 `front.*`；
3. 优先读取 `preloaded_embedding_skip_prefixes`，只有该键不存在时才回退到 `preloaded_skip_prefixes`；
4. 对 embedding 执行 `load_state_dict(..., strict=False)`。

第 4 步是非严格加载：被跳过的 ArcFace 参数可能不报错地停在构造器随机初始化的状态。架构兼容的完整预加载要显式清空通用和专项 skip 列表，并把 `preload_embedding_skip_front=false` 写进配置；skip 列表还留着时，这些参数就停在随机值上。

## checkpoint 证据

下面的 `weights_only=False` 只用于身份和来源都已核实的自有训练 checkpoint；它可能执行 pickle 对象。来源未知的 checkpoint 不加载，也不为了看形状去关掉这道安全限制。解释器以当前训练环境为准，代码里的历史绝对路径不是所有任务的默认入口。

```python
ckpt = torch.load(path, map_location="cpu", weights_only=False)
embedding = ckpt["embedding_model"]
assert "sv_arc_margin.weight" in embedding
print(tuple(embedding["sv_arc_margin.weight"].shape))
```

当前 SharedUtteranceSV 设置的预期形状为 `(n_speakers, 256)`。文件存在也不够：形状或 speaker vocabulary 映射对不上时，父 checkpoint 仍然不兼容。

## 启动验收证据

启动日志里 `sv_arc_margin.weight` 要显示为已加载；显示成跳过或缺失都不算通过。同架构的完整预加载，不该有没解释清楚的模型 missing keys。

需要更强的证据时，在第一个 optimizer step 之前比较运行中模型参数与 checkpoint 参数的稳定指纹，或执行精度合适的 `allclose`；比的是加载后、更新前的状态，训练更新造成的差异不会被误判成加载失败。

任务描述里的“已完整加载兼容权重”只是一句话；判断依据是有效配置、checkpoint 键与形状、加载摘要。
