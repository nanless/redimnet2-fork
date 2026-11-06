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

同时打印 trainer 的梯度累积和 checkpoint 保存间隔。不能只检查子 TOML：`extends` 链会递归合并。子配置中的 `preloaded_skip_prefixes` 不会覆盖从父配置继承的另一个键 `preloaded_embedding_skip_prefixes`。

## 关键加载语义

`trainer.py::_preload_model` 的策略顺序为：

1. 加载 extraction checkpoint，并应用 extraction skip allowlist；
2. 根据 `preload_embedding_skip_front` 决定是否移除 `front.*`；
3. 优先读取 `preloaded_embedding_skip_prefixes`，只有该键不存在时才回退到 `preloaded_skip_prefixes`；
4. 对 embedding 执行 `load_state_dict(..., strict=False)`。

由于第 4 步是非严格加载，被跳过的 ArcFace 参数可能无报错地停留在构造器随机初始化状态。对架构兼容的完整预加载，必须显式清空通用和专项 skip 列表，并设置 `preload_embedding_skip_front=false`。

## checkpoint 证据

下面的 `weights_only=False` 仅用于身份与来源已核实的自有训练checkpoint；它可能执行pickle对象。来源未知时不加载，也不为查看形状关闭安全限制。解释器以当前训练环境为准，代码中的历史绝对路径不是所有任务的默认入口。

```python
ckpt = torch.load(path, map_location="cpu", weights_only=False)
embedding = ckpt["embedding_model"]
assert "sv_arc_margin.weight" in embedding
print(tuple(embedding["sv_arc_margin.weight"].shape))
```

当前 SharedUtteranceSV 设置的预期形状为 `(n_speakers, 256)`。即使文件存在，只要形状或 speaker vocabulary 映射不一致，父 checkpoint 仍然不兼容。

## 启动验收证据

启动日志必须显示 `sv_arc_margin.weight` 已加载，而不是被跳过或缺失。对同架构完整预加载，不应存在未被明确解释的模型 missing keys。

需要更强证据时，在第一个 optimizer step 之前比较运行中模型参数与 checkpoint 参数的稳定指纹，或执行精度合适的 `allclose`；避免把训练更新后的差异误判为加载失败。

任务描述中的“已完整加载兼容权重”只是文字；最终以有效配置、checkpoint 键/形状和加载摘要为准。
