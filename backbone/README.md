# PSG Backbone

第一版连续 PSG 主干已经实现。数据输入由 `dataloader.as_signal_batch` 提供，JEPA、对比学习和 codebook 的目标/损失由外部任务组合。

```python
import yaml
from pathlib import Path
from dataloader import as_signal_batch
from backbone import BackboneConfig, build_backbone

batch = as_signal_batch(raw_batch)
config = BackboneConfig.from_dict(
    yaml.safe_load(Path("configs/backbone/base.yaml").read_text())
)
model = build_backbone(config, batch)
result = model(batch, outputs=("patch_tokens", "local", "features", "joint"))
```

`architectures/` 包含独立 CNN/Transformer 算法与通用 mask/时间传播；`modeling/` 包含 patch、序列、通道和模态职责。`configuration.py` 定义配置，`factory.py` 构建实例，`contracts.py` 定义输出。

- patch：direct Linear / CNN / Transformer / 混合 blocks，独立编码每个 patch。
- 序列：按 epoch 内真实局部窗口打包，默认30秒；空 blocks 为 identity。
- 通道：none / mean / attention，通道身份嵌入独立开关。
- 融合：none / pool / cross_attention，只接受相同时间网格。
- 遮挡：none / waveform / token，外部指定可见区域，默认零替换。

输入 patch 必须整除 epoch；序列不跨 epoch，末块可补齐。修改通道顺序、采样率或单位/尺度需要匹配 input_spec。当前数据处理可用时刻未知，输出 available_at_ns=-1。

逐步运行 [notebook](../notebooks/backbone_step_by_step.ipynb)，或查看 [完整规格](../docs/PSG_ARCHITECTURE_PROPOSAL.md)。代码中的小类与参数可直接查看，不需要额外注册框架。
