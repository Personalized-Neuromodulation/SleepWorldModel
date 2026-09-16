# PSG Backbone

新增的 `model.moe.enabled=true` 在同一 ModalityTimeFusion 内替换四层 Dense FFN：一个共享SwiGLU专家、每个模态一个专属专家、可配置路由专家池（默认4个、top-1、hidden256）。Criss-Cross attention继续负责模态和时间交互。`architectures/moe.py`不负责PSG数据读取或loss；它返回路由辅助量，由pretraining决定是否加入loss。

`model(..., outputs=("features", "fused_features", "joint"))` 增加 `output.fused_features[name]`：融合后尚未聚合的 `TokenSequence [B,N,256]`。`encode_modalities(...)` 提供相同的融合前编码路径，供EMA目标复用。没有第二套backbone或SignalBatch。

模态预测实验使用 `readout.modality_pooling=mean` 和 `readout.temporal_pooling=mean`；其预测loss不经过汇总pooling，不应保留未训练的attention query。channel attention仍参与训练。HF配置保存这些开关和MoE参数，teacher/predictor不导出到backbone。以下attention readout默认值和无动态索引的profiler约束属于MoE关闭时的LeJEPA基线；稀疏专家dispatch会使用动态索引，速度需实测。


SpO₂ 默认通过 `model.numeric_tokenizers.spo2: {hidden_dim: 32, activation: gelu}` 选择数值 tokenizer：输入 `[B,1,N,200]` 在每个 patch 内求 FP32 均值，然后 `Linear(1,32) → GELU → Linear(32,256)`；不加逐 patch 归一化，保持绝对水平差异。固定中心化在 dataloader 中完成。此映射按模态名配置，没有写死 SpO₂ 特例；不在映射中的模态继续使用配置的 CNN。后续 Temporal Encoder、Fusion 与 readout 不变，旧 checkpoint 缺少该字段时保持 CNN。代码位于 `architectures/numeric.py`，测试位于 `tests/test_numeric_tokenizer.py`，全局单步入口仍为 `tests/test_pipeline.py`。

`build_backbone` 只构建 `FoundationBackbone`，不再选择旧架构。模型参数位于 `configs/psg_model.yaml`，由 `configs/psg_training.yaml` 引用；`BackboneConfig()` 是不包含数据集模态名的通用 CNN 配置，YAML 显式指定 SpO₂ 数值 tokenizer。输入由 dataloader 提供，LeJEPA 位于 pretraining。

```python
from backbone import BackboneConfig, build_backbone
from pretraining.configuration import load_config
from dataloader import as_signal_batch
from dataloader.synthetic import synthetic_windows

batch = as_signal_batch(synthetic_windows(epochs=1), foundation=True)
config = BackboneConfig.from_dict(load_config()["model"])
model = build_backbone(config, batch)
output = model(batch, outputs=("patch_tokens", "local", "features", "joint"))
print(output.foundation_representation.shape)  # [B,256]
```

唯一调用路径：Patchifier → PatchTokenizer → FoundationModalityEncoder → ModalityTimeFusion → TemporalReadout。通用 CNN/CrissCrossBlock/TemporalBlock 位于 `architectures/`，PSG轴组合位于 `modeling/foundation.py`。每个模态权重独立；所有views共享同一主干。支持已知通道ID的子集、模态缺失和1–30秒输入。

校验只在数据边界执行：`as_signal_batch(raw, foundation=True, input_spec=model.input_spec)` 在 CPU 检查 shape、采样率、epoch QC、有效区域有限值、通道身份和时间网格，再 `.to(device, non_blocking=True)`。首次构建没有 input_spec 时可省略。直接构造/修改 SignalBatch 的调用方须先调用 `batch.validate(...)`；backbone 接收已验证数据，不重复检查张量值。MaskPlan 的结构和模型构建配置仍由模型校验。

所有 foundation 模态复用同一时间布局，位置编码缓存为 buffer；仅按需生成 patch/local 调试输出。HF 公开 tensor 入口由 `dataloader.public` 适配并校验；重复 GPU 推理建议在循环外完成适配，使用 `model(batch=validated_batch)`。

全局调试入口为 [tests/test_pipeline.py](../tests/test_pipeline.py)，可串起 reader、views、backbone、loss、backward、HF roundtrip。`tests/test_input_boundary.py` 检查校验只执行一次，并用 CUDA profiler 防止 GPU forward 重新出现张量值检查。

`SignalGroup.data_valid [B,C,E]` 在 dataloader 中生成并保存；`SignalBatch.data_valid` 只引用它，不重新组合 QC。`SignalGroup.visible` 将当前 channel_mask 广播为 `[B,C,E]`，pretraining 的 dropout 只改变可见性。裁剪保留原始 epoch QC，bucket 拼接仅沿 B 拼接已有 mask。

`Patchifier` 只应用 `data_valid & visible`、外部 MaskPlan 和 patch reshape，不读取原始 QC 字段。`sample_visible` 只包含遮挡，不混入 QC；坏通道可以 visible=True、data_valid=False，此时计算仍被屏蔽。`epoch_mask` 在模型中仅用于 padding 位置的时间布局。attention/pooling 的有效性归约、NaN 隔离和 loss 的 invalid-view 排除是必要计算，予以保留。

`PatchBatch/TokenGrid.data_valid` 在 foundation 路径保持 epoch 粒度 `[B,C,E]`，只通过 `token_valid` 广播到 token。聚合后的 `TokenSequence.data_valid` 描述特征支持，不产生新的 QC 判定。

HF入口见 `backbone.huggingface.SleepWorldModelConfig / SleepWorldModel`。公开输入是 `signals={name: [B,C,N,200]}`、可选epoch `data_valid={name:[B,C,1]}`、`visible={name:[B,C,N]}`、`channel_ids`与 `time_intervals_ns=[B,N,2]`。提供通道子集时必须给出对应ID；省略模态可直接不传。时间网格必须连续且整秒对齐，不跨30秒来源epoch。

模块位置：

- `modeling/patching.py`：mask 应用与 patch reshape。
- `architectures/cnn/configuration.py`：CNN 各层及归一化、激活、投影的配置。
- `architectures/cnn/patch_tokenizer.py`：按配置列表构建 CNN，200 samples → 256d，默认三层。
- `architectures/transformer/criss_cross.py`：Criss-Cross / Temporal blocks。
- `architectures/transformer/modeling.py`：SDPA Attention / DenseFFN。
- `modeling/foundation.py`：模态编码、融合、读出与唯一主干组合。
- `modeling/pooling.py`：masked pooling、位置和时间辅助函数。

已删除旧 PatchEncoder、SignalEncoder、PatchSequenceEncoder、通道聚合/Fusion 包装、旧 CNN/Transformer sequence blocks、旧配置及切换分支。旧配置中的 architecture、patch_encoder、sequence_encoder、fusion_heads 等字段不再接受；包含这些字段的历史导出需要按当前配置重新导出。不会静默丢弃旧字段或加载另一套模型。

测试从 `tests/test_pipeline.py` 进入；`tests/test_backbone.py` 的通用回归已迁至当前实现。默认网络的参数命名和数量保持不变。

网络参数统一在 `configs/psg_model.yaml` 的 `model` 中修改：

| 配置 | 控制内容 |
|---|---|
| `patch_tokenizer.layers` | 每项是一层 Conv1d；增加/删除项即可改变层数。各层配置 out_channels、kernel_size、stride、padding、dilation、bias；in_channels 自动衔接 |
| `patch_tokenizer.norm / group_norm_groups` | group_norm 或 none，以及 GN 分组数 |
| `patch_tokenizer.activation / dropout` | gelu/relu/silu，以及 CNN dropout |
| `patch_tokenizer.projection / output_norm` | none/linear，以及 layer_norm/none |
| `modality_encoder` | EEG、多通道、单通道 encoder 的层数 |
| `criss_cross / temporal` | 多通道 heads_per_branch、单通道 num_heads，以及各自 ffn_dim |
| `fusion` | depth、heads_per_branch、ffn_dim |
| `post_fusion_temporal_depth / dropout` | 融合后层数，以及 Transformer dropout；融合后块复用 temporal 配置 |
| `readout.encoder_output_norm` | 在每个 modality encoder 的 blocks 后、channel pooling 前沿 D 做无 affine 的 LayerNorm；不重新计算 QC |
| `readout.fusion_output_norm` | Fusion blocks 后、modality pooling 前沿 D 做无 affine 的 LayerNorm |
| `readout.modality_pooling` | mean / attention，仅改变模态聚合；不强制 channel/temporal pooling 同时改变 |
| `readout.preferred_modalities` | 默认空列表；可设 `[eeg, eog]` 优先汇总其融合后特征。该位置均缺失时回退所有有效模态；不改变融合、QC或预测目标 |
| `readout.score_kind` | linear / normalized_linear / cosine；独立控制 modality attention 的打分 |
| `readout.cosine_scale` | 固定正数，cosine logits 位于其正负范围内；初始诊断值 2.0，不是已证明的 PSG 最优值 |

输出归一化针对 latent D 轴，不是原始波形逐窗标准化。无 affine 参数使边界不再通过可学习 gain 放大尺度；输出继续应用原有 active mask，缺失位置仍为零。cosine 打分用 FP32 归一化 query/key、固定 scale；value 加权仍用当前 token dtype。通道与时间池化目前保持原实现。

这些开关服务于同一主干的消融，没有第二套模型。缺少 `readout` 字段的旧 foundation checkpoint 按未归一化、linear attention 解释，避免静默改变旧权重行为；新增字段会随 HF/config JSON 保存。cosine query 使用通常的小随机初始化，不能套用 linear 打分的全零初始化建议。

CNN 默认保持 `32×8→256`，不加 Linear。修改卷积后若 flatten 不等于 256，构建时会报错；可以调整卷积，或显式设 `projection: linear` 保持输出契约。输入 200 samples、输出 256 维仍是当前模型边界。配置校验在构建时完成，不进入 GPU forward。

命令行支持零起始的列表索引，例如 `model.patch_tokenizer.layers.0.kernel_size=25 model.patch_tokenizer.layers.0.padding=12`。修改层数推荐直接编辑 YAML 列表。修改后重新构建模型；HF 和训练 checkpoint 保存完整配置，重载时按相同配置重建网络。

GPU 训练中，通道/模态 embedding 随当前 token dtype 相加，避免把 bf16 特征提升回 FP32。attention pooling 使用 batched matmul 完成加权归约，减少中间张量。参数仍保留 FP32，由 autocast 控制计算精度。`tests/test_gpu_training.py` 用 profiler 检查完整 LeJEPA forward 没有 `.item()`、动态布尔索引或重复有限值检查造成的主机同步。
