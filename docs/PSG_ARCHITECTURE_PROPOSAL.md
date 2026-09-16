# PSG foundation backbone 与模态预测预训练

用户已批准实现本轮方案并训练 10K，每 1K 评价。输入范围仍为单个 30 秒 epoch。
`data_preprocess/**` 冻结。所有改动复用原 backbone、数据契约、HF API 和训练入口。

## 数据与调用关系

```text
SignalBatch：各模态 [B,1,C_m,6000]，epoch data_valid [B,C_m,1]
  │
  ├─ online：在原始输入上应用 MaskPlan
  │   → 1秒 patch [B,C_m,30,200]
  │   → tokenizer [B,C_m,30,256]
  │   → 独立 modality encoder ×4
  │   → channel attention pool → 各模态 [B,30,256]
  │   → stack [B,G,30,256]
  │   → modality-time attention + shared/private/routed MoE ×4
  │   → 保留融合 token [B,G,30,256]
  │       ├─ 各模态 predictor → 各模态预测 [B,30,256]
  │       └─ masked mean over G、N → 下游 [B,256]
  │
  └─ target（干净输入，无梯度）：EMA modality encoders
      → channel attention pool → 各模态目标 [B,30,256]
      → 沿 D 做 LayerNorm，作为被遮挡位置的预测目标
```

`B` 是来源样本数，`C_m` 是模态通道槽位数，`G` 是实际输入模态数，
`N` 是整秒 token 数，`P=200` 是每秒原始采样点，`D=256` 是 latent 维度。
P 和 D 没有数值对应关系。backbone 仍支持 N=1–30；本轮预训练固定 N=30。

`data_valid` 由 dataloader 在 CPU 一次生成：epoch 存在、通道存在、覆盖完整、
处理成功、artifact QC 通过。任何一项失败，整个 epoch 的该通道都无效，
不重新做 1 秒 QC。`night_grade` 只用于筛选和组成 batch，目前仅 Outstanding。

- `data_valid`：不可被增强改变的数据事实。
- `visible`：在线模型可读的位置。
- `target_mask`：待预测的位置，与数据有效性单独保存。
- attention key/value、pooling 使用 `active = broadcast(data_valid) & visible`。
- loss 只包含干净目标有效、被选中预测且仍有可见上下文的位置。
- 全无效 view 排除；不足两个有效来源样本时跳过更新，包括 EMA 更新。

## Backbone

CNN 默认三层，可通过 YAML layers 列表配置：

| 层 | 参数 | 输出 |
|---|---|---|
| Conv1 | 1→32，K49，S25，P24，GN8，GELU | [BCN,32,8] |
| Conv2/3 | 32→32，K3，S1，P1，GN8，GELU | [BCN,32,8] |
| flatten + LN | 32×8=256 | [B,C,N,256] |

Conv3 单位置感受野 149 samples=745ms；flatten 汇总完整 1 秒 patch。
不同模态 tokenizer 参数独立，同一模态跨通道、views 共享参数。
SpO₂ 保留已批准数值路径：CPU `(百分数−95)/5`，每秒均值→1→32→256 MLP，
不做逐 patch 归一化。HF 输入需在模型外完成同样预处理。

- EEG/EOG/EMG/respiratory 多通道：Channel-Time Criss-Cross ×4。
  D=128 channel +128 temporal，每支4 heads，head_dim32，Dense FFN1024。
- ECG/SpO₂ 单通道：Temporal ×4，8 heads，FFN1024，不建立 channel attention。
- 模态编码器出口沿 D 做无 affine LN；channel attention pooling 保留。
- Fusion ×4：128 modality +128 temporal，SDPA，每支4 heads。
  **FFN 换成 MoE，attention 仍负责跨模态交换信息。**
- 本轮不增加 post-fusion temporal block。

每层 MoE 包含一个共享专家、每个配置模态一个专属专家、4个 routed experts，top-1。
专家为 SwiGLU，hidden=256。共享和专属分支始终计算，路由只计算选中的专家。
输出为三个分支之和除以3；top-1保留完整softmax门值，保证 router 接收预测梯度。
路由使用FP32并有负载均衡损失，不采用容量丢弃 token。
这是 DeepSeek 思路的 PSG 扩展，不是原版 DeepSeek 的复现：固定 modality-private
专家与 token-routed expert 不等价。小专家循环允许，不能逐样本/通道/时间循环。
稀疏dispatch存在动态索引开销，速度需以实际GPU测试为准。

本轮预测损失不经过汇总 readout，因此 modality 和 temporal readout 采用 masked mean，
避免下游使用未被训练的 attention query。通道池化仍由预测损失训练。

可用 `readout.preferred_modalities: [eeg, eog]` 只在最终汇总时优先 EEG/EOG，
若该位置均无效则回退所有有效模态。六模态仍参与融合与预测；数据覆盖、256D输出和三项下游评价不变。

## 预训练目标

两个遮挡 views 使用同一 online backbone；EMA 仅复制 modality encoders（含 tokenizer、
channel pool），不复制 fusion。干净目标每个 batch 计算一次；两个在线 views 沿 B 拼接一次 forward。

默认每个 view/sample：25% 概率整模态遮挡，随机选择一个有效模态，保留其他模态；
否则采用各模态独立的局部时间遮挡：3秒连续块，遮挡40%=12秒。只有一个有效模态时
退回局部遮挡。所有 views 仍在同一30秒时间网格，**不使用原 LeJEPA 的 global/local 裁剪**。
遮挡发生在在线 tokenizer/temporal encoder 前，不能先完整编码再遮挡。

每个模态使用独立 predictor：模态可学习 query + 30个时间位置，cross-attention读取
可见的融合 token，再经 FFN512 和 Linear 输出256维。query不能读取目标latent。

```text
L = L_prediction + 0.01 L_SIGReg + 0.01 L_router
```

- prediction：逐模态对有效目标位置求256维MSE，再对有目标的模态等权平均。
  局部/整模态损失另外报告，便于识别无法从其他模态预测的信息。
- SIGReg：每个 view、每个可见模态的融合token沿时间masked mean，
  projector 256→512→128；保持 [V,G,B,128]，仅沿有效 B 统计，之后平均 V/G。
  默认2048 slices、17 knots，关键计算FP32；不把相关时间token当独立样本。
- EMA decay=.996，只在成功 optimizer.step 后更新。target 始终 eval，无梯度。
- 历史 LeJEPA 可通过 objective=lejepa 加载/对照，仍无teacher/predictor。
  新目标名为 modality_jepa，不再将预测损失标为 invariance。

## 文件职责与调试

```text
backbone/architectures/moe.py             MoE配置、SwiGLU、共享/专属/路由专家
backbone/architectures/transformer/       通用SDPA、Temporal、CrissCross
backbone/modeling/foundation.py          模态编码、二维fusion、pooling
backbone/contracts.py                    BackboneOutput新增fused_features
backbone/huggingface/                    同一HF config/model/output
pretraining/modality_jepa/masking.py      配置、MaskPlan采样及RNG恢复
pretraining/modality_jepa/model.py        目标encoder、predictor、loss和EMA
pretraining/lejepa/                      历史LeJEPA对照，复用projector
pretraining/objectives/sigreg.py          共享FP32 SIGReg
pretraining/factory.py / views.py        统一模型和sampler构建
pretraining/training.py / evaluation.py  统一训练、评价和日志
pretraining/checkpoint.py                模型/EMA/优化器/sampler/RNG恢复、HF导出
pretraining/cli.py                       唯一正式入口
experiment_logging/                     W&B
tests/test_pipeline.py                  同一全局调试入口，按objective进入对应流程
tests/test_modality_jepa.py              泄漏、EMA、缺失数据、路由及恢复测试
```

内部沿用 SignalBatch/SignalGroup/PatchBatch/TokenGrid/TokenSequence/MaskPlan，
`BackboneOutput.fused_features[name]` 是融合后、池化前的 TokenSequence [B,N,D]。
HF仍支持标准save/load、AutoModel及公开预处理tensor输入；加载环境安装本项目。
HF导出只包含online backbone，不包含teacher/predictor/projector/任务头。

所有实验仍用两文件配置。本轮快照：
`results/foundation/modality_jepa_moe_10k_20260914/psg_training.yaml` 引用同目录 `psg_model.yaml`。
仓库 `configs/` 保留 LeJEPA 基线默认值，新参数也可通过其 dotted overrides 配置。

```powershell
python -B tests/test_pipeline.py --config results/foundation/modality_jepa_moe_10k_20260914/psg_training.yaml --break-at loss training.batch_size=3 evaluation.wandb_mode=disabled
python -B -m pretraining.cli --config results/foundation/modality_jepa_moe_10k_20260914/psg_training.yaml
```

## 训练与评价

10K成功更新，每1K评价和保存；eval_every_seconds=0。B192，CUDA bf16，
fused AdamW，峰值lr1e-3，5%warmup，cosine至1e-5；原数据划分、RAM和SpO₂处理不变。
不承诺MoE一定提高F1；此次同时改变SSL目标和fusion，只能评价组合方案，不能据此单独归因。

W&B每步记录：总/预测/SIGReg/router损失、逐模态预测损失、local/whole损失与目标数、
目标表示std、router负载/熵，以及step/batch/epoch-equivalent、LR、梯度、显存。
每1K同时评价固定train/validation子集：sleep staging五类独立F1/recall/precision/support，
heart_rate、sao2原单位回归指标；独立充分拟合线性头保持eval_refit命名。
原 `eval/macro_f1` 仍用于最佳checkpoint选择，绝不使用test集调参。

验证要求：隐藏原始波形不泄漏、部分/全部缺失、所有在线分支梯度、teacher无梯度且只按step更新、
EMA和sampler精确恢复、CPU/CUDA bf16、实际B192显存/速度、HF roundtrip、完整pytest、相关Ruff。
实施末尾必须确认 `git diff HEAD -- data_preprocess` 为空。

真实六模态 B192、RTX4090D bf16 的8步验证：backbone 26,101,888参数，
EMA 16,913,024参数（冻结），总可训练参数29,861,120；稳定步耗时约0.312秒，
峰值allocated15.92GiB、reserved17.45GiB。此测量复用已读入的batch，
不含磁盘、下游probe和周期评价，不能直接等同完整训练速度。
原始测量保存在本轮结果目录 `gpu_smoke.json`。

## Approval Checklist

后续已批准30秒监督适配：`training.task=finetune` 从既有SSL checkpoint初始化，
仅解冻配置指定的末端 modality/fusion blocks及对应channel pool。分期梯度更新这些层，
回归任务使用detach特征。`finetuning/model.py → evaluation.py / training.py` 复用原数据契约、
CLI、checkpoint和HF导出。监督结果使用 `supervised_*`，不替代原冻结探针；新目标为 N1 F1≥0.70。

监督适配的可选消融：`train_temporal_readout=true` 解冻已有 TemporalReadout；
`model.readout.temporal_pooling=attention` 可从 mean checkpoint 迁移，新增 query
置零以保持初始均值输出。其余权重严格加载，仍输出 `[B,256]`，不增加上下文或网络深度。
是否用于下一轮正式训练，先由训练集内部对照决定。
`training.finetuning.train_tokenizers` 可指定参与监督更新的模态 tokenizer，默认空列表；
例如 `[eeg,eog,emg]` 沿用 backbone 学习率，其他 tokenizer/embedding 保持冻结。
该消融不增加参数或移除输入模态，但需计入反传到原始波形 CNN 的显存与计算开销。

- [x] 已获批准：可见信息融合后，预测各模态融合前的latent；局部和整模态遮挡。
- [x] shared/private/routed MoE放在fusion FFN，保留attention，D256/30秒边界。
- [x] 干净EMA模态目标、mask先于在线编码、独立predictor、FP32正则。
- [x] 保留既有QC、Outstanding筛选、SpO₂输入变换和HF接口。
- [x] 单一训练/调试入口，配置化，W&B记录，10K训练且每1K评价。
- [x] 最初 modality JEPA 实验从头训练；后续监督适配已获批从已完成的SSL权重初始化，不恢复用户已停止的进程。
