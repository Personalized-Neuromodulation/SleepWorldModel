# PSG Backbone 实现规格

版本：v15，第一版 backbone 已实现。入口为 `build_backbone(config, signal_batch)`，配置见 [base.yaml](../configs/backbone/base.yaml)，逐步验证见 [Jupyter notebook](../notebooks/backbone_step_by_step.ipynb)。Fast/Slow、长程编码、MoE、JEPA、对比学习、codebook 和世界模型训练仍为后续范围。

## 1. 包边界与文件树

```text
dataloader/                         # 数据读取、对齐、QC、adapter、SignalBatch
├── signals.py                     # SignalBatch/SignalGroup、as_signal_batch
└── synthetic.py                   # notebook 和测试用合成窗口

backbone/                           # SignalBatch → 连续 PSG 表示
├── __init__.py                     # 公共导出
├── configuration.py                # BackboneConfig 与组合校验
├── contracts.py                    # 内部状态、MaskPlan、各级输出
├── factory.py                      # 构建并注入 architecture/modeling 实例
├── architectures/                  # 通用序列算法，不识别 PSG 业务轴
│   ├── sequence.py                 # 通用 mask 和依赖区间传播
│   ├── cnn/
│   │   ├── configuration.py        # CNNConfig
│   │   └── modeling.py             # CNNSequenceBlock
│   └── transformer/
│       ├── configuration.py        # TransformerConfig、FFN 配方
│       ├── modeling.py             # Attention、FFN、Sequence/CrossAttentionBlock
│       └── modeling_moe.py         # 后续：MoE FFN，第一版不创建
└── modeling/                       # PSG 编码职责与组合
    ├── pooling.py                  # patch/通道/模态共用的安全汇聚与位置编码
    ├── patching.py                 # Patchifier 与 PatchLayout
    ├── masking.py                  # 执行 waveform/token MaskPlan
    ├── patch_encoder.py            # 单 patch 编码
    ├── patch_sequence_encoder.py   # patch 序列编码
    ├── channel_aggregator.py       # 通道身份与通道汇聚
    ├── fusion.py                   # 模态身份、时间对齐与融合
    ├── signal_encoder.py           # 单编码组流程
    ├── modality_encoder.py         # 单模态流程
    └── psg_backbone.py             # 多模态总入口

tokenization/                       # 可选离散 tokenizer，不属于 backbone
└── codebook/
    ├── configuration.py
    └── modeling.py

pretraining/                        # 组合 backbone、目标分支、head 和 loss
├── jepa/
├── contrastive/
├── masked_code/
└── objectives/                     # SIGReg 等可复用目标项

world_model/                        # 后续：历史 latent → 未来 latent

configs/backbone/base.yaml          # 可直接加载的第一版网络配置
notebooks/backbone_step_by_step.ipynb # 15个代码单元：输入、各层、梯度、重载
scripts/check_backbone_notebook.py   # 无界面执行 notebook，可选 --real / --device cuda
tests/test_backbone.py              # shape、泄漏、梯度、时序与非法配置
```

`architectures` 只接收通用序列，不识别 PSG 业务轴。CNN 和 Transformer 是可替换算法家族；Attention、FFN 属于 Transformer 内部。它对外提供保形的 sequence blocks，并为 Fusion 提供 `CrossAttentionBlock`。`modeling` 负责 PSG 的 B/C/N/L 轴、mask、时间映射和调用顺序。`factory.py` 只在构建时选择网络；forward 不读取 YAML。`BackboneConfig.to_dict/from_dict` 用于 YAML 与 checkpoint。

当前 `world_model/ssl` 保留为已有最小基线，等新 backbone 接口稳定后再迁移。

## 2. 模块组合

```text
PSGBackbone
├── modality_encoders: ModuleDict[str, ModalityEncoder]
│   └── ModalityEncoder
│       ├── SignalEncoder
│       │   ├── Patchifier
│       │   ├── MaskApplier
│       │   ├── PatchEncoder
│       │   │   ├── input projection / patch 内位置编码
│       │   │   ├── ModuleList[SequenceBlock]
│       │   │   └── patch 内 readout / output projection
│       │   └── PatchSequenceEncoder
│       │       ├── patch 时间编码
│       │       └── ModuleList[SequenceBlock]
│       └── ChannelAggregator
│           └── 通道身份编码 + masked mean/attention readout
└── fusion: Fusion | None
    └── 模态身份编码 + 时间对齐 + pool/注入的 CrossAttentionBlock
```

| 类 | 输入 → 输出 | 负责的轴 |
|---|---|---|
| Patchifier | `[B,E,C,S] → [B,C,N,L]` | 将波形切为 patch |
| PatchEncoder | `[B,C,N,L] → [B,C,N,D]` | 每个通道、每个 patch 内部 |
| PatchSequenceEncoder | `[B,C,N,D] → [B,C,N,D]` | 同一通道的相邻 patch |
| ChannelAggregator | `[B,C,N,D] → [B,N,D]` | 同一时间位置的通道 |
| Fusion | `dict[m,[B,N_m,D]] → [B,N_joint,D]` | 不同模态 |
| SignalEncoder | SignalGroup → patch_tokens/local | 组合两个 patch 编码阶段 |
| ModalityEncoder | SignalGroup → features | SignalEncoder + 通道汇聚 |
| PSGBackbone | SignalBatch → BackboneOutput | 多模态调度和输出选择 |

同组通道共享网络权重；不同编码组默认使用独立参数。任何参数共享都必须显式配置。

## 3. 输入数据与维度

统一符号：

| 符号 | 含义 | 第一版示例 |
|---|---|---:|
| B | batch 中的窗口数 | 8 |
| E | 每个窗口的 epoch 槽位数 | 20 |
| C | 当前编码组通道数 | 1、2、3 或 6 |
| S | 每通道每 epoch 的采样点数 | 6000 |
| L | 每个 patch 的采样点数 | 200 |
| N | 每个窗口的 patch 数 | 600 |
| P | patch 内子片段数 | 10 |
| H | architecture 内部维度 | 64 |
| D | backbone 对外特征维度 | 128 |
| Q | 一个上下文块内的 patch 数 | 30 |
| W | 每个窗口的上下文块数，逐 epoch 分块后相加 | 20 |
| K | codebook 大小 | 由预训练配置决定 |
| Z | 对比学习 projector 的输出维度 | 由预训练配置决定 |
| M、T | architecture 的独立序列数与序列长度 | patch 内为 B×C×N、P |

当前 HSP 为 200 Hz、30 秒 epoch，因此 `S=200×30=6000`。采用 1 秒、不重叠 patch 时：

```text
L = 200
N = E × 30 = 600
P = L / 20 = 10                 # 当 subpatch_samples=20
Q = 30                          # 30秒上下文
W = N / Q = E = 20
```

Reader 返回五个存储组；dataloader adapter 将 SpO₂ 从 respiratory 中拆出，提供六个编码组：

| 编码组 | 通道 | C | SignalGroup.values |
|---|---|---:|---|
| eeg | F3-M2、F4-M1、C3-M2、C4-M1、O1-M2、O2-M1 | 6 | `[B,E,6,6000]` |
| eog | E1、E2 | 2 | `[B,E,2,6000]` |
| ecg | ECG | 1 | `[B,E,1,6000]` |
| emg | Chin1-Chin2、LAT、RAT | 3 | `[B,E,3,6000]` |
| respiratory | Airflow、Snore | 2 | `[B,E,2,6000]` |
| spo2 | SpO₂ | 1 | `[B,E,1,6000]` |

SignalBatch 的必要字段：

```text
SignalBatch
├── groups: dict[str, SignalGroup]
├── epoch_mask                bool  [B,E]
├── epoch_in_record           int64 [B,E]
├── epoch_start_offset_ns     int64 [B,E]
├── start_sec                 float [B]
├── duration_sec              float [B]
├── recording_duration_sec    float [B]
├── recording_ids             strings，长度 B
└── night_grade               uint8 [B]，可选；只供筛选/分层

SignalGroup
├── values                    float32 [B,E,C,S]
├── coverage_valid            bool    [B,E,C]
├── processing_valid          bool    [B,E,C]
├── artifact_valid            bool    [B,E,C]
├── valid                     bool    [B,E,C]
├── hard_code                 uint8   [B,E,C]
├── available_mask            bool    [B,C]
├── channel_mask              bool    [B,C]
├── sample_rate_hz            float   [B]
├── channel_ids               strings，长度 C
└── units                     strings，长度 C
```

```python
valid = available_mask[:, None, :] & coverage_valid & processing_valid & artifact_valid
data_valid = epoch_mask[:, :, None] & valid
```

`night_grade`、subject/session、标签和 hard_code 不进入 backbone 数值特征。当前数据没有逐采样点 QC；waveform mask 是训练遮挡，不是质量标注。

## 4. 中间数据契约

```text
PatchBatch
├── values                    float [B,C,N,L]
├── sample_visible            bool  [B,C,N,L]
├── data_valid                bool  [B,C,N]
└── layout                    PatchLayout

SequenceState
├── tokens                    float [M,T,H]
├── data_valid                bool  [M,T]
├── visible                   bool  [M,T]
├── positions                 int64 [M,T]
├── time_intervals_ns         int64 [M,T,2]
├── context_intervals_ns      int64 [M,T,2]
├── available_at_ns           int64 [M,T]
└── connection_mask           bool  [M,T,T] 或可等价表达的局部规则

TokenGrid
├── tokens                    float [B,C,N,D]
├── data_valid                bool  [B,C,N]
├── visible                   bool  [B,C,N]
├── coverage                  float [B,C,N]
├── time_intervals_ns         int64 [B,N,2]
├── context_intervals_ns      int64 [B,N,2]
├── available_at_ns           int64 [B,N]
├── channel_ids               strings，长度 C
└── patch_layout              PatchLayout

TokenSequence
├── tokens                    float [B,N,D]
├── data_valid/visible        bool  [B,N]
├── coverage                  float [B,N]
└── time/context/available    与 N 对齐
```

PatchLayout 保存采样率、patch 长度、每 epoch 的 patch 数、原始 epoch/样本索引和时间映射。第一版仅支持逐 epoch、不重叠、可整除的 patch；stride 等于 patch 长度，epoch padding 沿用 epoch_mask。TokenSequence 另含 `support_count[B,N]`，融合时按原始通道数统计 coverage。

三类 mask 必须分开：

| mask | shape | 来源与作用 |
|---|---|---|
| data_valid | `[B,C,N]` | dataloader QC/padding，经 Patchifier 映射 |
| visible | `[B,C,N]` | 任务指定，控制 context 能读取的位置 |
| target_mask | `[B,C,N]` | 任务指定，控制需要预测的位置 |

```python
context_mask = data_valid & visible
loss_mask = target_data_valid & target_mask
```

`MaskPlan(stage, visible={group: bool_tensor})` 支持 none/waveform/token，True 表示可见，第一版采用零替换。waveform 遮挡使用 `[B,E,C,S]`，在投影前执行；token 遮挡使用 `[B,C,N]`，在 PatchEncoder 后执行。token 隐藏位置的原始值也预先清零，避免 NaN 梯度；patch 编码彼此独立，因此不会改变可见位置的结果。数据质量与目标资格保留。

时间使用 recording 相对整数纳秒的 `[start,end)`；padding/无贡献为 -1。原始定位不随网络改变，context 区间随连接更新。当前离线 QC 的可用时刻未知，`available_at_ns=-1`；causal 只约束给定输入的 patch 序列，不声明原始数据处理可实时运行。

## 5. 数据流与 shape

```text
SignalGroup.values
[B,E,C,S]
    │ waveform mask
    ▼
Patchifier
[B,C,N,L]
    ▼
PatchEncoder
  direct Linear: [B,C,N,L] → [B,C,N,D]
  sequence path: [B,C,N,L]
                 → [B×C×N,P,H]
                 → SequenceBlocks
                 → patch readout
                 → [B,C,N,D]
    │ token mask
    ▼
patch_tokens [B,C,N,D]
    ▼
PatchSequenceEncoder
  [B,C,N,D] → [B×C×W,Q,D]
              → SequenceBlocks
              → [B,C,N,D]
    ▼
local [B,C,N,D]
    ▼
ChannelAggregator
features [B,N,D]
    ▼
Fusion（可选）
joint [B,N_joint,D]
```

以 EEG、`B=8,E=20,C=6,S=6000,L=200,P=10,H=64,D=128,Q=30,W=20` 为例：

| 位置 | shape |
|---|---|
| EEG 输入 | `[8,20,6,6000]` |
| patches | `[8,6,600,200]` |
| patch 内序列 | `[28800,10,64]` |
| patch_tokens | `[8,6,600,128]` |
| 30 秒序列块 | `[960,30,128]` |
| local | `[8,6,600,128]` |
| EEG features | `[8,600,128]` |
| joint（公共时间网格） | `[8,600,128]` |

第一版 Fusion 要求各模态时间网格完全相同，因此 `N_joint=N`；不一致则明确报错。pool 对同一时间点的有效模态求均值；cross-attention 使用该均值作 query、同一时间点的模态向量作 source，不跨时间融合。序列窗口不跨 epoch，末块不足 Q 时补齐后再还原。

### 5.1 近似张量大小

以下按 float32、只计算单个张量，不包括梯度、优化器状态、临时激活和框架开销：

| 张量 | 元素数 | 大小 |
|---|---:|---:|
| 全部15通道原始输入 `[8,20,15,6000]` | 14,400,000 | 54.9 MiB |
| patches | 与原始输入相同 | view 时不新增存储；复制时再增加54.9 MiB |
| 全部15通道 patch tokens `[8,15,600,128]` | 9,216,000 | 35.2 MiB |
| EEG patch 内状态 `[28800,10,64]` | 18,432,000 | 70.3 MiB |
| 六组 features，各 `[8,600,128]` | 3,686,400 | 14.1 MiB |
| joint `[8,600,128]` | 614,400 | 2.34 MiB |

若显式保存 attention score，EEG 的4头、30秒窗口对应 `[960,4,30,30]`，约13.2 MiB/层；600个 patch 的全局 attention 对应 `[48,4,600,600]`，约263.7 MiB/层。实现实际切为 Q=30，并使用 PyTorch SDPA；上述是稠密 score 的理论大小，不等于实测峰值显存。

## 6. 调用与通信关系

运行调用方向：

```text
外部任务
  → PSGBackbone
    → ModalityEncoder
      → SignalEncoder
        → Patchifier / MaskApplier
        → PatchEncoder
          → CNNSequenceBlock / TransformerSequenceBlock
        → PatchSequenceEncoder
          → CNNSequenceBlock / TransformerSequenceBlock
      → ChannelAggregator
    → Fusion（可选）
```

| 调用边界 | 输入 | 返回 |
|---|---|---|
| 任务 → PSGBackbone | SignalBatch、MaskPlan、outputs 请求 | BackboneOutput |
| PSGBackbone → ModalityEncoder | 当前 SignalGroup 与共享时间信息 | ModalityOutput |
| SignalEncoder → PatchEncoder | PatchBatch `[B,C,N,L]` | TokenGrid `[B,C,N,D]` + aux |
| modeling → architecture | SequenceState `[M,T,H]` | 更新后的 SequenceState + aux |
| Fusion → CrossAttentionBlock | query/source SequenceState 与对齐约束 | 融合后的 SequenceState + aux |
| ModalityEncoder → ChannelAggregator | TokenGrid `[B,C,N,D]` | TokenSequence `[B,N,D]` |
| PSGBackbone → Fusion | `dict[str,TokenSequence]` | joint TokenSequence |

通信规则：

- 所有输入输出使用显式 dataclass/类型，不传共享可变字典。
- 特征必须与有效性、可见性、时间和布局一起传递。
- architecture 不导入 dataloader、modeling 或训练任务。
- modeling 不实现 CNN/Transformer 内部数学结构。
- factory 可以导入所有实现用于构建；其他模块不反向导入 factory。
- 第一版的 BackboneOutput.aux 为空，后续 MoE 才扩展辅助输出；loss 在任务侧计算。
- 未请求的输出阶段可以跳过；被配置禁用的模块不创建、不进入优化器。

## 7. JEPA、对比学习与 Codebook 接入

BackboneOutput 可按需返回：

| 输出 | shape | 用途 |
|---|---|---|
| patch_tokens[m] | `[B,C,N,D]` | patch 编码消融、token 目标 |
| local[m] | `[B,C,N,D]` | JEPA 局部目标、通道级预测 |
| features[m] | `[B,N,D]` | 单模态 JEPA/对比学习 |
| joint | `[B,N_joint,D]` | 多模态 JEPA/对比学习 |

JEPA：

```text
context SignalBatch + MaskPlan
    → online PSGBackbone
    → context representation [B,N,D]
    → predictor [B,N_target,D]

clean target SignalBatch
    → target PSGBackbone
    → stop-gradient target [B,N_target,D]
```

`pretraining/jepa` 拥有 online/target backbone、predictor、target 位置、EMA 和 loss。两个 backbone 使用同一类与配置，参数状态独立。

对比学习：

```text
view1 → shared PSGBackbone → selected tokens [B,N,D] → task readout [B,D] → projector [B,Z]
view2 → shared PSGBackbone → selected tokens [B,N,D] → task readout [B,D] → projector [B,Z]
```

`pretraining/contrastive` 拥有增强、配对、task readout、projector 和 loss。subject/session/recording/time 元数据只用于配对，不进入 backbone 特征。

Masked-code：

```text
clean target → codebook tokenizer → code_ids [B,C,N]
masked input → PSGBackbone.local  → predictor logits [B,C,N,K]
loss mask = code_valid & target_mask
```

`tokenization/codebook` 拥有码本与 CodebookOutput；`pretraining/masked_code` 拥有 predictor、更新/冻结策略和 loss。CodebookOutput 至少包含 code_ids、valid_mask、PatchLayout 和 codebook_version。第一版 codebook 只生成目标，不进入 context backbone。

## 8. 第一版实现范围

- 实现 dataloader adapter → SignalBatch。
- 实现 waveform/token 两个遮挡入口。
- 实现 direct Linear、CNN、Transformer 及混合 blocks 的 PatchEncoder。
- 实现按30秒真实分块的 PatchSequenceEncoder。
- 实现通道聚合和可关闭的 Fusion。
- 实现所有 shape、mask、时间、输出请求和依赖方向测试。
- 暂不实现 Fast/Slow、跨 epoch 长程编码、MoE、codebook、JEPA、对比训练和未来 latent dynamics。
