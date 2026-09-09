# 通用读取接口

`dataloader` 是独立包，不导入模型或预处理实现。`WindowDataset, collate_windows` 返回下述读取字典；`as_signal_batch` 在数据侧将其转换成 backbone 所需的 SignalBatch，训练目标由外部任务管理。

## SignalBatch adapter

`from dataloader import as_signal_batch`；调用 `as_signal_batch(raw_batch)` 将 respiratory 中的 spo2 拆为独立组，同时切分 QC、通道身份和单位。`split_spo2=False` 保留原分组；其他数据集组默认原样保留。

`scales={"eeg": scale, ...}` 可按固定训练集尺度除以信号，单位同步标记；不在 batch 中拟合统计、不修改源 tensor。返回值提供 `.validate()` 和 `.to(device)`，包含可选 `night_grade[B]`。collate 要求整批都有 night_grade 或整批都没有；HSP 发布读取器仍要求等级存在且有效。完整主干契约见 [实现规格](../docs/PSG_ARCHITECTURE_PROPOSAL.md)。

## Reader 扩展点

实现 `schema.Reader` 协议并通过 `register_reader(name, ReaderClass)` 注册。构造函数接受 `root: Path`、`version`、`tasks` 及该实现自己的可选参数。Reader 应可被 pickle，不序列化打开的文件句柄，并在 worker 中独立打开文件。

- `metadata: DatasetMetadata`：name、version、channels、sample_rates、units、epoch_seconds。
- `records: list[dict]`：每条至少包含 recording_id、subject_id、session_id、n_epochs；Reader 可增加文件路径和分片偏移等内部字段。相同 subject 的不同 session 必须使用相同 subject_id。
- `read_window(record, start, count)`：读取一个记录内的窗口，start 和 count 以 epoch 为单位，返回下述样本。
- `close()`：释放资源，可重复调用。

WindowDataset 负责按 subject 划分和窗口索引，Reader 负责格式、版本、发布状态及存储索引校验。注册同名 Reader 会报错。不同格式不能伪装成同一 schema；新版本应明确增加支持和回归测试。

## 样本与 batch

统一维度：`B` 为 batch 大小，`E` 为单窗口 epoch 数，`Emax` 为 batch 内补齐后的最大 epoch 数，`C` 为信号组通道数，`S` 为每通道每 epoch 采样点数，`F` 为任务字段数。当前发布数据为 200 Hz、30 秒 epoch，因此 `S=6000`；默认 `E=20`，记录末尾可以更短。

### 信号

所有信号为 float32。单样本使用 `[E,C,S]`，collate 后使用 `[B,Emax,C,S]`，padding 填 0。

| `signals` 键 | 通道顺序 | C | 单样本 | batch |
|---|---|---:|---|---|
| eeg | f3-m2、f4-m1、c3-m2、c4-m1、o1-m2、o2-m1 | 6 | `[E,6,6000]` | `[B,Emax,6,6000]` |
| eog | e1、e2 | 2 | `[E,2,6000]` | `[B,Emax,2,6000]` |
| ecg | ecg | 1 | `[E,1,6000]` | `[B,Emax,1,6000]` |
| emg | chin1-chin2、lat、rat | 3 | `[E,3,6000]` | `[B,Emax,3,6000]` |
| respiratory | airflow、snore、spo2 | 3 | `[E,3,6000]` | `[B,Emax,3,6000]` |

### 信号质量和通道状态

以下字段都按信号组存放。质量字段的单样本 shape 为 `[E,C]`，batch shape 为 `[B,Emax,C]`；padding 的 bool 值为 false，`hard_code` 填 0。

| 字段 | dtype | 含义 |
|---|---|---|
| `quality.coverage_valid` | bool | 完整覆盖该30秒来源区间 |
| `quality.processing_valid` | bool | 必需处理成功并产生有限的6000点 |
| `quality.artifact_valid` | bool | 适用 QC 已完成且没有 hard artifact |
| `quality.valid` | bool | `available_mask & coverage_valid & processing_valid & artifact_valid` |
| `quality.hard_code` | uint8 | hard artifact 主码；它是原因码，不是 mask |
| `available_mask` | bool `[C]` → `[B,C]` | 原始记录是否具有该通道，不随训练改变 |
| `channel_mask` | bool `[C]` → `[B,C]` | 本次输入允许使用的通道，初始复制 `available_mask` |

模型实际输入的 epoch/通道有效性应使用 `epoch_mask[:,:,None] & quality.valid & channel_mask[:,None,:]`，结果为 `[B,Emax,C]`。

### Epoch 与时间

| 字段 | dtype | 单样本 | batch | 含义与 padding |
|---|---|---|---|---|
| `epoch_mask` | bool | `[E]` | `[B,Emax]` | 真实 epoch 为 true，padding 为 false |
| `epoch_in_record` | int64 | `[E]` | `[B,Emax]` | epoch 在整夜记录中的下标，padding 为 -1 |
| `epoch_start_offset_ns` | int64 | `[E]` | `[B,Emax]` | 相对记录起点的纳秒偏移，padding 为 -1 |
| `start_sec` | float | 标量 | float64 `[B]` | 当前窗口相对记录起点的位置 |
| `duration_sec` | float | 标量 | float64 `[B]` | 当前窗口实际包含的完整 epoch 时长 |
| `recording_duration_sec` | float | 标量 | float64 `[B]` | 源记录声明时长，可能包含未发布的不足30秒尾段 |

### 整夜等级

`night_grade` 是整夜可用信号时长等级，不是窗口 mask。单样本为 Python int，batch 为 uint8 `[B]`；同一 recording 的所有窗口取值相同。

| 值 | 名称 |
|---:|---|
| 5 | Outstanding |
| 4 | Excellent |
| 3 | Good |
| 2 | Fair |
| 1 | Poor |

### 任务标签

`tasks=()` 时不读取标签。启用任务后，所有标签与信号使用同一 epoch 网格；`valid` 只描述标签有效性，训练时再与所需信号 mask 取交集。

| 任务 | `field_names` | F | `labels` 单样本 → batch |
|---|---|---:|---|
| sleep_stage | stage | 1 | uint8 `[E]` → `[B,Emax]` |
| heart_rate | max_bpm、min_bpm、mean_bpm | 3 | float32 `[E,3]` → `[B,Emax,3]` |
| sao2 | max_percent、min_percent | 2 | float32 `[E,2]` → `[B,Emax,2]` |

每个任务还返回：`valid` bool `[E]` → `[B,Emax]`、`qc_code` uint8 `[E]` → `[B,Emax]`、`field_valid` bool `[E,F]` → `[B,Emax,F]`、`field_qc_code` uint8 `[E,F]` → `[B,Emax,F]`，以及不参与拼接的 `field_names`。无效标签的原始数值可以保留，使用方必须读取 `valid/field_valid`。

### 身份与静态元数据

| 字段 | 单样本 | batch |
|---|---|---|
| `recording_id`、`subject_id`、`session_id`、`path` | 字符串 | 长度 B 的列表 |
| `sample_rates[group]` | float 标量 | float32 `[B]` |
| `channel_names[group]`、`units[group]` | 长度 C 的 tuple | 保持一份共享 tuple |
| `dataset`、`version` | 字符串 | 保持一份共享字符串 |

collate 要求 batch 内 dataset、version、通道顺序、单位、采样率及 task 字段一致。跨数据集混合训练需要先制定明确的通道和时间对齐规则。

## 当前 HSP 发布格式

根目录为 `I:\HSP\I0002-preprocess\processed\v1.0.0`，通过版本描述、records/shards manifest 和逐 shard commit 读取已发布文件。支持旧内部 schema 1.0.0-pilot.2 与恢复后的 1.0.0-full.1。

读取后仍为 eeg、eog、ecg、emg、respiratory 五组，15 路、200 Hz、30 秒 epoch；respiratory 顺序为 airflow、snore、spo2。QC 不重新计算，标签不重新标注，发布文件不改写。默认按窗口切片读取和有界句柄缓存，不全量加载多 GiB 分片。

`tasks=()` 默认不读取任务；可以指定 sleep_stage、heart_rate、sao2。`verify_checksums=True` 会在每次重新打开文件时计算完整 SHA256，默认只检查提交摘要一致性、文件大小、绑定及窗口索引。

旧原始 H5 实现迁入 readers/hsp_raw，继续提供原有 HSPDataset、collate 和 manifest 工具。其旧格式契约不同，不作为 WindowDataset 的发布 Reader；训练 CLI 通过显式 `--dataset hsp_raw` 支持旧实验。
