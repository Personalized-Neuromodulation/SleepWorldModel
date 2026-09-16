# 通用读取接口

`dataloader` 是独立包，不导入模型或预处理实现。`WindowDataset, collate_windows` 返回下述读取字典；`as_signal_batch` 在数据侧将其转换成 backbone 所需的 SignalBatch，训练目标由外部任务管理。

## SignalBatch adapter

GPU 训练的 `collate_signal_windows` 在 CPU 完成 collate、SignalBatch 适配和校验，返回 `signal_batch` 及可选 probe 使用的 `tasks`。DataLoader 通过 `SignalBatch/SignalGroup.pin_memory()` 固定适配后的张量（包括新生成的 data_valid），再由训练入口异步传输；启用 num_workers 时适配在 worker 内完成。复用现有 SignalBatch 数据契约。

`from dataloader import as_signal_batch`；调用 `as_signal_batch(raw_batch)` 将 respiratory 中的 spo2 拆为独立组，同时切分 QC、通道身份和单位。`split_spo2=False` 保留原分组；其他数据集组默认原样保留。

`scales={"eeg": scale, ...}` 可按固定训练集尺度除以信号，单位同步标记；不在 batch 中拟合统计、不修改源 tensor。返回值提供 `.validate()` 和 `.to(device)`，包含可选 `night_grade[B]`。collate 要求整批都有 night_grade 或整批都没有；HSP 发布读取器仍要求等级存在且有效。完整主干契约见 [实现规格](../docs/PSG_ARCHITECTURE_PROPOSAL.md)。

`scales` 也接受固定仿射变换，例如 `{"spo2": {"offset": 95.0, "divisor": 5.0}}` 表示对原始百分数执行 `(x−95)/5`，输入单位记为 `(%-95)/5`。数字写法仍只表示除法，与旧导出兼容。适配在 CPU collation 执行一次，不按 batch/epoch 拟合统计，不改变 QC、原始数据或任务标签；同一配置用于训练、评价和单步调试。仅支持按编码组配置，本次不扩展 respiratory 的通道缩放。

## Reader 扩展点

Foundation训练使用 `context_epochs=1`。`WindowDataset(..., night_grades=(3,4,5))`
可按整夜等级筛选；`NightGradeBatchSampler`先等概率选等级，再在该等级中均匀抽窗口，
保证单个batch内等级一致，不创建全数据集的窗口索引张量。subject划分仍先由稳定哈希确定。

`SignalBatch.data_valid` 返回每组 `[B,C,E]` 的来源epoch QC，包含epoch padding，
不包含channel dropout。`view_start_samples[B]`记录local在来源epoch内的整秒裁剪起点；
裁剪后保留 `epoch_in_record / epoch_start_offset_ns` 及原始QC。
`.to(device, non_blocking=True)`支持异步GPU搬运。

`dataloader.public.from_modality_tensors`仅适配已处理的 `[B,C,N,200]`
与epoch QC，不读取文件或重新生成QC，供HF公开接口调用。

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

适配到模型时，在CPU调用 `as_signal_batch(raw, foundation=True, input_spec=...)` 完成一次校验；首次构建模型可省略 `input_spec`。它检查epoch QC组合、有效区有限值及foundation的200 Hz、1–30整秒、通道身份、单位和统一时间网格。之后传入backbone的SignalBatch被视为已验证数据，不在GPU forward重复扫描。直接构造或修改SignalBatch时由调用方在数据边界调用 `validate(...)`；crop/dropout仅做已验证数据的裁剪和mask变换，不重新执行QC。HF公开tensor同样由 `dataloader.public.from_modality_tensors` 适配与校验。

### 提供给模型的 mask

- `SignalGroup.data_valid [B,C,E]`：适配时保存 `group.valid & epoch_mask`，包含来源 QC 与 epoch 存在性，不包含 dropout。`SignalBatch.data_valid` 返回这些张量的引用。
- `SignalGroup.visible [B,C,E]`：由当前 `channel_mask [B,C]` 广播；初始来自读取结果，pretraining 的 view sampler 可以继续 dropout。
- crop 保留来源 epoch 的 data_valid，不重新判定局部秒级 QC；拼接 views 时 mask 随 B 轴拼接。
- 自行构造 SignalGroup 时必须提供 data_valid；若修改源 QC、epoch 存在性或通道布局，应在数据边界同步更新并 validate，不能只修改源字段后直接送入模型。

## 当前 HSP 发布格式

根目录为 `I:\HSP\I0002-preprocess\processed\v1.0.0`，通过版本描述、records/shards manifest 和逐 shard commit 读取已发布文件。支持旧内部 schema 1.0.0-pilot.2 与恢复后的 1.0.0-full.1。

读取后仍为 eeg、eog、ecg、emg、respiratory 五组，15 路、200 Hz、30 秒 epoch；respiratory 顺序为 airflow、snore、spo2。QC 不重新计算，标签不重新标注，发布文件不改写。默认按窗口切片读取和有界句柄缓存，不全量加载多 GiB 分片。

`tasks=()` 默认不读取任务；可以指定 sleep_stage、heart_rate、sao2。`verify_checksums=True` 会在每次重新打开文件时计算完整 SHA256，默认只检查提交摘要一致性、文件大小、绑定及窗口索引。

旧原始 H5 工具位于 readers/hsp_raw，不属于当前 foundation 训练输入路径。

## HDD + 128 GB RAM 训练读取

`buffered.py::BufferedWindowLoader` 复用 HSP Reader、collator 和 SignalBatch。
默认只选 Outstanding (`night_grades: [5]`)，先完成 subject split，再建立读取计划。

```text
shard 内连续读取（最多 1024 epochs / read_window）
    → RAM 当前池 + RAM 预读池（合计最多 64 GiB）
    → 池内随机排列 epoch → collate / CPU QC → pin batch → GPU
```

- 一个后台线程独占自己的 Reader；主线程组 batch。`training.num_workers=0`，不会生成多份大缓存。
- 两个池各最多 32 GiB，包含正在加载的池。只 pin 当前 batch，不把整个缓存锁页。
- 每个池从最多 8 个同时活跃的 shard 轮流选块，再按 shard/物理 epoch 顺序读取。耗尽的 shard 会由后续 shard 补充，因此大池可覆盖超过 8 个 shard。
- 池内混洗跨记录的 epoch，同一 batch 保持 night_grade 一致。一个完整数据遍历内各窗口恰好出现一次，池尾允许小 batch；下一遍重新打乱 shard 顺序。不在小缓存内无限重复采样。
- 第一个池最多 8 块，降低启动等待；以后扩大到预算上限。小数据集自然用更少 RAM。
- 配置为 `data.ram_pool`：`cache_gib` 是两个池总预算，`reserve_gib` 在启动时从可用内存中保留至少 24 GiB，必要时缩小预算。64 GiB 是 tensor payload 上限，不是进程 RSS 上限；Python 元数据、HDF5 cache、临时 batch 和其他进程需另外预留。
- 当前 15 路、200 Hz、30 s float32 波形约 0.343 MiB/epoch；64 GiB 双池约容纳 18.8 万 epochs（含 QC/标签预留），远小于全量 Outstanding 数据，必须持续轮换。
- 读取错误会传回主线程，提前停止会等待当前块读完并关闭 Reader、释放两个池。读取不改写数据文件。
- 验证仍使用固定 subject split 子集及独立 Reader，默认 `evaluation.num_workers=0`；本次没有把整个验证集缓存到 RAM。评价期间后台最多填满一个预读池后停止。

W&B 每步新增 `train/input_{budget_bytes,resident_bytes,peak_bytes,blocks_read,epochs_read,read_seconds,wait_seconds}`。
`wait_seconds` 仅统计等待池就绪的累计时间，不含 collation 和 GPU 时间；判断是否仍受 HDD 限制时看其增量。
第一次预热及池切换仍可能等待硬盘，不能把 RAM 命中速度当成全程训练吞吐。

通过 `tests/test_pipeline.py` 可逐步进入同一个 RAM 读取路径；交互调试只预热一个 batch，避免为了单步调试加载几十 GiB。
`tests/test_buffered.py` 检查读取结果、QC/标签、覆盖率、内存上限、异常退出及 CUDA pin/transfer。

2026-09-12 小规模真实 Outstanding 输入短测（没有训练更新）：同一记录预热后，32 epochs 连续读取中位数 26 ms，逐条读取 336 ms；RAM 已就绪时 batch=32 的取数/组批耗时 13–35 ms。第一次未控制缓存状态的连续读取耗时 17.6 s。这不是冷盘公平基准，也不代表训练加速倍数；尚未进行 64 GiB 满池长时间吞吐测试。原始记录保存在 `results/foundation/ram_pool_benchmark_20260912.json`。
