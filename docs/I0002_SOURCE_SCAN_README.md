# HSP I0002 源数据扫描规范

状态：**草案，等待确认；全量扫描尚未开始。**

本目录用于保存重新清洗 I0002 之前的只读源数据画像。扫描先回答实际有哪些文件和通道、不同设备使用了哪些采样率和滤波参数、各类配置覆盖多少夜、记录与标签时长是否一致，以及后续伪迹阈值应如何分层制定。

扫描阶段不得生成 signal shard，不得修改 H5、EDF、TSV、JSON 或 CSV，不得执行新的生理频带滤波。

## 1. 输入

### 本地 HSP 数据

```text
I:\HSP\I0002\<subject>\<session>\eeg\
  <subject>_<session>.h5
  <subject>_<session>_task-PSG_channels.tsv
  <subject>_<session>_task-PSG_eeg.json
  <subject>_<session>_task-psg_sleep_annotations.csv
  <subject>_<session>_task-psg_events_annotations.csv
```

本地 H5 是头信息扫描的 H5 来源。

### 网络原始数据

```text
\\172.16.6.5\sleep\HSP\I0002\<subject>\<session>\eeg\
  <subject>_<session>_task-PSG_eeg.edf
  <subject>_<session>.h5
```

网络 EDF 是原始 EDF 头信息来源。网络 H5只用于文件存在性和本地副本一致性检查，不在头信息阶段重复读取完整信号。

### PSG 元数据

```text
I:\HSP\psg-metadata\I0002_psg_metadata_2025-09-08.csv
```

元数据是 session 清单的唯一入口。目录中存在但 metadata 未登记的文件另列为孤立文件，不能自动加入数据集。

## 2. 文件选择规则

1. 人工睡眠分期只接受标准 `*_task-psg_sleep_annotations.csv`。
2. 文件名含 `caisr` 的文件完全不进入扫描：不枚举、不读取属性或内容、不统计，也不生成 finding。
3. 文件缺失时可以列出候选文件，但不能静默选择第一个候选。
4. ECG/EKG、CHIN/Chin1-Chin2、SaO2/SpO2等别名保留原始名称和映射依据。
5. Airflow、NPT、PTAF、THERM只作为映射候选；没有元数据和波形证据前不能假定为同一路信号。
6. H5、EDF、TSV的参数分别保存；任何一方不能预先指定为绝对真值。

## 3. 扫描阶段

### 阶段 A：全量头信息扫描

覆盖 metadata 中全部 session，只读取：

- metadata 行；
- 文件名、大小、修改时间和头摘要；
- H5根属性及 `/signals` dataset 的shape、dtype、attrs；
- EDF固定头和逐通道头；
- channels.tsv和BIDS EEG JSON；
- 人工注释CSV的列名、行数和epoch索引；

阶段 A 不读取整夜信号样本。

### 阶段 B：分层波形预扫描

阶段 A 完成后，按通道、采样率、filter profile、记录年份和异常类型分层抽样。罕见profile全部进入预扫描；常见profile从正常、边界和异常时长中抽样。

阶段 B 分块读取波形，用于确定伪迹阈值和 H5/EDF 转换关系，不生成最终训练 mask。

### 阶段 C：规则确认后的全量 QC

只有阶段 A、B报告确认后，才执行全量30秒QC、重采样、signal shard和task打包。阶段C不属于本次源数据画像扫描。

## 4. 阶段 A 字段

### `sessions.parquet`

每个 metadata session 一行：

```text
metadata_index
patient_id
subject_id
session_id
session_key
recording_start
recording_end
recording_year
metadata_duration_seconds
session_count_per_patient
metadata_duplicate
metadata_row_hash
local_eeg_directory_exists
network_eeg_directory_exists
h5_exists
edf_exists
channels_tsv_exists
eeg_json_exists
manual_sleep_csv_exists
h5_subject_id
h5_session_id
h5_duration_seconds
edf_duration_seconds
annotation_duration_seconds
scan_status
scan_error
```

### `files.parquet`

每个相关文件一行：

```text
session_key
file_role
source_location
path
exists
selection_status
selection_reason
size_bytes
mtime_ns
header_sha256
read_status
error_type
error_message
```

文件角色至少包括：`local_h5`、`network_h5`、`network_edf`、`channels_tsv`、`eeg_json`、`manual_sleep_csv`、`events_csv`、`events_json`。

完整文件 SHA256 会读取全部数据，不在头信息阶段默认执行。正式处理信号时再生成所选数据的流式摘要。

### `channels.parquet`

每个session、来源、原始通道一行：

```text
session_key
source
raw_channel_name
normalized_channel_name
declared_channel_type
canonical_candidate
mapping_status
mapping_confidence
selected_for_15ch
selection_reason
alias_priority
transducer
signal_type
dtype
ndim
sample_count
samples_per_record
fs_hz
channel_duration_seconds
unit_raw
canonical_unit
unit_scale_factor
value_domain
```

扫描所有原始通道；15路目标通道只是其中一个视图。

### `calibration_profiles.parquet`

```text
session_key
source
raw_channel_name
canonical_candidate
dig_min
dig_max
phys_min
phys_max
calibration_slope
calibration_intercept
unit_raw
canonical_unit
unit_scale_factor
calibrated_phys_min
calibrated_phys_max
calibration_status
calibration_finding
```

检查数字和物理端点、单位换算、SpO2的fraction/percent/规范化状态，以及H5与EDF端点差异是否能由单位转换解释。原始端点不相等只能记为difference，不能直接记为错误。

### `filter_profiles.parquet`

每夜、每通道、每来源一行：

```text
session_key
source
raw_channel_name
canonical_candidate
selected_for_15ch
prefilter_raw
hp_hz
lp_hz
notch_hz
time_constant_seconds
filter_parse_status
filter_definition_known
filter_profile_id
h5_vs_edf_status
h5_vs_tsv_status
hp_difference_hz
lp_difference_hz
finding_severity
```

规则：

- 0.30、0.32、0.531等数值分别统计，不提前归并；
- 原始字符串必须保留；
- cutoff定义、阶数、滚降和相位未知时明确记为unknown；
- notch只统计，不作为纳入限制；
- filter差异不修改信号质量mask。

### `sampling_profiles.parquet`

```text
session_key
source
canonical_candidate
source_fs_hz
source_nyquist_hz
declared_lp_hz
target_fs_hz
sampling_action
resampling_ratio
needs_antialias
declared_lp_vs_target_nyquist
```

`sampling_action`取 `none`、`upsample`、`downsample`、`unsupported` 或 `unknown`。

### `duration_profiles.parquet`

```text
session_key
metadata_duration_seconds
h5_duration_seconds
edf_duration_seconds
annotation_duration_seconds
selected_channel_min_duration
selected_channel_max_duration
h5_minus_metadata
edf_minus_metadata
h5_minus_edf
annotation_minus_signal
partial_last_epoch_seconds
edf_continuity_type
duration_status
```

### `annotation_profiles.parquet`

```text
session_key
annotation_source
selection_status
column_names
row_count
epoch_min
epoch_max
duplicate_epoch_count
conflicting_epoch_count
missing_epoch_count
stage_token_counts
invalid_stage_token_count
sao2_columns_present
heart_rate_columns_present
annotation_duration_seconds
```

### `findings.parquet`

```text
session_key
scope
source
channel
severity
code
detail
evidence_fields
```

严重度为：

- `INFO`：覆盖率或可解释差异；
- `WARNING`：需要规则确认或波形复核；
- `BLOCKING`：文件不可读、身份冲突、结构损坏等确定问题。

通道缺失、filter不同和TSV差异不能自动升级为BLOCKING。

## 5. 阶段 B 波形指标

每个30秒、每通道至少计算：

```text
finite_fraction
mean
median
std
mad
minimum
maximum
peak_to_peak
unique_value_count
zero_fraction
longest_constant_run_seconds
adc_low_rail_fraction
adc_high_rail_fraction
repeated_min_fraction
repeated_max_fraction
max_absolute_derivative
robust_derivative_zmax
step_count
impulse_count
linear_trend_slope
line_power_fraction_50hz
line_power_fraction_60hz
high_frequency_fraction
modality_band_powers
rule_evaluated_bits
```

SpO2额外计算：

```text
fraction_50_100
fraction_50_110
fraction_below_5
fraction_above_100
maximum_change_per_second
dropout_run_seconds
constant_run_seconds
```

跨通道指标：

```text
exact_duplicate
affine_duplicate
zero_lag_correlation
lagged_correlation
shared_reference_suspected
ecg_leakage_score
left_right_swap_suspected
```

扩展指标在人工核验阈值以前只作为finding，不直接排除epoch。

## 6. 最终统计

### 数据规模

- metadata记录数；
- patient、subject、session数量；
- 每位patient的session数分布；
- 记录年份分布；
- 总H5/EDF字节数。

### 文件完整性

- H5、EDF、TSV、JSON、人工分期CSV的存在夜数和占比；
- 本地H5与网络H5的一致/不同/未知数量；
- 重复、孤立、命名异常和不可读文件数量；
- BLOCKING/WARNING/INFO数量。

### 记录时长

metadata、H5、EDF、注释及目标通道分别报告：

```text
N, min, P1, P5, P25, median, mean, P75, P95, P99, max
```

分桶统计：`<4h`、`4–5h`、`5–8h`、`8–12h`、`>12h`。另列出H5/EDF/metadata相差超过1秒、30秒和一个epoch的记录数。

### 通道覆盖率

- 所有原始通道名称及夜数；
- 15个目标通道各自的存在夜数和占比；
- 每种别名和映射路径的夜数；
- 同一目标通道存在多个候选的夜数；
- 每晚具备0–15个目标通道的分布；
- 各模态类别的候选覆盖率。

### Filter分布

- 每种原始prefilter字符串和HP/LP/notch组合；
- 每种组合的夜数；
- 占全部session比例；
- 占该通道存在session比例；
- 采样率×filter profile联合分布；
- 年份×filter profile联合分布；
- H5与EDF完全一致、数值不同、无法比较的数量；
- H5与TSV一致、不同、无法比较的数量；
- 0.30、0.32、0.531等HPF的独立占比。

### 采样率与重采样需求

- 每个目标通道的原始采样率分布；
- 不变、上采样、下采样和不支持的夜数；
- 各重采样比例的夜数；
- 需要抗混叠处理的夜数；
- LP100 Hz与200 Hz目标奈奎斯特边界相关的夜数。

### 单位与标定

- dtype和单位分布；
- 数字/物理端点profile分布；
- 能明确转换到µV/%的夜数；
- 单位未知、端点退化和校准冲突数量；
- H5与EDF仅单位换算不同的数量；
- SpO2 percent/fraction/规范化/未知数量。

### 标签结构

- 人工分期CSV覆盖率及stage token分布；
- 缺失、重复和冲突epoch数量；
- HR、SaO2字段覆盖率；
- 标签比信号长/短的记录数和时长分布；

### 分层波形预扫描

- 每种filter、采样率和设备profile的波形指标分布；
- HSP基础伪迹模式的候选触发率；
- 阶跃、冻结、削顶、重复通道等finding比例；
- 各候选阈值下排除率的敏感性分析；
- 需要人工复核的正常、边界和异常样例。

## 7. 比例口径

Filter和通道统计同时报告：

```text
percent_all = profile_nights / metadata_total_nights
percent_present = profile_nights / channel_present_nights
```

缺失通道不进入该通道filter profile的 `percent_present` 分母。每夜同一路通道只计一次，不按epoch数或记录长度加权。

## 8. 结果目录

```text
I:\HSP\I0002-preprocess\
├── README.md
└── source_scan\v1.0.0\
    ├── manifest.json
    ├── schema.json
    ├── config\scan_spec.json
    ├── logs\progress.json
    ├── logs\source_scan_<run_id>.jsonl
    ├── checkpoints\
    ├── tables\sessions.parquet
    ├── tables\files.parquet
    ├── tables\channels.parquet
    ├── tables\channel_mapping_candidates.parquet
    ├── tables\calibration_profiles.parquet
    ├── tables\filter_profiles.parquet
    ├── tables\sampling_profiles.parquet
    ├── tables\duration_profiles.parquet
    ├── tables\annotation_profiles.parquet
    ├── tables\findings.parquet
    ├── reports\summary.json
    ├── reports\report.md
    ├── reports\channel_coverage.csv
    ├── reports\filter_distribution.csv
    ├── reports\sampling_distribution.csv
    ├── reports\duration_distribution.csv
    ├── reports\calibration_distribution.csv
    └── reports\finding_counts.csv
```

Parquet保存完整明细；CSV和Markdown保存便于人工检查的汇总。扫描未完成时 `manifest.status` 为 `building`、`stopped`或`failed`；只有全量完成且表间数量核对通过后才写 `completed`。

## 9. 验收条件

1. metadata中的每条记录都有session结果，包括失败记录；
2. 所有表可以通过 `session_key` 关联；
3. 所有比例同时提供分子和分母；
4. 未知值保留为unknown，不用0代替；
5. 扫描器不访问、不统计文件名含 `caisr` 的文件；
6. 阶段A没有读取信号dataset内容；
7. 源文件没有被修改；
8. 汇总计数可以从Parquet明细重算；
9. 未经确认不启动阶段B或清洗；
10. 不生成任何signal shard。

## 10. 当前状态

- 已验证本地H5路径和网络EDF路径均可访问。
- 100夜技术试扫只用于发现字段和映射问题，不作为正式统计。
- 正式全量头信息扫描已于 2026-09-04 11:10（Asia/Shanghai）启动。
- 并发配置为8个进程、每进程2个线程，最多同时扫描16个session。
- 实时进度见 `source_scan\v1.0.0\progress.json`，逐session日志见 `source_scan\v1.0.0\scan.jsonl`。
- 阶段B波形预扫描、阶段C清洗和signal shard生成均未启动。
