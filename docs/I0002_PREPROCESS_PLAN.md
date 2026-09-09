# HSP I0002 v1.0.0 清洗与 QC 方案

状态：**新版pilot自动验收通过，用户已确认测试通过并授权全量清洗；按获批的QC/PSD核心启动v1.0.0全量任务。旧pilot只保留为历史结果。**
日期：2026-09-05
方案与源扫描目录：`I:\HSP\I0002-preprocess`
预处理产物根目录：`I:\HSP\I0002-preprocess\processed\v1.0.0`

## 1. 范围与原则

- 原始H5：`I:\HSP\I0002`，只读，是波形、标定和prefilter的主来源。
- EDF：`\\172.16.6.5\sleep\HSP\I0002`，只读，仅用于通道、采样率、时长、标定和prefilter审计，不回填H5。
- 不读取、检查或使用文件名含 `caisr` 的文件。
- 仅发布 `v1.0.0`；15路均输出为200 Hz、30秒、float32物理值，不做任何输入normalization。
- 存储层将Airflow、Snore、SpO2合并到同一个 `respiratory` dataset，固定顺序为 `Airflow,Snore,SpO2`；三路不混合数值，仍各自保留mask和QC。
- QC发布粒度只有30秒；不保存1秒或逐样本质量mask。
- 来源HP/LP明确时只记录，不重滤；仅 `SOURCE_FILTER_UNKNOWN` 才补滤。Notch只记录，不要求一致，也不新增陷波。
- filter finding和处理状态按“整夜/通道”保存；滤波在整夜内的连续可处理区间执行，不逐30秒epoch单独滤波。
- `v1.0.0` 不进行跨通道重复波形扫描，不计算跨通道相关性或逐样本相等率。
- QC audit只允许冻结的hard code；不生成warning、candidate或自由文本epoch描述。通道availability、滤波和重采样属于record/channel处理状态，不得伪装成artifact。
- signal与task分开；每个task独立H5，并使用相同的 `recording_id + epoch_in_record + epoch_start_offset_ns` 对齐。
- 本方案只定义通用信号清洗、QC和数据发布，不定义任何训练目标、loss、训练划分、loader或模型配置。发布数据保持split-neutral。

固定排除以下3个session；它们只进入排除台账，不进入signal、task或整夜等级：

| session_key | H5时长 | EDF时长 | 原因 |
|---|---:|---:|---|
| `sub-I0002150026681/ses-2` | 79 s | 27,899 s | `SOURCE_DURATION_CONFLICT` |
| `sub-I0002150030931/ses-1` | 15,447 s | 26,578 s | `SOURCE_DURATION_CONFLICT` |
| `sub-I0002150035292/ses-1` | 22 s | 29,730 s | `SOURCE_DURATION_CONFLICT` |

源扫描共15,031夜；固定排除后候选15,028夜。只有H5整体不可读、record身份/起点/时长无法建立、时长不足30秒等record级错误才排除整夜。单通道缺失或失败只影响该通道。

## 2. 通道与处理目标

固定逻辑顺序：

`F3-M2,F4-M1,C3-M2,C4-M1,O1-M2,O2-M1,E1,E2,ECG,Chin1-Chin2,LAT,RAT,Airflow,Snore,SpO2`

| 索引 | 模态 | 通道 | 单位 | 明确来源筛选目标 | UNKNOWN补滤 |
|---:|---|---|---|---|---|
| 0–5 | EEG | F3-M2、F4-M1、C3-M2、C4-M1、O1-M2、O2-M1 | µV | 0.3–35 Hz | 0.3–35 Hz |
| 6–7 | EOG | E1、E2 | µV | 0.3–35 Hz | 0.3–35 Hz |
| 8 | ECG | ECG | µV | 0.3–70 Hz | 0.3–70 Hz |
| 9–11 | EMG | Chin1-Chin2、LAT、RAT | µV | 10–100 Hz | 源200 Hz：10–95 Hz；源500 Hz：10–100 Hz |
| 12 | Respiratory | Airflow | µV | 0.159–15 Hz | 源200/500 Hz：0.159–15 Hz |
| 13 | Respiratory | Snore | H5来源物理单位 | 10–100 Hz | 源200 Hz：10–95 Hz；源500 Hz：10–100 Hz |
| 14 | Respiratory（存储） | SpO2 | % | 不要求HP/LP | 不补滤 |

Airflow只允许H5 `/signals/airflow`；缺失时写 `CHANNEL_MISSING`，不得用 `ptaf/npt/cflow/therm` 或EDF替代。EEG/EOG保留来源参考，不重新参考。SpO2并入respiratory只改变物理存储位置，不改变其单位、重采样、专属QC或整夜等级中的oximetry语义。

## 3. Record网格、标定与通道门槛

### 3.1 唯一时间网格

- 仅使用有限且满足 `0 < duration_sec <= 72 h` 的H5根属性 `duration_sec` 建立record网格；缺失或无效则排除整夜。
- `N_record=floor(duration_sec/30)`；不再使用旧流程的“root时长与最长通道取max后ceil”。尾部不足30秒不发布。
- epoch `k` 为 `[30k,30(k+1))`，输出必须正好6000点。
- 通道覆盖按其 `sample_count/source_fs_hz` 核验，浮点比较只允许最多0.5个原生采样点误差。超出record的尾部丢弃；覆盖不足的epoch仅令该通道 `coverage_valid=false`，不另生成epoch描述或hard code。
- 没有显式逐样本时间戳时只能核对长度，不能声称检测到隐藏采集缺口；统一保存 `records/channel_processing/coverage_detection`，枚举为 `LENGTH_ONLY/TIMESTAMP_CHECKED/NOT_EVALUATED`。

### 3.2 数字码转物理值

`physical = phys_min + (digital-dig_min)*(phys_max-phys_min)/(dig_max-dig_min)`

四个属性必须有限，且 `dig_max>dig_min`、`phys_max>phys_min`。转换使用float64中间量，验收后写float32。单位缺失或无法解析时，该夜该通道写 `CALIBRATION_OR_UNIT_INVALID`。数字码只用于ADC rail和Airflow数字恒定检查，不再执行freeze检查。

### 3.3 Airflow硬门槛

严格 `/signals/airflow` 在15,031夜中存在13,961夜；固定排除后存在13,959夜。

| phys范围 | unit | 全部夜数 | 候选夜数 | 处理 |
|---|---|---:|---:|---|
| `-3200/3200` | uV | 13,952 | 13,950 | 继续处理 |
| `-12003000/12003000` | uV | 7 | 7 | Airflow整夜通道无效 |
| `-328000000/328000000` | uV | 1 | 1 | Airflow整夜通道无效 |
| `0/0` | 空 | 1 | 1 | Airflow整夜通道无效 |

只有 `phys_min=-3200`、`phys_max=3200`、`unit=uV` 通过。其他或未来未审核profile均fail closed：该夜Airflow的 `channel_available=false`、所有epoch `valid=false`、signal填0；原因只存record/channel状态，不写epoch hard code。同夜其他14路继续。

### 3.4 SpO2硬门槛

| phys范围 | unit | 全部夜数 | 候选夜数 | 处理 |
|---|---|---:|---:|---|
| `0/100` | % | 14,871 | 14,868 | 继续处理 |
| `-328/328` | % | 158 | 158 | SpO2整夜通道无效 |
| `-12/12` | % | 2 | 2 | SpO2整夜通道无效 |

只有 `0/100%` 通过。其余160夜SpO2的 `channel_available=false`、所有epoch `valid=false`、signal填0；状态不写epoch hard code，同夜其他14路继续。通过profile后仍须执行SpO2专属30秒QC。

未通过Airflow/SpO2 profile时，不读取波形做伪迹、补滤或重采样；只保留来源元数据，并写：

- `effective_filter_state=PROCESSING_SKIPPED_PROFILE_EXCLUDED`
- `filter_reapply_result=NOT_RUN_PROFILE_EXCLUDED`
- `resample_result=NOT_RUN_PROFILE_EXCLUDED`
- `channel_usable_hours=0`

## 4. 滤波与重采样

### 4.1 状态

- `source_filter_state`：`NOT_EVALUATED`、`SOURCE_FILTER_EXPLICIT`、`SOURCE_FILTER_UNKNOWN`，不可覆盖来源事实。
- `effective_filter_state`：`SOURCE_FILTER_AS_RECORDED`、`SOURCE_FILTER_REAPPLY`、`FILTER_NOT_REQUIRED`、`REAPPLY_FAILED`、`PROCESSING_SKIPPED_PROFILE_EXCLUDED`。
- `SOURCE_FILTER_UNKNOWN`：需要滤波的通道H5 HP/LP缺失、不可解析或声明为0/0。H5/TSV或H5/EDF冲突只记finding，以H5为准，不触发补滤。
- 明确来源与目标的关系记录为 `MATCH/SOURCE_WIDER/SOURCE_NARROWER/MIXED/UNKNOWN/NOT_REQUIRED`，不直接改变mask。
- 上述值只存 `records/channel_processing/* [R,15]`；它们不是artifact，不得进入epoch `hard_code` 或 `hard_flags`。

### 4.2 UNKNOWN补滤

- HP与LP分别使用Butterworth 4阶、SOS、双向零相位；profile为 `HP4_LP4_SOS_ZEROPHASE`。
- 200 Hz EMG/Snore必须使用10–95 Hz；500 Hz使用10–100 Hz后再降到200 Hz。不得静默把100 Hz裁成其他值。
- Airflow 200/500 Hz使用0.159–15 Hz。严格Airflow的8个25 Hz记录均已被量程门槛排除。
- 补滤按整夜通道执行，只在真实缺失/NaN处断开；不得因artifact hard code把连续信号切成多个滤波段，否则会人为增加边界并丢失邻近epoch。
- 使用反射padding。脉冲响应衰减至峰值 `1e-4` 的guard只作为record/channel处理元数据，不生成 `PROCESSING_BOUNDARY`，也不直接排除30秒epoch。边缘若确实异常，只能由第5节定义的hard code触发。
- 短到无法稳定执行双向滤波的有限连续段写 `processing_valid=false`；它不是artifact，不写hard code。同夜其他可处理区间继续。
- 补滤前指标可进入运行日志用于比较；发布波形QC在补零前的200 Hz候选输出上判定，原生数字码检查及来源NaN证据按第5节映射到同一epoch。不能发布 `SOURCE_ARTIFACT_CORRECTED` 等额外epoch标签。

### 4.3 重采样

- EEG/EOG/ECG/EMG/Airflow/Snore：冻结Kaiser FIR的有理数polyphase；500→200使用 `up=2,down=5`，200→200不重复重采样。
- SpO2以previous-sample hold输出到200 Hz；不跨真实缺口、不外推。
- 保存profile、输入/输出样本数、比例和验收结果。滤波或重采样失败只改变 `processing_valid` 和record/channel状态，不产生artifact code。
- 被profile排除的通道不执行重采样；shard中的6000点仅为0占位，不是重采样输出。

## 5. 30秒QC

### 5.1 HSP依据与适用阶段

HSP 3.0的30秒SQA定义六类artifact：flat-line、saturation、high amplitude、50/60 Hz power-line interference、>70 Hz high-frequency noise和NaN；SpO2另有range检查，近零SpO2作为整通道availability处理。依据：[HSP v3.0 Signal Quality Assessment](https://bdsp.io/content/hsp/3.0/)。HSP未公开class-specific flat/high-amplitude的全部数值，本项目阈值须在pilot人工复核后冻结，不能声称为HSP官方阈值。

- 波形hard QC在200 Hz物理值候选输出上执行一次，必须在非有限值填0之前取证；来源及候选输出中的NaN/Inf保留映射到输出epoch的证据，不能因填0而消失。未形成候选输出的处理异常按processing失败记录。
- saturation和Airflow数字恒定检查使用同一30秒区间的原生数字码，分别核对ADC rail及数字码差，不对重采样物理值做数字码判定。
- 每条规则是否实际执行写QC shard根数据集 `/hard_evaluated_flags`；未执行或不适用都不是通过。规则对通道的静态适用关系由 `config/qc.json` 的 `hard_applicable bool[7,15]` 给出，不逐epoch重复存储。
- 不再执行或发布数字冻结、fixed-tone、hold、rapid-change、rail-touch、低活动、滤波边界或任何warning/candidate。

### 5.2 唯一hard code表

| hard code | 名称 | 适用通道 | 30秒条件 |
|---:|---|---|---|
| 0 | `NO_HARD_ARTIFACT_DETECTED` | 全部 | 已执行规则均未触发；仍须结合三层mask判断是否可用 |
| 1 | `NAN` | 全部 | 完整覆盖区间中，来源NaN或Inf数 `>0`，或补零前200 Hz候选输出的NaN或Inf数 `>0`；缺通道/缺时间段占位不冒充NAN |
| 2 | `FLAT_LINE` | EEG/EOG | 标准差 `<0.5 µV` |
| 2 | `FLAT_LINE` | ECG | 标准差 `<5 µV` |
| 2 | `FLAT_LINE` | Airflow | 整个epoch原生数字码差 `max(digital)-min(digital)<=2` 且不同数字code不超过2个 |
| 3 | `SATURATION` | EEG/EOG/ECG/EMG | 上下ADC rail并集占比 `>=5%` |
| 4 | `HIGH_AMPLITUDE` | EEG | 30个连续、互不重叠的1秒子窗峰峰值均 `>1000 µV`，即持续30秒 |
| 4 | `HIGH_AMPLITUDE` | EOG | 30个连续、互不重叠的1秒子窗峰峰值均 `>2000 µV`，即持续30秒 |
| 4 | `HIGH_AMPLITUDE` | ECG | 30个连续、互不重叠的1秒子窗峰峰值均 `>10000 µV`，即持续30秒 |
| 4 | `HIGH_AMPLITUDE` | EMG | 30个连续、互不重叠的1秒子窗峰峰值均 `>5000 µV`，即持续30秒 |
| 5 | `POWER_LINE_INTERFERENCE` | EEG/EOG/ECG/EMG | `(49–51 Hz + 59–61 Hz)/(0.5 Hz–Nyquist)`功率 `>40%` |
| 6 | `HIGH_FREQUENCY_NOISE` | EEG/EOG/ECG | `>70 Hz/(0.5 Hz–Nyquist)`功率 `>50%` |
| 7 | `SPO2_OUT_OF_RANGE` | SpO2 | `[50,110]%`内有限样本占比 `<70%` |

`1 LSB=(phys_max-phys_min)/(dig_max-dig_min)`。rail定义为 `digital<=dig_min+1` 或 `digital>=dig_max-1`。去除旧版连续rail `>=0.5 s`附加条件，严格使用HSP的5% epoch比例。`HIGH_AMPLITUDE` 的持续时间指标 `high_amplitude_duration_seconds` 为30个1秒子窗中超过阈值的子窗数，取值0–30；仅当其等于30时触发hard code。这是1秒分辨率的超阈值子窗覆盖量，并非逐样本测得的连续高幅时间；每秒一个短尖峰也可能满足，用户已接受此局限；测试须固定该情形仍按规则命中，不再作为待确认项。这里的1秒窗只是30秒判定的内部统计，不发布1秒QC或mask；也不要求每个样本的绝对值均超过阈值，因为正常过零的双极生理信号无法满足该定义。200 Hz输出可按处理chunk直接reshape为 `[B,C,30,200]`，以 `max(axis=-1)-min(axis=-1)` 一次得到全部1秒峰峰值，先计算 `over=p2p>threshold`，再对布尔over用 `sum(axis=-1)` 得超阈值子窗数、用 `all(axis=-1)` 得hard结果；禁止逐epoch或逐秒Python循环。先向量化检查finite；有非有限证据的epoch触发 `NAN`，跳过其他依赖完整有限波形的规则，其evaluated位为false，不使用 `nanmax/nanmin`掩盖缺失。Saturation和high amplitude只用于biopotential，不用于Airflow、Snore或SpO2。HSP明确关闭EMG和SpO2的flat-line；本项目也关闭Snore flat-line，因为无鼾声可以是真实生理状态。Airflow采用极保守的数字恒定判据以避免把呼吸暂停误判为脱落。EMG不执行 `HIGH_FREQUENCY_NOISE`；其 `hard_evaluated_flags` 对应位必须为false，不能解释为检查通过。

### 5.3 多hard code与完整性原则

- `hard_flags` 为uint8 bitset，bit 0–6对应hard code 1–7；一个epoch可以同时触发多个hard code。
- `hard_code` 为uint8主码，仅为快速筛选；优先级固定为 `NAN > SATURATION > HIGH_AMPLITUDE > FLAT_LINE > POWER_LINE_INTERFERENCE > HIGH_FREQUENCY_NOISE > SPO2_OUT_OF_RANGE`。
- hard artifact只令 `valid=false`，不得把对应有限波形清零、截断或从shard删除；训练端决定是否使用mask。
- 缺通道、标定失败、profile拒绝、覆盖不足、滤波/重采样失败不是artifact，不得写入hard code。它们分别通过record/channel `channel_available`、epoch `coverage_valid` 和 `processing_valid` 表达。
- SpO2全通道 `<5%`占比 `>=90%` 或整夜中位数 `<5%` 时写 `channel_available=false`；不生成 `SPO2_NEAR_ZERO` epoch hard code。
- 不生成任何warning字段、warning code、candidate或自由文本epoch标签。

### 5.4 冻结的PSD计算

采用Welch功率谱密度；实现依据：[SciPy Welch](https://scipy.github.io/devdocs/reference/generated/scipy.signal.welch.html)。以下参数全部显式指定，不依赖库默认值：

- 输入为补零前、完整有限的200 Hz、6000点候选输出，使用已量化到float32的波形转float64计算。
- 周期Hann窗：`signal.windows.hann(800,sym=False)`；`nperseg=800`（4秒），`noverlap=400`（2秒），`nfft=800`。
- `detrend="constant"`（每个4秒窗去均值）、`return_onesided=true`、`scaling="density"`、`average="mean"`、`axis=-1`。
- 一个30秒epoch恰好14个重叠窗；频率为0至100 Hz、步长0.25 Hz，无额外补零。
- 频段功率统一为选中频率bin的PSD求和乘0.25；分母为 `[0.5,100] Hz`；工频分子为 `[49,51]∪[59,61] Hz`；高频分子为 `(70,100] Hz`。区间端点按上述开闭规则选择，不改用梯形积分。
- 分母等于0时，比例定义为0并标为已执行；PSD含非有限值/负分母属于计算失败，记录日志并阻止按通过发布。
- 使用float64比例比较 `line>0.40`、`HF>0.50`，再将metric写float32；重验存储metric时对阈值附近1e-7内的浮点舍入差异不作相反推断。
- 每批最多64个epoch，向量化沿末维计算；只在适用且完整有限的通道epoch上执行。EMG不执行或写入HF metric。
- 高幅值计算仍为30个1秒窗全部超阈值；每秒短尖峰亦可触发的局限已接受。
- 运行时保存NumPy/SciPy/h5py版本和当前代码摘要，便于复现。

## 6. Mask、编码与存储

### 6.1 三层mask

| mask | 形状 | 含义 |
|---|---|---|
| `records/channel_available` | bool `[R,15]` | 通道存在、可读、标定/单位/profile可接受；SpO2整夜availability也在此处理 |
| `quality/<modality>/coverage_valid` | bool `[N,C]` | 该epoch具有完整30秒来源覆盖 |
| `quality/<modality>/processing_valid` | bool `[N,C]` | 必需滤波和重采样成功，且输出为有限6000点 |
| `quality/<modality>/artifact_valid` | bool `[N,C]` | 所有适用规则已完成且 `hard_flags==0`；按第6.4节公式生成 |
| `quality/<modality>/valid` | bool `[N,C]` | 上述四项逻辑与 |

`coverage_valid[k,c]=true` 的精确定义是：通道c的来源时间轴完整覆盖该记录第k个半开区间 `[30k,30(k+1))` 秒，按源采样率应有的样本均存在，没有已知时间戳断点或数据缺口，而且重采样不需要在来源边界之外外推。它只回答“这30秒是否有完整来源覆盖”，不评价滤波、伪迹或数值是否生理合理。若源文件没有逐样本时间戳，只能依据起点、采样率和样本计数判断，并在record元数据写 `records/channel_processing/coverage_detection=LENGTH_ONLY`；这时不能声称排除了隐藏的内部丢样。发布网格只取完整30秒epoch，因此正常的末尾不足30秒尾段不进入N。整夜缺失的通道同时令 `channel_available=false`，并令其全部epoch的 `coverage_valid=false`。

不存在统一的非artifact `reason_code`。通道不可用的原因存 `records/channel_status_code [R,15]`，处理结果存 `records/channel_processing/*`；二者使用各自冻结枚举，但不进入QC audit。这样 `hard_code` 永远只代表第5.2节的artifact。

### 6.2 H5形状

一个shard含R夜、N个epoch：

| 数据 | 形状 |
|---|---|
| `signals/eeg` | float32 `[N,6,6000]` |
| `signals/eog` | float32 `[N,2,6000]` |
| `signals/ecg` | float32 `[N,1,6000]` |
| `signals/emg` | float32 `[N,3,6000]` |
| `signals/respiratory` | float32 `[N,3,6000]`，固定为Airflow、Snore、SpO2 |
| `quality/<modality>/valid` | bool `[N,C_modality]` |
| `quality/<modality>/coverage_valid`、`processing_valid`、`artifact_valid` | bool `[N,C_modality]` |
| `quality/<modality>/hard_code` | uint8 `[N,C_modality]`，仅0–7 |
| `records/*` | `[R]`或 `[R,15]` |
| `segments/record_index`、`epoch_in_record`、`epoch_start_offset_ns` | `[N]` |

每个 `signals/qc/qc-xxxxx.h5` 与同编号signal shard一一对应，并保存根数据集 `/hard_flags uint8[N,15]`、`/hard_code uint8[N,15]`、`/hard_evaluated_flags uint8[N,15]` 和冻结的 `/metrics/*`。两个flags数据集均为bitset：bit 0–6分别对应hard code 1–7；`hard_flags` 表示规则触发，`hard_evaluated_flags` 表示规则在该epoch/通道确实完成计算。静态 `hard_applicable=false` 且evaluated bit为0表示“不适用”；静态applicable=true但evaluated bit为0表示 `NOT_EVALUATED`，必须结合availability、coverage、processing状态解释，不能视为通过。metrics按指标分别存 `[N,15]`，不使用含义不明的统一M维。QC audit不存在warning或处理状态描述。signal shard不重复保存合并的 `[N,15,6000]`，也不保存逐样本或1秒mask；训练常规读取不需要打开 `/hard_evaluated_flags`。

record/channel至少保存：来源存在、源名、源fs、样本数、单位、dig/phys量程、标定profile与有效性、来源/effective filter状态、HP/LP/notch、filter relation/result、resample method/result、usable hours。record保存 `recording_id/subject_id/session_id/start_time_raw/timezone_known/first_epoch/n_epochs/night_grade`。

缺失或无法生成有限波形的位置填 `0.0`；各mask按各自原因独立记录：缺通道令availability/coverage/processing均false；profile拒绝令availability/processing为false但coverage仍可由元数据确定；短通道令受影响epoch的coverage/processing为false；处理失败令processing为false，不能连带篡改availability或coverage。非有限值的证据先记入QC再填0；零占位不得参与flat-line、幅值或频谱QC。不能依据数值0推断有效性。凡能生成有限波形的epoch，包括触发hard artifact者，均保留处理后数值，不清零、不删除，是否纳入只由mask决定。

chunk/压缩仅在pilot比较4或8个epoch/chunk及LZF、gzip-1、无压缩后冻结；目标是随机连续epoch读取吞吐，单shard目标约6 GiB、硬上限8 GiB、约不超过16夜，同一夜不跨shard。

### 6.3 Task时长与signal对齐QC

每个task使用独立H5，但与对应signal shard共享完全相同的segment网格。共同字段为 `segments/record_index int32 [N]`、`segments/epoch_in_record int32 [N]`、`segments/epoch_start_offset_ns int64 [N]`、`valid bool [N]`、`field_valid bool [N,F]`、`qc_code uint8 [N]` 和 `field_qc_code uint8 [N,F]`。record级字段为 `records/recording_id string [R]`、`records/first_epoch int64 [R]`、`records/n_epochs int32 [R]`、`records/structure_qc_code uint8 [R]`、`records/structure_qc_flags uint8 [R]`、`records/source_missing_epochs int32 [R]` 和 `records/source_extra_epochs int32 [R]`；根属性另存shard级 `structure_qc_code uint8` 和 `structure_qc_flags uint8`（标量）。其中 `valid=all(field_valid,axis=1)`；它只表示任务标签有效，不与signal或模态mask预先合并，训练读取时再按所需模态取交集。

| task文件 | `labels`形状/类型 | F及固定字段顺序 | 缺失sentinel | 有效值规则 |
|---|---|---|---|---|
| `tasks/sleep_stage/shards/task-xxxxx.h5` | uint8 `[N]` | 1：`stage` | 0 | 1=N3、2=N2、3=N1、4=REM、5=Wake；只接受1–5 |
| `tasks/heart_rate/shards/task-xxxxx.h5` | float32 `[N,3]` | 3：`max_bpm, min_bpm, mean_bpm` | NaN | 三项有限、20–300 bpm，且 `min<=mean<=max` |
| `tasks/sao2/shards/task-xxxxx.h5` | float32 `[N,2]` | 2：`max_percent, min_percent` | NaN | 两项有限、20–100%，且 `min<=max` |

task的epoch级QC使用以下唯一代码。多字段任务由 `field_qc_code` 保留各字段结果，`qc_code` 按表中从上到下的非零优先级聚合；`qc_code=0` 当且仅当 `valid=true`。

| code | 名称 | 含义 |
|---:|---|---|
| 0 | `TASK_INCLUDED` | 所有必需字段存在且通过任务值域/顺序检查 |
| 1 | `TASK_SOURCE_EPOCH_MISSING` | 源task没有对应signal epoch；写sentinel |
| 2 | `TASK_VALUE_MISSING_OR_NONFINITE` | 源字段为空、使用缺失占位值，或浮点值为NaN/Inf |
| 3 | `TASK_VALUE_OUT_OF_RANGE_OR_UNKNOWN_CLASS` | 数值超出冻结范围，或睡眠分期不在1–5 |
| 4 | `TASK_FIELD_ORDER_INVALID` | heart rate或SaO2的min/mean/max次序不成立 |
| 5 | `TASK_SOURCE_TIME_UNVERIFIED` | H5显式30秒网格缺口；对应epoch三个task的原本有效字段均排除 |
| 6 | `TASK_H5_STAGE_CONFLICT` | H5同一epoch有不同分期；仅排除该epoch分期标签 |
| 7 | `TASK_SOURCE_STAGE_MISMATCH` | CSV已知分期与无冲突H5分期不一致；仅排除该epoch分期标签 |

保留原始标签值，不补分期、不平移时间轴。code 1–4已有原因优先保留；code 5–7只覆盖原本有效字段。所有来源对齐问题的完整epoch索引另存提交记录的`task_alignment`，逐shard QC报告显示具体原因及相对秒区间，避免主码掩盖并存问题。这些是task代码，不进入signal artifact hard code。时间证据缺失不等于已证实错位；H5分期冲突也不自动否定同一时间网格内HR/SaO2数值。

细则：整行源epoch缺失用code 1；行存在但字段为空、非有限或HR/SaO2值为0用code 2；睡眠stage为0或无法映射到1–5用code 3。仅在参与比较的字段都已通过存在性、有限性和值域检查后执行顺序检查；顺序失败时，所有参与该约束的字段均写 `field_qc_code=4` 且 `field_valid=false`，保证 `valid=all(field_valid)` 与主 `qc_code` 一致。task H5必须保存 `field_names` 属性，防止仅靠列位置猜测语义。

结构级task QC按record和shard记录，不混入epoch `qc_code`。`structure_qc_flags` 的bit 0–4对应code 1–5，可同时保留多个失败；`structure_qc_code` 是主码，非零优先级固定为 `TASK_N_MISMATCH > TASK_RECORD_RANGE_MISMATCH > TASK_EPOCH_KEY_MISMATCH > TASK_DURATION_MISMATCH > TASK_SOURCE_COVERAGE_MISMATCH`：

| code | 名称 | 发布规则 |
|---:|---|---|
| 0 | `TASK_STRUCTURE_PASS` | 结构通过 |
| 1 | `TASK_SOURCE_COVERAGE_MISMATCH` | 允许发布；必须报告缺失与超出signal网格的epoch数 |
| 2 | `TASK_N_MISMATCH` | FAIL，禁止发布 |
| 3 | `TASK_RECORD_RANGE_MISMATCH` | FAIL，禁止发布 |
| 4 | `TASK_EPOCH_KEY_MISMATCH` | FAIL，禁止发布 |
| 5 | `TASK_DURATION_MISMATCH` | FAIL，禁止发布 |

- 每个task shard与对应signal shard一一绑定，task第0维必须等于signal的N；每个record的 `task_n_epochs` 必须等于 `signal_n_epochs`。
- `signal_duration_seconds=signal_n_epochs*30`，`task_duration_seconds=task_n_epochs*30`，二者必须严格相等。不能用源标注的声明时长替代实际发布长度。
- task必须逐项匹配 `recording_id`、`record_index`、`epoch_in_record` 和 `epoch_start_offset_ns`；必须依据源epoch编号/时间映射，不得只靠CSV行号重新编号制造对齐。缺失/非法/重复CSV epoch键、无法解析H5分期网格或已发布容器键错位仍为结构FAIL。可定位到具体epoch的H5时间证据缺口、分期冲突/不一致改用task code 5–7，不拒绝整夜signal；`task_alignment.status=PARTIAL`不能解释为所有task标签通过。
- 源task标签较短、较长或有缺口时，不修改signal网格：缺失epoch或字段写上述sentinel并设置对应QC；超出signal网格的标签丢弃并计数。结构输出仍必须与signal等长，否则禁止发布。
- 显式H5 30秒标注网格用于核对CSV的一基Epoch时间对应；已知分期逐项比较。整段未分期但时间网格完整时，分期标签仍无效，结构检查可通过；记录known_anchors=0，不将“无已知类别”当成时间错位。

### 6.4 QC字段清单与关系审核

维度：N=当前shard全部30秒epoch数，R=当前shard记录数，C=当前模态通道数，F=当前task标签字段数。全局15通道顺序固定为第2节；模态切片固定为EEG `0:6`、EOG `6:8`、ECG `8:9`、EMG `9:12`、respiratory `12:15`。以下均为发布方案，需在新版pilot实现后验证。

Signal文件 `signals/shards/signal-xxxxx.h5`：

- `/quality/<modality>/coverage_valid`：bool `[N,C]`，完整来源时间覆盖。
- `/quality/<modality>/processing_valid`：bool `[N,C]`，完整6000点成功处理；占位补零不构成成功。
- `/quality/<modality>/artifact_valid`：bool `[N,C]`，适用规则全部完成且没有hard命中。
- `/quality/<modality>/valid`：bool `[N,C]`，最终通道epoch纳入mask。
- `/quality/<modality>/hard_code`：uint8 `[N,C]`，由QC文件的hard flags及固定优先级生成；不能独立计算另一套主码。
- 上述五项在EEG/EOG/ECG/EMG/respiratory中分别为 `[N,6]/[N,2]/[N,1]/[N,3]/[N,3]`。
- `/records/channel_available`：bool `[R,15]`，整夜通道门槛结果。
- `/records/channel_status_code`：uint8 `[R,15]`，通道门槛主状态；0=AVAILABLE，1=CHANNEL_MISSING，2=CHANNEL_UNREADABLE，3=CALIBRATION_OR_UNIT_INVALID，4=PROFILE_EXCLUDED，5=SPO2_NEAR_ZERO。主状态优先级为missing、unreadable、profile、calibration/unit、near-zero；`channel_available == (channel_status_code==0)`。此枚举独立于artifact hard code。
- `/records/channel_processing/{source_filter_state,effective_filter_state,filter_relation,filter_reapply_result,resample_result,coverage_detection}`：每项uint8 `[R,15]`；数字与第4节状态名的映射保存在processing.json。来源filter为NOT_EVALUATED=0/EXPLICIT=1/UNKNOWN=2；effective filter依次为NOT_EVALUATED=0/AS_RECORDED=1/REAPPLY=2/NOT_REQUIRED=3/FAILED=4/PROFILE_EXCLUDED=5；filter relation为NOT_EVALUATED=0/MATCH=1/WIDER=2/NARROWER=3/MIXED=4/UNKNOWN=5/NOT_REQUIRED=6；处理结果为NOT_EVALUATED=0/NOT_REQUIRED=1/SUCCESS=2/FAILED=3/UNSUPPORTED=4/PROFILE_EXCLUDED=5/PARTIAL=6；coverage detection为NOT_EVALUATED=0/LENGTH_ONLY=1/TIMESTAMP_CHECKED=2。不可仅用整夜状态代替epoch processing mask；混合成功/失败的连续段需在处理结果枚举中表达PARTIAL。
- `/records/channel_processing/{source_fs_hz,source_hp_hz,source_lp_hz,effective_hp_hz,effective_lp_hz,notch_hz}`：每项float64 `[R,15]`；未知/不适用为NaN，不用0代替未知。
- `/records/channel_usable_hours`：float64 `[R,15]`，从最终valid按record汇总。
- `/records/night_grade`：uint8 `[R]`；`/records/grade_evaluated`：bool `[R]`。已发布record必须完成评级；固定排除记录只在records.parquet台账中记0/false，不混入shard的R。
- `/records/source_hsp_likert_scale`：uint8 `[R]`，缺失为0，原始评级有值时独立保存。

QC文件 `signals/qc/qc-xxxxx.h5`：

- `/hard_flags`：uint8 `[N,15]`，每条规则是否命中。
- `/hard_code`：uint8 `[N,15]`，优先级主码，0–7。
- `/hard_evaluated_flags`：uint8 `[N,15]`，每条规则是否完成。
- `/metrics/nonfinite_output_count`：uint16 `[N,15]`，补零前6000点中的非有限数，0–6000，未取得候选输出为65535。
- `/metrics/nonfinite_source_count`：uint32 `[N,15]`，该epoch原生数字数据中的非有限数，未检查为4294967295；NAN计算可使用来源或候选输出证据，不把时间覆盖缺失当成非有限数据。
- `/metrics/std_physical`：float32 `[N,15]`，执行标准差型flat-line规则时的总体标准差，单位沿用通道单位。
- `/metrics/digital_peak_to_peak_codes`：float64 `[N,15]`，Airflow原生数字码差，量纲为数字码步长。
- `/metrics/digital_unique_count_capped3`：uint8 `[N,15]`，Airflow不同码数量截断到3（3代表至少3种），未检查为255。
- `/metrics/saturation_fraction`：float32 `[N,15]`，原生数字样本上下rail并集比例。
- `/metrics/high_amplitude_duration_seconds`：uint8 `[N,15]`，超阈值1秒子窗数0–30，未检查为255。
- `/metrics/power_line_fraction`、`/metrics/high_frequency_fraction`：各float32 `[N,15]`，对应功率比例0–1。
- `/metrics/spo2_in_range_fraction`：float32 `[N,15]`，完整有限6000点中落在[50,110]%的比例。
- 所有浮点metric在不适用/未执行时写NaN，不能写0假装通过；整数使用上述sentinel。频谱分母为0时两项比例约定为0并记录规则已执行，避免无功率信号产生除零；flat-line按适用范围独立判断。PSD参数按第5.4节冻结在qc.json，并由独立FFT实现与已知频率合成信号验证。

配置 `config/qc.json`：

- `hard_applicable`：JSON布尔二维数组，逻辑shape `[7,15]`，行是hard code 1–7，列是全局通道。
- 将每列打包成uint8适用位图A：EEG/EOG/ECG为63，EMG为29，Airflow为3，Snore为1，SpO2为65。EMG的HIGH_FREQUENCY_NOISE（bit 5）必须为0。
- 该静态配置必须与schema中的通道顺序绑定；不为每个epoch重复保存。

每个Task文件 `tasks/<task>/shards/task-xxxxx.h5`：

- `/labels`：sleep_stage为uint8 `[N]`；heart_rate为float32 `[N,3]`；sao2为float32 `[N,2]`。字段顺序、单位和值域见6.3。
- `/valid`、`/qc_code`：分别bool、uint8 `[N]`，任务所有字段的聚合结果。
- `/field_valid`、`/field_qc_code`：分别bool、uint8 `[N,F]`；三个task的F分别为1、3、2，睡眠分期这里仍保留第二维1。
- `/records/structure_qc_code`、`/records/structure_qc_flags`：各uint8 `[R]`。
- `/records/source_missing_epochs`、`/records/source_extra_epochs`：各int32 `[R]`，只计整行缺失/超出，不与字段缺失数混淆。
- 根属性 `structure_qc_code`、`structure_qc_flags`：各uint8标量 `[]`，包含record聚合结果及shard级结构检查失败。

关联字段与必须成立的关系：

- Signal和Task都存 `/records/recording_id string[R]`、`first_epoch int64[R]`、`n_epochs int32[R]`，以及 `/segments/record_index int32[N]`、`epoch_in_record int32[N]`、`epoch_start_offset_ns int64[N]`。`sum(n_epochs)==N`；record区间连续、不重叠；每夜epoch编号为0至n_epochs-1；`epoch_start_offset_ns=epoch_in_record*30_000_000_000`。
- Signal、QC、Task根属性共同保存 `shard_id`、`schema_sha256`、`processing_config_sha256`、`qc_config_sha256`、`epoch_index_sha256`、`channel_order_sha256`；均为字符串标量。epoch索引摘要包含有序record身份及segment键，不能只哈希0、1、2等局部编号。QC和Task另存最终关闭后的signal文件 `signal_sha256`，manifest保存各文件SHA256；QC无需重复整份segment数组，同编号文件名本身不足以证明绑定正确。
- 令H=hard_flags，E=hard_evaluated_flags，A=适用位图按epoch广播；必须 `H & (~E & 127)==0`、`E & (~A & 127)==0`，所有flags的保留bit 7为0。不能出现“未执行却命中”或“不适用却执行”。
- `qc_complete=(E & A)==A` 为派生量，不新增存储字段。`artifact_valid=qc_complete & (H==0)`；`valid=channel_available[record_index,channel] & coverage_valid & processing_valid & artifact_valid`。所有适用规则都作为必需规则；意外计算失败必须在日志中显式记录并重试/解决，不能按通过发布。
- 缺通道或跳过处理的占位epoch为H=0、E=0、hard_code=0、artifact_valid=false、valid=false。发现NAN而跳过其余规则时H包含NAN、E至少包含NAN，artifact_valid=false。Snore只适用NAN，artifact_valid=true只说明通过了有限性检查，不能据此声称鼾声波形质量全面合格。
- QC的hard_code必须可由H重新推导；Signal各模态hard_code与QC对应通道切片逐值相同。metrics与触发条件、evaluated位须可互相核验。
- `channel_usable_hours[r,c]=sum(valid[record_range(r),c])/120`；night_grade由这些时长和第7节规则重算。task缺标签不改变signal可用时长或整夜等级。
- Task的 `field_valid=(field_qc_code==0)`、`valid=all(field_valid,axis=1)=(qc_code==0)`；聚合遵循6.3优先级。有限但越界或次序不合法的原始标签保留数值，用QC排除；不通过改值使标签变成有效。
- Task结构flags只允许bit 0–4；主码从flags重算；shard flags为record flags按位或并入shard级错误。code 2–5任一对应bit非零即禁止发布，不能被code 1覆盖。N一致和容器时长一致仅证明输出结构，真实时间对应仍须核实来源epoch身份/起点。
- QC报告分别统计适用数、已执行数、命中数；命中率分母为该规则已执行epoch数，分母为0报告NOT_EVALUATED。不同mask的排除数会重叠，不能直接相加作为总排除数；总数按最终valid取反计算。

本轮审核修正：未执行规则被误当作artifact通过、补零掩盖非有限证据、三个mask原因混淆、数字码和物理LSB混用、task文件命名不一致、task顺序QC覆盖优先级、QC sidecar缺少内容绑定。processing.json各枚举已由当前实现固定输出；qc.json按第5.4节固定PSD。HIGH_AMPLITUDE每秒短尖峰的局限已获用户接受。按本文件执行新版pilot，报告必须区分自动验收通过与人工Review待完成。

## 7. 整夜等级

| grade | 名称 | 条件 |
|---:|---|---|
| 5 | Outstanding | EEG、EOG、EMG、ECG、Airflow、SpO2六类均 `>5 h` |
| 4 | Excellent | EEG、EOG、EMG、Airflow、SpO2均 `>5 h` |
| 3 | Good | EEG、Airflow、SpO2均 `>5 h` |
| 2 | Fair | EEG、Airflow、SpO2均 `>4 h` |
| 1 | Poor | 其余 |

`usable_hours=sum(valid)*30/3600`。EEG/EOG/EMG使用该类最佳单通道，不拼接互不重叠的通道。虽然三路共用 `signals/respiratory`，等级仍分别计算Airflow respiratory和SpO2 oximetry；Snore不能替代Airflow。Airflow或SpO2整夜通道无效时，该记录必为Poor，但其他通道和记录仍保留。固定排除记录使用 `night_grade=0, grade_evaluated=false`。

规则名：`I0002_15CH_AIRFLOW_RESP_V1`。若源数据有官方 `likert_scale`，另存 `source_hsp_likert_scale`，不得覆盖项目等级。

## 8. 精简目录

```text
I:\HSP\I0002-preprocess\
├── README.md
├── preprocess.md
├── source_scan\v1.0.0\
│   ├── report.md
│   ├── scan.jsonl
│   └── summary.json
└── processed\v1.0.0\              # 所有预处理产物只放在此子目录
    ├── config\
    │   ├── schema.json              # 通道、形状、编码
    │   ├── processing.json          # 标定、filter、resample
    │   └── qc.json                  # 伪迹、整夜等级、task对齐
    ├── manifests\
    │   ├── records.parquet          # 纳入/排除及record→shard索引
    │   ├── shards.parquet
    │   └── release.json
    ├── signals\
    │   ├── shards\signal-xxxxx.h5
    │   ├── qc\qc-xxxxx.h5
    │   └── dataset.json
    ├── tasks\<task>\
    │   ├── shards\task-xxxxx.h5
    │   ├── manifest.parquet
    │   └── dataset.json
    ├── reports\
    │   ├── qc.md
    │   ├── qc_summary.parquet
    │   └── io.md
    └── logs\
        ├── run.log
        ├── errors.jsonl
        └── progress.json
```

`preprocess.md` 和只读源扫描保留在根目录；config、manifest、signal、task、QC报告和运行日志全部写入 `processed\v1.0.0`。删除原方案中的重复catalog、独立exclusions、多个summary/config文件和splits目录。

新版pilot单独放在 `processed/v1.0.0/pilot-<运行时间>/`，内部复用上述目录，旧 `pilot/` 保留为历史结果；正式版本号仍为v1.0.0，pilot内部schema revision为1.0.0-pilot.2。每个pilot保存冻结方案、代码快照、运行依赖版本、错误日志与progress.json；最终状态为PILOT_REVIEW_REQUIRED，自动通过不能触发全量。

## 9. 执行与验收

全量运行参数：8个session进程×4个通道线程，BLAS线程数为1，单writer与下一批计算重叠；至多缓存当前及下一shard的结果。临时记录写E盘专用work目录，正式产物写I盘；每shard最多16夜、目标6 GiB、硬上限8 GiB，不拆夜。progress.json每5秒更新并原子替换；ETA在首个完整shard提交后根据包含打包、校验与跨盘复制的实际吞吐估计，前三个shard标记WARMUP。恢复必须核对代码、输入清单及配置摘要，只读取已提交manifest；异常记录重试1次后记入FAILED_PENDING_REVIEW，不能伪装为清洗通过。manifest/commits和reports/shards用于增量提交与逐shard报告。

2026-09-05恢复修订：Windows共享锁导致原子替换失败时有限退避重试；进度快照失败记录`PROGRESS_WRITE_DEFERRED`并保留旧快照，不中断数据计算，run.log/console日志作为进度备份。数据与commit写入仍须成功才能发布。以21个提交凭证恢复全局清单，保留294个已发布记录及全部既有H5；42个已提交失败记录重新加入未发布队列（第43个失败记录尚未提交，原本就在待处理范围）。按原inventory顺序重组剩余记录，保持每夜唯一、已发布record→shard映射不变。原配置和代码归档到`config/history/pre-task-alignment-recovery/`以解析旧文件摘要；公开版本仍为v1.0.0，新task代码为向后兼容扩展，内部schema为1.0.0-full.1。后续恢复使用已冻结的新shard_plan，禁止再次加入已发布记录。重启后ETA前三个新提交为WARMUP。

1. 冻结本文件和三个config的SHA256。
2. 实现并单元测试：时间网格、标定门槛、整夜补滤、重采样、七个hard code、三层mask和断点续跑。
3. pilot覆盖常见/少见采样率、全部filter profile、UNKNOWN分支、9夜Airflow及160夜SpO2拒绝profile、缺失通道，以及每种hard code和近阈值阴性；逐通道人工复核。
4. pilot冻结filter/resample profile、七类hard阈值、chunk、压缩和QC profile revision。
5. 全量构建：按session多进程；每进程限制数值库线程；worker生成独立session临时结果，单writer组装HDF5并原子提交。日志记录吞吐、错误、重试和ETA。
6. 全量QC：逐文件重开，核对SHA256、200 Hz、6000点、dtype、通道顺序、时间轴、mask不变量、处理状态、整夜等级，以及每个task与signal的N、逐record时长和epoch键。
7. 只有signal、QC、task、manifest及摘要全部一致时，`release.json` 才标为accepted。

QC报告至少包含：

- record总数、固定/运行期排除、通道覆盖和时长分布；
- 每通道单位、标定profile、源/输出fs、HP/LP/notch及filter relation；
- Airflow 9夜和SpO2 160夜profile导致的channel availability计数，且处理调用数必须为0；
- 每通道每种hard code的夜数、epoch数、比例和usable hours；不得出现未定义code或warning字段；
- 每个hard阈值两侧的人工复核抽样结果，以及三层mask各自排除的epoch数；
- signal/QC/task的N、record、epoch、时间和SHA256一致性；逐task报告源标签覆盖不足/超出数、task容器时长不匹配数（发布必须为0）和epoch键错位数（发布必须为0）；
- shard大小、压缩率、随机/顺序读取吞吐、峰值内存；
- `PASS/FAIL/NOT_EVALUATED`、未解决问题及自动QC边界。

## 10. 本轮Review结论

已确认并写入正文：

1. 原 `ALIGN-01`：H5 root `duration_sec` + floor建立唯一epoch网格；不使用max+ceil。
2. UNKNOWN补滤按整夜有限连续段执行，不按artifact切段；guard只记处理元数据，不生成epoch hard code或自动排除。
3. QC采用HSP六类artifact，加HSP SpO2专属range检查；hard code固定为0–7。
4. 删除数字冻结、fixed-tone、连续rail附加条件、所有warning/candidate及 `PROCESSING_BOUNDARY`；saturation只保留biopotential的5% rail比例。
5. Airflow、Snore、SpO2合并存入 `signals/respiratory[N,3,6000]`，但mask、QC及整夜等级语义仍逐通道独立。
6. 每个task容器必须与signal逐record等长并逐epoch键一致；结构不一致阻止发布，源标签覆盖差异只通过task mask和QC报告表达。
7. 所有正式预处理产物统一写入 `I:\HSP\I0002-preprocess\processed\v1.0.0`。
8. `v1.0.0` 不进行跨通道重复波形扫描，也不因通道间相关性修改mask。
9. hard artifact不清零或删除有限波形；最终纳入由channel availability、coverage、processing和artifact四项共同决定。
10. `HIGH_AMPLITUDE` 只在30个1秒子窗全部超过阈值时触发，并保存0–30秒持续时间指标；不发布1秒QC。
11. task冻结为三个独立H5及固定shape/字段顺序；epoch标签QC与结构对齐QC使用互不混淆的代码空间。
12. `HIGH_AMPLITUDE` 使用 `[B,C,30,200]` 向量化峰峰值计算；取消EMG的 `HIGH_FREQUENCY_NOISE` 判定并将对应evaluated位设为false。

`report.md` 在名称映射层面没有报告重复，严格Airflow也只取 `/signals/airflow`；该结论不代表波形层面已经验证无重复。本版本按上述决定不追加波形重复检查。既有pilot的reason 8–21、warning和边界标记均属旧schema，必须废止后重建，不能迁移到新QC。

## 11. 限制

- 来源prefilter是采集/导出声明，不是硬件传递函数的独立测量。
- 没有显式时间戳时，样本长度检查无法发现隐藏采集缺口。
- 自动QC不能证明电极接线正确，也不能替代临床判读。
- `source_scan/v1.0.0/report.md` 是统计依据，不是波形质量检查结果。
