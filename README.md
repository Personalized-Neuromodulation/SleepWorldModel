# SleepWorldModel

本轮新增 **modality JEPA + shared/private/routed MoE fusion**：遮挡后的可见融合token预测各模态融合前的干净EMA目标，输入固定30秒。设计与数据流见 [架构文档](docs/PSG_ARCHITECTURE_PROPOSAL.md)。历史LeJEPA仍可按原配置加载和对照，下面的global/local说明属于LeJEPA。

当前 N1 监督适配的两份配置保存在 `results/finetuning/n1_last2_balanced_10k_20260915/`。用户已批准仅用训练标签更新末端网络层，保持30秒输入；实现与梯度边界见 [finetuning](finetuning/README.md)。训练/单步调试仍共用同一入口：

```powershell
python -B -m pretraining.cli --config results/finetuning/n1_last2_balanced_10k_20260915/psg_training.yaml
python -B tests/test_pipeline.py --config results/finetuning/n1_last2_balanced_10k_20260915/psg_training.yaml --break-at loss training.batch_size=8 evaluation.wandb_mode=disabled
```

每1K成功更新评价train/validation并保存checkpoint，监督分期指标使用 `supervised_*`，按 N1 F1 选择最佳权重，同时报告其他阶段、心率和血氧。原冻结探针结果保留在 `results/foundation/`，不得与监督成绩混用。`configs/`默认仍保留LeJEPA基线；以上配置选择监督适配。


HSP 睡眠信号预处理与多模态自监督学习。当前 I0002 数据由 `data_preprocess.i0002_v100` 生成；训练与调试默认通过 `dataloader` 读取这套正式分片。

## 目录与入口

| 目录 | 职责 |
|---|---|
| [data_preprocess/i0002_v100](data_preprocess/i0002_v100) | 当前 I0002 清洗、QC、任务标签及全量发布 |
| [scripts](scripts) | 源数据扫描、统计报告及 F3-M2 标定抽样复核 |
| [docs](docs) | 源扫描规范与冻结清洗方案 |
| [dataloader](dataloader) | 独立的通用数据读取、窗口、QC 与 batch 接口 |
| [backbone](backbone/README.md) | 连续 PSG 主干：patch 编码、局部序列、通道聚合和模态融合 |
| [notebooks](notebooks/backbone_step_by_step.ipynb) | 逐步检查主干各层、真实数据、梯度和权重重载 |
| [pretraining](pretraining/README.md) | 唯一 LeJEPA/SIGReg 训练入口 |
| [finetuning](finetuning/README.md) | 复用同一 CLI 的监督适配、独立评价与梯度隔离 |
| [experiment_logging](experiment_logging) | 独立 W&B 日志 |
| [tests](tests) | 当前预处理及独立功能的回归测试 |

[代码使用依据与清理记录](data_preprocess/CODE_MAP.md) · [原始 H5 接口](#legacy-loader) · [SSL](#training)

多模态 foundation backbone 与 LeJEPA/SIGReg 已接入，见 [架构与验收](docs/PSG_ARCHITECTURE_PROPOSAL.md)。训练统一使用 `pretraining.cli`，旧 SSL 实现及入口已删除。

## 全局单步调试

从 [tests/test_pipeline.py](tests/test_pipeline.py) 的 `run_pipeline()` 开始。每一阶段保留清晰的局部变量，可在 IDE 设置断点后按 F11 进入调用模块，或使用命令行断点：

```powershell
python -B tests/test_pipeline.py --device cpu --break-at data
python -B tests/test_pipeline.py --device cuda --break-at all pretraining.sigreg.num_slices=8
python -B -m pytest tests/test_pipeline.py tests/test_input_boundary.py -q
```

断点可选 `data/model/views/backbone/loss/backward/save-load/all`，进入 pdb 后用 `n` 下一行、`s` 进入函数、`c` 继续。数据源统一读取 YAML 的 `data.mode` 和 `data.root`，默认真实数据；合成调试可设置 `data.mode: synthetic`。`--output` 指定调试产物目录。入口只更新一步并保存 HF 权重，输出各层 shape、loss、梯度和重载结果；诊断指标来自当前调试 batch，不是泛化评分。训练与 W&B 评价仍使用下方统一训练入口。

IDE 中直接 Run/Debug `test_global_pipeline` 也可运行：pytest 临时文件默认位于 `artifacts/pytest`，缓存位于 `artifacts/pytest-cache`，无需额外传 `--basetemp`。每次运行使用 pytest 独立编号目录；显式指定的 `--basetemp` 或 `PYTEST_DEBUG_TEMPROOT` 仍优先。

N1 诊断使用同一个入口的 `--mode diagnose`：固定 checkpoint，比较类别加权、MLP、当前30秒内的时间分类头和融合前模态特征。参数位于 `evaluation.diagnostics`，流程与输出见 [诊断说明](pretraining/README.md#n1-checkpoint-diagnosis-independent-30-s)。单步运行 `python -B tests/test_pipeline.py --mode diagnose --break-at features`；完整运行 `python -B -m pretraining.cli --mode diagnose`。诊断结果单独记录，不覆盖原始 `eval/macro_f1`。

输入只在 dataloader 适配阶段验证，再传入 backbone；GPU forward 应用 QC/visibility mask，不重复扫描原始数据或比较时间网格。自行构造 SignalBatch 时，在送入 GPU 前调用 `validate(foundation=True, input_spec=...)`。

`test_pipeline` 的 `data` 断点可分别查看已经保存的 `data_valid` 和当前 `visible`；`views` 断点可观察 dropout 如何改变可见性，同时复用原 QC。`patching.py` 仅应用 mask 和切分 patch，不再组合 epoch/QC 规则。`tests/test_mask_flow.py` 验证这条数据流、padding、NaN 隔离以及全无效 view 排除。

## Foundation backbone 与 LeJEPA

配置分为两个文件：`configs/psg_model.yaml` 保存网络结构、LeJEPA/SIGReg、views 和增强参数；`configs/psg_training.yaml` 保存数据模式/路径、训练和 W&B 设置，并通过 `model_config: psg_model.yaml` 引用前者。CLI 默认读取训练配置，自动合并模型配置；文件引用相对于训练配置所在目录解析。各模态编码器及 Fusion 默认4层；ECG/SpO₂只使用 temporal attention。输入为一个30秒来源 epoch，局部 view 为1–15整秒，同一 batch 内统一长度；输出逐秒256维和汇总256维表示。QC仍按来源epoch/通道判定，坏通道通过mask排除。

评价频率由 `evaluation.eval_every_seconds: 600.0` 控制：每训练10分钟，在下一次成功更新后进行完整train/validation评价、W&B记录和checkpoint保存；评价与保存耗时不计入下一间隔，结束时仍评价。设为0才使用 `eval_every_steps`，训练总步数不受影响。

训练默认使用 CUDA、bf16 和 fused AdamW，SIGReg 保持 FP32。数据读取、适配和 QC 在 CPU 完成；真实 DataLoader 在适配后 pin memory，再异步传输到 GPU。CPU 调试使用 `training.device=cpu`；不支持 bf16 的 GPU 可设置 `training.precision=float32`。模型训练、评价和 `tests/test_pipeline.py` 共用 `pretraining/runtime.py` 的运行配置。

机械硬盘与 128 GB RAM 配置：默认仅 Outstanding，`data.ram_pool.cache_gib: 64.0` 为当前池与预读池的总预算。单后台 Reader 连续读取，RAM 内跨记录混洗，再将 batch 送入 GPU；`training.num_workers: 0` 避免多个进程重复大缓存。内存逐步填充，不预分配全部预算；配置、限制及 W&B 输入指标见 [RAM 读取说明](dataloader/FORMAT.md#hdd--128-gb-ram-训练读取)。单步入口仍是 `tests/test_pipeline.py`。

CNN 层数由 `model.patch_tokenizer.layers` 列表长度决定；每层的 channels/kernel/stride/padding/dilation/bias、归一化、激活和 dropout 均可配置。Criss-Cross、Temporal、Fusion 的 heads 和 FFN 宽度也由 YAML 设置，详见 [backbone 配置说明](backbone/README.md)。`test_pipeline` 同时回归默认网络和非默认网络，并在报告中保存实际模型配置。

SpO₂ 使用独立数值路径：`data.scales.spo2: {offset: 95.0, divisor: 5.0}` 在 CPU 将原始百分数变为 `(SpO₂−95)/5`；`model.numeric_tokenizers.spo2: {hidden_dim: 32, activation: gelu}` 将每个 1 秒 patch 的均值通过 `1→32→256` MLP 编码，不使用逐 patch GroupNorm/LayerNorm。其他模态继续使用 CNN，epoch QC、标签和通道维度保持原定义。旧配置没有 `numeric_tokenizers` 时仍构建原 CNN。HF 公开输入需先完成相同变换，新模型的 SpO₂ 输入单位记录为 `(%-95)/5`，不能直接输入百分数或仅 `/100` 的值。

Readout 默认在 modality encoder 出口沿 D 做无 affine LayerNorm，Fusion 出口保留原值，模态池化使用固定 scale=2 的 cosine 打分。`model.readout` 可独立配置输出归一化与池化方式；数值边界、梯度和小规模任务对照见 [验证报告](results/foundation/readout_validation/REPORT.md)。这不是下游收敛结论，旧 checkpoint 不会自动改用新结构。修改源码后开发环境建议 `python -m pip install -e . --no-deps`，保证其他工作目录下的 HF 加载也使用当前代码。

```powershell
# 合成数据：一步训练、固定验证集评价、HF导出；不需要HSP文件。
python -B -m pretraining.cli data.mode=synthetic training.max_steps=1 evaluation.max_batches=1 pretraining.sigreg.num_slices=8 evaluation.wandb_mode=disabled

# 正式数据：设置数据路径，按subject划分；night_grade用于筛选和组batch。
python -B -m pretraining.cli

# 可覆盖网络深度、loss、增强；每次运行都保存解析后的完整配置。
python -B -m pretraining.cli data.mode=synthetic training.max_steps=1 model.modality_encoder.eeg_depth=6 pretraining.lejepa.lambda_sigreg=0.05 pretraining.augmentation.channel_dropout_prob=0.2
```

默认 W&B online，自动读取项目根目录 `.env` 中的 `WANDB_API_KEY`（已有环境变量优先），密钥不写入实验配置。`evaluation.wandb_mode=offline/disabled` 可改为离线或关闭。评价记录 SSL loss、有效 view 比例、embedding 标准差/effective rank；默认 `evaluation.frozen_linear_probe=true`，同时训练并验证 sleep_stage、heart_rate、sao2 的独立线性头，梯度不传回 backbone。心率和血氧回归记录原始单位的 MAE、RMSE、逐字段 R² 与有效样本数；无效字段由 `field_valid` 排除。synthetic 无标签，需设置 `evaluation.frozen_linear_probe=false`。默认输出 `results/foundation/`，其中 `backbone/` 是HF模型、`training.pt`含SSL/优化器状态、`config.json`保存实验配置，`best/`保留最佳验证checkpoint。

当前配置为 **10K训练、每1K评价和保存完整checkpoint**。AdamW使用5%warmup＋余弦衰减；SIGReg按view统计。固定评价原始输入缓存到RAM，独立的`train_refit/*`、`eval_refit/*`报告充分拟合线性头的结果；原`eval/macro_f1`仍用于选择最佳模型。恢复路径可配置为`training.resume_from`，具体说明见[预训练文档](pretraining/README.md#current-10k-training-and-restart)。

脚本调试 `python -B tests/test_pipeline.py` 同样遵循 YAML 的 W&B 模式，在流程完成后上传当前 batch 的 `debug/*`、`debug_batch/*` 指标。pytest 自动测试默认关闭联网；运行 `python -B -m pytest tests/test_pipeline.py --wandb-mode online -s` 开启，IDE 的 pytest 运行配置也可填写 `--wandb-mode online`。

HF加载环境需要安装本项目及其依赖：

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("results/foundation/backbone", trust_remote_code=True)
model.eval()
with torch.no_grad():
    output = model(
        signals={"ecg": torch.randn(2, 1, 8, 200)},
        data_valid={"ecg": torch.ones(2, 1, 1, dtype=torch.bool)},  # epoch QC
    )
print(output.last_hidden_state.shape)  # [2,8,256]
print(output.pooler_output.shape)      # [2,256]
```

调试按以下顺序打开两个notebook：先 [组件单步调试](notebooks/backbone_step_by_step.ipynb)，再 [端到端运行逻辑](notebooks/architecture_debug_walkthrough.ipynb)。修改配置后重新运行构建cell；代码cell调用仓库模块，支持在forward中设置断点。

```powershell
python -B scripts/check_backbone_notebook.py
python -B scripts/check_backbone_notebook.py --notebook architecture_debug_walkthrough
python -B scripts/inspect_foundation.py --device cuda
```

## 环境

Python ≥3.11。从仓库根目录运行；安装包包含 backbone、dataloader、pretraining、experiment_logging。

```powershell
Set-Location 'E:\Code\SleepWorldModel'
$sleepPython = 'C:\Users\user\miniconda3\envs\SleepWM\python.exe'
& $sleepPython -m pip install -r requirements.txt
& $sleepPython -m pip install -e . --no-deps
```

`requirements.txt` 合并数据处理、训练和测试依赖；`pyproject.toml` 保留项目打包信息及依赖声明。

## Backbone 单步测试

在 Jupyter/VS Code 打开 `notebooks/backbone_step_by_step.ipynb`，选择 **Python (SleepWM)** kernel，按 Shift+Enter 逐个执行。数据模式及路径默认读取 YAML，`DEVICE='cuda'` 使用 GPU；验证脚本通过 `PSG_NOTEBOOK_REAL` 显式覆盖数据模式。每层变量可直接查看，网络块输入通过临时 hook 展示。

```powershell
& $sleepPython -m jupyterlab notebooks/backbone_step_by_step.ipynb
# 无界面执行同一份 notebook，结果保存在 artifacts/backbone：
& $sleepPython scripts/check_backbone_notebook.py
& $sleepPython scripts/check_backbone_notebook.py --real --device cuda
```

首次在其他环境使用时，执行 `python -m ipykernel install --user --name sleepwm --display-name "Python (SleepWM)"` 注册 kernel。唯一配置入口为 `configs/psg_training.yaml`。已移除旧 backbone 配置和实现，`build_backbone` 始终构建当前 CNN/Criss-Cross 主干。

## 本地产物与可清理文件

以下为 2026-09-07 检查结果，大小按 MiB 统计。

| 路径 | 内容 | 清理建议 |
|---|---|---|
| `artifacts/pytest-*`、`artifacts/pytest_sampling_*` | 自动测试生成的合成数据，约 86 MiB | 测试结束后可删除 |
| `artifacts/i0002-full-work`、`artifacts/qc-cache` | 当前为空的处理工作目录和缓存 | 可删除；后续运行可能重建 |
| `artifacts/qc-pilot-20260903-01` | 旧版 QC pilot 数据和报告，约 5,090 MiB | 当前正式发布不依赖它；不再需要历史复核时可删除 |
| `artifacts/i0002_flagged_epochs` | 旧 pilot 标记 epoch 的导出与检查材料，约 323 MiB | 不再需要人工复核材料时可删除 |
| `artifacts/source-scan-*`、`artifacts/audit-pilot-*`、`artifacts/docs-backups` | 历史扫描、审计日志和文档备份 | 确定不需要追溯后可删除 |
| `artifacts/checkpoints` | 7 个模型权重文件，约 371 MiB | 删除会失去对应模型状态，按需要保留 |
| `artifacts/wandb` | 训练实验记录和离线同步文件 | 需要查询或同步实验时保留 |
| `identifier.db` | 12 KiB 空 DuckDB 数据库，无表、无自定义 schema，当前代码无引用 | 可以删除；仅凭现有文件无法确定创建来源 |
| `sleep_world_model.egg-info` | setuptools 生成的包名、依赖和命令入口等安装元数据 | 可以重新生成；删除可能影响依赖该目录的安装发现或命令入口，随后执行 `python -m pip install -e . --no-deps` 恢复 |

`artifacts` 是本地运行产物目录，正式 I0002 分片在 I 盘。它包含有价值的训练和复核结果，不应把整个目录当作临时缓存。`identifier.db` 与 `sleep_world_model.egg-info` 已按用户要求删除，历史产物按需保留。W&B 记录内的历史 `requirements.txt` 是当时环境快照，保留原样。

## 正式分片读取与训练

```text
dataloader/           # 独立顶层包，供不同模型复用
  schema.py           # Reader 协议与数据元信息
  dataset.py          # 通用记录窗口索引、读取器注册
  sampler.py          # subject 级数据划分
  collate.py          # 补齐、QC 与标签组装
  readers/
    hsp.py            # v1.0.0 发布格式
    hsp_raw/          # 从旧 data/hsp 迁入的原始 H5 工具
backbone/             # 连续 PSG 表示主干
  architectures/     # CNN、Transformer 等可替换算法家族
  modeling/          # patch、序列、通道、模态编码与组合
pretraining/          # 组合 backbone、目标分支、head 与 loss
  lejepa/             # 共享 backbone、projector、多视图一致性
  views.py            # raw crop、通道/模态 dropout
  objectives/         # SIGReg 等可复用目标项
experiment_logging/   # 独立 W&B 日志，不依赖模型
scripts/Copy-HSP-I0002.ps1  # 源数据复制工具
```

```python
from torch.utils.data import DataLoader
from dataloader import WindowDataset, collate_windows

with WindowDataset(
    root=r"I:\HSP\I0002-preprocess\processed\v1.0.0",
    dataset="hsp", version="v1.0.0",
    split="train", split_seed=42, split_ratios=(0.8, 0.1, 0.1),
    context_epochs=20, stride_epochs=20,
    tasks=("sleep_stage", "heart_rate", "sao2"),
) as dataset:
    loader = DataLoader(dataset, batch_size=2, num_workers=0,
                        collate_fn=collate_windows)
    batch = next(iter(loader))
```

Windows 多进程脚本应把 DataLoader 创建和迭代放入 `if __name__ == "__main__":`。句柄不随 Dataset 序列化，worker 各自以只读模式打开文件；`max_open_files` 默认 8，限制每进程的打开文件数。

- root 指向发布版本目录。版本由 `manifests/release.json` 验证；只读取 manifest 与 commit 一致的文件，拒绝 .partial、越界路径和身份/时间索引不一致的记录。
- 支持发布中并存的旧内部 schema `1.0.0-pilot.2` 和恢复后的 `1.0.0-full.1`，它们都属于公开版本 v1.0.0。
- 一个窗口只属于一个 recording。默认保留末尾短窗口并在 batch 中补齐；`drop_last=True` 可仅保留完整窗口。epoch_mask 表示真实 epoch，补齐区标签和 QC 有效性均为 false。
- 信号保持存储中的物理值、200 Hz 和五个模态组。batch 包含 signals、quality、available_mask、tasks、通道/单位信息和记录/epoch 标识；数值标签即使无效也保留原值，必须结合 valid / field_valid 使用。
- tasks 默认不加载；按需传入任务名，SSL 训练不需要加载下游标签。
- 默认检查大小、提交摘要一致性、绑定及读取窗口的索引；`verify_checksums=True` 额外在打开文件时校验整文件 SHA256，有额外 I/O 成本。
- 数据划分由 `dataset + subject_id + split_seed` 的稳定哈希决定，同一 subject 的所有 session 在同一 split。默认 80/10/10，是哈希分配概率而非精确计数；不写回发布文件，也不沿用旧 raw manifest 的划分。

在 `configs/psg_training.yaml` 中选择数据源：

```yaml
data:
  mode: real  # real：正式数据；synthetic：合成调试数据
  root: 'I:\HSP\I0002-preprocess\processed\v1.0.0'
```

CLI 默认直接读取该配置，PyCharm 参数可留空：

```powershell
& $sleepPython -m pretraining.cli
```

适配在 `dataloader/signals.py` 完成，将 respiratory 中的 SpO₂ 拆为独立编码组；QC 保持来源 epoch/通道粒度。其他模型可直接复用 dataloader，无需导入训练包。接口见 [读取接口契约](dataloader/FORMAT.md)。

项目分发名称仍为 `sleep-world-model`。重新安装后仅保留训练命令 `train-psg-foundation`，它与 `python -m pretraining.cli` 调用同一个 `main`；数据 manifest 工具独立保留。

新增数据集时，实现 `Reader` 协议的 metadata、records、read_window、close，并用 `register_reader("name", ReaderClass)` 注册；记录元数据包含 recording_id、subject_id、session_id 和 n_epochs，窗口字典遵循 hsp.py 的输出字段及张量形状。同格式的数据复用读取器，不按 cohort 复制代码；新格式或 schema 必须明确实现并验证，不能仅改版本字符串。注册应在入口创建 Dataset 前完成。只有当前 HSP 发布读取器已用真实数据验证。

## I0002 预处理

正式产物位于 `I:\HSP\I0002-preprocess\processed\v1.0.0`。发布清单记录状态 COMPLETE：15,028 条记录、942 个分片、13,669,728 个 epoch；固定排除 3 条，失败 0 条。

- [完整清洗方案](docs/I0002_PREPROCESS_PLAN.md)：15 通道、200 Hz、30 秒 epoch、float32 物理值；signal、QC 和三个 task 分开存储。
- `core.py`：物理值校准、来源滤波判断、重采样、冻结 PSD 与 hard QC。
- `pipeline.py`：原始 H5 / CSV 标签读取、时间对齐、mask、配置、分片打包、验证及 pilot。
- `full.py`：全量调度、断点恢复、跨盘复制校验、提交清单和进度。
- `__main__.py`：无 `--full` 时进入 pilot，带 `--full` 时进入全量流程。

```powershell
& $sleepPython -m data_preprocess.i0002_v100 --help
& $sleepPython -m data_preprocess.i0002_v100 --full --help
```

三个任务为 sleep_stage、heart_rate、sao2，直接读取原始 CSV，通过 `recording_id + epoch_in_record + epoch_start_offset_ns` 与信号对齐。标签有效性由 task code 和 mask 指定。数据读取以 `manifests/shards.parquet`、`manifests/records.parquet` 与提交凭证为入口；发布读取器按窗口读取 H5，并为每个进程维护有界文件句柄缓存，不把整个分片加载到内存。

已有正式数据无需重新运行。若进行受控复现或恢复，应先核对所有 CLI 路径、批准 pilot 和冻结代码；发布记录引用的默认 pilot 目录当前已不存在，不能将 `--resume` 当作无需准备即可运行的命令。三个核心文件和方案参与 SHA256 校验，整理目录时保持原始字节。

## 源扫描工具

| 脚本 | 用途 |
|---|---|
| `scripts/scan_hsp_source_metadata.py` | 只读扫描 H5、EDF、BIDS 头信息，输出源数据画像 |
| `scripts/generate_hsp_source_report.py` | 从扫描结果生成统计 Markdown |
| `scripts/analyze_f3_m2_calibrated_ranges.py` | 标定后 F3-M2 量程抽样及 H5/EDF 对照 |

这些工具独立运行。扫描结果位于 `I:\HSP\I0002-preprocess\source_scan\v1.0.0`，其中 sessions.parquet 是正式全量流程的 inventory。报告中的标定抽样章节对应第三个脚本；它所需的详细扫描表是否齐全，应在重跑前检查。

```powershell
& $sleepPython scripts/scan_hsp_source_metadata.py --help
& $sleepPython scripts/generate_hsp_source_report.py --help
& $sleepPython scripts/analyze_f3_m2_calibrated_ranges.py --help
```

<a id="legacy-loader"></a>
## 原始 H5 数据接口

本节保留现有 package 的使用方式，用于兼容旧实验及调试。其通道布局、采样率、mask 含义不等同于新分片。

### Session manifest

~~~powershell
& $sleepPython -m dataloader.readers.hsp_raw.cli `
  --root 'I:\HSP\I0002' `
  --output 'artifacts\hsp_i0002.jsonl'
~~~

每行对应一个 H5 session，记录通道、采样率、长度、标定、注释及被试级 train/validation/test 划分。`--limit 100` 可做小规模扫描，完整扫描每 100 个 session 报告进度。

### Dataset 与 DataLoader

~~~python
from torch.utils.data import DataLoader, RandomSampler
from dataloader.readers.hsp_raw import (
    HSPDataset, RandomChannelDropout, hsp_collate_fn,
)

dataset = HSPDataset(
    root=r"I:\HSP\I0002",
    manifest_path="artifacts/hsp_i0002.jsonl",
    split="train",
    channel_profile="hsp_full",
    sampling_mode="context",
    epoch_seconds=30,
    context_epochs=20,
    missing_channel="mask",
    target_sample_rates={
        "eeg": 100, "eog": 100, "emg": 100,
        "ecg": 100, "resp": 25, "spo2": 25,
    },
    normalization="none",
    transform=RandomChannelDropout(
        probability=0.2, modalities=("eeg",), min_remaining=1,
    ),
)
batch_size = 8
sampler = RandomSampler(
    dataset, replacement=True, num_samples=10_000 * batch_size,
)
loader = DataLoader(
    dataset, batch_size=batch_size, sampler=sampler,
    num_workers=4, persistent_workers=True, collate_fn=hsp_collate_fn,
)
~~~

Windows 下将带多 worker 的可执行脚本入口放在 `if __name__ == "__main__":` 保护中。上面的 replacement sampler 避免为所有重叠 context 建立巨大排列，**并不解决机械硬盘随机读取问题，也不应照搬到整 shard 缓存上**。

`sampling_mode` 支持 `epoch`（一个 30 秒片段）、`context`（连续多个 epoch）、`night`（整夜，由 collate 补齐）。默认按需读窗口，只有显式 night 模式才读取整夜。整夜样本可能占数百 MB；长范围实验可先用 90 分钟 context，或将整夜任务建立在缓存的 epoch embedding 上。

单个模态形状为 `[epochs,channels,samples_per_epoch]`，batch 后为 `[B,epochs,channels,samples_per_epoch]`。输出字典含 `signals`、`sample_rates`、`available_mask`、`channel_mask`、`epoch_mask`、`sample_mask`。

`available_mask` 表示源通道存在，`channel_mask` 表示增强/dropout 后可见通道。缺失通道零填充仅是张量占位，必须用 mask 忽略。这些存在性/增强/补齐 mask **不是** 新的 30 秒 SQA 结果。

未提供 `target_sample_rates` 时不重采样。内置 windowed-sinc 抗混叠重采样仅支持整数倍降采样，按每个通道自身采样率处理；100/25 Hz 默认值兼容常见 25/200/500 Hz 来源。不能用它代替新流程的 500→200 Hz 或 SpO2 25→200 Hz 处理。

`normalization="zscore"`、`"robust"` 在当前样本/context 内计算，可能消除跨夜幅度差异；默认 `none`。若需整夜统计归一化，可通过明确设计的 transform 提供。

<a id="training"></a>
## 训练与调试

训练使用 [pretraining.cli](pretraining/cli.py)，配置使用 `configs/psg_training.yaml`，详见 [预训练说明](pretraining/README.md)。
`tests/test_pipeline.py` 和 notebook 调用同一套模型、loss、runtime 与日志组件，提供逐步调试，不保留第二套 SSL 实现。
历史最小 SSL checkpoint 保留为实验产物，当前模型不加载该格式。

## 测试

```powershell
& $sleepPython -m pytest tests/test_i0002_v100.py -q
& $sleepPython -m pytest tests -q
```

当前预处理测试覆盖信号处理、PSD、任务 code / mask、分片打包校验、排除记录、恢复清单及 Windows 文件锁处理。若系统 pytest 临时目录不可写，可指定一个新的 `--basetemp` 路径并加 `-p no:cacheprovider`。
