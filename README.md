# SleepWorldModel

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
| [world_model](world_model) | 模型、训练与调试 |
| [tests](tests) | 当前预处理及独立功能的回归测试 |

[代码使用依据与清理记录](data_preprocess/CODE_MAP.md) · [原始 H5 接口](#legacy-loader) · [SSL](#training)

独立 backbone 已实现，见 [实现规格](docs/PSG_ARCHITECTURE_PROPOSAL.md) 和 [逐步 notebook](notebooks/backbone_step_by_step.ipynb)。现有 SSL 入口仍使用 `world_model.ssl` 基线；新主干的正式预训练方法后续接入。

## 环境

Python ≥3.11。从仓库根目录运行；安装包包含 backbone、dataloader、world_model 以及预留的 pretraining/tokenization 包。

```powershell
Set-Location 'E:\Code\SleepWorldModel'
$sleepPython = 'C:\Users\user\miniconda3\envs\SleepWM\python.exe'
& $sleepPython -m pip install -r requirements.txt
& $sleepPython -m pip install -e . --no-deps
```

`requirements.txt` 合并数据处理、训练和测试依赖；`pyproject.toml` 保留项目打包信息及依赖声明。

## Backbone 单步测试

在 Jupyter/VS Code 打开 `notebooks/backbone_step_by_step.ipynb`，选择 **Python (SleepWM)** kernel，按 Shift+Enter 逐个执行。默认合成数据；参数单元格将 `USE_REAL_DATA=True` 切到 I 盘 HSP，`DEVICE='cuda'` 切换 GPU。每层变量可直接查看，网络块输入通过临时 hook 展示。

```powershell
& $sleepPython -m jupyterlab notebooks/backbone_step_by_step.ipynb
# 无界面执行同一份 notebook，结果保存在 artifacts/backbone：
& $sleepPython scripts/check_backbone_notebook.py
& $sleepPython scripts/check_backbone_notebook.py --real --device cuda
```

首次在其他环境使用时，执行 `python -m ipykernel install --user --name sleepwm --display-name "Python (SleepWM)"` 注册 kernel。主干默认配置位于 `configs/backbone/base.yaml`。

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
tokenization/         # 可选的离散表示生成器
  codebook/           # masked-code 等任务使用的码本
pretraining/          # 组合 backbone、目标分支、head 与 loss
  jepa/               # online/target backbone 与 predictor
  contrastive/        # 双视图、projector 与配对目标
  masked_code/        # 离散 code 预测
  objectives/         # SIGReg 等可复用目标项
world_model/
  ssl/                # 当前最小 SSL 基线，待新 backbone 实现后迁移
  training/           # 模型专用输入适配、训练和日志
  debugging/          # 单步调试
  io/                 # H5 I/O 与源数据复制工具
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

正式分片训练入口：

```powershell
& $sleepPython -m world_model.training.ssl_cli `
  --dataset hsp --version v1.0.0 `
  --root 'I:\HSP\I0002-preprocess\processed\v1.0.0' `
  --split train --split-seed 42 --split-ratios 0.8 0.1 0.1 `
  --context-epochs 20 --batch-size 2 --num-workers 2 `
  --device cuda --wandb-mode disabled
```

训练从数据元信息构造输入配置，保留发布数据的五组模态：EEG 6、EOG 2、ECG 1、EMG 3、respiratory 3，全部 200 Hz。respiratory 内含 Airflow、Snore、SpO2。

QC 按每个 epoch、每个通道生效；一个坏 epoch 不影响该通道的其他有效 epoch。无效通道先置零，整组无效时屏蔽该模态的特征，所有模态均无效的 epoch 不参与损失。

`dataloader` 不导入 `world_model` 或预处理代码，其他模型可直接复用它。模型输入适配位于 `world_model/training/batch_adapter.py`。完整接口见 [读取接口契约](dataloader/FORMAT.md)。模型根据元信息建立卷积输入层；改变通道布局需要重新建立模型，不能直接套用旧权重。

Python 包名为 `world_model`，项目分发名称保留 `sleep-world-model`；运行 `python -m pip install -e . --no-deps` 可刷新本机安装入口。

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
## SSL 模型、训练与调试

当前基线为小型 CNN、两视图一致性损失和 SIGReg。实现与 Review 记录见 [SSL 模块说明](world_model/ssl/SSL.md)。

每个 30 秒 epoch 独立编码：各模态经过两层一维卷积与全局池化，拼接后生成 embedding，再经过共享投影头。训练时对同一 epoch 做两次随机时间裁剪，默认保留 80%；同一视图内各模态按相对时间对齐。模型内部使用 `sign(x) * log1p(abs(x))` 压缩幅度，读取器和磁盘上的物理值保持原样。

损失只有两项：`(1 − λ) × MSE(z1, z2) + λ × (SIGReg(z1) + SIGReg(z2)) / 2`，默认 λ=0.05。SIGReg 将随机单位方向上的投影分布约束到标准高斯，采用特征函数和梯形积分计算。每个 batch 至少需要两个有效 epoch；不足时明确报错。SIGReg 始终使用 FP32，CUDA 编码器训练使用 BF16。

`context-epochs` 控制一次读取的窗口大小；当前模型不学习 epoch 之间的时序关系。用于下游任务的完整 epoch 表示来自 `model.encode(batch)`，形状为 `[batch, epochs, embedding_dim]`，并返回有效 epoch mask。

### 训练

```powershell
& $sleepPython -m world_model.training.ssl_cli `
  --root 'I:\HSP\I0002-preprocess\processed\v1.0.0' `
  --context-epochs 20 --batch-size 8 --num-workers 2 `
  --max-steps 1000 --device cuda --wandb-mode disabled
```

默认使用 AdamW、固定学习率 0.001、梯度裁剪 1.0。可通过 `--hidden-dim`、`--embedding-dim`、`--projection-dim` 调整模型大小；默认分别为 32、128、64。SIGReg 默认 256 个方向、17 个积分点，分别由 `--sigreg-projections` 和 `--sigreg-frequencies` 控制。完整参数运行 `--help` 查看。

W&B 默认关闭；`--wandb-mode offline` 在 `artifacts/wandb` 保存日志，`online` 在配置凭据后上传。训练记录总损失、一致性损失、SIGReg、embedding 标准差、梯度范数、学习率和有效 epoch 数。

默认权重路径为 `artifacts/checkpoints/ssl_minimal.pt`，保存模型配置、读取配置、损失配置、模型及优化器状态。新版 schema 3 / `minimal-sigreg-v1` 与旧层级 SSL 权重不兼容；旧权重保留在原位置。当前入口从头训练，不提供断点续训参数。

### 调试与提取表示

调试入口读取一个 batch，执行 forward、loss、backward，不更新参数或保存权重。不带参数时使用合成数据。

```powershell
& $sleepPython -m world_model.debugging.ssl_step --synthetic
& $sleepPython -m world_model.debugging.ssl_step `
  --root 'I:\HSP\I0002-preprocess\processed\v1.0.0' `
  --context-epochs 4 --batch-size 2 --break-at model
```

断点支持 `batch/model/loss/backward/none`；`--forward-only` 跳过反向传播。也可在 IDE 中调试对应模块。

```python
import torch
from dataloader import WindowDataset, collate_windows
from world_model.ssl import SSLConfig, SSLModel
from world_model.training.batch_adapter import prepare_batch

saved = torch.load("artifacts/checkpoints/ssl_minimal.pt", map_location="cpu", weights_only=False)
model = SSLModel(SSLConfig(**saved["model_config"]))
model.load_state_dict(saved["model"])
model.eval()
with WindowDataset(r"I:\HSP\I0002-preprocess\processed\v1.0.0", context_epochs=4) as dataset:
    batch = prepare_batch(collate_windows([dataset[0]]), model.config, torch.device("cpu"))
    with torch.no_grad():
        embeddings, valid = model.encode(batch)
```

## 测试

```powershell
& $sleepPython -m pytest tests/test_i0002_v100.py -q
& $sleepPython -m pytest tests -q
```

当前预处理测试覆盖信号处理、PSD、任务 code / mask、分片打包校验、排除记录、恢复清单及 Windows 文件锁处理。若系统 pytest 临时目录不可写，可指定一个新的 `--basetemp` 路径并加 `-p no:cacheprovider`。
