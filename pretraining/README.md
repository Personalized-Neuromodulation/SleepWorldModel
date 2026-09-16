# Pretraining

监督分期适配通过同一 CLI 的 `training.task: finetune` 启动，模块与梯度边界见
[finetuning/README.md](../finetuning/README.md)。监督成绩使用独立 `supervised_*` 指标。

## 本轮模态 JEPA + MoE 实验

`pretraining.objective=modality_jepa` 使用 `modality_jepa/`，复用原factory、CLI、training/evaluation和checkpoint。`objective=lejepa`保留下面的历史基线；两个目标不混用EMA语义。

- 每个30秒样本产生2个mask view，拼接后一次online forward；干净EMA modality encoder只计算一次。
- 默认25%概率遮挡一个有效模态；否则每模态遮挡40%的3秒块。没有另一套global/local backbone，不裁出短local窗口。
- `MaskPlan`作用于在线原始输入，预测器读取可见的 `[B,G,30,256]` 融合token，目标是干净、融合前的各模态 `[B,30,256]` latent。
- 目标encoder（含tokenizer/channel pool）冻结、始终eval，成功更新后以decay=.996做EMA；跳过batch不更新。训练checkpoint包含teacher、predictor、sampler及RNG；HF只导出online backbone。
- 总loss为prediction + `.01*SIGReg + .01*router_balance`；不同模态的预测MSE等权汇总。SIGReg保持view/modality独立，沿有效来源B统计，projector256→512→128，FP32。
- 新损失不训练汇总readout，要求modality/temporal masked mean；channel attention仍可训练。

完整运行配置位于 `results/foundation/modality_jepa_moe_10k_20260914/psg_training.yaml`（引用同目录模型配置）。配置全部可覆盖：`model.moe.*`、`pretraining.modality_jepa.*`和`pretraining.sigreg.*`；无额外入口或依赖。

每步新增W&B指标：`loss_prediction`、`prediction/<modality>`、`prediction/local`、`prediction/whole_modality`、`targets/local`、`targets/whole_modality`、`loss_sigreg`、`loss_router`、`router_entropy`、`router_load/<expert>`、`target_std/<modality>`。均有train/train_eval/eval前缀，不再误标loss_inv。路由负载是按有效token计数的专家选择频率；target_std排除缺失来源。固定mean readout的权重不代表学到的模态重要性。

本轮10K，每1K评价/保存，显式 `eval_every_seconds=0`。固定train/validation cohort、原在线probe以及refit指标保持原定义。改变了fusion和objective两项，结果只能检验组合方案；要归因于MoE本身还需要Dense-FFN同目标消融。

`tests/test_pipeline.py --config <本轮运行配置> --break-at views/loss/backward` 可分别观察MaskPlan、各模态prediction/target、MoE和teacher梯度。`tests/test_modality_jepa.py`覆盖信息泄漏、whole/local masks、无效数据、EMA更新、CPU/CUDA、HF加载和精确checkpoint恢复。

## 历史 LeJEPA 基线


This package implements LeJEPA around one shared backbone.

SpO₂ 的固定预处理由 `data.scales.spo2: {offset: 95.0, divisor: 5.0}` 配置；数值 tokenizer 由 `model.numeric_tokenizers.spo2` 配置。CPU 读取/合并后执行 `(原始百分数−95)/5`，每秒均值经 MLP 生成 256 维 token。训练与评价共用相同入口，任务标签仍为原始单位。改变输入变换或 tokenizer 后必须从头训练，不能当作原 CNN checkpoint 的精确 resume。HF 推理也需在模型外完成同样变换。

```text
pretraining/
├── lejepa/            # shared backbone, projector and multi-view objective
├── views.py           # aligned raw crops and mask dropout
└── objectives/        # reusable objective terms such as SIGReg
```

Unimplemented JEPA/contrastive/masked-code/codebook placeholder packages were removed. Future methods can reuse the same backbone contracts when implemented.

The old SSL package and its training/debug entry points have been removed.
The only training entry is `pretraining.cli:main`; `train-psg-foundation` invokes
that same function. `test_pipeline` and notebooks are debugging clients of this
implementation. W&B logging lives independently in `experiment_logging`.

## LeJEPA implementation

`python -m pretraining.cli` is the unified entry. It reads `configs/psg_training.yaml`
(data, training and evaluation), which references `psg_model.yaml` through
`model_config`. The model file contains `model` and `pretraining` settings.
Relative model-file paths resolve against the training YAML directory, not the
process working directory. `--config` can select another training YAML.
Checkpoint config.json and W&B receive the merged values for reproducibility.
Set `data.mode: synthetic`, `evaluation.frozen_linear_probe: false` and `training.max_steps: 1` in YAML for an unlabeled debug run. Dotted overrides are
validated before constructing any network. The default backbone has 19,549,824 parameters.

CNN depth is the length of `model.patch_tokenizer.layers`; edit the YAML list to
add/remove convolutions. Indexed overrides are supported, for example
`model.patch_tokenizer.layers.0.kernel_size=25 model.patch_tokenizer.layers.0.padding=12`.
`model.criss_cross`, `model.temporal` and `model.fusion` expose attention heads and
FFN widths. See [backbone configuration](../backbone/README.md) for all network settings.
HF exports preserve the resolved architecture as well as its weights.

GPU execution is configured in `psg_training.yaml`:

```yaml
training:
  device: cuda
  precision: bf16
  fused_adamw: true
  num_workers: 0  # RAM pool owns its background reader; no process copies.
```

`runtime.py` shares device, autocast and optimizer construction with `test_pipeline`.
CPU debugging explicitly uses `training.device=cpu`; it runs FP32 and standard AdamW.
GPU hardware without bf16 support can use `training.precision=float32`.
The real-data collator adapts and validates SignalBatch on CPU (inside workers when
enabled), then DataLoader pins the resulting tensors, including derived masks.
`prepare` transfers them with `non_blocking=True` without repeating QC.

Full global views reuse input waveform storage. A bucket containing one view does
not concatenate/copy it. Dropout packs the tiny modality masks into one H2D copy
per view. SIGReg uses masked sums over a fixed tensor shape; invalid rows have zero
contribution and gradients. `skip_update` remains a scalar GPU tensor until the
training loop consumes it. Standalone SIGReg calls still validate inputs; LeJEPA
checks the final loss at the training boundary. The loop retains necessary loss,
gradient and skip checks, and transfers scalar logging metrics together.

```bash
python -B tests/test_pipeline.py --break-at model
python -B -m pretraining.cli
```

Both commands default to GPU and read `data.mode` and `data.root` from YAML.
`real` selects the published dataset; `synthetic` generates debug inputs and ignores
`data.root`. The separate `--synthetic` flag has been removed. Real mode preserves
subject splits, night-grade filtering and the configured `max_steps`; it does not
mean combining all splits or making exactly one sequential pass over the dataset.

PyCharm can run/debug `pretraining/cli.py` directly, or use module name
`pretraining.cli`; both call the same `main`. For a one-step debug run, set
`data.mode: synthetic`, `evaluation.frozen_linear_probe: false` and `training.max_steps: 1` in YAML, leave parameters empty, and put a breakpoint inside
`main` or `train`. The default YAML resolves relative to this project regardless
of the IDE working directory. Explicit relative paths in overrides (such as
`training.output_dir`) still resolve against the working directory.

- `configuration.py / factory.py`: validate YAML, build the existing backbone recipe,
  shared projector, SIGReg and raw view sampler.
- `views.py`: crop raw inputs on whole seconds; preserve source epoch QC.
  Static `SignalGroup.data_valid` is prepared by the dataloader and reused by
  every crop/dropout view. Dropout changes `channel_mask` (and thus `visible`)
  only. Bucket concatenation includes the prepared QC tensor along B.
  Local lengths follow a seeded shuffled 1–15 cycle, shared by all local views in
  one batch. Adjacent batches differ when the configured range has more than one length.
- `lejepa/model.py`: bucket views by length, execute one batched forward per bucket,
  compute valid-view mean invariance and per-view global/local SIGReg.
- `objectives/sigreg.py`: FP32 sliced Gaussian characteristic-function matching.
- `evaluation.py`: fixed train/validation subsets and shared representation/task metrics.
- `probes.py`: detached linear heads sharing one clean 30-second foundation
  representation. Sleep-stage labels 1–5 map to indices 0–4; heart-rate and
  saturation regression use field masks from the existing dataloader.
- `training.py / cli.py`: AdamW, W&B, best/final checkpoints and HF export.

No EMA, teacher or predictor is used in LeJEPA. SIGReg retains `[V,B,D]`,
estimates each view's distribution across valid source samples, then averages
views with at least two valid sources. Repeating identical views does not multiply
the regularizer. All computations remain FP32; no Python sample/view forward loop.
At least two source samples and a paired sample are needed for a training update.
Fully dropped/invalid views are excluded. If all attempted batches are invalid,
training raises an error. A finite loader may finish with fewer updates than
`max_steps`; the returned `steps` and `skipped_batches` make this explicit.

W&B is online by default and reads WANDB_API_KEY from the project-root .env;
existing environment variables take priority. Automated tests default to disabled
except the offline logging test; the global test accepts --wandb-mode online.
Checkpoints contain the resolved configuration, backbone input
specification, optimizer and SSL state. `backbone/` contains the Hugging Face
export, which loads with `AutoModel.from_pretrained(..., trust_remote_code=True)`
in an environment with this project installed. No full dataset training is
started automatically by validation scripts or notebooks.

## Simultaneous task validation

W&B progress is logged at initialization, after every consumed training batch,
and at the end of training. Training and validation charts use
`train/global_step` (successful SSL optimizer updates) as their default x-axis.
W&B's internal history row counter is separate, so skipped batches and final
metrics are retained even when the optimizer step has not changed.

| Progress metric | Meaning |
|---|---|
| `train/global_step` | Completed SSL optimizer updates; task-head updates do not increment it |
| `train/batches_seen` | Batches consumed by the training loop, including skipped batches |
| `train/samples_seen` | Actual source windows consumed, including skipped batches; views are not counted again |
| `train/samples_trained` | Actual source windows in successful SSL batches, before per-view masking |
| `train/skipped_batches` | Batches excluded from SSL updates |
| `train/dataset_windows` | Number of windows in the filtered training dataset |
| `train/epoch_equivalent` | `samples_seen / dataset_windows`, including repeats and skipped batches |
| `train/progress_percent` | `100 * global_step / max_steps` |
| `train/elapsed_seconds` | Elapsed training-loop wall time, including evaluation |

Epoch equivalent is sampled volume, not unique dataset coverage or PSG's
30-second scoring epoch. Current sampling uses replacement and balances night
grades. Synthetic lists and custom loaders without a dataset length omit epoch
equivalent rather than inventing a denominator.

Training losses, valid-view ratio, gradient norm, learning rate, throughput and
GPU memory are logged after every successful update. Evaluation SSL losses,
cosine similarity, embedding standard deviation/effective rank and all enabled
task metrics for BOTH splits are logged together every
`evaluation.eval_every_steps` (currently 1,000), and at the final update.
A loader ending early triggers a final evaluation if needed.
`evaluation.max_batches` limits batches separately for each evaluated split, not
the logging interval. No new run or process restart is performed by code edits;
an already-running process keeps the code it loaded when it started.

The runtime YAML enables `evaluation.frozen_linear_probe: true` with
`probe_tasks: [sleep_stage, heart_rate, sao2]` and `probe_learning_rate: 0.001`.
Heads train on the training subject split and are evaluated, without updates, on
both the training and validation subject splits. They receive the same clean backbone features;
no separate encoder forward is needed for each task. These are online detached
probes of an evolving SSL encoder, not fully converged post-training benchmarks.

Three W&B groups distinguish live optimization from periodic evaluation:

- `train/*`: per-update loss and training progress.
- `train_eval/*`: periodic, fixed training-set subset metrics in evaluation mode.
- `eval/*`: periodic, fixed validation-set subset metrics (existing names retained).

When task probes are enabled, their existing clean forward also supplies readout
diagnostics. `training.readout_log_every_steps` defaults to 100; it adds
`train/readout/modality_entropy`, `modality_max_weight`, and
`train/readout/<modality>/{weight,feature_norm}` without another forward.
`train/readout/fused_feature_norm` follows the pooled Fusion output so growth
after the encoder normalization boundary remains visible.
Periodic train/validation evaluation reports the same fields under its own
prefix. Norm averages exclude missing positions. Entropy is in nats and depends
on available modality count; it is not a target to maximize. When probes are
disabled these clean-pass metrics are omitted. The global debug pipeline also
includes `global_view_readout`, measured on its augmented global view.

Every metric below also appears with `train_eval/` instead of `eval/`, as do
SSL losses, validity and representation diagnostics. The CLI samples each
subset uniformly without replacement from the corresponding filtered subject
split, once at startup, using independent fixed seeds. Indices are sorted for
disk locality. Both subsets reuse the same evaluation code and view seed.
They do not iterate the training sampler, alter training RNG, or update either
the backbone or probe heads. Test-set data is not used. With batch_size=192
and max_batches=128, up to 24,576 windows are evaluated per split. The values
describe these fixed subsets, not the entire multi-million-window datasets.
The additional training-set pass increases periodic evaluation cost.

Custom `train(...)` clients using a DataLoader or one-shot training iterator must
pass a separate `train_evaluation_loader`. Reusable small lists/tuples used by
tests may be evaluated directly. `test_pipeline` keeps the explicit debug-batch
namespace rather than labeling the same debug batch as held-out data.

| Task | Outputs | W&B metrics |
|---|---|---|
| sleep_stage | 5 logits | `eval/macro_f1`, `eval/balanced_accuracy`, `eval/probe_samples` |
| heart_rate | max_bpm, min_bpm, mean_bpm | `eval/heart_rate/{mae,rmse,samples,valid_values}` and `eval/heart_rate/<field>/{mae,rmse,r2,samples}` |
| sao2 | max_percent, min_percent | `eval/sao2/{mae,rmse,samples,valid_values}` and `eval/sao2/<field>/{mae,rmse,r2,samples}` |

Sleep staging also reports overall `eval/accuracy` and per-stage
`eval/sleep_stage/<stage>/{f1,accuracy,precision,support,predicted_samples}`;
the same fields are available under `train_eval/` and debug-batch prefixes.
The published HSP label order is 1=N3, 2=N2, 3=N1, 4=REM, 5=WAKE.
Per-stage accuracy means `TP / true-stage support` (recall), not one-vs-rest
accuracy inflated by true negatives from other stages. When support is zero,
accuracy is a zero placeholder, not an observed score; consult `support` and
exclude that stage from balanced accuracy. Macro-F1 continues averaging all five
stages for compatibility. Overall accuracy is the confusion-matrix trace / total.

Regression heads train with MSE on labels divided by the fixed scale 100;
predictions are multiplied by 100 for metrics (bpm or percentage points).
No validation statistics are used for scaling. Task-level MAE/RMSE pool all
valid field values; `samples` counts rows with at least one valid field, whereas
`valid_values` counts scalar targets. Per-field R² uses all validation batches
together and is omitted for constant labels or fewer than two valid samples.
An empty task reports counts only, without invented zero errors. Each task uses
its own label mask intersected with backbone validity; regression uses
`field_valid` rather than the all-fields `task.valid`. NaN/Inf targets are excluded
before loss arithmetic. Missing label files for an explicitly selected task
remain a reader error, rather than silently evaluating an untrained head.

Best-checkpoint selection continues to use sleep-stage macro-F1 when that probe
is enabled; without it, selection uses SSL validation loss. Regression metrics
are reported alongside classification and do not change the selection rule.
Probe parameters are saved in `training.pt`; HF backbone exports exclude task heads.

The same implementation is exercised by `tests/test_pipeline.py --break-at probe`.
Its `debug_batch/*` values follow one head update on the same batch and are only
debug diagnostics. Real training uses held-out validation. Synthetic inputs have
no task labels: disable probes explicitly for synthetic smoke runs.

Reference: [LeJEPA official minimal example](https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md).


## Current 10K training and restart

Evaluation now defaults to `evaluation.eval_every_seconds: 600.0`: after ten
minutes of training and input waiting, the next successful update triggers full
train/validation evaluation, W&B logging and a checkpoint. Evaluation/checkpoint
time is excluded from the next interval, and a final evaluation always runs.
Set seconds to `0` to use `eval_every_steps`. This does not alter the learning-rate
schedule or maximum optimizer steps. On resume the timer starts a fresh interval;
changing evaluation cadence is supported but changes evaluation history and may
change which checkpoint is selected as best.

The default run uses 10,000 updates and evaluation/checkpoints every ten minutes,
AdamW peak lr1e-3, 5% linear warmup and cosine decay to1e-5. Scheduler advances
only after successful optimizer updates. Network depths and augmentation are unchanged.

`evaluation.ram_cache_gib: 20` bounds both fixed raw evaluation caches together.
The first evaluation reads the same fixed subjects/windows in the same order;
later evaluations reuse CPU inputs and re-encode them with the current model.
Only each transferred batch is pinned. The training RAM pool reserves space for
these caches in addition to its system reserve. Set cache to0 to disable.

After each split evaluation returns, the training loop releases unused CUDA
allocator cache before entering the next phase. This happens only at evaluation
boundaries; live model/optimizer tensors and the CPU input caches remain intact.
W&B records `train_eval/cuda_reserved_before_release_bytes` and
`train_eval/cuda_reserved_after_release_bytes`, plus the corresponding `eval/`
fields. On this Windows GPU, two controlled real-batch benchmarks reproduced
roughly 3x slower GPU steps after evaluation and recovery after cache release.
This excludes normal data loading and is not a whole-run speedup estimate.

`evaluation.refit_probe` adds independent `train_refit/*` and `eval_refit/*`.
It reuses clean features from the current evaluation, fits normalization and
linear heads on the fixed training cohort only, then evaluates both cohorts.
Classification uses regularized multinomial logistic regression (CPU LBFGS);
regression uses ridge with intercept. Defaults: max_iter200, l2=.01,
ridge_alpha=100, no class weighting. Solver iteration count is logged. No feature
cache is reused across model updates. The online probe remains unchanged and
`eval/macro_f1` remains best-checkpoint selection and the0.7 target metric.

Every evaluation writes `checkpoints/step-00001000/` etc. Each contains an HF
backbone and atomic `training.pt` with model/projector, AdamW, online heads and
their optimizer, scheduler, view RNG/pending lengths, Python/NumPy/Torch/CUDA RNG,
RAM sampler position/capacity/signature, progress and refit heads. `best/` keeps
the best online validation checkpoint; the output root stores the final state.

Resume with the same model/data/training schedule and evaluation configuration:

```powershell
python -B -m pretraining.cli "training.resume_from=results/foundation/run/checkpoints/step-00001000/training.pt"
```

`training.output_dir`, `resume_from` and W&B display settings may differ.
Changing the schedule horizon or objective is a new experiment, not exact resume.
For an explicit new experiment, `training.initialize_from` loads all model/EMA
weights strictly and restarts the optimizer, scheduler, sampler and online probe.
It cannot be combined with `resume_from`; the source checkpoint/step is logged.

`model.readout.preferred_modalities: [eeg, eog]` optionally pools available
post-fusion EEG/EOG features into the same 256D representation. Empty (default)
uses all modalities. When preferred sources are missing, pooling falls back to
all valid sources at that position. Fusion, modality JEPA targets and losses
still use all modalities. All downstream tasks use this same readout; any
heart-rate/SpO2 tradeoff must be evaluated alongside staging.
The RAM loader reconstructs metadata/RNG and reads only the current and future
pools. It refuses a different dataset/order or insufficient RAM for the saved
capacity. Older incomplete checkpoints are rejected for exact resume.

`tests/test_pipeline.py` steps through the scheduler and per-view objective,
HF roundtrip and optional same-batch refit debug flow; that debug score is not a
held-out result. `tests/test_training_restart.py` verifies exact continuation.

## N1 checkpoint diagnosis (independent 30 s)

Use the same CLI with the checkpoint list and head parameters in
`evaluation.diagnostics` in `configs/psg_training.yaml`:

```powershell
python -B -m pretraining.cli --mode diagnose
python -B tests/test_pipeline.py --mode diagnose --break-at features
```

The debugger also accepts `--break-at heads` and `--break-at report`. Open
`pretraining/diagnostics.py` and `pretraining/diagnostic_heads.py` to step through
feature extraction, fitting, and metric calculation. Run the global synthetic
CPU/CUDA test with `python -B -m pytest tests/test_pipeline.py -k frozen_diagnostics`.

Diagnosis freezes each backbone and compares the saved online head with:

- Refit linear heads, unweighted and inverse-square-root class weighted.
- A fixed train-prior logit correction; its strength is configurable.
- EEG/EOG/EMG features before fusion, using masked temporal mean readout.
- A small MLP on the existing 256-dimensional representation.
- Two temporal convolutions and attention readout on the 30 temporal tokens.

The temporal head uses only the current 30 s. Class counts and normalization
come from training data. Neural-head epoch selection holds out training subjects,
then refits from scratch on the full training cohort; validation never selects
head epochs. Missing-modality heads have their own valid sample counts, so compare
them on a common cohort before attributing differences to fusion.

Both checkpoints consume each fixed raw evaluation batch once. CPU feature
caches retain checkpoint/source/data fingerprints and are reused only when those
match. No pretrained weights, optimizer state, or original evaluation results
are overwritten. Cache files contain recording/subject identifiers and stay local.

The output directory contains `progress.json`, `REPORT.md`, `results.json`,
and per-checkpoint features, fitted heads and logits. Metrics include all stage
F1/precision/recall/support, N1 average precision, confusion matrices and modality
missingness strata. Aggregate scalar metrics use a separate W&B `diagnostic/`
namespace. They do not replace `eval/macro_f1` or redefine its 0.7 target.

`training.sampler_seed` controls real training order independently of model RNG and
fixed evaluation cohorts. `null` or omitted inherits `training.seed`; synthetic
data is unchanged. Change it only for a new weights-initialized run, not exact
resume. Subject splits still use `data.split_seed`; evaluation still uses
`training.seed + 20000/30000`. The global debug entry honors the same sampler seed.
