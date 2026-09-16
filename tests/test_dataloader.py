"""Published-format integration tests, including masks, boundaries and workers."""

import hashlib
import json
import pickle

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from dataloader import WindowDataset, collate_windows, register_reader
from dataloader.readers.hsp import CHANNELS, QUALITY, TASKS
from dataloader.sampler import subject_split


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "release"
    records, artifacts = [], []
    for shard, specs in [
        ("00000", [("subject-a", "1", 3), ("subject-b", "1", 2)]),
        ("00001", [("subject-a", "2", 2)]),
    ]:
        shard_records = []
        total = 0
        for subject, session, count in specs:
            shard_records.append(
                dict(
                    recording_id=f"{subject}-{session}",
                    subject_id=subject,
                    session_id=session,
                    shard_id=shard,
                    first_epoch=total,
                    n_epochs=count,
                    duration_sec=count * 30.0,
                    night_grade=5 if subject == "subject-a" else 1,
                    status="QC_PASSED",
                )
            )
            total += count
        records.extend(shard_records)
        bindings = {
            key: f"{shard}-{key}"
            for key in (
                "schema_sha256",
                "processing_config_sha256",
                "qc_config_sha256",
                "epoch_index_sha256",
                "channel_order_sha256",
            )
        }
        attrs = dict(
            bindings,
            shard_id=shard,
            version="v1.0.0",
            release_status="QC_PASSED",
            schema_version="1.0.0-pilot.2" if shard == "00000" else "1.0.0-full.1",
            epoch_count=total,
            epoch_samples=6000,
            sample_rate_hz=200,
            epoch_seconds=30,
        )

        def index(f):
            for key in ("recording_id", "subject_id", "session_id"):
                f.create_dataset(
                    f"records/{key}",
                    data=[r[key] for r in shard_records],
                    dtype=h5py.string_dtype(),
                )
            for key in ("first_epoch", "n_epochs"):
                f[f"records/{key}"] = [r[key] for r in shard_records]
            f["segments/record_index"] = np.concatenate(
                [np.full(r["n_epochs"], i) for i, r in enumerate(shard_records)]
            )
            f["segments/epoch_in_record"] = np.concatenate(
                [np.arange(r["n_epochs"]) for r in shard_records]
            )
            f["segments/epoch_start_offset_ns"] = (
                f["segments/epoch_in_record"][:] * 30_000_000_000
            )

        shard_artifacts = []

        def commit_file(path):
            row = dict(
                artifact=str(path.relative_to(root)).replace("/", "\\"),
                bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                epochs=total,
                shard_id=shard,
            )
            shard_artifacts.append(row)
            return row["sha256"]

        path = root / f"signals/shards/signal-{shard}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs.update(attrs, artifact_kind="signal")
            index(f)
            f.create_dataset(
                "channel_names",
                data=[c for v in CHANNELS.values() for c in v],
                dtype=h5py.string_dtype(),
            )
            f.create_dataset(
                "channel_units", data=["uV"] * 14 + ["%"], dtype=h5py.string_dtype()
            )
            f["records/channel_available"] = np.ones((len(specs), 15), bool)
            f["records/night_grade"] = np.asarray(
                [r["night_grade"] for r in shard_records], np.uint8
            )
            f["records/grade_evaluated"] = np.ones(len(specs), bool)
            for group, channels in CHANNELS.items():
                values = np.broadcast_to(
                    np.arange(total, dtype=np.float32)[:, None, None] + 1,
                    (total, len(channels), 6000),
                ).copy()
                f.create_dataset(f"signals/{group}", data=values, compression="lzf")
                for key in QUALITY:
                    array = (
                        np.zeros((total, len(channels)), np.uint8)
                        if key == "hard_code"
                        else np.ones((total, len(channels)), bool)
                    )
                    if (
                        group == "eeg"
                        and total > 1
                        and key in ("valid", "artifact_valid")
                    ):
                        array[1, 0] = False
                    if group == "eeg" and total > 1 and key == "hard_code":
                        array[1, 0] = 4
                    f[f"quality/{group}/{key}"] = array
        signal_hash = commit_file(path)
        for task in TASKS:
            path = root / f"tasks/{task}/shards/task-{shard}.h5"
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "w") as f:
                fields = (
                    ("stage",)
                    if task == "sleep_stage"
                    else ("max_bpm", "min_bpm", "mean_bpm")
                    if task == "heart_rate"
                    else ("max_percent", "min_percent")
                )
                f.attrs.update(
                    attrs,
                    artifact_kind=f"task:{task}",
                    task=task,
                    signal_sha256=signal_hash,
                    field_names=fields,
                )
                index(f)
                values = np.full(
                    total if task == "sleep_stage" else (total, len(fields)),
                    5 if task == "sleep_stage" else 80,
                    np.uint8 if task == "sleep_stage" else np.float32,
                )
                f["labels"] = values
                valid = np.ones(total, bool)
                valid[-1] = False
                f["valid"] = valid
                f["qc_code"] = np.where(valid, 0, 5).astype(np.uint8)
                f["field_valid"] = np.repeat(valid[:, None], len(fields), axis=1)
                f["field_qc_code"] = np.where(f["field_valid"][:], 0, 5).astype(
                    np.uint8
                )
            commit_file(path)
        _write_json(
            root / f"manifests/commits/shard-{shard}.json",
            dict(shard_id=shard, status="QC_PASSED", artifacts=shard_artifacts),
        )
        artifacts.extend(shard_artifacts)
    _write_json(
        root / "manifests/release.json", dict(version="v1.0.0", status="COMPLETE")
    )
    pq.write_table(pa.Table.from_pylist(records), root / "manifests/records.parquet")
    pq.write_table(pa.Table.from_pylist(artifacts), root / "manifests/shards.parquet")
    return root


def test_windows_tasks_and_padding_preserve_physical_values(release):
    with WindowDataset(release, context_epochs=2, tasks=TASKS) as ds:
        assert len(ds) == 4
        first, tail, next_record = ds[0], ds[1], ds[2]
        assert (
            tail["recording_id"] == first["recording_id"] != next_record["recording_id"]
        )
        assert tail["epoch_in_record"].tolist() == [2]
        assert tail["epoch_start_offset_ns"].tolist() == [60_000_000_000]
        assert torch.all(first["signals"]["eeg"][1] == 2)
        assert not first["quality"]["eeg"]["valid"][1, 0]
        assert first["night_grade"] == 5
        assert next_record["tasks"]["sleep_stage"]["labels"].tolist() == [5, 5]
        assert not next_record["tasks"]["sleep_stage"]["valid"][-1]
        batch = collate_windows([first, tail])
        assert batch["signals"]["eeg"].shape == (2, 2, 6, 6000)
        assert batch["night_grade"].dtype == torch.uint8
        assert batch["night_grade"].tolist() == [5, 5]
        assert batch["epoch_mask"].tolist() == [[True, True], [True, False]]
        assert batch["epoch_in_record"][1].tolist() == [2, -1]
        assert not batch["quality"]["eeg"]["valid"][1, 1].any()
        assert not batch["tasks"]["sleep_stage"]["valid"][1, 1]
        assert ds[-1]["session_id"] == "2"
        with pytest.raises(IndexError):
            ds[len(ds)]


def test_split_is_subject_level_and_stable(release):
    selected = subject_split("hsp", "subject-a")
    with WindowDataset(release, split=selected, context_epochs=1) as ds:
        assert (
            len({r["session_id"] for r in ds.records if r["subject_id"] == "subject-a"})
            == 2
        )
    assert subject_split("hsp", "subject-a") == selected
    with pytest.raises(ValueError, match="ratios"):
        WindowDataset(release, split_ratios=(1, 1, 1))


def test_worker_pickle_and_lru_handles(release):
    with WindowDataset(release, context_epochs=2, tasks=TASKS, max_open_files=1) as ds:
        expected = ds[0]
        assert len(ds.reader._handles) <= 1
        clone = pickle.loads(pickle.dumps(ds))
        assert not clone.reader._handles
        clone.close()
        batch = next(
            iter(
                DataLoader(ds, batch_size=2, num_workers=1, collate_fn=collate_windows)
            )
        )
        assert torch.equal(batch["signals"]["eeg"][0], expected["signals"]["eeg"])


@pytest.mark.parametrize(
    "problem", ["missing_commit", "commit_mismatch", "wrong_version", "partial"]
)
def test_reject_unpublished_or_incompatible_artifacts(release, problem):
    if problem == "missing_commit":
        (release / "manifests/commits/shard-00000.json").unlink()
    elif problem == "commit_mismatch":
        p = release / "manifests/commits/shard-00000.json"
        c = json.loads(p.read_text())
        c["artifacts"][0]["sha256"] = "bad"
        _write_json(p, c)
    elif problem == "wrong_version":
        _write_json(
            release / "manifests/release.json", dict(version="v2", status="COMPLETE")
        )
    else:
        p = release / "manifests/shards.parquet"
        rows = pq.read_table(p).to_pylist()
        rows[0]["artifact"] += ".partial"
        pq.write_table(pa.Table.from_pylist(rows), p)
    with pytest.raises((ValueError, FileNotFoundError)):
        with WindowDataset(release) as ds:
            ds[0]


def test_task_alignment_and_checksum_are_checked(release):
    path = release / "tasks/sleep_stage/shards/task-00000.h5"
    with h5py.File(path, "r+") as f:
        f["segments/epoch_start_offset_ns"][0] = 1
    with WindowDataset(release, tasks=("sleep_stage",)) as ds:
        with pytest.raises(ValueError, match="alignment"):
            ds[0]
    with WindowDataset(release, tasks=("sleep_stage",), verify_checksums=True) as ds:
        with pytest.raises(ValueError, match="checksum"):
            ds[0]


def test_night_grade_is_bound_to_signal_shard(release):
    path = release / "signals/shards/signal-00000.h5"
    with h5py.File(path, "r+") as f:
        f["records/night_grade"][0] = 1
    with WindowDataset(release) as ds:
        with pytest.raises(ValueError, match="night grade"):
            ds[0]


def test_reader_extension_without_editing_training_or_indexing(release):
    from dataloader.readers.hsp import HSPReleaseReader

    register_reader("test_other", HSPReleaseReader)
    with WindowDataset(release, dataset="test_other", context_epochs=2) as ds:
        assert len(ds) == 4
    with pytest.raises(ValueError, match="registered"):
        register_reader("test_other", HSPReleaseReader)


def test_published_batch_model_forward_backward_and_qc(release):
    from dataloader import as_signal_batch
    from pretraining.factory import build_pretraining
    from tests.test_foundation import small_config

    torch.set_num_threads(2)
    with WindowDataset(release, context_epochs=1) as dataset:
        raw = collate_windows([dataset[0], dataset[1]])
        raw["signals"]["eeg"][1, 0, 0] = float("nan")
        batch = as_signal_batch(raw, foundation=True)
        assert batch.groups["emg"].values.shape[2] == 3
        assert batch.groups["respiratory"].values.shape[2] == 2
        assert batch.groups["spo2"].values.shape[2] == 1
        assert not batch.groups["eeg"].data_valid[1, 0].any()
        model, sampler = build_pretraining(small_config(), batch)
        output = model(sampler(batch))
        assert not output.skip_update and torch.isfinite(output.loss)
        output.loss.backward()
        assert model.projector[0].weight.grad.abs().sum() > 0


def test_training_cli_reads_release_and_writes_checkpoint(release, tmp_path):
    import yaml
    from transformers import AutoModel

    from pretraining.cli import main
    from pretraining.configuration import load_config

    output_dir = tmp_path / "foundation"
    config = load_config(
        overrides=[
            "data.mode=real",
            f"data.root={release}",
            "data.split_seed=9",  # fixture: subject-a train, subject-b validation
            "data.night_grades=[1,5]",  # Validation fixture intentionally has grade 1.
            f"training.output_dir={output_dir}",
            "training.device=cpu",
            "training.max_steps=1",
            "training.readout_log_every_steps=1",
            "training.batch_size=2",
            "model.modality_encoder.eeg_depth=1",
            "model.modality_encoder.multichannel_depth=1",
            "model.modality_encoder.single_channel_depth=1",
            "model.fusion.depth=1",
            "pretraining.sigreg.num_slices=8",
            "evaluation.max_batches=1",
            "evaluation.wandb_mode=disabled",
        ]
    )
    config_path = tmp_path / "real.yaml"
    model_path = tmp_path / "model.yaml"
    model_path.write_text(
        yaml.safe_dump({key: config[key] for key in ("model", "pretraining")}),
        encoding="utf-8",
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "model_config": model_path.name,
                **{key: config[key] for key in ("data", "training", "evaluation")},
            }
        ),
        encoding="utf-8",
    )
    assert main(["--config", str(config_path)]) == 0
    saved = torch.load(output_dir / "training.pt", weights_only=True)
    assert saved["step"] == 1
    assert "train/readout/modality_entropy" in saved["metrics"]
    assert "eval/readout/fused_feature_norm" in saved["metrics"]
    assert saved["config"]["data"]["dataset"] == "hsp"
    restored = AutoModel.from_pretrained(
        output_dir / "backbone", trust_remote_code=True
    )
    with torch.no_grad():
        result = restored(signals={"ecg": torch.randn(2, 1, 5, 200)})
    assert result.pooler_output.shape == (2, 256)


def test_foundation_grade_filter_and_batch_composition(release):
    from dataloader.sampler import NightGradeBatchSampler

    with WindowDataset(release, context_epochs=1, night_grades=(5,)) as dataset:
        assert all(r["night_grade"] == 5 for r in dataset.records)
        assert len(dataset) == 5
    with WindowDataset(release, context_epochs=1) as dataset:
        sampler = NightGradeBatchSampler(dataset, batch_size=3, num_batches=20, seed=8)
        batches = list(sampler)
        assert batches == list(sampler)
        grades = set()
        for indices in batches:
            batch = collate_windows([dataset[i] for i in indices])
            assert len(batch["night_grade"].unique()) == 1
            grades.add(int(batch["night_grade"][0]))
        assert grades == {1, 5}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_foundation_training_real_schema_with_linear_probe(release, tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from pretraining.configuration import load_config
    from pretraining.training import train

    torch.set_num_threads(2)
    config = load_config(
        overrides=[
            f"training.device={device}",
            "model.modality_encoder.eeg_depth=1",
            "model.modality_encoder.multichannel_depth=1",
            "model.modality_encoder.single_channel_depth=1",
            "model.fusion.depth=1",
            "pretraining.sigreg.num_slices=8",
            "training.max_steps=1",
            "evaluation.frozen_linear_probe=true",
            "evaluation.wandb_mode=disabled",
            "evaluation.max_batches=1",
        ]
    )
    config["training"]["output_dir"] = str(tmp_path / "foundation")
    with WindowDataset(release, context_epochs=1, tasks=TASKS) as dataset:
        train_batch = collate_windows([dataset[0], dataset[1]])  # subject-a
        val_batch = collate_windows([dataset[3], dataset[4]])  # subject-b
        result = train(config, [train_batch], [val_batch])
    assert result["steps"] == 1
    assert result["metrics"]["eval/probe_samples"] == 1
    assert result["metrics"]["train_eval/probe_samples"] == 2
    assert 0 <= result["metrics"]["eval/macro_f1"] <= 1
    for task in ("heart_rate", "sao2"):
        assert result["metrics"][f"train_eval/{task}/samples"] == 2
        assert result["metrics"][f"train_eval/{task}/mae"] >= 0
        assert result["metrics"][f"eval/{task}/samples"] == 1
        assert result["metrics"][f"eval/{task}/mae"] >= 0
        assert result["metrics"][f"eval/{task}/rmse"] >= 0
    checkpoint = torch.load(tmp_path / "foundation/training.pt", weights_only=True)
    assert set(checkpoint["probe"]) == {
        f"heads.{task}.{parameter}"
        for task in TASKS
        for parameter in ("weight", "bias")
    }


def test_pipeline_debugs_all_three_tasks(release, tmp_path, monkeypatch):
    import test_pipeline

    from pretraining.configuration import load_config

    config = load_config(
        overrides=[
            f"data.root={release}",
            "data.split_seed=9",
            "training.device=cpu",
            "training.batch_size=2",
            "model.modality_encoder.eeg_depth=1",
            "model.modality_encoder.multichannel_depth=1",
            "model.modality_encoder.single_channel_depth=1",
            "model.fusion.depth=1",
            "pretraining.sigreg.num_slices=8",
            "evaluation.max_batches=1",
            "evaluation.wandb_mode=disabled",
        ]
    )
    logged = {}
    monkeypatch.setattr(
        test_pipeline.WandbLogger,
        "log",
        lambda self, metrics, step: logged.update(metrics),
    )
    report = test_pipeline.run_pipeline(config, tmp_path / "pipeline")
    assert report["hf_roundtrip"]
    assert "debug_batch/macro_f1" in logged
    for task in ("heart_rate", "sao2"):
        assert logged[f"debug_batch/{task}/samples"] > 0
        assert logged[f"debug_batch/{task}/mae"] >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("workers", [0, 1])
def test_foundation_worker_adapter_pins_derived_tensors(release, workers):
    from dataclasses import fields
    from functools import partial

    from torch.utils.data import DataLoader

    from dataloader.signals import collate_signal_windows
    from pretraining.evaluation import prepare

    with WindowDataset(release, context_epochs=1, tasks=("sleep_stage",)) as dataset:
        loader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=workers,
            collate_fn=partial(collate_signal_windows, scales={"eeg": 2.0}),
            pin_memory=True,
        )
        raw = next(iter(loader))
        batch = raw["signal_batch"]
        assert raw["tasks"]["sleep_stage"]["labels"].is_pinned()
        for obj in [batch, *batch.groups.values()]:
            assert all(
                getattr(obj, f.name).is_pinned()
                for f in fields(obj)
                if isinstance(getattr(obj, f.name), torch.Tensor)
            )
        moved = prepare(raw, torch.device("cuda"))
        assert moved.groups["eeg"].data_valid.device.type == "cuda"
        torch.testing.assert_close(
            moved.groups["eeg"].values.cpu(), batch.groups["eeg"].values, equal_nan=True
        )
