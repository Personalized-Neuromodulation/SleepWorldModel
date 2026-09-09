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
from world_model.ssl import SSLConfig, SSLLoss, SSLModel
from world_model.training.batch_adapter import prepare_batch
from world_model.training.input import model_input_config


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
    torch.set_num_threads(1)
    with WindowDataset(release, context_epochs=2) as ds:
        batch = collate_windows([ds[0], ds[1]])
        config = SSLConfig(
            **model_input_config(ds.metadata),
            hidden_dim=8,
            embedding_dim=16,
            projection_dim=8,
        )
        prepared = prepare_batch(batch, config, torch.device("cpu"))
        assert config.channels["emg"] == 3 and config.channels["respiratory"] == 3
        assert config.sample_rates["respiratory"] == 200
        assert prepared["valid"]["eeg"][0, 0, 0]
        assert not prepared["valid"]["eeg"][0, 1, 0]
        assert not prepared["signals"]["eeg"][0, 1, 0].any()
        assert not prepared["valid"]["eeg"][1, 1].any()
        model = SSLModel(config)
        loss = SSLLoss(num_projections=4, num_frequencies=5)(model(prepared)).total
        loss.backward()
        assert torch.isfinite(loss)
        assert model.projector[0].weight.grad is not None


def test_training_cli_reads_release_and_writes_checkpoint(release, tmp_path):
    from world_model.training.ssl_cli import main

    torch.set_num_threads(1)
    checkpoint = tmp_path / "model.pt"
    assert (
        main(
            [
                "--root",
                str(release),
                "--split",
                "all",
                "--device",
                "cpu",
                "--max-steps",
                "1",
                "--batch-size",
                "2",
                "--context-epochs",
                "2",
                "--embedding-dim",
                "16",
                "--projection-dim",
                "8",
                "--hidden-dim",
                "8",
                "--sigreg-projections",
                "4",
                "--sigreg-frequencies",
                "5",
                "--wandb-mode",
                "disabled",
                "--checkpoint",
                str(checkpoint),
            ]
        )
        == 0
    )
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["schema_version"] == 3
    assert saved["architecture"] == "minimal-sigreg-v1"
    assert saved["input_config"]["dataset"] == "hsp"
    assert saved["model_config"]["channels"]["emg"] == 3
    restored = SSLModel(SSLConfig(**saved["model_config"]))
    restored.load_state_dict(saved["model"])
    with WindowDataset(release, context_epochs=2) as ds:
        sample = prepare_batch(
            collate_windows([ds[0]]), restored.config, torch.device("cpu")
        )
        embeddings, valid = restored.encode(sample)
        assert embeddings.shape == (1, 2, 16) and valid.all()


def test_debug_cli_runs_one_real_format_batch(release):
    from world_model.debugging.ssl_step import main

    torch.set_num_threads(1)
    assert (
        main(
            [
                "--root",
                str(release),
                "--split",
                "all",
                "--device",
                "cpu",
                "--context-epochs",
                "2",
                "--batch-size",
                "2",
            ]
        )
        == 0
    )
