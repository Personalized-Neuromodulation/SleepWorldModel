from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dataloader.readers.hsp_raw import (
    HSPDataError,
    HSPDataset,
    RandomChannelDropout,
    build_hsp_manifest,
    hsp_collate_fn,
    load_hsp_manifest,
)


def make_hsp_file(
    root: Path,
    subject: str,
    session: str,
    *,
    duration: int,
    channels: dict[str, float],
) -> Path:
    eeg_dir = root / subject / session / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    path = eeg_dir / f"{subject}_{session}.h5"

    with h5py.File(path, "w") as handle:
        handle.attrs["subject_id"] = subject
        handle.attrs["session_id"] = session
        handle.attrs["duration_sec"] = float(duration)
        signal_group = handle.create_group("signals")
        for channel_index, (name, fs) in enumerate(channels.items()):
            length = int(duration * fs)
            raw = ((np.arange(length) + channel_index * 17) % 201 - 100).astype(
                np.int16
            )
            dataset = signal_group.create_dataset(name, data=raw, chunks=True)
            dataset.attrs["fs"] = float(fs)
            dataset.attrs["unit"] = "uV"
            dataset.attrs["dig_min"] = -100.0
            dataset.attrs["dig_max"] = 100.0
            dataset.attrs["phys_min"] = -10.0
            dataset.attrs["phys_max"] = 10.0
            dataset.attrs["type"] = "raw"

        stage = handle.create_group("annotations/expert_1/stage")
        starts = np.arange(0, duration, 30, dtype=np.float32)
        stage.create_dataset("starts", data=starts)
        stage.create_dataset("durations", data=np.full_like(starts, 30.0))
        stage.create_dataset("codes", data=np.arange(len(starts), dtype=np.int16))

    return path


@pytest.fixture()
def small_hsp(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "I0002"
    common = {
        "c3-m2": 4.0,
        "c4-m1": 4.0,
        "e1": 2.0,
        "chin1-chin2": 4.0,
        "ecg": 4.0,
        "abd": 2.0,
        "chest": 2.0,
        "spo2": 1.0,
    }
    make_hsp_file(root, "sub-a", "ses-1", duration=120, channels=common)
    make_hsp_file(root, "sub-a", "ses-2", duration=90, channels=common)
    make_hsp_file(
        root,
        "sub-b",
        "ses-1",
        duration=60,
        channels={name: fs for name, fs in common.items() if name != "c4-m1"},
    )
    manifest = tmp_path / "manifest.jsonl"
    build_hsp_manifest(root, manifest)
    return root, manifest


def test_manifest_is_session_level_and_subject_split_is_stable(
    small_hsp: tuple[Path, Path],
) -> None:
    _, manifest = small_hsp
    records = load_hsp_manifest(manifest)

    assert len(records) == 3
    assert all(record["status"] == "ok" for record in records)
    sub_a_splits = {record["split"] for record in records if record["subject_id"] == "sub-a"}
    assert len(sub_a_splits) == 1
    assert records[0]["signals"]["c3-m2"]["fs"] == 4.0
    assert "expert_1/stage" in records[0]["annotations"]


def test_context_slice_calibration_and_annotation_alignment(
    small_hsp: tuple[Path, Path],
) -> None:
    root, manifest = small_hsp
    dataset = HSPDataset(
        root,
        manifest,
        split=None,
        channel_groups={"eeg": ["c3-m2", "c4-m1"], "spo2": ["spo2"]},
        sampling_mode="context",
        context_epochs=2,
        missing_channel="drop",
        return_annotations=True,
    )

    # sub-a/ses-1 contributes three two-epoch windows; ses-2 contributes two.
    assert len(dataset) == 5
    sample = dataset[0]
    assert sample["signals"]["eeg"].shape == (2, 2, 120)
    assert sample["signals"]["spo2"].shape == (2, 1, 30)
    assert sample["signals"]["eeg"].dtype == torch.float32
    assert sample["signals"]["eeg"][0, 0, 0].item() == pytest.approx(-10.0)
    assert sample["sample_rates"] == {"eeg": 4.0, "spo2": 1.0}
    assert sample["annotations"]["stage"].tolist() == [0, 1]
    assert sample["annotations"]["stage_mask"].tolist() == [True, True]

    shifted = dataset[1]
    assert shifted["start_sec"] == 30.0
    assert shifted["annotations"]["stage"].tolist() == [1, 2]


def test_missing_channel_mask_and_dropout(small_hsp: tuple[Path, Path]) -> None:
    root, manifest = small_hsp
    dataset = HSPDataset(
        root,
        manifest,
        split=None,
        channel_groups={"eeg": ["c3-m2", "c4-m1"]},
        sampling_mode="night",
        missing_channel="mask",
    )

    missing_sample = next(sample for sample in dataset if sample["subject_id"] == "sub-b")
    assert missing_sample["available_mask"]["eeg"].tolist() == [True, False]
    assert torch.count_nonzero(missing_sample["signals"]["eeg"][:, 1]) == 0

    full_sample = next(
        sample
        for sample in dataset
        if sample["subject_id"] == "sub-a" and sample["session_id"] == "ses-1"
    )
    dropout = RandomChannelDropout(probability=1.0, modalities=("eeg",), min_remaining=1)
    dropped = dropout(full_sample)
    assert dropped["available_mask"]["eeg"].sum().item() == 2
    assert dropped["channel_mask"]["eeg"].sum().item() == 1


def test_integer_downsampling_has_consistent_batch_shape(
    small_hsp: tuple[Path, Path],
) -> None:
    root, manifest = small_hsp
    dataset = HSPDataset(
        root,
        manifest,
        split=None,
        channel_groups={"eeg": ["c3-m2", "c4-m1"], "spo2": ["spo2"]},
        target_sample_rates={"eeg": 2.0, "spo2": 1.0},
        sampling_mode="context",
        context_epochs=2,
        missing_channel="drop",
    )

    sample = dataset[0]
    assert sample["signals"]["eeg"].shape == (2, 2, 60)
    assert sample["signals"]["spo2"].shape == (2, 1, 30)
    assert sample["sample_rates"] == {"eeg": 2.0, "spo2": 1.0}
    assert torch.isfinite(sample["signals"]["eeg"]).all()


def test_mixed_channel_rates_are_resampled_to_one_modality_rate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "I0002"
    make_hsp_file(
        root,
        "sub-a",
        "ses-1",
        duration=60,
        channels={"abd": 20.0, "airflow": 5.0},
    )
    manifest = tmp_path / "manifest.jsonl"
    build_hsp_manifest(root, manifest)

    dataset = HSPDataset(
        root,
        manifest,
        split=None,
        channel_groups={"resp": ["abd", "airflow"]},
        target_sample_rates={"resp": 5.0},
        sampling_mode="epoch",
        missing_channel="error",
    )

    sample = dataset[0]
    assert sample["signals"]["resp"].shape == (1, 2, 150)
    assert sample["sample_rates"] == {"resp": 5.0}
    assert sample["available_mask"]["resp"].tolist() == [True, True]


def test_mixed_channel_rates_require_an_explicit_target(tmp_path: Path) -> None:
    root = tmp_path / "I0002"
    make_hsp_file(
        root,
        "sub-a",
        "ses-1",
        duration=60,
        channels={"abd": 20.0, "airflow": 5.0},
    )
    manifest = tmp_path / "manifest.jsonl"
    build_hsp_manifest(root, manifest)

    with pytest.raises(HSPDataError, match="mixed channel sample rates"):
        HSPDataset(
            root,
            manifest,
            split=None,
            channel_groups={"resp": ["abd", "airflow"]},
            sampling_mode="epoch",
            missing_channel="error",
        )


def test_drop_and_error_missing_channel_policies(small_hsp: tuple[Path, Path]) -> None:
    root, manifest = small_hsp
    dropped = HSPDataset(
        root,
        manifest,
        split=None,
        channel_groups={"eeg": ["c3-m2", "c4-m1"]},
        sampling_mode="night",
        missing_channel="drop",
    )
    assert len(dropped) == 2

    with pytest.raises(HSPDataError, match="requested channels are missing"):
        HSPDataset(
            root,
            manifest,
            split=None,
            channel_groups={"eeg": ["c3-m2", "c4-m1"]},
            sampling_mode="night",
            missing_channel="error",
        )


def test_night_collate_pads_epochs_and_keeps_masks(
    small_hsp: tuple[Path, Path],
) -> None:
    root, manifest = small_hsp
    dataset = HSPDataset(
        root,
        manifest,
        split=None,
        channel_groups={"eeg": ["c3-m2"]},
        sampling_mode="night",
    )
    batch = hsp_collate_fn([dataset[0], dataset[2]])

    assert batch["signals"]["eeg"].shape == (2, 4, 1, 120)
    assert batch["epoch_mask"].tolist() == [
        [True, True, True, True],
        [True, True, False, False],
    ]
    assert not batch["sample_mask"]["eeg"][1, 2:].any()


def test_multiworker_dataloader_does_not_share_hdf5_handles(
    small_hsp: tuple[Path, Path],
) -> None:
    root, manifest = small_hsp
    dataset = HSPDataset(
        root,
        manifest,
        split=None,
        channel_groups={"eeg": ["c3-m2"]},
        sampling_mode="epoch",
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        collate_fn=hsp_collate_fn,
        persistent_workers=False,
    )

    batch = next(iter(loader))
    assert batch["signals"]["eeg"].shape == (2, 1, 1, 120)
    assert batch["epoch_mask"].all()
