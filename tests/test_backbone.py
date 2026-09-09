"""Behavioral tests: leakage, padding, gradients, timing and public composition."""

from dataclasses import replace

import pytest
import torch

from backbone import (
    BackboneConfig,
    BlockConfig,
    MaskPlan,
    PatchEncoderConfig,
    SequenceEncoderConfig,
    build_backbone,
)
from dataloader import as_signal_batch
from dataloader.synthetic import synthetic_windows


@pytest.fixture
def batch():
    return as_signal_batch(synthetic_windows(sample_rate=20, epoch_seconds=4))


def small_config(**kwargs):
    return BackboneConfig(
        feature_dim=8,
        patch_encoder=PatchEncoderConfig(
            subpatch_samples=5,
            hidden_dim=8,
            blocks=(BlockConfig(kind="cnn"), BlockConfig(kind="transformer")),
        ),
        sequence_encoder=SequenceEncoderConfig(window_seconds=2),
        **kwargs,
    )


def changed_values(batch, name, change):
    group = batch.groups[name]
    values = group.values.clone()
    change(values)
    return replace(batch, groups={**batch.groups, name: replace(group, values=values)})


def test_adapter_keeps_masks_and_splits_spo2(batch):
    assert batch.groups["respiratory"].values.shape == (2, 2, 2, 80)
    assert batch.groups["spo2"].channel_ids == ("spo2",)
    assert batch.night_grade.dtype == torch.uint8
    assert not batch.epoch_mask[1, -1]
    assert not batch.groups["eeg"].available_mask[1, -1]
    batch.to("cpu").validate()


@pytest.mark.parametrize("fusion", ["none", "pool", "cross_attention"])
def test_shapes_finite_backward_and_support(batch, fusion):
    model = build_backbone(small_config(fusion=fusion), batch)
    requested = ("patch_tokens", "local", "features") + (
        ("joint",) if fusion != "none" else ()
    )
    result = model(batch, outputs=requested)
    for name, group in batch.groups.items():
        c = len(group.channel_ids)
        assert result.local[name].tokens.shape == (2, c, 8, 8)
        assert result.features[name].tokens.shape == (2, 8, 8)
        assert torch.isfinite(result.local[name].tokens).all()
        assert not result.local[name].active[1, :, 4:].any()
    if result.joint is not None:
        assert result.joint.tokens.shape == (2, 8, 8)
        assert torch.allclose(result.joint.coverage[1, :4], torch.full((4,), 14 / 15))
    target = result.joint.tokens if result.joint else result.features["eeg"].tokens
    target.square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)


@pytest.mark.parametrize("stage", ["waveform", "token"])
@pytest.mark.parametrize("variant", ["direct_linear", "sequence"])
def test_hidden_raw_values_cannot_affect_context_or_gradients(batch, stage, variant):
    config = small_config(fusion="cross_attention")
    config = replace(
        config, patch_encoder=replace(config.patch_encoder, variant=variant)
    )
    model = build_backbone(config, batch).eval()
    group = batch.groups["eeg"]
    if stage == "token":
        visible = torch.ones(2, 6, 8, dtype=torch.bool)
        visible[..., 0] = False
        hidden_samples = 20
    else:
        visible = torch.ones_like(group.values, dtype=torch.bool)
        visible[:, 0, :, :10] = False  # partial-patch hiding
        hidden_samples = 10

    def hide(x):
        x[:, 0, :, :hidden_samples].fill_(float("nan"))

    plan = MaskPlan(stage, {"eeg": visible})
    changed = changed_values(batch, "eeg", hide)
    changed.groups["eeg"].values.requires_grad_()
    expected = model(batch, plan, ("joint",)).joint.tokens
    actual = model(changed, plan, ("joint",)).joint.tokens
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    gradient = changed.groups["eeg"].values.grad
    hidden_samples = 20 if stage == "token" else 10
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[:, 0, :, :hidden_samples]) == 0
    assert torch.isfinite(batch.groups["eeg"].values[0]).all()


def test_wholly_hidden_modalities_are_finite_and_invalid(batch):
    model = build_backbone(small_config(fusion="cross_attention"), batch)
    visible = {
        name: torch.zeros(2, len(g.channel_ids), 8, dtype=torch.bool)
        for name, g in batch.groups.items()
    }
    result = model(batch, MaskPlan("token", visible), ("joint",)).joint
    assert torch.isfinite(result.tokens).all()
    assert torch.count_nonzero(result.tokens) == 0
    assert result.data_valid.any() and not result.visible.any()
    result.tokens.sum().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_patch_isolation_local_windows_and_time_gaps(batch):
    model = build_backbone(small_config(), batch).eval()
    changed = changed_values(batch, "eeg", lambda x: x[:, 0, :, 40:].add_(50))
    a = model(batch, outputs=("patch_tokens", "local"))
    b = model(changed, outputs=("patch_tokens", "local"))
    torch.testing.assert_close(
        a.patch_tokens["eeg"].tokens[..., :2, :],
        b.patch_tokens["eeg"].tokens[..., :2, :],
    )
    torch.testing.assert_close(
        a.local["eeg"].tokens[..., :2, :], b.local["eeg"].tokens[..., :2, :]
    )
    shifted = replace(batch, epoch_start_offset_ns=batch.epoch_start_offset_ns.clone())
    shifted.epoch_start_offset_ns[0, 1] += 10_000_000_000
    grid = model(shifted, outputs=("local",)).local["eeg"]
    assert grid.time_intervals_ns[0, 4, 0] == 14_000_000_000
    assert grid.context_intervals_ns[0, 4, 0] >= 14_000_000_000
    assert grid.context_intervals_ns[0, 0, 1] == 2_000_000_000
    assert (grid.available_at_ns == -1).all()  # offline QC availability is unknown


@pytest.mark.parametrize("kind", ["cnn", "transformer"])
def test_causal_blocks_do_not_use_later_patches(batch, kind):
    config = small_config()
    config = replace(
        config,
        sequence_encoder=SequenceEncoderConfig(
            window_seconds=4, causal=True, blocks=(BlockConfig(kind=kind, depth=2),)
        ),
    )
    model = build_backbone(config, batch).eval()
    changed = changed_values(batch, "eeg", lambda x: x[:, 0, :, 20:].add_(99))
    a = model(batch, outputs=("local",)).local["eeg"]
    b = model(changed, outputs=("local",)).local["eeg"]
    torch.testing.assert_close(a.tokens[..., 0, :], b.tokens[..., 0, :])
    assert a.context_intervals_ns[0, 0, 1] <= 1_000_000_000


def test_output_requests_identity_and_invalid_configs(batch):
    config = small_config()
    config = replace(
        config,
        channel_aggregation="none",
        sequence_encoder=replace(config.sequence_encoder, blocks=()),
    )
    model = build_backbone(config, batch)
    result = model(batch, outputs=("patch_tokens", "local"))
    assert result.patch_tokens["eeg"] is result.local["eeg"]
    assert not any("channel_aggregator" in key for key in model.state_dict())
    with pytest.raises(ValueError, match="aggregation"):
        model(batch)
    with pytest.raises(ValueError, match="joint"):
        build_backbone(small_config(), batch)(batch, outputs=("joint",))
    normal = build_backbone(small_config(), batch)
    encoder = normal.modality_encoders["eeg"].signal_encoder.sequence_encoder
    handle = encoder.register_forward_pre_hook(
        lambda *_: pytest.fail("unexpected sequence call")
    )
    normal(batch, outputs=("patch_tokens",))
    handle.remove()


def test_save_reload_and_channel_order(batch, tmp_path):
    model = build_backbone(small_config(fusion="pool"), batch).eval()
    path = tmp_path / "weights.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_spec": model.input_spec,
            "config": model.config.to_dict(),
        },
        path,
    )
    saved = torch.load(path, weights_only=True)
    restored = build_backbone(
        BackboneConfig.from_dict(saved["config"]), saved["input_spec"]
    ).eval()
    restored.load_state_dict(saved["state_dict"])
    torch.testing.assert_close(
        model(batch, outputs=("joint",)).joint.tokens,
        restored(batch, outputs=("joint",)).joint.tokens,
    )
    group = batch.groups["eeg"]
    wrong = replace(
        batch,
        groups={
            **batch.groups,
            "eeg": replace(group, channel_ids=group.channel_ids[::-1]),
        },
    )
    with pytest.raises(ValueError, match="channel order"):
        restored(wrong)


def test_fusion_rejects_unaligned_grids(batch):
    model = build_backbone(small_config(fusion="pool"), batch)
    features = model(batch).features
    features["eog"] = replace(
        features["eog"], time_intervals_ns=features["eog"].time_intervals_ns + 1
    )
    with pytest.raises(ValueError, match="identical time"):
        model.fusion(features)


def test_window_tail_padding_and_attention_readout(batch):
    config = small_config()
    config = replace(
        config,
        patch_encoder=replace(config.patch_encoder, readout="attention"),
        sequence_encoder=replace(config.sequence_encoder, window_seconds=3),
    )
    model = build_backbone(config, batch)
    result = model(batch, outputs=("local",)).local["eeg"]
    assert result.tokens.shape == (2, 6, 8, 8)
    assert result.context_intervals_ns[0, 3].tolist() == [3_000_000_000, 4_000_000_000]
    assert result.context_intervals_ns[0, 4].tolist() == [4_000_000_000, 7_000_000_000]
    assert not result.active[1, :, 4:].any()
    result.tokens.square().sum().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_disabled_identity_and_linear_do_not_create_parameters(batch):
    config = small_config()
    config = replace(
        config,
        channel_embedding=False,
        modality_embedding=False,
        patch_encoder=replace(config.patch_encoder, variant="direct_linear"),
    )
    model = build_backbone(config, batch)
    keys = tuple(model.state_dict())
    assert not any("identity" in key or "patch_encoder.blocks" in key for key in keys)
    assert not any("patch_encoder.readout" in key for key in keys)


def test_invalid_input_quality_rates_and_dimensions(batch):
    raw = synthetic_windows(sample_rate=20, epoch_seconds=4)
    raw["quality"]["eeg"]["valid"][0, 0, 0] = False
    with pytest.raises(ValueError, match="quality masks"):
        as_signal_batch(raw)
    with pytest.raises(ValueError, match="divisible"):
        build_backbone(replace(small_config(), patch_seconds=0.3), batch)
    with pytest.raises(ValueError, match="one epoch"):
        build_backbone(
            replace(small_config(), sequence_encoder=SequenceEncoderConfig()), batch
        )
    raw = synthetic_windows(sample_rate=20, epoch_seconds=4)
    raw["sample_rates"]["eeg"][1] = 21
    with pytest.raises(ValueError, match="sample rate"):
        as_signal_batch(raw)
    invalid = changed_values(batch, "eeg", lambda x: x[0, 0, 0, 0].fill_(float("nan")))
    with pytest.raises(ValueError, match="nonfinite"):
        build_backbone(small_config(), batch)(invalid)


def test_external_prediction_heads_receive_gradients(batch):
    """Toy external heads verify compatibility, not JEPA/codebook training quality."""
    model = build_backbone(small_config(), batch)
    mask = torch.ones(2, 6, 8, dtype=torch.bool)
    mask[..., 1] = False
    output = model(batch, MaskPlan("token", {"eeg": mask})).features["eeg"]
    weights = output.active.float()
    pooled = (output.tokens * weights[..., None]).sum(1) / weights.sum(
        1, keepdim=True
    ).clamp_min(1)
    jepa = torch.nn.Linear(8, 8)
    projection = torch.nn.Linear(8, 4)
    code_prediction = torch.nn.Linear(8, 16)
    with torch.no_grad():
        target = model(batch).features["eeg"].tokens[:, 1]
    loss = (jepa(pooled) - target).square().mean()
    loss += projection(pooled).square().mean()
    loss += torch.nn.functional.cross_entropy(
        code_prediction(pooled), torch.zeros(2, dtype=torch.long)
    )
    loss.backward()
    for head in (jepa, projection, code_prediction):
        assert head.weight.grad.abs().sum() > 0
    gradient = model.modality_encoders[
        "eeg"
    ].signal_encoder.patch_encoder.projection.weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_adapter_optional_grade_and_fixed_scale():
    raw = synthetic_windows(sample_rate=20, epoch_seconds=4)
    raw.pop("night_grade")
    batch = as_signal_batch(raw, scales={"eeg": 10})
    assert batch.night_grade is None
    torch.testing.assert_close(
        batch.groups["eeg"].values, raw["signals"]["eeg"] / 10, equal_nan=True
    )
    assert batch.groups["eeg"].units == ("a.u./10",) * 6


def test_patch_encoder_is_safe_when_called_independently(batch):
    signal = (
        build_backbone(small_config(), batch).modality_encoders["eeg"].signal_encoder
    )
    patches = signal.patchifier(batch.groups["eeg"], batch)
    values = patches.values.clone()
    visible = patches.sample_visible.clone()
    visible[..., :5] = False
    values[..., :5] = float("nan")
    values.requires_grad_()
    output = signal.patch_encoder(
        replace(patches, values=values, sample_visible=visible)
    )
    output.tokens.square().sum().backward()
    assert torch.isfinite(values.grad).all()
    assert torch.count_nonzero(values.grad[..., :5]) == 0
    assert all(
        torch.isfinite(p.grad).all() for p in signal.parameters() if p.grad is not None
    )
