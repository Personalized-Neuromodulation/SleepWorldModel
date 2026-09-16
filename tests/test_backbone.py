"""Regression coverage migrated to the single CNN/Criss-Cross PSG backbone."""

from dataclasses import replace

import pytest
import torch

from backbone import BackboneConfig, FoundationBackbone, MaskPlan, build_backbone
from dataloader import as_signal_batch
from dataloader.synthetic import synthetic_windows
from pretraining.configuration import load_config


@pytest.fixture
def batch():
    torch.set_num_threads(2)
    torch.manual_seed(9)
    return as_signal_batch(synthetic_windows(epochs=1), foundation=True)


def small_config(**kwargs):
    return BackboneConfig(
        modality_encoder={
            "eeg_depth": 1,
            "multichannel_depth": 1,
            "single_channel_depth": 1,
        },
        fusion_depth=1,
        **kwargs,
    )


def test_default_config_builds_the_only_backbone(batch):
    config = BackboneConfig.from_dict(load_config()["model"])
    assert config == BackboneConfig.from_dict(config.to_dict())
    assert not BackboneConfig().numeric_tokenizers  # generic constructor has no dataset names
    model = build_backbone(config, batch)
    assert type(model) is FoundationBackbone
    assert not hasattr(model.modality_encoders["eeg"], "signal_encoder")
    assert model.modality_encoders["eeg"].tokenizer.convolutions[0].kernel_size == (49,)


@pytest.mark.parametrize(
    "key,value",
    [
        ("architecture", "legacy"),
        ("patch_encoder", {"variant": "direct_linear"}),
        ("sequence_encoder", {"causal": True}),
        ("fusion_heads", 4),
    ],
)
def test_removed_configuration_is_rejected(key, value):
    with pytest.raises(TypeError):
        BackboneConfig.from_dict({key: value})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feature_dim": 128},
        {"patch_seconds": 0.5},
        {"sample_rate": 20},
        {"channel_aggregation": "none"},
        {"fusion_depth": 0},
        {"post_fusion_temporal_depth": 3},
        {"dropout": 1.0},
    ],
)
def test_invalid_current_configuration(kwargs):
    with pytest.raises(ValueError):
        BackboneConfig(**kwargs)


@pytest.mark.parametrize("stage", ["waveform", "token"])
def test_hidden_waveform_cannot_affect_output_or_gradients(batch, stage):
    model = build_backbone(small_config(), batch).eval()
    group = batch.groups["eeg"]
    if stage == "token":
        visible = torch.ones(2, 6, 30, dtype=torch.bool)
        visible[..., 0] = False
        hidden_samples = 200
    else:
        visible = torch.ones_like(group.values, dtype=torch.bool)
        visible[..., :10] = False
        hidden_samples = 10
    values = group.values.clone()
    values[..., :hidden_samples] = float("nan")
    values.requires_grad_()
    changed = replace(
        batch, groups={**batch.groups, "eeg": replace(group, values=values)}
    )
    plan = MaskPlan(stage, {"eeg": visible})
    expected = model(batch, plan).foundation_representation
    actual = model(changed, plan).foundation_representation
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert torch.isfinite(values.grad).all()
    assert not values.grad[..., :hidden_samples].any()


@pytest.mark.parametrize("pool", ["attention", "mean"])
def test_disabled_identity_and_readout_ablation(batch, pool):
    config = small_config(
        channel_embedding=False, modality_embedding=False, channel_aggregation=pool
    )
    model = build_backbone(config, batch)
    assert not any("identity" in key for key in model.state_dict())
    result = model(batch, outputs=("patch_tokens", "local", "features"))
    assert result.foundation_representation.shape == (2, 256)
    assert result.patch_tokens["eeg"].data_valid.shape == (2, 6, 1)
    assert result.patch_tokens["eeg"].token_valid.shape == (2, 6, 30)
    assert (model.modality_encoders["eeg"].channel_pool.score is None) == (
        pool == "mean"
    )


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_optional_post_fusion_blocks_receive_gradients(batch, depth):
    model = build_backbone(small_config(post_fusion_temporal_depth=depth), batch)
    assert len(model.temporal_readout.blocks) == depth
    result = model(batch)
    result.foundation_representation.square().mean().backward()
    for block in model.temporal_readout.blocks:
        assert block.attention.value.weight.grad.abs().sum() > 0


def test_save_reload_and_input_metadata_validation(batch, tmp_path):
    model = build_backbone(small_config(), batch).eval()
    path = tmp_path / "backbone.pt"
    torch.save(
        {
            "config": model.config.to_dict(),
            "input_spec": model.input_spec,
            "state": model.state_dict(),
        },
        path,
    )
    saved = torch.load(path, weights_only=True)
    restored = build_backbone(
        BackboneConfig.from_dict(saved["config"]), saved["input_spec"]
    ).eval()
    restored.load_state_dict(saved["state"])
    with torch.no_grad():
        torch.testing.assert_close(
            model(batch).foundation_representation,
            restored(batch).foundation_representation,
        )
    group = batch.groups["eeg"]
    wrong = replace(
        batch, groups={**batch.groups, "eeg": replace(group, units=("wrong",) * 6)}
    )
    with pytest.raises(ValueError, match="units"):
        wrong.validate(foundation=True, input_spec=model.input_spec)


def test_external_prediction_heads_receive_gradients(batch):
    """Continuous features still support external predictors/classifiers."""
    model = build_backbone(small_config(), batch)
    pooled = model(batch).foundation_representation
    predictor = torch.nn.Linear(256, 256)
    projection = torch.nn.Linear(256, 128)
    code_head = torch.nn.Linear(256, 16)
    loss = (predictor(pooled) - pooled.detach()).square().mean()
    loss += projection(pooled).square().mean()
    loss += torch.nn.functional.cross_entropy(
        code_head(pooled), torch.zeros(2, dtype=torch.long)
    )
    loss.backward()
    for head in (predictor, projection, code_head):
        assert head.weight.grad.abs().sum() > 0
    gradient = model.modality_encoders["eeg"].tokenizer.convolutions[0].weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_adapter_optional_grade_fixed_scale_and_padding():
    raw = synthetic_windows()
    raw.pop("night_grade")
    batch = as_signal_batch(raw, scales={"eeg": 10})
    assert batch.night_grade is None
    assert not batch.data_valid["eeg"][1, :, 1].any()
    torch.testing.assert_close(
        batch.groups["eeg"].values, raw["signals"]["eeg"] / 10, equal_nan=True
    )
    assert batch.groups["eeg"].units == ("a.u./10",) * 6
    batch.to("cpu").validate()
