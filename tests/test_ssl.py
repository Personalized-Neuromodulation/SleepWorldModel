"""Numerical and behavioral tests for the minimal SSL baseline."""

import numpy as np
import pytest
import torch
from scipy.integrate import trapezoid

from world_model.ssl import SIGReg, SSLConfig, SSLLoss, SSLModel, SSLOutput


def test_sigreg_matches_independent_complex_characteristic_function():
    values = np.array([-1.4, -0.1, 0.2, 1.8], dtype=np.float64)
    frequencies = np.linspace(0, 3, 17)
    empirical = np.exp(1j * values[:, None] * frequencies).mean(axis=0)
    gaussian = np.exp(-(frequencies**2) / 2)
    expected = (
        2
        * len(values)
        * trapezoid(abs(empirical - gaussian) ** 2 * gaussian, frequencies)
    )
    # In one dimension every unit projection is +1 or -1, with identical statistic.
    actual = SIGReg(num_projections=8)(torch.tensor(values[:, None])).item()
    assert actual == pytest.approx(expected, rel=2e-6)


def test_sigreg_detects_collapse_and_non_gaussian_matching_moments():
    torch.manual_seed(12)
    gaussian = torch.randn(4096, 1)
    collapsed = torch.zeros_like(gaussian)
    two_points = torch.cat((torch.ones(2048, 1), -torch.ones(2048, 1)))
    regularizer = SIGReg(num_projections=8)
    normal_loss = regularizer(gaussian)
    assert regularizer(collapsed) > normal_loss * 10
    assert regularizer(two_points) > normal_loss * 10
    assert regularizer(gaussian + 2) > normal_loss * 10


def test_sigreg_ignores_masked_nonfinite_rows_and_their_gradients():
    x = torch.randn(8, 4, requires_grad=True)
    with torch.no_grad():
        x[-2] = float("nan")
        x[-1] = float("inf")
    valid = torch.tensor([True] * 6 + [False] * 2)
    regularizer = SIGReg(num_projections=16)
    torch.manual_seed(8)
    actual = regularizer(x, valid)
    torch.manual_seed(8)
    expected = regularizer(x[:6])
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(x.grad).all()
    assert not x.grad[-2:].any()
    assert x.grad[:6].abs().sum() > 0


def test_sigreg_fp32_under_autocast():
    x = torch.randn(32, 8, dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = SIGReg(num_projections=8)(x)
    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("count", [0, 1])
def test_sigreg_requires_multiple_samples(count):
    with pytest.raises(ValueError, match="at least two"):
        SIGReg()(torch.zeros(count, 3))


def test_sigreg_rejects_nonfinite_valid_values():
    with pytest.raises(ValueError, match="NaN"):
        SIGReg()(torch.full((2, 3), float("nan")))


def _example():
    torch.set_num_threads(1)
    config = SSLConfig(
        channels={"eeg": 2, "respiratory": 3},
        sample_rates={"eeg": 32, "respiratory": 16},
        epoch_seconds=4,
        hidden_dim=8,
        embedding_dim=16,
        projection_dim=8,
    )
    batch = {
        "signals": {
            name: torch.randn(2, 3, c, int(4 * config.sample_rates[name]))
            for name, c in config.channels.items()
        },
        "valid": {
            name: torch.ones(2, 3, c, dtype=torch.bool)
            for name, c in config.channels.items()
        },
        "epoch_mask": torch.tensor([[True, True, True], [True, True, False]]),
    }
    return config, batch


def test_model_both_views_backpropagate_to_shared_encoder():
    config, batch = _example()
    model = SSLModel(config).train()
    output = model(batch)
    output.view1.retain_grad()
    output.view2.retain_grad()
    losses = SSLLoss(num_projections=8)(output)
    losses.total.backward()
    assert output.view1.shape == (2, 3, 8)
    assert output.representation.shape == (2, 3, 16)
    assert not output.valid[1, 2]
    for view in (output.view1, output.view2):
        assert view.grad[output.valid].abs().sum() > 0
        assert not view.grad[~output.valid].any()
    assert torch.isfinite(model.encoders["eeg"][0].weight.grad).all()


def test_masked_channels_padding_and_epoch_independence():
    config, batch = _example()
    model = SSLModel(config).eval()
    batch["valid"]["eeg"][0, 1, 0] = False
    expected, valid = model.encode(batch)
    batch["signals"]["eeg"][0, 1, 0] = float("nan")
    for name in batch["signals"]:
        batch["signals"][name][1, 2] = float("nan")
    actual, _ = model.encode(batch)
    torch.testing.assert_close(actual, expected)
    assert not actual[~valid].any()
    batch["signals"]["respiratory"][0, 1] *= 10
    changed, _ = model.encode(batch)
    torch.testing.assert_close(changed[0, 0], actual[0, 0])
    output = model(batch)
    torch.testing.assert_close(output.view1, output.view2)


def test_loss_rejects_all_invalid_epochs():
    z = torch.zeros(1, 1, 4, requires_grad=True)
    output = SSLOutput(z, z, z, torch.zeros(1, 1, dtype=torch.bool))
    with pytest.raises(ValueError, match="valid epochs"):
        SSLLoss()(output)


def test_disabled_logger_creates_no_artifacts(tmp_path):
    from world_model.training.wandb_logging import WandbLogger

    directory = tmp_path / "absent"
    logger = WandbLogger(
        project="test", run_name=None, mode="disabled", config={}, directory=directory
    )
    logger.log({"loss": 1}, 0)
    logger.finish()
    assert not directory.exists()
