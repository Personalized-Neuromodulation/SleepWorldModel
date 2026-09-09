"""Convert a dataloader batch into model input, preserving per-epoch QC."""

import torch

from world_model.ssl import SSLConfig


def prepare_batch(batch, config: SSLConfig, device):
    epoch_mask = batch["epoch_mask"].to(device=device, dtype=torch.bool)
    signals, valid = {}, {}
    for name, channels in config.channels.items():
        source = batch["signals"][name]
        samples = round(config.sample_rates[name] * config.epoch_seconds)
        if source.shape != (*epoch_mask.shape, channels, samples):
            raise ValueError(f"{name}: unexpected signal shape {tuple(source.shape)}")
        rates = torch.as_tensor(batch["sample_rates"][name])
        if not torch.all(rates == config.sample_rates[name]):
            raise ValueError(f"{name}: sample rate does not match model config")
        availability = batch["available_mask"][name].to(device=device, dtype=torch.bool)
        channel_mask = batch["channel_mask"][name].to(device=device, dtype=torch.bool)
        if (
            availability.shape != (source.shape[0], channels)
            or channel_mask.shape != availability.shape
        ):
            raise ValueError(f"{name}: channel availability shape mismatch")
        mask = (availability & channel_mask).unsqueeze(1) & epoch_mask.unsqueeze(-1)
        if name in batch.get("quality", {}):
            quality = batch["quality"][name]["valid"].to(
                device=device, dtype=torch.bool
            )
            if quality.shape != source.shape[:3]:
                raise ValueError(f"{name}: quality mask shape mismatch")
            mask = mask & quality
        values = source.to(device=device, dtype=torch.float32)
        values = torch.where(mask.unsqueeze(-1), values, 0.0)
        if not torch.isfinite(values).all():
            raise ValueError(f"{name}: valid signal contains NaN or infinity")
        signals[name], valid[name] = values, mask
    return {"signals": signals, "valid": valid, "epoch_mask": epoch_mask}
