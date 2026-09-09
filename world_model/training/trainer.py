"""A direct training loop: prepare batch, forward, loss, backward, step."""

import time
from dataclasses import asdict
from pathlib import Path

import torch

from .batch_adapter import prepare_batch


def train_ssl(
    *,
    model,
    objective,
    data_loader,
    optimizer,
    device,
    max_steps,
    logger,
    checkpoint_path,
    gradient_clip=1.0,
    input_config=None,
):
    if max_steps < 1 or gradient_clip <= 0:
        raise ValueError("max_steps and gradient_clip must be positive")
    model.to(device).train()
    objective.to(device)
    started = time.perf_counter()
    completed = 0
    last_metrics = {}
    for cpu_batch in data_loader:
        batch = prepare_batch(cpu_batch, model.config, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(batch)
            losses = objective(output)
        if not torch.isfinite(losses.total):
            raise FloatingPointError("training loss is not finite")
        losses.total.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), gradient_clip, error_if_nonfinite=True
        )
        optimizer.step()
        representations = output.representation[output.valid].detach().float()
        last_metrics = {
            "loss/total": float(losses.total.detach()),
            "loss/invariance": float(losses.invariance),
            "loss/sigreg": float(losses.sigreg),
            "representation/std": float(
                representations.std(dim=0, unbiased=False).mean()
            ),
            "optimization/gradient_norm": float(norm),
            "optimization/learning_rate": optimizer.param_groups[0]["lr"],
            "data/valid_epochs": int(output.valid.sum()),
        }
        logger.log(last_metrics, completed)
        completed += 1
        if completed >= max_steps:
            break
    if not completed:
        raise RuntimeError("data loader produced no batches")
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 3,
            "architecture": "minimal-sigreg-v1",
            "model_config": asdict(model.config),
            "input_config": input_config,
            "loss_config": {
                "sigreg_weight": objective.sigreg_weight,
                "num_projections": objective.sigreg.num_projections,
                "num_frequencies": len(objective.sigreg.frequencies),
            },
            "model": model.state_dict(),
            "objective": objective.state_dict(),
            "optimizer": optimizer.state_dict(),
            "steps": completed,
        },
        checkpoint,
    )
    return {
        "steps": completed,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint.resolve()),
        "last_metrics": last_metrics,
    }
