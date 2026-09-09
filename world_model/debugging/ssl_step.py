"""Read one batch and inspect the minimal SSL forward/loss/backward path."""

import argparse
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader, Subset

from world_model.ssl import SSLConfig, SSLLoss, SSLModel
from world_model.training.batch_adapter import prepare_batch
from world_model.training.input import (
    add_input_arguments,
    model_input_config,
    open_dataset,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--context-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument(
        "--break-at",
        choices=("batch", "model", "loss", "backward", "none"),
        default="none",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.context_epochs < 1:
        raise ValueError("batch size and context epochs must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = None
    try:
        if args.synthetic or (args.root is None and args.manifest is None):
            config = SSLConfig(
                channels={"eeg": 2, "ecg": 1}, sample_rates={"eeg": 100, "ecg": 100}
            )
            b, e = args.batch_size, args.context_epochs
            batch = {
                "signals": {
                    k: torch.randn(b, e, c, 3000, device=device)
                    for k, c in config.channels.items()
                },
                "valid": {
                    k: torch.ones(b, e, c, dtype=torch.bool, device=device)
                    for k, c in config.channels.items()
                },
                "epoch_mask": torch.ones(b, e, dtype=torch.bool, device=device),
            }
        else:
            dataset, collate_fn, metadata = open_dataset(args)
            config = SSLConfig(**model_input_config(metadata))
            indices = [
                (args.sample_index + i) % len(dataset) for i in range(args.batch_size)
            ]
            cpu_batch = next(
                iter(
                    DataLoader(
                        Subset(dataset, indices),
                        batch_size=args.batch_size,
                        collate_fn=collate_fn,
                    )
                )
            )
            batch = prepare_batch(cpu_batch, config, device)
        print("Config:", asdict(config))
        print("Signals:", {k: tuple(v.shape) for k, v in batch["signals"].items()})
        if args.break_at == "batch":
            breakpoint()
        model, objective = SSLModel(config).to(device), SSLLoss().to(device)
        if args.break_at == "model":
            breakpoint()
        output = model(batch)
        print(
            "Projections:",
            tuple(output.view1.shape),
            "valid epochs:",
            int(output.valid.sum()),
        )
        if args.break_at == "loss":
            breakpoint()
        losses = objective(output)
        print({k: float(v.detach()) for k, v in zip(losses._fields, losses)})
        if not args.forward_only:
            if args.break_at == "backward":
                breakpoint()
            losses.total.backward()
            print("Backward complete; no optimizer step or checkpoint write.")
        return 0
    finally:
        if dataset is not None:
            dataset.close()


if __name__ == "__main__":
    raise SystemExit(main())
