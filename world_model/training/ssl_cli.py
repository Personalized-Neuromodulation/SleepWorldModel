"""Train the minimal CNN + two-view consistency + SIGReg baseline."""

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, RandomSampler

from world_model.ssl import SSLConfig, SSLLoss, SSLModel

from .input import add_input_arguments, model_input_config, open_dataset
from .trainer import train_ssl
from .wandb_logging import WandbLogger


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser, required_root=True)
    parser.add_argument("--context-epochs", type=int, default=20)
    parser.add_argument("--stride-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--crop-fraction", type=float, default=0.8)
    parser.add_argument("--sigreg-weight", type=float, default=0.05)
    parser.add_argument("--sigreg-projections", type=int, default=256)
    parser.add_argument("--sigreg-frequencies", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--wandb-project", default="SleepWorldModel")
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="disabled"
    )
    parser.add_argument("--wandb-dir", type=Path, default=Path("artifacts/wandb"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/checkpoints/ssl_minimal.pt")
    )
    return parser


def main(argv=None):
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    if args.max_steps < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("steps and batch size must be positive, workers nonnegative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset, collate_fn, metadata = open_dataset(args)
    logger = None
    try:
        config = SSLConfig(
            **model_input_config(metadata),
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            projection_dim=args.projection_dim,
            crop_fraction=args.crop_fraction,
        )
        model = SSLModel(config)
        objective = SSLLoss(
            args.sigreg_weight, args.sigreg_projections, args.sigreg_frequencies
        )
        device = torch.device(args.device)
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        sampler = RandomSampler(
            dataset, replacement=True, num_samples=args.max_steps * args.batch_size
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
        )
        input_config = {
            "dataset": args.dataset,
            "version": metadata.version,
            "root": str(args.root),
            "manifest": str(args.manifest) if args.manifest else None,
            "split": args.split,
            "split_seed": args.split_seed,
            "split_ratios": list(args.split_ratios),
            "context_epochs": args.context_epochs,
            "stride_epochs": args.stride_epochs,
            "metadata": asdict(metadata),
            "qc_policy": "per_epoch_per_channel",
        }
        logger = WandbLogger(
            project=args.wandb_project,
            run_name=args.wandb_run_name,
            mode=args.wandb_mode,
            config={
                "input": input_config,
                "model": asdict(config),
                "sigreg_weight": args.sigreg_weight,
                "seed": args.seed,
            },
            directory=args.wandb_dir,
        )
        summary = train_ssl(
            model=model,
            objective=objective,
            data_loader=loader,
            optimizer=optimizer,
            device=device,
            max_steps=args.max_steps,
            logger=logger,
            checkpoint_path=args.checkpoint,
            gradient_clip=args.gradient_clip,
            input_config=input_config,
        )
    finally:
        if logger is not None:
            logger.finish()
        dataset.close()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
