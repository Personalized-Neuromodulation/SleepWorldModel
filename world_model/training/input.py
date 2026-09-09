"""One input configuration shared by training and debugging."""

from pathlib import Path

from dataloader import DatasetMetadata, WindowDataset, collate_windows


def add_input_arguments(parser, *, required_root=False):
    parser.add_argument(
        "--root",
        type=Path,
        required=required_root,
        help="Published release root; raw H5 root only for hsp_raw",
    )
    parser.add_argument(
        "--dataset",
        default="hsp",
        help="Registered reader name; hsp_raw is the legacy loader",
    )
    parser.add_argument("--version", default="v1.0.0")
    parser.add_argument(
        "--manifest", type=Path, help="Legacy hsp_raw JSONL manifest only"
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "test", "all"), default="train"
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    parser.add_argument(
        "--channel-profile",
        default="hsp_full",
        help="Legacy hsp_raw channel profile only",
    )


def open_dataset(args):
    if args.root is None:
        raise ValueError("real data requires --root")
    split = None if args.split == "all" else args.split
    if args.dataset != "hsp_raw":
        if args.manifest is not None:
            raise ValueError(
                "published readers use manifests inside --root; omit --manifest"
            )
        dataset = WindowDataset(
            args.root,
            dataset=args.dataset,
            version=args.version,
            split=split,
            split_seed=args.split_seed,
            split_ratios=tuple(args.split_ratios),
            context_epochs=args.context_epochs,
            stride_epochs=getattr(args, "stride_epochs", args.context_epochs),
        )
        return dataset, collate_windows, dataset.metadata
    from dataloader.readers.hsp_raw import HSPDataset, hsp_collate_fn
    from dataloader.readers.hsp_raw.channels import CHANNEL_PROFILES

    from .defaults import HSP_SSL_TARGET_SAMPLE_RATES

    if args.manifest is None:
        raise ValueError("hsp_raw requires --manifest")
    dataset = HSPDataset(
        root=args.root,
        manifest_path=args.manifest,
        split=split,
        channel_profile=args.channel_profile,
        sampling_mode="context",
        epoch_seconds=30,
        context_epochs=args.context_epochs,
        stride_epochs=getattr(args, "stride_epochs", args.context_epochs),
        missing_channel="mask",
        target_sample_rates=HSP_SSL_TARGET_SAMPLE_RATES,
        normalization="none",
    )
    channels = {
        name: tuple(spec.name for spec in specs)
        for name, specs in CHANNEL_PROFILES[args.channel_profile].items()
    }
    metadata = DatasetMetadata(
        "hsp_raw",
        "raw",
        channels,
        {k: HSP_SSL_TARGET_SAMPLE_RATES[k] for k in channels},
        {k: () for k in channels},
    )
    return dataset, hsp_collate_fn, metadata


def model_input_config(metadata):
    """Preserve the dataset's modalities, channel counts and sample rates."""
    return {
        "channels": {
            name: len(channels) for name, channels in metadata.channels.items()
        },
        "sample_rates": dict(metadata.sample_rates),
        "epoch_seconds": metadata.epoch_seconds,
    }
