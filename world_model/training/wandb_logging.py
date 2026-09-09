"""Optional experiment logging; disabled mode has no files or SDK side effects."""

from pathlib import Path


class WandbLogger:
    def __init__(self, *, project, run_name, mode, config, directory):
        self.run = None
        if mode == "disabled":
            return
        if mode not in ("offline", "online"):
            raise ValueError("wandb mode must be disabled, offline or online")
        import wandb

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(
            project=project,
            name=run_name,
            mode=mode,
            config=dict(config),
            dir=str(directory),
        )

    def log(self, metrics, step):
        if self.run is not None:
            self.run.log(dict(metrics), step=step)

    def finish(self):
        if self.run is not None:
            self.run.finish()
