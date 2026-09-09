"""Public PSG backbone; no task heads, objectives or target networks."""

from torch import nn

from dataloader.signals import SignalBatch

from ..contracts import BackboneOutput, MaskPlan


class PSGBackbone(nn.Module):
    def __init__(self, config, input_spec, modality_encoders, fusion):
        super().__init__()
        self.config = config
        self.input_spec = input_spec
        self.modality_encoders = nn.ModuleDict(modality_encoders)
        self.fusion = fusion

    def _check_input(self, batch):
        if set(batch.groups) != set(self.input_spec):
            raise ValueError("batch encoding groups differ from backbone input_spec")
        for name, group in batch.groups.items():
            spec = self.input_spec[name]
            if tuple(group.channel_ids) != tuple(spec["channel_ids"]):
                raise ValueError(f"{name}: channel order differs from input_spec")
            if tuple(group.units) != tuple(spec["units"]):
                raise ValueError(f"{name}: units/scaling differ from input_spec")
            if group.values.shape[-1] != spec["epoch_samples"]:
                raise ValueError(f"{name}: epoch sample count differs from input_spec")

    def layout(self, batch: SignalBatch):
        self._check_input(batch)
        return {
            name: model.signal_encoder.patchifier.layout(batch.groups[name], batch)
            for name, model in self.modality_encoders.items()
        }

    def forward(
        self,
        batch: SignalBatch,
        mask_plan: MaskPlan | None = None,
        outputs=("features",),
    ) -> BackboneOutput:
        self._check_input(batch)
        requested = set(outputs)
        if not requested or requested - {"patch_tokens", "local", "features", "joint"}:
            raise ValueError(
                "outputs must select patch_tokens, local, features or joint"
            )
        need_features = bool(requested & {"features", "joint"})
        need_local = need_features or "local" in requested
        if need_features and self.config.channel_aggregation == "none":
            raise ValueError("features/joint require channel aggregation")
        if "joint" in requested and self.fusion is None:
            raise ValueError("joint requires enabled fusion")
        plan = mask_plan or MaskPlan()
        if set(plan.visible) - set(batch.groups):
            raise ValueError("MaskPlan contains unknown groups")
        result = BackboneOutput()
        features = {}
        for name, model in self.modality_encoders.items():
            output = model(
                batch.groups[name],
                batch,
                plan.stage,
                plan.visible.get(name),
                need_local,
                need_features,
            )
            if "patch_tokens" in requested:
                result.patch_tokens[name] = output.patch_tokens
            if "local" in requested:
                result.local[name] = output.local
            if need_features:
                features[name] = output.features
        if "features" in requested:
            result.features = features
        if "joint" in requested:
            result.joint = self.fusion(features)
        return result
