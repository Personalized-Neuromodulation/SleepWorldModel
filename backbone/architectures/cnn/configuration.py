"""Serializable CNN layer specifications for the patch tokenizer."""

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ConvLayerConfig:
    out_channels: int = 32
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    dilation: int = 1
    bias: bool = True

    def __post_init__(self):
        for name in ("out_channels", "kernel_size", "stride", "dilation"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"CNN {name} must be a positive integer")
        if type(self.padding) is not int or self.padding < 0:
            raise ValueError("CNN padding must be a nonnegative integer")
        if type(self.bias) is not bool:
            raise ValueError("CNN bias must be bool")


@dataclass(frozen=True)
class PatchTokenizerConfig:
    layers: list[ConvLayerConfig] = field(
        default_factory=lambda: [
            ConvLayerConfig(kernel_size=49, stride=25, padding=24),
            ConvLayerConfig(),
            ConvLayerConfig(),
        ]
    )
    norm: str = "group_norm"
    group_norm_groups: int = 8
    activation: str = "gelu"
    dropout: float = 0.0
    projection: str = "none"
    output_norm: str = "layer_norm"

    def __post_init__(self):
        if (
            not isinstance(self.layers, list)
            or not self.layers
            or not all(isinstance(layer, ConvLayerConfig) for layer in self.layers)
        ):
            raise ValueError("CNN layers must be a nonempty list of ConvLayerConfig")
        if self.norm not in ("group_norm", "none"):
            raise ValueError("CNN norm must be group_norm or none")
        if type(self.group_norm_groups) is not int or self.group_norm_groups < 1:
            raise ValueError("group_norm_groups must be a positive integer")
        if self.norm == "group_norm" and any(
            layer.out_channels % self.group_norm_groups for layer in self.layers
        ):
            raise ValueError("CNN out_channels must be divisible by group_norm_groups")
        if self.activation not in ("gelu", "relu", "silu"):
            raise ValueError("CNN activation must be gelu, relu or silu")
        if not 0 <= self.dropout < 1:
            raise ValueError("CNN dropout must be in [0,1)")
        if self.projection not in ("none", "linear"):
            raise ValueError("CNN projection must be none or linear")
        if self.output_norm not in ("layer_norm", "none"):
            raise ValueError("CNN output_norm must be layer_norm or none")

    def output_shape(self, samples):
        length = samples
        for layer in self.layers:
            length = (
                length
                + 2 * layer.padding
                - layer.dilation * (layer.kernel_size - 1)
                - 1
            ) // layer.stride + 1
            if length < 1:
                raise ValueError("CNN layer produces an empty temporal axis")
            if (
                self.norm == "group_norm"
                and layer.out_channels // self.group_norm_groups * length <= 1
            ):
                raise ValueError(
                    "GroupNorm needs more than one value per group for single-patch inputs"
                )
        return self.layers[-1].out_channels, length

    def validate_output(self, samples, feature_dim):
        channels, length = self.output_shape(samples)
        if self.projection == "none" and channels * length != feature_dim:
            raise ValueError(
                f"CNN flatten {channels}*{length}={channels * length}, expected {feature_dim}; adjust layers or set projection=linear"
            )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if "layers" in values:
            values["layers"] = [ConvLayerConfig(**layer) for layer in values["layers"]]
        return cls(**values)
