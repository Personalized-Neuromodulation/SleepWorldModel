from __future__ import annotations

from pathlib import Path
from typing import Any


def read_h5(h5_path: str | Path, load_arrays: bool = False) -> dict[str, Any]:
    """Read an HDF5 tree, optionally loading dataset values."""
    import h5py

    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"H5 file not found: {path}")

    def read_attrs(obj: Any) -> dict[str, Any]:
        return {
            key: value.tolist() if hasattr(value, "tolist") else value
            for key, value in obj.attrs.items()
        }

    def read_node(node: Any) -> dict[str, Any]:
        if isinstance(node, h5py.Dataset):
            info: dict[str, Any] = {
                "type": "dataset",
                "shape": node.shape,
                "dtype": str(node.dtype),
                "attrs": read_attrs(node),
            }
            if load_arrays:
                value = node[()]
                info["data"] = value.tolist() if hasattr(value, "tolist") else value
            return info
        if isinstance(node, h5py.Group):
            return {
                "type": "group",
                "attrs": read_attrs(node),
                "children": {name: read_node(child) for name, child in node.items()},
            }
        return {"type": type(node).__name__}

    with h5py.File(path, "r") as handle:
        return {
            "path": str(path),
            "attrs": read_attrs(handle),
            "children": {name: read_node(child) for name, child in handle.items()},
        }


def list_signals(h5_path: str | Path) -> dict[str, dict[str, Any]]:
    """Return all signals with shape, dtype, and attributes."""
    import h5py

    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"H5 file not found: {path}")
    with h5py.File(path, "r") as handle:
        if "signals" not in handle:
            return {}
        return {
            name: {
                "shape": dataset.shape,
                "dtype": str(dataset.dtype),
                "attrs": {
                    key: value.tolist() if hasattr(value, "tolist") else value
                    for key, value in dataset.attrs.items()
                },
            }
            for name, dataset in handle["signals"].items()
        }


def read_signal(
    h5_path: str | Path,
    signal_name: str,
    start: int | None = None,
    stop: int | None = None,
) -> Any:
    """Read a complete signal or a sample slice from `/signals`."""
    import h5py

    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"H5 file not found: {path}")
    with h5py.File(path, "r") as handle:
        signal_path = f"signals/{signal_name}"
        if signal_path not in handle:
            available = ", ".join(handle["signals"].keys()) if "signals" in handle else "none"
            raise KeyError(
                f"Signal not found: {signal_name}. Available signals: {available}"
            )
        dataset = handle[signal_path]
        return dataset[()] if start is None and stop is None else dataset[start:stop]


def print_h5_tree(h5_path: str | Path) -> None:
    """Print HDF5 groups and datasets without loading full arrays."""
    import h5py

    path = Path(h5_path)

    def print_node(name: str, node: Any) -> None:
        indent = "  " * name.count("/")
        short_name = name.rsplit("/", 1)[-1]
        if isinstance(node, h5py.Dataset):
            print(f"{indent}- {short_name}: dataset shape={node.shape}, dtype={node.dtype}")
        else:
            print(f"{indent}+ {short_name}: group")

    with h5py.File(path, "r") as handle:
        print(f"H5 file: {path}")
        handle.visititems(print_node)


__all__ = ["list_signals", "print_h5_tree", "read_h5", "read_signal"]
