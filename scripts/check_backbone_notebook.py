"""Execute the same interactive notebook on synthetic or real PSG windows.

Run from the repository root with the Python environment containing torch.
Outputs are written separately, so the source notebook remains easy to review.
"""

import argparse
import os
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--notebook",
        default="backbone_step_by_step",
        choices=("backbone_step_by_step", "architecture_debug_walkthrough"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / f"notebooks/{args.notebook}.ipynb", as_version=4)
    nbformat.validate(notebook)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client = NotebookClient(notebook, km=manager, timeout=240)
    env = {
        **os.environ,
        "PSG_NOTEBOOK_REAL": "1" if args.real else "0",
        "PSG_NOTEBOOK_DEVICE": args.device,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        executed = client.execute(cwd=str(root), env=env)
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
    name = f"{args.notebook}_{'real' if args.real else 'synthetic'}_{args.device}.ipynb"
    target = root / "artifacts" / "backbone" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(executed, target)
    print(
        f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} cells: {target}"
    )


if __name__ == "__main__":
    main()
