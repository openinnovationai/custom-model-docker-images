"""Download a pinned model snapshot into the image at build time."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    repo_id = os.environ["MODEL_ID"]
    revision = os.environ["MODEL_REVISION"]  # a commit SHA, never a branch name
    target = Path(os.environ["MODEL_DIR"])

    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=target,
        # Only what the runtime loads - skip ONNX/TF/duplicate .bin weights.
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        token=os.environ.get("HF_TOKEN") or None,
    )
    shutil.rmtree(target / ".cache", ignore_errors=True)

    if not (target / "config.json").is_file() or not any(target.glob("*.safetensors")):
        raise RuntimeError(f"snapshot of {repo_id}@{revision} is incomplete")
    print(f"{repo_id}@{revision} installed at {target}")


if __name__ == "__main__":
    main()
