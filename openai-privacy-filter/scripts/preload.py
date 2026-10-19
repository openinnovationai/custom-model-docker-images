"""Download a pinned OPF checkpoint into the container image."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_REPOSITORY = "openai/privacy-filter"


def main() -> None:
    target = Path(os.environ["OPF_CHECKPOINT"])
    revision = os.environ["OPF_MODEL_REVISION"]
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        shutil.rmtree(target)

    temporary = Path(tempfile.mkdtemp(prefix="privacy-filter-", dir=target.parent))
    try:
        snapshot_download(
            repo_id=MODEL_REPOSITORY,
            revision=revision,
            local_dir=temporary,
            allow_patterns=["original/*"],
        )
        source = temporary / "original"
        if not (source / "config.json").is_file() or not any(source.glob("*.safetensors")):
            raise RuntimeError("downloaded Privacy Filter checkpoint is incomplete")
        source.rename(target)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    print(f"Privacy Filter checkpoint {revision} installed at {target}")


if __name__ == "__main__":
    main()
