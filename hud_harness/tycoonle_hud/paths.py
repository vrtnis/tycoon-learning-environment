from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_python_path() -> None:
    """Prefer the local checkout's tycoonle_jax package when available."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "python"
        if (candidate / "tycoonle_jax").is_dir():
            path = str(candidate)
            if path not in sys.path:
                sys.path.insert(0, path)
            return
