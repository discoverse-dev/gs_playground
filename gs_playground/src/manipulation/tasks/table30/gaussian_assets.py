from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from gs_playground import ROOT_PATH
from gs_playground.src.utils.path_utils import as_posix_if_exists


def build_task_gaussians(task_dir: Path, mapping: Dict[str, Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name, rel_path in mapping.items():
        p = task_dir / rel_path
        posix = as_posix_if_exists(p)
        if posix:
            out[name] = posix
    return out


__all__ = [
    "franka_gaussians",
    "franka_background_ply",
    "build_task_gaussians",
]
