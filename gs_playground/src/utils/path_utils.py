from __future__ import annotations

from pathlib import Path
from typing import Optional


def as_posix_if_exists(path: Path) -> Optional[str]:
    """Return path.as_posix() if the file exists, else None."""
    return path.as_posix() if path.exists() else None


__all__ = ["as_posix_if_exists"]
