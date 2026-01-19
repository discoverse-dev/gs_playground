"""RoboArena package initialization."""

import os
from etils import epath

__version__ = "0.1.0"

# Root path is used for loading XML strings directly using etils.epath.
ROOT_PATH = epath.Path(__file__).parent

GS_GYM_ENVS_PATH = ROOT_PATH / "gs_playground/src/env"