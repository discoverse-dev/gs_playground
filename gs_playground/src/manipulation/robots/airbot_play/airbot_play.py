from __future__ import annotations
import motrixsim as mtx
from motrixsim import ik
from typing import Dict, Optional
from pathlib import Path
from gs_playground import ROOT_PATH

from ..base_robot import BaseRobot

ASSETS_AIRBOT_PLAY_DIR = ROOT_PATH / "models" / "robots" / "manipulation" / "airbot_play"

class AirbotPlay(BaseRobot):
    """
    Airbot Play helper.
    Handles IK, joint control, and observation extraction.
    Also provides a list of gaussian assets for rendering.
    """
    
    GAUSSIANS: Dict[str, Path] = {
        "arm_base": (ASSETS_AIRBOT_PLAY_DIR / "3dgs" / "arm_base.ply"),
        "link1": (ASSETS_AIRBOT_PLAY_DIR / "3dgs" / "link1.ply"),
        "link2": (ASSETS_AIRBOT_PLAY_DIR / "3dgs" / "link2.ply"),
        "link3": (ASSETS_AIRBOT_PLAY_DIR / "3dgs" / "link3.ply"),
        "link4": (ASSETS_AIRBOT_PLAY_DIR / "3dgs" / "link4.ply"),
        "link5": (ASSETS_AIRBOT_PLAY_DIR / "3dgs" / "link5.ply"),
        "link6": (ASSETS_AIRBOT_PLAY_DIR / "3dgs" / "link6.ply"),
        "left": (ASSETS_AIRBOT_PLAY_DIR / "3dgs" / "left.ply"),
        "right": (ASSETS_AIRBOT_PLAY_DIR / "3dgs" / "right.ply"),
    }
    
    BACKGROUND_PLY = ASSETS_AIRBOT_PLAY_DIR / "3dgs" / "background.ply"

    num_dof_arm = 6
    ee_site_name = "endpoint"
    gripper_actuator_name = "gripper"

    def __init__(self, mx_model: mtx.SceneModel):
        super().__init__(mx_model)
        
        # IK Setup (copied from previous impl)
        self.chain = ik.IkChain(
            self.mx_model,
            start_link="link1",
            end_link="link6",
            end_effector_offset=[0.0, 0.0, 0.023, 0.707, 0.0, 0.0, 0.707],
        )
        self.solver = ik.DlsSolver(
            max_iter=50,
            step_size=0.5,
            tolerance=1e-3,
            damping=1e-3,
        )