from __future__ import annotations
import numpy as np
import motrixsim as mtx
from motrixsim import ik
import gymnasium as gym
from typing import Dict, Optional
from pathlib import Path
from gs_playground import ROOT_PATH
from gs_playground.src.utils.path_utils import as_posix_if_exists

from ..base_robot import BaseRobot

ASSETS_FRANKA_DIR = ROOT_PATH / "models" / "robots" / "manipulation" / "franka_robotiq"

class FrankaRobotiq(BaseRobot):
    """
    Franka Emika Panda + Robotiq 2F-85 gripper helper.
    Handles IK, joint control, and observation extraction.
    Also provides a list of gaussian assets for rendering.
    """
    
    GAUSSIANS: Dict[str, Path] = {
        "link0": ASSETS_FRANKA_DIR / "3dgs" / "franka" / "link0.ply",
        "link1": ASSETS_FRANKA_DIR / "3dgs" / "franka" / "link1.ply",
        "link2": ASSETS_FRANKA_DIR / "3dgs" / "franka" / "link2.ply",
        "link3": ASSETS_FRANKA_DIR / "3dgs" / "franka" / "link3.ply",
        "link4": ASSETS_FRANKA_DIR / "3dgs" / "franka" / "link4.ply",
        "link5": ASSETS_FRANKA_DIR / "3dgs" / "franka" / "link5.ply",
        "link6": ASSETS_FRANKA_DIR / "3dgs" / "franka" / "link6.ply",
        "link7": ASSETS_FRANKA_DIR / "3dgs" / "franka" / "link7.ply",
        "robotiq_base": ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "robotiq_base.ply",
        "left_driver": ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "left_driver.ply",
        "left_coupler": ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "left_coupler.ply",
        "left_spring_link": ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "left_spring_link.ply",
        "left_follower": ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "left_follower.ply",
        "right_driver": ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "right_driver.ply",
        "right_coupler": ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "right_coupler.ply",
        "right_spring_link": ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "right_spring_link.ply",
        "right_follower": ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "right_follower.ply",
    }
    
    BACKGROUND_PLY = ASSETS_FRANKA_DIR / "3dgs" / "background.ply"

    num_dof_arm = 7
    ee_site_name = "gripper"
    gripper_actuator_name = "fingers_actuator"

    def __init__(self, mx_model: mtx.SceneModel):
        super().__init__(mx_model)
        
        # IK Setup (copied from previous impl)
        self.chain = ik.IkChain(
            self.mx_model,
            start_link="link1",
            end_link="robotiq_base",
            end_effector_offset=[0.0, 0.0, 0.1489, 0.0, 0.0, 0.0, 1.0],
        )
        self.solver = ik.DlsSolver(
            max_iter=50,
            step_size=0.5,
            tolerance=1e-3,
            damping=1e-3,
        )

    @property
    def action_space(self) -> gym.Space:
        limits = self.mx_model.actuator_ctrl_limits
        return gym.spaces.Box(low=limits[0], high=limits[1], dtype=np.float32)

    @classmethod
    def robot_gaussians(cls) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for name, path in cls.GAUSSIANS.items():
            posix = as_posix_if_exists(path)
            if posix:
                out[name] = posix
        return out

    @classmethod
    def robot_background_ply(cls) -> Optional[str]:
        return as_posix_if_exists(cls.BACKGROUND_PLY)
