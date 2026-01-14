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

ASSETS_UR5E_DIR = ROOT_PATH / "models" / "robots" / "manipulation" / "universal_robots_ur5e_robotiq"

class UR5eRobotiq(BaseRobot):
    """
    Universal Robots UR5e + Robotiq 2F-85 gripper helper.
    Handles IK, joint control, and observation extraction.
    Also provides a list of gaussian assets for rendering.
    """
    
    GAUSSIANS: Dict[str, Path] = {
        "base": ASSETS_UR5E_DIR / "3dgs" / "ur5e" / "base.ply",
        "shoulder_link": ASSETS_UR5E_DIR / "3dgs" / "ur5e" / "shoulder_link.ply",
        "upper_arm_link": ASSETS_UR5E_DIR / "3dgs" / "ur5e" / "upper_arm_link.ply",
        "forearm_link": ASSETS_UR5E_DIR / "3dgs" / "ur5e" / "forearm_link.ply",
        "wrist_1_link": ASSETS_UR5E_DIR / "3dgs" / "ur5e" / "wrist_1_link.ply",
        "wrist_2_link": ASSETS_UR5E_DIR / "3dgs" / "ur5e" / "wrist_2_link.ply",
        "wrist_3_link": ASSETS_UR5E_DIR / "3dgs" / "ur5e" / "wrist_3_link.ply",
        "robotiq_base": ASSETS_UR5E_DIR / "3dgs" / "robotiq" / "robotiq_base.ply",
        "left_driver": ASSETS_UR5E_DIR / "3dgs" / "robotiq" / "left_driver.ply",
        "left_coupler": ASSETS_UR5E_DIR / "3dgs" / "robotiq" / "left_coupler.ply",
        "left_spring_link": ASSETS_UR5E_DIR / "3dgs" / "robotiq" / "left_spring_link.ply",
        "left_follower": ASSETS_UR5E_DIR / "3dgs" / "robotiq" / "left_follower.ply",
        "right_driver": ASSETS_UR5E_DIR / "3dgs" / "robotiq" / "right_driver.ply",
        "right_coupler": ASSETS_UR5E_DIR / "3dgs" / "robotiq" / "right_coupler.ply",
        "right_spring_link": ASSETS_UR5E_DIR / "3dgs" / "robotiq" / "right_spring_link.ply",
        "right_follower": ASSETS_UR5E_DIR / "3dgs" / "robotiq" / "right_follower.ply",
    }
    
    BACKGROUND_PLY = ASSETS_UR5E_DIR / "3dgs" / "background.ply"

    num_dof_arm = 6
    ee_site_name = "gripper"
    gripper_actuator_name = "fingers_actuator"

    def __init__(self, mx_model: mtx.SceneModel):
        super().__init__(mx_model)
        
        # IK Setup (copied from previous impl)
        self.chain = ik.IkChain(
            self.mx_model,
            start_link="shoulder_link",
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
