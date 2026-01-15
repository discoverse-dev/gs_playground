from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import motrixsim as mtx
import numpy as np
from motrixsim import SceneData

from gs_playground import ROOT_PATH
from gs_playground.src.env.registry import envcfg, env
from gs_playground.src.manipulation.tasks.task_env import TaskEnvCfg, TaskEnv

ASSETS_TASK_DIR = ROOT_PATH / "models" / "tasks" / "table30" / "_02_stack_color_blocks" / "3dgs"
TASK_GAUSSIANS = {
    "cube_blue"   : ASSETS_TASK_DIR /  "cube_blue.ply",
    "cube_yellow" : ASSETS_TASK_DIR / "cube_yellow.ply",
    "cube_orange" : ASSETS_TASK_DIR / "cube_orange.ply",
}

@envcfg("table30/stack_color_blocks")
@dataclass
class StackColorBlocksEnvCfg(TaskEnvCfg):
    # model / sim
    robot_name: str = "ur5e_robotiq"
    model_file: str = str((ROOT_PATH / "models" / "robots" / 
                           "manipulation" / "universal_robots_ur5e_robotiq" / 
                           "xmls" / "table30_02_stack_color_blocks.xml").as_posix())

    # control
    action_mode: str = "joint"  # "joint" or "eef"

    # rendering
    img_width: int = 320
    img_height: int = 240
 
    # observation / prompt
    instruction: str = "Stack the blue cube on top of the yellow cube."

    # task entities
    cube_names: Tuple[str, str, str] = ("cube_blue", "cube_yellow", "cube_orange")
    
    # task params
    success_dist_xy: float = 0.05
    success_delta_z_min: float = 0.02
    success_delta_z_max: float = 0.10
    gripper_close_thresh: float = 0.2
    grasp_dist_thresh: float = 0.03


@env("table30/stack_color_blocks", "np")
class StackColorBlocksEnv(TaskEnv):
    """
    Task: Stack color blocks.
    Robot: UR5e + Robotiq 2F-85.
    Backend: MotrixSim (np).
    """

    def __init__(self, cfg: StackColorBlocksEnvCfg, num_envs: int = 32):
        super().__init__(cfg, num_envs=num_envs)

        self.cube_bodies = [self.model.get_body(self.model.get_body_index(n)) for n in cfg.cube_names]

        B = self._num_envs
        self.top_idx = np.zeros((B,), dtype=np.int32)
        self.base_idx = np.zeros((B,), dtype=np.int32)
        self.grasp_latched = np.zeros((B,), dtype=bool)
        self.success_latched = np.zeros((B,), dtype=bool)

    # ---- Task hooks ----
    def task_gaussians(self) -> Dict[str, str]:
        return TASK_GAUSSIANS

    def _reset_task_state(self, done: np.ndarray):
        """Reset internal task state variables for done environments."""
        rng = np.random.default_rng()
        
        n_done = np.sum(done)
        if n_done > 0:
            perms = np.argsort(rng.random((n_done, len(self._cfg.cube_names))), axis=1)
            
            self.top_idx[done] = perms[:, 0]
            self.base_idx[done] = perms[:, 1]

            self.grasp_latched[done] = False
            self.success_latched[done] = False

    # ---- helpers ----
    def _compute_reward(self, data: SceneData) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        B = self._num_envs
        cube_pose = np.stack(
            [np.asarray(b.get_pose(data), dtype=np.float32) for b in self.cube_bodies],
            axis=1,
        )
        
        ee_pos = self.robot.get_ee_pose(data)[:, :3]

        idx = np.arange(B)
        top_pos = cube_pose[idx, self.top_idx, :3]
        base_pos = cube_pose[idx, self.base_idx, :3]

        dist_ee_obj = np.linalg.norm(ee_pos - top_pos, axis=1)
        dist_xy = np.linalg.norm(top_pos[:, :2] - base_pos[:, :2], axis=1)
        dz = top_pos[:, 2] - base_pos[:, 2]

        grip_cmd = np.asarray(data.actuator_ctrls)[:, self.robot.gripper_act_id]
        grip_closed = grip_cmd > float(self._cfg.gripper_close_thresh)

        is_grasp = (dist_ee_obj < float(self._cfg.grasp_dist_thresh)) & grip_closed
        self.grasp_latched = self.grasp_latched | is_grasp

        is_stack_pos = (dist_xy < float(self._cfg.success_dist_xy)) & (
            (dz > float(self._cfg.success_delta_z_min)) & (dz < float(self._cfg.success_delta_z_max))
        )
        is_success = is_stack_pos & (~grip_closed)
        self.success_latched = self.success_latched | is_success

        reach_r = -dist_ee_obj
        stack_r = -dist_xy
        grasp_r = 0.5 * self.grasp_latched.astype(np.float32)
        success_r = 5.0 * self.success_latched.astype(np.float32)

        reward = reach_r + stack_r + grasp_r + success_r
        return reward.astype(np.float32), {
            "is_success": self.success_latched.copy(),
            "is_grasped": self.grasp_latched.copy(),
            "reach_dist": dist_ee_obj,
            "stack_xy": dist_xy,
            "dz": dz,
        }