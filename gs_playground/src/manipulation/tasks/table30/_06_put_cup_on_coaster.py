from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import gymnasium as gym
from motrixsim import SceneData

from gs_playground import ROOT_PATH
from gs_playground.src.env.registry import envcfg, env
from gs_playground.src.manipulation.tasks.task_env import TaskEnvCfg, TaskEnv
from gs_playground.src.env.motrix_env.render_env import RenderEnvState

# -----------------------------------------------------------------------------
# Asset Paths
# -----------------------------------------------------------------------------
_ASSETS_TASK_DIR = ROOT_PATH / "models" / "tasks" / "table30" / "_06_put_cup_on_coaster"

TASK_GAUSSIANS = {
    "cup": _ASSETS_TASK_DIR / "3dgs" / "cup.ply",
    "coaster": _ASSETS_TASK_DIR / "3dgs" / "coaster.ply",
}


@envcfg("table30/cup_on_coaster_franka")
@dataclass
class CupOnCoasterEnvCfg(TaskEnvCfg):
    # model / sim
    robot_name: str = "franka_robotiq"
    model_file: str = str((ROOT_PATH / "models" / "robots" / 
                           "manipulation" / "franka_emika_panda_robotiq" / 
                           "xmls" / "table30_06_put_cup_on_coaster.xml").as_posix())

    # control
    action_mode: str = "eef_relative" 

    # rendering
    img_width: int = 320
    img_height: int = 240
    cam_id: int = 0

    # observation / prompt
    instruction: str = "Place the cup onto the coaster."

    # task entities
    cup_name: str = "cup"
    coaster_name: str = "coaster"
    
    # task params
    success_dist_xy: float = 0.05
    gripper_close_thresh: float = 0.2
    grasp_dist_thresh: float = 0.05
    
    # Sensors
    touch_name_cup: str = "cup_touch"
    touch_name_coaster: str = "coaster_touch"
    touch_thresh: float = 1e-3

    # Randomization
    rand_xy_range: float = 0.15


@env("table30/cup_on_coaster_franka", "np")
class CupOnCoasterEnv(TaskEnv):
    """
    Task: Place the Cup on the Coaster.
    Robot: Franka Emika Panda + Robotiq 2F-85.
    Backend: MotrixSim (np).
    """

    def __init__(self, cfg: CupOnCoasterEnvCfg, num_envs: int = 32):
        super().__init__(cfg, num_envs=num_envs)

        self.cup_body = self.model.get_body(self.model.get_body_index(cfg.cup_name))
        self.coaster_body = self.model.get_body(self.model.get_body_index(cfg.coaster_name))
        
        # Helper list for iteration
        self.task_bodies = [self.cup_body, self.coaster_body]

        # State trackers
        self.grasp_latched = np.zeros((self.num_envs,), dtype=bool)
        self.success_latched = np.zeros((self.num_envs,), dtype=bool)

    # ---- Task hooks ----
    def task_gaussians(self) -> Dict[str, str]:
        return TASK_GAUSSIANS

    def _randomize(self, data: SceneData, done_mask: np.ndarray, phase: str = "reset"):
        """
        Randomize Cup and Coaster positions ensuring no overlap.
        Uses rejection sampling logic similar to StackColorBlocksEnv.
        """
        if data.shape[0] == 0:
            return

        num_objs = len(self.task_bodies)
        # Coaster radius ~6cm, Cup radius ~3cm, margin ~2cm -> min dist ~0.11m
        min_xy_dist = 0.12 

        # 1. Get current poses for the subset: (B_subset, num_objs, 7)
        obj_pose = np.stack(
            [np.asarray(b.get_pose(data), dtype=np.float32) for b in self.task_bodies],
            axis=1,
        )

        base_xy = obj_pose[..., :2]
        new_pose = obj_pose.copy()

        remaining = np.ones((data.shape[0],), dtype=bool)
        eye_mask = np.eye(num_objs, dtype=np.float32) * 1e6  # ignore self-distance

        # 2. Try a few times to satisfy min distance
        for _ in range(10):
            if not remaining.any():
                break
                
            n_rem = remaining.sum()
            # Sample jitters only for remaining envs
            jitter = self._rng.uniform(
                -self._cfg.rand_xy_range, 
                self._cfg.rand_xy_range, 
                size=(n_rem, num_objs, 2)
            ).astype(np.float32)
            
            candidate_xy = base_xy[remaining] + jitter

            # Calculate pairwise distances: (N, n_obj, n_obj)
            diff = candidate_xy[:, :, None, :] - candidate_xy[:, None, :, :]
            dist = np.linalg.norm(diff, axis=-1) + eye_mask[None]
            
            # Check if all mutual distances are valid
            ok = dist.min(axis=(1, 2)) >= min_xy_dist

            if ok.any():
                rem_indices = np.where(remaining)[0]
                
                # Update pose buffer for successful samples
                new_pose_view = new_pose[remaining]
                new_pose_view[ok, :, :2] = candidate_xy[ok]
                new_pose[remaining] = new_pose_view
                
                # Update remaining mask
                remaining[rem_indices[ok]] = False

        # 3. Write back poses using set_dof_pos
        for env_idx in range(data.shape[0]):
            for i, body in enumerate(self.task_bodies):
                body.set_dof_pos(
                    data[env_idx],
                    new_pose[env_idx, i],
                    include_floatingbase=True,
                )

    def _reset_task_state(self, done: np.ndarray):
        """Reset latches."""
        self.grasp_latched[done] = False
        self.success_latched[done] = False

    def _read_sensors(self, data: SceneData) -> Tuple[np.ndarray, np.ndarray]:
        """Read cup and coaster touch sensors safely."""
        try:
            cup_touch = np.asarray(self.model.get_sensor_value(self._cfg.touch_name_cup, data), dtype=np.float32)
            coaster_touch = np.asarray(self.model.get_sensor_value(self._cfg.touch_name_coaster, data), dtype=np.float32)
            
            if cup_touch.ndim > 1: cup_touch = cup_touch[..., 0]
            if coaster_touch.ndim > 1: coaster_touch = coaster_touch[..., 0]
            
            return cup_touch, coaster_touch
        except Exception:
            return np.zeros(self.num_envs), np.zeros(self.num_envs)

    def _compute_reward(self, state: RenderEnvState) -> np.ndarray:
        data: SceneData = state.data
        info: Dict[str, np.ndarray] = state.info
        
        # 1. Robot State
        ee_pos = self.robot.get_ee_pose(data)[:, :3]
        grip_cmd = np.asarray(data.actuator_ctrls)[:, self.robot.gripper_act_id]
        grip_closed = grip_cmd > float(self._cfg.gripper_close_thresh)

        # 2. Object States
        cup_pos = np.asarray(self.cup_body.get_pose(data), dtype=np.float32)[:, :3]
        coaster_pos = np.asarray(self.coaster_body.get_pose(data), dtype=np.float32)[:, :3]
        
        # 3. Sensors
        cup_touch_val, coaster_touch_val = self._read_sensors(data)
        is_touching_cup = cup_touch_val > self._cfg.touch_thresh
        is_touching_coaster = coaster_touch_val > self._cfg.touch_thresh 

        # 4. Distances
        dist_ee_cup = np.linalg.norm(ee_pos - cup_pos, axis=1)
        dist_cup_coaster = np.linalg.norm(cup_pos - coaster_pos, axis=1)
        dist_xy_cup_coaster = np.linalg.norm(cup_pos[:, :2] - coaster_pos[:, :2], axis=1)

        # 5. Logic
        is_grasp_dist = dist_ee_cup < self._cfg.grasp_dist_thresh
        is_grasped = is_grasp_dist & grip_closed & is_touching_cup
        self.grasp_latched = self.grasp_latched | is_grasped

        is_aligned = dist_xy_cup_coaster < self._cfg.success_dist_xy
        is_released = ~grip_closed
        is_success = is_aligned & is_released & is_touching_coaster
        self.success_latched = self.success_latched | is_success

        # 6. Rewards
        reach_r = -dist_ee_cup
        place_r = -dist_cup_coaster 
        grasp_r = 2.0 * self.grasp_latched.astype(np.float32)
        success_r = 5.0 * self.success_latched.astype(np.float32)

        reward = reach_r + place_r + grasp_r + success_r

        # 7. Update Info
        info["is_success"] = self.success_latched.copy()
        info["is_grasped"] = self.grasp_latched.copy()
        info["cup_touch"] = cup_touch_val
        info["coaster_touch"] = coaster_touch_val
        info["dist_ee_cup"] = dist_ee_cup
        
        return reward.astype(np.float32)