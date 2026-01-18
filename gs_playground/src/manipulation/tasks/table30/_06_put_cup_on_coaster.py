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
_ASSETS_FRANKA_DIR = ROOT_PATH.parent / "models" / "robots" / "manipulation" / "franka_robotiq"
_ASSETS_TASK_DIR = ROOT_PATH.parent / "models" / "tasks" / "table30" / "_06_put_cup_on_coaster"

TASK_GAUSSIANS = {
    "cup": _ASSETS_TASK_DIR / "3dgs" / "cup.ply",
    "coaster": _ASSETS_TASK_DIR / "3dgs" / "coaster.ply",
}


@envcfg("table30/cup_on_coaster_franka")
@dataclass
class CupOnCoasterEnvCfg(TaskEnvCfg):
    # model / sim
    robot_name: str = "franka_robotiq"
    model_file: str = str((_ASSETS_FRANKA_DIR / "xmls" / "cup_on_coaster.xml").as_posix())

    # control
    # 推荐使用 "eef" (绝对位姿控制) 配合 Runner 的 IK 求解器
    action_mode: str = "eef" 

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
    
    # Sensors (Important for this task)
    touch_name_cup: str = "cup_touch"
    touch_name_coaster: str = "coaster_touch"
    touch_thresh: float = 1e-3

    # Randomization range
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

        # State trackers
        self.grasp_latched = np.zeros((self.num_envs,), dtype=bool)
        self.success_latched = np.zeros((self.num_envs,), dtype=bool)

    # ---- Task hooks ----
    def task_gaussians(self) -> Dict[str, str]:
        return TASK_GAUSSIANS

    def _randomize(self, data: SceneData, done_mask: np.ndarray, phase: str = "reset"):
        """
        Randomize Cup and Coaster positions on the XY plane.
        """
        if data.shape[0] == 0:
            return

        # Bodies to randomize
        bodies = [self.cup_body, self.coaster_body]
        min_dist = 0.10 # Minimum distance between cup and coaster to avoid overlap

        # Get current poses (B_subset, 2, 7)
        # 0: cup, 1: coaster
        current_poses = np.stack(
            [np.asarray(b.get_pose(data), dtype=np.float32) for b in bodies],
            axis=1
        )
        base_xy = current_poses[..., :2].copy() # Store original positions
        
        # We want to randomize around the original position (or a generic center)
        # Assuming original position is the center of workspace
        
        new_xy = np.zeros_like(base_xy)
        remaining = np.ones((data.shape[0],), dtype=bool)
        
        # Simple Rejection Sampling for collision free placement
        for _ in range(10):
            if not remaining.any():
                break
            
            n_rem = remaining.sum()
            jitter = self._rng.uniform(
                -self._cfg.rand_xy_range, 
                self._cfg.rand_xy_range, 
                size=(n_rem, 2, 2)
            ).astype(np.float32)
            
            # Candidate positions: Base + Jitter
            candidates = base_xy[remaining] + jitter
            
            # Check distance between Cup (0) and Coaster (1)
            diff = candidates[:, 0, :] - candidates[:, 1, :]
            dist = np.linalg.norm(diff, axis=-1)
            
            ok = dist > min_dist
            
            if ok.any():
                # Update valid positions
                rem_indices = np.where(remaining)[0]
                valid_indices = rem_indices[ok]
                
                # Assign only the valid ones
                # We need to map back to the subset index logic
                # Since 'candidates' corresponds to 'remaining', and 'ok' corresponds to 'candidates'
                
                # Specifically update the 'new_xy' buffer at the correct indices
                # Note: This logic is slightly complex due to masking. 
                # Simpler approach: update new_xy for the OK ones, update mask.
                
                # Fill the buffer rows corresponding to valid_indices
                # Since 'remaining' tracks indices in 'data', we iterate carefully or use boolean indexing
                
                # Let's use boolean indexing on the full subset array 'new_xy'
                # Create a mask for the FULL subset based on 'remaining' AND 'ok'
                full_ok_mask = np.zeros_like(remaining)
                full_ok_mask[remaining] = ok
                
                new_xy[full_ok_mask] = candidates[ok]
                remaining[full_ok_mask] = False

        # Apply positions (fallback to original base_xy if sampling failed)
        # If remaining is True, it means we didn't find a valid spot, use base_xy + small noise or just base
        if remaining.any():
             new_xy[remaining] = base_xy[remaining]

        # Set DoF Pos
        for env_i in range(data.shape[0]):
            # Set Cup
            self.cup_body.set_translation(data[env_i], np.append(new_xy[env_i, 0], current_poses[env_i, 0, 2]), include_floatingbase=True)
            # Set Coaster
            self.coaster_body.set_translation(data[env_i], np.append(new_xy[env_i, 1], current_poses[env_i, 1, 2]), include_floatingbase=True)

    def _reset_task_state(self, done: np.ndarray):
        """Reset latches."""
        self.grasp_latched[done] = False
        self.success_latched[done] = False

    def _read_sensors(self, data: SceneData) -> Tuple[np.ndarray, np.ndarray]:
        """Read cup and coaster touch sensors safely."""
        try:
            cup_touch = np.asarray(self.model.get_sensor_value(self._cfg.touch_name_cup, data), dtype=np.float32)
            coaster_touch = np.asarray(self.model.get_sensor_value(self._cfg.touch_name_coaster, data), dtype=np.float32)
            
            # Handle shape (B, 1) or (B,)
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
        grip_closed = grip_cmd > float(self._cfg.gripper_close_thresh) # 0.82 is close usually

        # 2. Object States
        cup_pos = np.asarray(self.cup_body.get_pose(data), dtype=np.float32)[:, :3]
        coaster_pos = np.asarray(self.coaster_body.get_pose(data), dtype=np.float32)[:, :3]
        
        # 3. Sensors
        cup_touch_val, coaster_touch_val = self._read_sensors(data)
        is_touching_cup = cup_touch_val > self._cfg.touch_thresh
        is_touching_coaster = coaster_touch_val > self._cfg.touch_thresh # Coaster touching implies cup is on it (if physics is stable)

        # 4. Distances
        dist_ee_cup = np.linalg.norm(ee_pos - cup_pos, axis=1)
        dist_cup_coaster = np.linalg.norm(cup_pos - coaster_pos, axis=1) # 3D distance
        dist_xy_cup_coaster = np.linalg.norm(cup_pos[:, :2] - coaster_pos[:, :2], axis=1)

        # 5. Logic
        # Grasp: EE close to Cup AND Gripper Closed AND Sensor Active
        is_grasp_dist = dist_ee_cup < self._cfg.grasp_dist_thresh
        is_grasped = is_grasp_dist & grip_closed & is_touching_cup
        self.grasp_latched = self.grasp_latched | is_grasped

        # Success: Cup on Coaster (XY align) AND Cup released AND Coaster sensor active
        is_aligned = dist_xy_cup_coaster < self._cfg.success_dist_xy
        is_released = ~grip_closed
        # We check coaster sensor to confirm physical contact between cup and coaster
        is_success = is_aligned & is_released & is_touching_coaster
        self.success_latched = self.success_latched | is_success

        # 6. Rewards
        reach_r = -dist_ee_cup
        place_r = -dist_cup_coaster # Guide cup to coaster
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