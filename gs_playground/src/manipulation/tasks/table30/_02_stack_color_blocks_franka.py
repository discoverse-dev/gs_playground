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
from gs_playground.src.env.motrix_env.render_env import RenderEnvState

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
    robot_name: str = "franka_robotiq"
    model_file: str = str((ROOT_PATH / "models" / "robots" / 
                           "manipulation" / "franka_emika_panda_robotiq" / 
                           "xmls" / "table30_02_stack_color_blocks.xml").as_posix())

    # control
    action_mode: str = "joint"  # "joint" or "eef"

    # rendering
    img_width: int = 320
    img_height: int = 240
 
    # observation / prompt
    instruction: str = "Stack the yellow block on top of the orange block."

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


        self.top_idx = np.zeros((self.num_envs,), dtype=np.int32)
        self.base_idx = np.zeros((self.num_envs,), dtype=np.int32)
        self.grasp_latched = np.zeros((self.num_envs,), dtype=bool)
        self.success_latched = np.zeros((self.num_envs,), dtype=bool)

    # ---- Task hooks ----
    def task_gaussians(self) -> Dict[str, str]:
        return TASK_GAUSSIANS

    def _randomize(self, data: SceneData, done_mask: np.ndarray, phase: str = "reset"):
        """
        Example randomization: jitter cube xy positions for the envs being reset.

        Args:
            data: SceneData view for the subset of envs being reset (len == sum(done_mask)).
            done_mask: boolean mask over all envs (not used directly here).
            phase: "reset" or "auto_reset" for potential differentiated logic.
        """
        if data.shape[0] == 0:
            return

        num_cubes = len(self.cube_bodies)
        min_xy_dist = 0.05

        # Get current poses for the subset: (B_subset, num_cubes, 7)
        cube_pose = np.stack(
            [np.asarray(b.get_pose(data), dtype=np.float32) for b in self.cube_bodies],
            axis=1,
        )

        base_xy = cube_pose[..., :2]
        new_pose = cube_pose.copy()

        remaining = np.ones((data.shape[0],), dtype=bool)
        eye_mask = np.eye(num_cubes, dtype=np.float32) * 1e6  # ignore self-distance

        # Try a few times to satisfy min distance; fallback to base if not achieved
        for _ in range(10):
            if not remaining.any():
                break
            # Sample jitters only for remaining envs
            jitter = self._rng.uniform(-0.08, 0.08, size=(remaining.sum(), num_cubes, 2)).astype(np.float32)
            candidate_xy = base_xy[remaining] + jitter

            diff = candidate_xy[:, :, None, :] - candidate_xy[:, None, :, :]
            dist = np.linalg.norm(diff, axis=-1) + eye_mask[None]
            ok = dist.min(axis=(1, 2)) >= min_xy_dist

            if ok.any():
                # Assign successful candidates back into new_pose
                rem_indices = np.where(remaining)[0]
                new_pose_view = new_pose[remaining]
                new_pose_view[ok, :, :2] = candidate_xy[ok]
                new_pose[remaining] = new_pose_view
                # Update remaining mask
                remaining[rem_indices[ok]] = False

        # Write back poses using set_dof_pos (include floating base)
        for env_idx in range(data.shape[0]):
            for cube_idx, body in enumerate(self.cube_bodies):
                body.set_dof_pos(
                    data[env_idx],
                    new_pose[env_idx, cube_idx],
                    include_floatingbase=True,
                )

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
    def _compute_reward(self, state: RenderEnvState) -> np.ndarray:
        data: SceneData = state.data
        info: Dict[str, np.ndarray] = state.info
        cube_pose = np.stack(
            [np.asarray(b.get_pose(data), dtype=np.float32) for b in self.cube_bodies],
            axis=1,
        )
        
        ee_pos = self.robot.get_ee_pose(data)[:, :3]

        idx = np.arange(self.num_envs)
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

        # Write metrics into info in-place
        info["is_success"] = self.success_latched.copy()
        info["is_grasped"] = self.grasp_latched.copy()
        info["reach_dist"] = dist_ee_obj
        info["stack_xy"] = dist_xy
        info["dz"] = dz

        return reward.astype(np.float32)

    def _check_success(self, state: RenderEnvState) -> np.ndarray:
        data: SceneData = state.data
        B = self.num_envs
        cube_pose = np.stack(
            [np.asarray(b.get_pose(data), dtype=np.float32) for b in self.cube_bodies],
            axis=1,
        )
        name_to_idx = {name: i for i, name in enumerate(self._cfg.cube_names)}
        yellow_i = name_to_idx["cube_yellow"]
        orange_i = name_to_idx["cube_orange"]

        yellow = cube_pose[:, yellow_i, :3]
        orange = cube_pose[:, orange_i, :3]

        dist_xy = np.linalg.norm(yellow[:, :2] - orange[:, :2], axis=1)
        dz = np.abs(yellow[:, 2] - (orange[:, 2] + 0.05))

        success = (dist_xy < 0.01) & (dz < 0.01)
        # latch
        self.success_latched = self.success_latched | success
        return self.success_latched.copy()

    def update_state(self, state: RenderEnvState, obs_required: bool = True) -> RenderEnvState:
        state = super().update_state(state, obs_required=obs_required)

        # Fail-fast check: if any cube leaves workspace bounds, terminate env
        data: SceneData = state.data
        cube_pose = np.stack(
            [np.asarray(b.get_pose(data), dtype=np.float32) for b in self.cube_bodies],
            axis=1,
        )
        xy = cube_pose[..., :2]  # (B, num_cubes, 2)
        x_ok = (xy[..., 0] >= -0.75) & (xy[..., 0] <= -0.45)
        y_ok = (xy[..., 1] >= -0.20) & (xy[..., 1] <= 0.20)
        in_bounds = x_ok & y_ok
        out_of_bounds = ~np.all(in_bounds, axis=1)

        if np.any(out_of_bounds):
            terminated = state.terminated.copy()
            terminated[out_of_bounds] = True
            state.terminated = terminated
            state.info["out_of_bounds"] = out_of_bounds

        return state