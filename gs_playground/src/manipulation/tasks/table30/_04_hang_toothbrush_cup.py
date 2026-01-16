from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from motrixsim import SceneData
from scipy.spatial.transform import Rotation

from gs_playground import ROOT_PATH
from gs_playground.src.env.registry import envcfg, env
from gs_playground.src.manipulation.tasks.task_env import TaskEnvCfg, TaskEnv
from gs_playground.src.env.motrix_env.render_env import RenderEnvState

ASSETS_TASK_DIR = ROOT_PATH / "models" / "tasks" / "table30" / "04_hang_toothbrush_cup" / "3dgs"
TASK_GAUSSIANS = {
    "toothbrush_cup": ASSETS_TASK_DIR / "toothbrush_cup.ply",
    "rack": ASSETS_TASK_DIR / "rack.ply",
}


@envcfg("table30/hang_toothbrush_cup")
@dataclass
class HangToothbrushCupEnvCfg(TaskEnvCfg):
    # model / sim
    robot_name: str = "franka_robotiq"
    model_file: str = str((ROOT_PATH / "models" / "robots" / 
                           "manipulation" / "franka_emika_panda_robotiq" / 
                           "xmls" / "table30_04_hang_toothbrush_cup.xml").as_posix())

    # control
    action_mode: str = "eef_relative"  # "joint" or "eef"

    # rendering
    img_width: int = 320
    img_height: int = 240
 
    # instruction
    instruction: str = "Hang the orange toothbrush cup on the cup holder"

    # entities (XML names)
    cup_name: str = "toothbrush_cup"
    rack_name: str = "rack"

    # sites (XML names)
    grasp_site_name: str = "bottle_grasp_site"
    hook_site_name: str = "rack_hook_site"

    # sensors (XML names)
    sensor_grasp: str = "bottle_grasp_touch"
    sensor_hook: str = "rack_hook_touch"

    # reward params
    touch_threshold: float = 1e-3
    grasp_dist_threshold: float = 0.05

    reach_scale: float = 5.0
    move_scale: float = 2.0

    grasp_reward_bonus: float = 2.0          # latch 后固化
    hang_reward_bonus: float = 5.0           # 稀疏奖励（仅首次成功给）

    hang_height_margin: float = 0.05         # cup_z > hook_z - margin

    pre_hang_offset: Tuple[float, float, float] = (-0.04, -0.07, 0.02)
    hang_offset: Tuple[float, float, float] = (0, 0, -0.035)
    
    pre_hang_dist_threshold: float = 0.03
    hang_dist_threshold: float = 0.02

    # randomization
    xy_jitter: float = 0.03                # uniform[-xy_jitter, xy_jitter] (meters)

    reset_enabled: bool = True
    reset_keyframe: int | str = "home"  


@env("table30/hang_toothbrush_cup", "np")
class HangToothbrushCupEnv(TaskEnv):
    """
    Task: Hang the toothbrush cup on the rack.
    Robot: Franka + Robotiq.
    Backend: MotrixSim (np).
    """

    def __init__(self, cfg: HangToothbrushCupEnvCfg, num_envs: int = 32):
        super().__init__(cfg, num_envs=num_envs)
        self._cfg: HangToothbrushCupEnvCfg = cfg

        # bodies
        self.cup_body = self.model.get_body(self.model.get_body_index(cfg.cup_name))
        self.rack_body = self.model.get_body(self.model.get_body_index(cfg.rack_name))

        # sites
        self.grasp_site = self.model.get_site(cfg.grasp_site_name)
        self.hook_site = self.model.get_site(cfg.hook_site_name)

        # task latch state
        self.is_grasped = np.zeros((self.num_envs,), dtype=bool)
        self.is_pre_hang = np.zeros((self.num_envs,), dtype=bool)
        self.is_hung = np.zeros((self.num_envs,), dtype=bool)
        self.success_latched = np.zeros((self.num_envs,), dtype=bool)



    # ---- Task hooks ----
    def task_gaussians(self) -> Dict[str, str]:
        return TASK_GAUSSIANS

    def _randomize(self, data: SceneData, done_mask: np.ndarray, phase: str = "reset"):
        """
        Randomization: jitter rack + cup XY positions for the envs being reset.

        Args:
            data: SceneData view for the subset of envs being reset (len == sum(done_mask)).
            done_mask: boolean mask over all envs (not used directly here).
            phase: "reset" or "auto_reset" for potential differentiated logic.
        """
        if data.shape[0] == 0:
            return

        # Get current poses for subset: (B_subset, 2, 7)
        poses = np.stack(
            [
                np.asarray(self.rack_body.get_pose(data), dtype=np.float32),
                np.asarray(self.cup_body.get_pose(data), dtype=np.float32),
            ],
            axis=1,
        )

        # Small XY jitters (keep z/orientation unchanged)
        xy_jitter = self._rng.uniform(-0.03, 0.03, size=poses[..., :2].shape).astype(np.float32)
        new_pose = poses.copy()
        new_pose[..., :2] = poses[..., :2] + xy_jitter

        # Write back poses using set_dof_pos (include floating base)
        for env_idx in range(data.shape[0]):
            self.rack_body.set_dof_pos(
                data[env_idx],
                new_pose[env_idx, 0],
                include_floatingbase=True,
            )
            self.cup_body.set_dof_pos(
                data[env_idx],
                new_pose[env_idx, 1],
                include_floatingbase=True,
            )

    def _reset_task_state(self, done: np.ndarray):
        done = np.asarray(done, dtype=bool)
        if done.size == 0 or not np.any(done):
            return
        self.is_grasped[done] = False
        self.is_hung[done] = False
        self.is_pre_hang[done] = False
        self.success_latched[done] = False

    def _compute_reward(self, state: RenderEnvState) -> np.ndarray:
        data: SceneData = state.data
        info = state.info
        cfg = self._cfg
        B = self.num_envs

        # poses
        ee_pos = self.robot.get_ee_pose(data)[:, :3]
        cup_grasp_pos = np.asarray(self.grasp_site.get_pose(data), dtype=np.float32)[:, :3]
        hook_pos = np.asarray(self.hook_site.get_pose(data), dtype=np.float32)[:, :3]

        # sensors
        grasp_touch = np.asarray(self.model.get_sensor_value(cfg.sensor_grasp, data), dtype=np.float32).reshape(B, -1)[:, 0]
        hook_touch = np.asarray(self.model.get_sensor_value(cfg.sensor_hook, data), dtype=np.float32).reshape(B, -1)[:, 0]

        touching_cup = grasp_touch > cfg.touch_threshold
        touching_hook = hook_touch > cfg.touch_threshold

        # distances
        d_ee_cup = np.linalg.norm(ee_pos - cup_grasp_pos, axis=1)

        # --- stage targets: pre_hang vs hang ---
        pre_hang_tgt = hook_pos + np.asarray(cfg.pre_hang_offset, dtype=np.float32).reshape(1, 3)
        hang_tgt = hook_pos + np.asarray(cfg.hang_offset, dtype=np.float32).reshape(1, 3)

        d_cup_pre_hang = np.linalg.norm(cup_grasp_pos - pre_hang_tgt, axis=1)
        d_cup_hang = np.linalg.norm(cup_grasp_pos - hang_tgt, axis=1)
        # print("d_cup_pre_hang",d_cup_pre_hang)
        # print("d_cup_hang",d_cup_hang)
        # 1) Dense reach-to-cup
        r_reach = 1.0 - np.tanh(cfg.reach_scale * d_ee_cup)

        # 2) Grasp latch + fixed grasp reward
        grasp_now = touching_cup & (d_ee_cup < cfg.grasp_dist_threshold)
        self.is_grasped = self.is_grasped | grasp_now
        r_grasp_fixed = self.is_grasped.astype(np.float32) * cfg.grasp_reward_bonus

        # 3) Two-stage move shaping:
        #    stage A: move to pre_hang until reached
        #    stage B: then move to hang target
        pre_hang_reached_now = self.is_grasped & (~self.is_pre_hang) & (d_cup_pre_hang < cfg.pre_hang_dist_threshold)
        self.is_pre_hang = self.is_pre_hang | pre_hang_reached_now
        if self.is_pre_hang.any() :
            print("d_cup_pre_hang",d_cup_pre_hang)
            print("d_cup_hang",d_cup_hang)
            print("hang_tgt",hang_tgt)
            print("cup_grasp_pos",cup_grasp_pos)

        r_move_pre = (1.0 - np.tanh(cfg.move_scale * d_cup_pre_hang)) * (self.is_grasped & (~self.is_pre_hang)).astype(np.float32)
        r_move_hang = (1.0 - np.tanh(cfg.move_scale * d_cup_hang)) * (self.is_grasped & self.is_pre_hang).astype(np.float32)
        r_move = r_move_pre + r_move_hang

        # 4) Sparse success: must correspond to FINAL hang_offset
        high_enough = cup_grasp_pos[:, 2] > (hang_tgt[:, 2] - cfg.hang_height_margin)

        success_now = (
            touching_hook
            & high_enough
            & self.is_grasped
            & self.is_pre_hang              # 先对准到 pre_hang
            & (d_cup_hang < cfg.hang_dist_threshold)  # 再到 hang 点
            & (~self.is_hung)
        )

        if success_now.any() :
            print("d_cup_pre_hang",d_cup_pre_hang)
            print("d_cup_hang",d_cup_hang)

        self.is_hung = self.is_hung | success_now
        self.success_latched = self.success_latched | self.is_hung

        r_success_sparse = success_now.astype(np.float32) * cfg.hang_reward_bonus

        total_reward = r_reach + r_grasp_fixed + r_move + r_success_sparse
        print(self.success_latched)
        # stash info
        info["d_ee_cup"] = d_ee_cup
        info["d_cup_pre_hang"] = d_cup_pre_hang
        info["d_cup_hang"] = d_cup_hang
        info["grasp_touch"] = grasp_touch
        info["hook_touch"] = hook_touch
        info["is_grasped"] = self.is_grasped.copy()
        info["is_pre_hang"] = self.is_pre_hang.copy()
        info["is_hung"] = self.is_hung.copy()
        info["is_success"] = self.success_latched.copy()
        info["success_now"] = success_now

        return total_reward.astype(np.float32)
