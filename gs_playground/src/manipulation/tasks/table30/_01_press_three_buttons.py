from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
from motrixsim import SceneData

from gs_playground import ROOT_PATH
from gs_playground.src.env.registry import envcfg, env
from gs_playground.src.manipulation.tasks.task_env import TaskEnvCfg, TaskEnv
from gs_playground.src.env.motrix_env.render_env import RenderEnvState

ASSETS_TASK_DIR = ROOT_PATH / "models" / "tasks" / "table30" / "_01_press_three_buttons" / "3dgs"
TASK_GAUSSIANS = {
    "button_blue":  ASSETS_TASK_DIR / "button_blue.ply",
    "button_green": ASSETS_TASK_DIR / "button_green.ply",
    "button_pink":  ASSETS_TASK_DIR / "button_pink.ply",
}


@envcfg("table30/press_three_buttons")
@dataclass
class PressThreeButtonsEnvCfg(TaskEnvCfg):
    # model / sim
    robot_name: str = "franka_robotiq"
    model_file: str = str((ROOT_PATH / "models" / "robots" / 
                           "manipulation" / "franka_emika_panda_robotiq" / 
                           "xmls" / "table30_01_press_three_buttons.xml").as_posix())

    # control
    action_mode: str = "eef_relative"  # "joint" or "eef"

    # rendering
    img_width: int = 320
    img_height: int = 240
 
    # observation / prompt
    instruction: str = "Press the pink, blue and green buttons in sequence."

    # task entities
    button_names: Tuple[str, str, str] = ("button_blue", "button_green", "button_pink")
    button_touch_names: Tuple[str, str, str] = ("button_blue_touch", "button_green_touch", "button_pink_touch")

    # reward params
    touch_threshold: float = 1e-3
    dist_reward_scale: float = 1.0
    touch_reward_bonus: float = 0.5
    stage_complete_reward: float = 5.0


@env("table30/press_three_buttons", "np")
class PressThreeButtonsEnv(TaskEnv):
    """
    Task: Press three buttons (fixed order: blue -> green -> pink).
    Robot: Franka + Robotiq.
    Backend: MotrixSim (np).
    """

    def __init__(self, cfg: PressThreeButtonsEnvCfg, num_envs: int = 32):
        super().__init__(cfg, num_envs=num_envs)

        self.button_bodies = [self.model.get_body(self.model.get_body_index(n)) for n in cfg.button_names]


        self.current_btn_idx = np.zeros((self.num_envs,), dtype=np.int32)   # 当前目标按钮 index: 0/1/2
        self.btn_pressed_mask = np.zeros((self.num_envs, 3), dtype=bool)    # 记录哪些按钮已经按过
        self.success_latched = np.zeros((self.num_envs,), dtype=bool)       # 终止 latch（TaskEnv 通常会读它）

    # ---- Task hooks ----
    def task_gaussians(self) -> Dict[str, str]:
        return TASK_GAUSSIANS

    def _randomize(self, data: SceneData, done_mask: np.ndarray, phase: str = "reset"):
        """
        Randomization: jitter button XY positions for the envs being reset.

        Args:
            data: SceneData view for the subset of envs being reset (len == sum(done_mask)).
            done_mask: boolean mask over all envs (not used directly here).
            phase: "reset" or "auto_reset" for potential differentiated logic.
        """
        if data.shape[0] == 0:
            return

        # Get current poses for subset: (B_subset, 3, 7)
        btn_pose = np.stack(
            [np.asarray(b.get_pose(data), dtype=np.float32) for b in self.button_bodies],
            axis=1,
        )

        # Small XY jitters (keep z/orientation unchanged)
        xy_jitter = self._rng.uniform(-0.01, 0.01, size=btn_pose[..., :2].shape).astype(np.float32)
        new_pose = btn_pose.copy()
        new_pose[..., :2] = btn_pose[..., :2] + xy_jitter

        # Write back poses using set_dof_pos (include floating base)
        for env_idx in range(data.shape[0]):
            for btn_idx, body in enumerate(self.button_bodies):
                body.set_dof_pos(
                    data[env_idx],
                    new_pose[env_idx, btn_idx],
                    include_floatingbase=True,
                )

    def _reset_task_state(self, done: np.ndarray):
        return
    # ---- helpers ----
    def _compute_reward(self, state: RenderEnvState) -> np.ndarray:
        data: SceneData = state.data
        info: Dict[str, np.ndarray] = state.info
        cfg = self._cfg
        btn_poses = np.stack([np.asarray(b.get_pose(data), dtype=np.float32) for b in self.button_bodies], axis=1)

        ee_pos = self.robot.get_ee_pose(data)[:, :3]
        btn_pos_xyz = btn_poses[:, :, :3]

        # 3) current target (fixed order by current_btn_idx)
        cur_idx_clamped = np.clip(self.current_btn_idx, 0, 2)
        target_pos = btn_pos_xyz[np.arange(self.num_envs), cur_idx_clamped, :]  
        target_pos[:, 2] += 0.05

        # 4) distance to target
        dist = np.linalg.norm(ee_pos - target_pos, axis=1)  

        touch_all = []
        for s_name in cfg.button_touch_names:
            v = np.asarray(self.model.get_sensor_value(s_name, data), dtype=np.float32)
            v = v.reshape(self.num_envs, -1)[:, 0].astype(np.float32)
            touch_all.append(v)
        touch_all = np.stack(touch_all, axis=1)  

        # current target touch value
        touch_vals = touch_all[np.arange(self.num_envs), cur_idx_clamped]  
        is_touched = touch_vals > cfg.touch_threshold

        # 6) state transition: mark success for this stage
        already_pressed = self.btn_pressed_mask[np.arange(self.num_envs), cur_idx_clamped]
        is_success_now = is_touched & (dist < 0.01) & (~already_pressed)

        if np.any(is_success_now):
            env_ids = np.where(is_success_now)[0]
            self.btn_pressed_mask[env_ids, cur_idx_clamped[env_ids]] = True
            self.current_btn_idx[env_ids] += 1

        # 7) rewards
        num_completed = np.sum(self.btn_pressed_mask, axis=1) 
        all_done = (num_completed >= 3)

        r_stage = num_completed * cfg.stage_complete_reward
        r_reach = cfg.dist_reward_scale * (1.0 - np.tanh(5.0 * dist))
        r_reach[all_done] = 1.0
        r_touch = is_touched.astype(np.float32) * cfg.touch_reward_bonus

        total_reward = r_stage + r_reach + r_touch

        # 8) latch termination
        self.success_latched = self.success_latched | all_done

        # Write metrics into info in-place
        info["is_success"] = self.success_latched.copy()
        info["cur_idx"] = self.current_btn_idx.copy()
        info["completed"] = num_completed
        info["dist"] = dist
        info["touch_val"] = touch_vals

        return total_reward.astype(np.float32)