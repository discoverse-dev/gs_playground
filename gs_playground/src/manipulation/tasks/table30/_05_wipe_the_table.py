from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from motrixsim import SceneData

from gs_playground import ROOT_PATH
from gs_playground.src.env.registry import envcfg, env
from gs_playground.src.manipulation.tasks.task_env import TaskEnvCfg, TaskEnv
from gs_playground.src.env.motrix_env.render_env import RenderEnvState


ASSETS_TASK_DIR = ROOT_PATH / "models" / "tasks" / "table30" / "_05_wipe_the_table"
TASK_GAUSSIANS = {
    "blue_duck": (ASSETS_TASK_DIR / "3dgs" / "blue_duck.ply"),
    "chicken_doll": (ASSETS_TASK_DIR / "3dgs" / "chicken_doll.ply"),
    "dog": (ASSETS_TASK_DIR / "3dgs" / "dog.ply"),
    "yellow_brush": (ASSETS_TASK_DIR / "3dgs" / "yellow_brush.ply"),
    "transparent_tape_paper": (ASSETS_TASK_DIR / "3dgs" / "transparent_tape_paper.ply"),
    "box": (ASSETS_TASK_DIR / "3dgs" / "box.ply"),
}


@envcfg("table30/wipe_the_table")
@dataclass
class WipeTheTableEnvCfg(TaskEnvCfg):
    # model / sim
    robot_name: str = "franka_robotiq"
    model_file: str = str((ROOT_PATH / "models"/ "robots"/ 
                           "manipulation" / "franka_emika_panda_robotiq"/ 
                           "xmls" / "table30_05_wipe_the_table.xml").as_posix())

    # control
    action_mode: str = "eef_relative"  # "joint" or "eef"
    max_episode_steps = 1500
    # rendering
    img_width: int = 320
    img_height: int = 240
 
    # prompt
    instruction: str = ("Place all the clutter on the desk into the white basket")

    # entities
    box_name: str = "box"
    box_site: str = "box_site"
    target_obj_names: Tuple[str, str, str] = (
        "blue_duck",
        "chicken_doll",
        "transparent_tape_paper",
    )
    target_touch_names: Tuple[str, str, str] = (
        "blue_duck_touch",
        "chicken_doll_touch",
        "transparent_tape_paper_touch",
    )

    # thresholds / logic
    touch_threshold: float = 1e-3
    grasp_dist_thresh: float = 0.10
    gripper_close_thresh: float = 0.2

    box_half_extents: Tuple[float, float, float] = (0.14, 0.14, 0.01)
    box_margin: float = 0.01
    box_z_allow: float = 0.20

    # rewards
    stage_reward: float = 1.0
    reach_reward_scale: float = 1.0
    move_reward_scale: float = 1.0

    reset_enabled: bool = True
    reset_keyframe: int | str = "home"


@env("table30/wipe_the_table", "np")
class WipeTheTableEnv(TaskEnv):
    """
    Task:
      - Only targets: blue_duck, chicken_doll, transparent_tape_paper must be placed into box.
      - yellow_brush and dog are assumed already in the box by initialization (XML/keyframe),
        so the environment does NOT enforce them at reset.
    """

    def __init__(self, cfg: WipeTheTableEnvCfg, num_envs: int = 32):
        super().__init__(cfg, num_envs=num_envs)

        self.target_bodies = [self.model.get_body(self.model.get_body_index(n)) for n in cfg.target_obj_names]
        self.box_body = self.model.get_body(self.model.get_body_index(cfg.box_name))

        # optional site
        self.box_site_handle = None
        try:
            self.box_site_handle = self.model.get_site(cfg.box_site)
        except Exception:
            self.box_site_handle = None

        N = len(cfg.target_obj_names)
        self.current_obj_idx = np.zeros((self.num_envs,), dtype=np.int32)  # 0..N
        self.placed_mask = np.zeros((self.num_envs, N), dtype=bool)
        self.is_grasped = np.zeros((self.num_envs,), dtype=bool)
        self.success_latched = np.zeros((self.num_envs,), dtype=bool)

        self._dbg_print_ctr = 0

    # ---- Task hooks ----
    def task_gaussians(self) -> Dict[str, str]:
        return {k: str(v) for k, v in TASK_GAUSSIANS.items()}

    def _randomize(self, data: SceneData, done_mask: np.ndarray, phase: str = "reset"):
        """
        No-op by default.
        (If you later want to jitter target objects at reset, implement it here,
         similar to ArrangeFruitsEnv._randomize.)
        """
        return

    def _reset_task_state(self, done: np.ndarray):
        done = np.asarray(done, dtype=bool)
        if done.size == 0 or (not np.any(done)):
            return
        self.current_obj_idx[done] = 0
        self.placed_mask[done] = False
        self.is_grasped[done] = False
        self.success_latched[done] = False

    # ---- helpers ----
    def _box_pos(self, data: SceneData) -> np.ndarray:
        if self.box_site_handle is not None:
            return np.asarray(self.box_site_handle.get_pose(data), dtype=np.float32)[:, :3]
        return np.asarray(self.box_body.get_pose(data), dtype=np.float32)[:, :3]

    # ---- reward / success ----
    def _compute_reward(self, state: RenderEnvState) -> np.ndarray:
        cfg = self._cfg
        data: SceneData = state.data
        info: Dict[str, np.ndarray] = state.info

        N = len(cfg.target_obj_names)
        cur_idx = np.clip(self.current_obj_idx, 0, N - 1)

        ee_pos = self.robot.get_ee_pose(data)[:, :3]
        box_pos = self._box_pos(data)

        obj_pos_all = np.stack(
            [np.asarray(b.get_pose(data), dtype=np.float32)[:, :3] for b in self.target_bodies],
            axis=1,  # (B,N,3)
        )
        target_pos = obj_pos_all[np.arange(self.num_envs), cur_idx, :]
        target_pos_above = target_pos.copy()
        target_pos_above[:, 2] += 0.05

        dist_ee_obj = np.linalg.norm(ee_pos - target_pos_above, axis=1)
        dist_obj_box = np.linalg.norm(target_pos - box_pos, axis=1)

        # gripper gate
        grip_cmd = np.asarray(data.actuator_ctrls)[:, self.robot.gripper_act_id]
        grip_closed = grip_cmd > float(cfg.gripper_close_thresh)
        grip_open = ~grip_closed

        # touch (targets only)
        touch_all = []
        for s_name in cfg.target_touch_names:
            v = np.asarray(self.model.get_sensor_value(s_name, data), dtype=np.float32)
            v = v.reshape(self.num_envs, -1)[:, 0].astype(np.float32)
            touch_all.append(v)
        touch_all = np.stack(touch_all, axis=1)  # (B,N)

        touch_val = touch_all[np.arange(self.num_envs), cur_idx]
        is_touched = touch_val > float(cfg.touch_threshold)

        # grasp latch: touch + close + near
        newly_grasped = (~self.is_grasped) & is_touched & grip_closed & (dist_ee_obj < float(cfg.grasp_dist_thresh))
        self.is_grasped = self.is_grasped | newly_grasped

        # inside box (world-axis approx)
        sx, sy, _sz = cfg.box_half_extents
        m = float(cfg.box_margin)
        rel = target_pos - box_pos
        inside = (
            (np.abs(rel[:, 0]) < float(sx - m))
            & (np.abs(rel[:, 1]) < float(sy - m))
            & (rel[:, 2] > 0.0)
            & (rel[:, 2] < float(cfg.box_z_allow))
        )

        already = self.placed_mask[np.arange(self.num_envs), cur_idx]
        place_now = self.is_grasped & inside & grip_open & (~already)
        print("rel",rel)
        print("place_now",place_now)

        self._dbg_print_ctr += 1
        if np.any(place_now):
            envs = np.where(place_now)[0]
            self.placed_mask[envs, cur_idx[envs]] = True
            self.current_obj_idx[envs] += 1
            self.is_grasped[envs] = False

        completed = np.sum(self.placed_mask, axis=1).astype(np.int32)
        all_done = completed >= N
        self.success_latched = self.success_latched | all_done

        # rewards
        r_stage = completed.astype(np.float32) * float(cfg.stage_reward)
        r_reach = (1.0 - np.tanh(5.0 * dist_ee_obj)) * (~self.is_grasped).astype(np.float32)
        r_move = (1.0 - np.tanh(2.0 * dist_obj_box)) * self.is_grasped.astype(np.float32)
        reward = r_stage + float(cfg.reach_reward_scale) * r_reach + float(cfg.move_reward_scale) * r_move

        # info
        info["is_success"] = self.success_latched.copy()
        info["cur_idx"] = self.current_obj_idx.copy()
        info["completed"] = completed
        info["is_grasped"] = self.is_grasped.copy()
        info["touch_val"] = touch_val
        info["dist_ee_obj"] = dist_ee_obj
        info["dist_obj_box"] = dist_obj_box
        info["inside_box"] = inside

        return reward.astype(np.float32)

    def _check_success(self, state: RenderEnvState) -> np.ndarray:
        completed = np.sum(self.placed_mask, axis=1)
        success = completed >= len(self._cfg.target_obj_names)
        self.success_latched = self.success_latched | success
        return self.success_latched.copy()
