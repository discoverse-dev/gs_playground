from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from motrixsim import SceneData

from gs_playground import ROOT_PATH
from gs_playground.src.env.registry import envcfg, env
from gs_playground.src.manipulation.tasks.task_env import TaskEnvCfg, TaskEnv
from gs_playground.src.env.motrix_env.render_env import RenderEnvState

ASSETS_TASK_DIR = ROOT_PATH / "models" / "tasks" / "table30" / "03_arrange_fruits_in_basket"
TASK_GAUSSIANS = {
    "fruit_avocado":    ASSETS_TASK_DIR / "3dgs" / "fruit_avocado.ply",
    "fruit_banana":     ASSETS_TASK_DIR / "3dgs" / "fruit_banana.ply",
    "fruit_carambola":  ASSETS_TASK_DIR / "3dgs" / "fruit_carambola.ply",
    "fruit_mangosteen": ASSETS_TASK_DIR / "3dgs" / "fruit_mangosteen.ply",
    "basket":           ASSETS_TASK_DIR / "3dgs" / "basket.ply",
}

@envcfg("table30/arrange_fruits")
@dataclass
class ArrangeFruitsEnvCfg(TaskEnvCfg):
    # model / sim
    robot_name: str = "ur5e_robotiq"
    model_file: str = str((ROOT_PATH / "models"/ "robots"/ 
                           "manipulation" / "universal_robots_ur5e_robotiq"/ 
                           "xmls" / "table30_03_arrange_fruits_in_basket.xml").as_posix())

    # control
    action_mode: str = "eef_relative"  # "joint" or "eef"
    max_episode_steps = 1500
    # rendering
    img_width: int = 320
    img_height: int = 240
 
    # observation / prompt
    instruction: str = "Place the four fruits into the nearby basket one by one."

    # task entities
    basket_name: str = "basket"
    basket_site: str = "basket_site"
    fruit_names: Tuple[str, str, str, str] = (
        "fruit_avocado",
        "fruit_banana",
        "fruit_carambola",
        "fruit_mangosteen",
    )
    fruit_touch_names: Tuple[str, str, str, str] = (
        "touch_fruit_avocado",
        "touch_fruit_banana",
        "touch_fruit_carambola",
        "touch_fruit_mangosteen",
    )

    # reward / logic
    touch_threshold: float = 0.01
    grasp_dist_thresh: float = 0.10
    gripper_close_thresh: float = 0.2

    basket_xy_thresh: float = 0.25
    basket_z_thresh: float = 0.05
    place_dist_thresh: float = 0.2

    # stage reward: 4 fruits * 2.5 = 10
    stage_reward: float = 2.5

    # shaping (optional)
    reach_reward_scale: float = 1.0
    move_reward_scale: float = 1.0

    reset_enabled: bool = True
    reset_keyframe: int | str = "home"


@env("table30/arrange_fruits", "np")
class ArrangeFruitsEnv(TaskEnv):
    """
    Task: Arrange fruits into the basket (fixed order: cfg.fruit_names).
    Robot: UR5e + Robotiq.
    Backend: MotrixSim (np).
    """

    def __init__(self, cfg: ArrangeFruitsEnvCfg, num_envs: int = 32):
        super().__init__(cfg, num_envs=num_envs)

        self.fruit_bodies = [self.model.get_body(self.model.get_body_index(n)) for n in cfg.fruit_names]
        self.basket_body = self.model.get_body(self.model.get_body_index(cfg.basket_name))
        self.basket_site = self.model.get_site(cfg.basket_site)
         
        N = len(cfg.fruit_names)

        self.current_obj_idx = np.zeros((self.num_envs,), dtype=np.int32)   # 0..N
        self.fruit_placed_mask = np.zeros((self.num_envs, N), dtype=bool)   # per-fruit placed
        self.is_grasped = np.zeros((self.num_envs,), dtype=bool)            # current fruit grasp latch
        self.success_latched = np.zeros((self.num_envs,), dtype=bool)       # terminal latch
        self._dbg_print_ctr = 0

    # ---- Task hooks ----
    def task_gaussians(self) -> Dict[str, str]:
        return TASK_GAUSSIANS

    def _randomize(self, data: SceneData, done_mask: np.ndarray, phase: str = "reset"):
        """
        Randomization: jitter fruit XY positions for the envs being reset.
        Keep a minimum inter-fruit XY separation (simple rejection sampling).
        """
        if data.shape[0] == 0:
            return

        num_fruits = len(self.fruit_bodies)
        min_xy_dist = 0.01
        jitter_scale = 0.02
        # Current poses (Bsub, N, 7)
        fruit_pose = np.stack(
            [np.asarray(b.get_pose(data), dtype=np.float32) for b in self.fruit_bodies],
            axis=1,
        )
        base_xy = fruit_pose[..., :2]
        new_pose = fruit_pose.copy()

        remaining = np.ones((data.shape[0],), dtype=bool)
        eye_mask = np.eye(num_fruits, dtype=np.float32) * 1e6  # ignore self-distance

        for _ in range(10):
            if not remaining.any():
                break

            jitter = self._rng.uniform(-jitter_scale, jitter_scale, size=(remaining.sum(), num_fruits, 2)).astype(
                np.float32
            )
            cand_xy = base_xy[remaining] + jitter

            diff = cand_xy[:, :, None, :] - cand_xy[:, None, :, :]
            dist = np.linalg.norm(diff, axis=-1) + eye_mask[None]
            ok = dist.min(axis=(1, 2)) >= min_xy_dist

            if ok.any():
                rem_idx = np.where(remaining)[0]
                new_pose_view = new_pose[remaining]
                new_pose_view[ok, :, :2] = cand_xy[ok]
                new_pose[remaining] = new_pose_view
                remaining[rem_idx[ok]] = False

        # Write back
        for env_i in range(data.shape[0]):
            for f_i, body in enumerate(self.fruit_bodies):
                body.set_dof_pos(
                    data[env_i],
                    new_pose[env_i, f_i],
                    include_floatingbase=True,
                )

    def _reset_task_state(self, done: np.ndarray):
        done = np.asarray(done, dtype=bool)
        if done.size == 0 or not np.any(done):
            return

        self.current_obj_idx[done] = 0
        self.fruit_placed_mask[done] = False
        self.is_grasped[done] = False
        self.success_latched[done] = False

    # ---- helpers ----
    def _compute_reward(self, state: RenderEnvState) -> np.ndarray:
        cfg = self._cfg
        data: SceneData = state.data
        info: Dict[str, np.ndarray] = state.info

         
        N = len(cfg.fruit_names)

        cur_idx = np.clip(self.current_obj_idx, 0, N - 1)

        ee_pos = self.robot.get_ee_pose(data)[:, :3]
        basket_pos = np.asarray(self.basket_site.get_pose(data), dtype=np.float32)[:, :3]

        fruit_pos_all = np.stack(
            [np.asarray(b.get_pose(data), dtype=np.float32)[:, :3] for b in self.fruit_bodies],
            axis=1,  # (self.num_envs, N, 3)
        )
        target_pos = fruit_pos_all[np.arange(self.num_envs), cur_idx, :]  # (self.num_envs, 3)

        target_pos_above = target_pos.copy()
        target_pos_above[:, 2] += 0.05

        dist_ee_fruit = np.linalg.norm(ee_pos - target_pos_above, axis=1)
        dist_fruit_basket = np.linalg.norm(target_pos - basket_pos, axis=1)

        # gripper gate
        grip_cmd = np.asarray(data.actuator_ctrls)[:, self.robot.gripper_act_id]
        grip_closed = grip_cmd > float(cfg.gripper_close_thresh)
        grip_open = ~grip_closed

        # touch: same style as button task (model.get_sensor_value)
        touch_all = []
        for s_name in cfg.fruit_touch_names:
            v = np.asarray(self.model.get_sensor_value(s_name, data), dtype=np.float32)
            v = v.reshape(self.num_envs, -1)[:, 0].astype(np.float32)
            touch_all.append(v)
        touch_all = np.stack(touch_all, axis=1)  # (self.num_envs, N)

        touch_val = touch_all[np.arange(self.num_envs), cur_idx]
        is_touched = touch_val > float(cfg.touch_threshold)

        # grasp latch (touch + close + near)
        newly_grasped = (~self.is_grasped) & is_touched & grip_closed & (dist_ee_fruit < float(cfg.grasp_dist_thresh))
        self.is_grasped = self.is_grasped | newly_grasped

        # place (require grasped -> in basket -> release)
        in_basket_xy = np.linalg.norm(target_pos[:, :2] - basket_pos[:, :2], axis=1) < float(cfg.basket_xy_thresh)
        in_basket_z = (target_pos[:, 2] - basket_pos[:, 2]) < float(cfg.basket_z_thresh)
        in_basket = in_basket_xy & in_basket_z

        already_placed = self.fruit_placed_mask[np.arange(self.num_envs), cur_idx]
        place_now = (
            self.is_grasped
            &  (dist_fruit_basket < float(cfg.place_dist_thresh))
             
            & grip_open
            & (~already_placed)
        )
        self._dbg_print_ctr += 1

        if np.any(place_now):
            envs = np.where(place_now)[0]
            self.fruit_placed_mask[envs, cur_idx[envs]] = True
            self.current_obj_idx[envs] += 1
            self.is_grasped[envs] = False

        # if (self._dbg_print_ctr % 10) == 0:
        #     print("    is_grasped      :", self.is_grasped.astype(int))
        #     print("    in_basket       :", in_basket.astype(int))
        #     print("    in_basket_xy    :", in_basket_xy.astype(int))
        #     print("    in_basket_z     :", in_basket_z.astype(int))
        #     print("    dist_ok         :", (dist_fruit_basket < float(cfg.place_dist_thresh)).astype(int))
        #     print("    grip_open       :", grip_open.astype(int))
        #     print("    not_placed      :", (~already_placed).astype(int))
        #     print("    place_now       :", place_now.astype(int))
        #     print("    dist_fruit_bask :", np.round(dist_fruit_basket, 4))
        #     print("    touch_val       :", np.round(touch_val, 4))
        #     print("    grip_cmd        :", np.round(grip_cmd, 4))
        #     print("fruit_placed_mask",self.fruit_placed_mask)

        completed = np.sum(self.fruit_placed_mask, axis=1).astype(np.int32)
        all_done = completed >= N
        self.success_latched = self.success_latched | all_done

        # rewards
        # stage reward: each placed fruit => +2.5, total 10 for 4 fruits
        r_stage = completed.astype(np.float32) * float(cfg.stage_reward)

        # shaping (same pattern as buttons: reach when not grasped; move when grasped)
        r_reach = (1.0 - np.tanh(5.0 * dist_ee_fruit)) * (~self.is_grasped).astype(np.float32)
        r_move = (1.0 - np.tanh(2.0 * dist_fruit_basket)) * self.is_grasped.astype(np.float32)

        reward = r_stage + float(cfg.reach_reward_scale) * r_reach + float(cfg.move_reward_scale) * r_move

        info["is_success"] = self.success_latched.copy()
        info["cur_idx"] = self.current_obj_idx.copy()
        info["completed"] = completed
        info["is_grasped"] = self.is_grasped.copy()
        info["touch_val"] = touch_val
        info["dist_ee_fruit"] = dist_ee_fruit
        info["dist_fruit_basket"] = dist_fruit_basket
        info["in_basket"] = in_basket

        return reward.astype(np.float32)

    def _check_success(self, state: RenderEnvState) -> np.ndarray:
        completed = np.sum(self.fruit_placed_mask, axis=1)
        success = completed >= len(self._cfg.fruit_names)
        self.success_latched = self.success_latched | success

        return self.success_latched.copy()
