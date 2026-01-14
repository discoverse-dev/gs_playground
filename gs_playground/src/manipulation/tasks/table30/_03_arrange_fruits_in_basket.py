from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List

import gymnasium as gym
import motrixsim as mtx
import numpy as np
from motrixsim import SceneData, forward_kinematic

from gs_playground import ROOT_PATH
from gs_playground.src.env.motrix_env.render_env import RenderEnvCfg, NpRenderEnv, RenderEnvState
from gs_playground.src.env.registry import envcfg, env

# [Robot] UR5e
from gs_playground.src.manipulation.robots.universal_robots_ur5e_robotiq.ur5e_robotiq import UR5eRobotiq

# [Assets]
from gs_playground.src.manipulation.tasks.table30.gaussian_assets import (
    build_task_gaussians,
)
from gs_playground.src.manipulation._tasks.common.safe_access import read_touch_scalar_safe

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ASSETS_UR5E_DIR = ROOT_PATH.parent / "models" / "robots" / "manipulation" / "universal_robots_ur5e_robotiq"
# 资源依然复用 03 文件夹
ASSETS_TASK_DIR = ROOT_PATH.parent / "models" / "tasks" / "table30" / "03_arrange_fruits_in_basket"

# [修改] 仅保留 XML 中定义的 4 种水果
FRUIT_NAMES = [
    "fruit_avocado",
    "fruit_banana",
    "fruit_carambola",
    "fruit_mangosteen",
]

# [修改] 对应 XML 中的 sensor name
TOUCH_NAMES = {
    "fruit_avocado": "touch_fruit_avocado",
    "fruit_banana": "touch_fruit_banana",
    "fruit_carambola": "touch_fruit_carambola",
    "fruit_mangosteen": "touch_fruit_mangosteen",
}


@envcfg("table30/arrange_fruits")
@dataclass
class ArrangeFruitsEnvCfg(RenderEnvCfg):
    # model / sim
    # 请确保你将新的 XML 保存为了这个文件名
    model_file: str = str((ASSETS_UR5E_DIR / "xmls" / "03_arrange_fruits_in_basket.xml").as_posix())
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02

    # control
    max_episode_steps: int = 1200 
    action_mode: str = "eef"

    # observation / prompt
    prompt_template: str = "What action should the robot take to {task_description}?"
    instruction: str = "Pick up all fruits and arrange them in the basket."

    # rendering
    img_width: int = 320
    img_height: int = 240
    cam_id: Tuple[int, ...] = (0,)

    # assets
    gs_background_ply: str = ""
    gs_robot_gaussians: Optional[Dict[str, str]] = None

    # entities
    basket_name: str = "basket"
    basket_site: str = "basket_site"
    
    # reward params
    touch_threshold: float = 0.01
    grasp_bonus: float = 2.0
    place_bonus: float = 5.0

    # reset
    reset_enabled: bool = True
    reset_keyframe: int | str = "home"


@env("table30/arrange_fruits", "np")
class ArrangeFruitsEnv(NpRenderEnv):
    """
    Task: Arrange 4 fruits into a basket using UR5e.
    """

    def __init__(self, cfg: ArrangeFruitsEnvCfg, num_envs: int = 32):
        cfg.cam_id = tuple(cfg.cam_id) if not isinstance(cfg.cam_id, tuple) else cfg.cam_id
        super().__init__(cfg, num_envs=num_envs)
        self._cfg: ArrangeFruitsEnvCfg = cfg

        # 1. Initialize Robot
        self.robot = UR5eRobotiq(self.model)

        # 2. Initialize Task Handles
        self.fruit_bodies = [self.model.get_body(self.model.get_body_index(n)) for n in FRUIT_NAMES]
        self.basket_body = self.model.get_body(self.model.get_body_index(cfg.basket_name))
        self.basket_site = self.model.get_site(cfg.basket_site)

        # [State Tracking]
        B = self._num_envs
        self.current_obj_idx = np.zeros(B, dtype=np.int32)
        self.is_grasped = np.zeros(B, dtype=bool)
        self.completed_mask = np.zeros((B, len(FRUIT_NAMES)), dtype=bool)

        # 3. Init Renderer
        gauss = UR5eRobotiq.robot_gaussians()
        
        # Mapping: fruit_avocado -> 3dgs/fruit_avocado.ply
        task_gaussians = {}
        for fname in FRUIT_NAMES:
            task_gaussians[fname] = f"3dgs/{fname}.ply"
        task_gaussians[cfg.basket_name] = f"3dgs/{cfg.basket_name}.ply"

        gauss.update(build_task_gaussians(ASSETS_TASK_DIR, {k: ASSETS_TASK_DIR / v for k, v in task_gaussians.items()}))
        
        if cfg.gs_robot_gaussians:
            gauss.update(cfg.gs_robot_gaussians)
            
        bg = cfg.gs_background_ply.strip() or UR5eRobotiq.robot_background_ply()
        self.init_renderer(body_gaussians=gauss, background_ply=bg, minibatch=self._num_envs)

        self._state = None

    @property
    def observation_space(self) -> gym.Space:
        cam_spaces = {f"pixels/view_{i}": gym.spaces.Box(0, 255, (self._img_h, self._img_w, 3), np.uint8) for i, _ in enumerate(self._cam_ids)}
        
        obs_spaces = {
            **cam_spaces,
            "qpos": gym.spaces.Box(-np.inf, np.inf, (6,), np.float32),
            "gripper": gym.spaces.Box(0, 1, (1,), np.float32),
            "ee_pose": gym.spaces.Box(-np.inf, np.inf, (6,), np.float32),
            "basket_pose": gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
            "target_fruit_pose": gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
            "target_idx": gym.spaces.Box(0, len(FRUIT_NAMES), (1,), np.int32),
            "prompt": gym.spaces.Text(max_length=256),
        }
        return gym.spaces.Dict(obs_spaces)

    @property
    def action_space(self) -> gym.Space:
        return self.robot.action_space

    def init_state(self) -> RenderEnvState:
        data = SceneData(self._model, batch=[self._num_envs])
        
        obs_struct = self.observation_space
        obs = {}
        for k, s in obs_struct.items():
            if isinstance(s, gym.spaces.Text):
                obs[k] = np.empty((self._num_envs,), dtype=object)
            else:
                obs[k] = np.zeros((self._num_envs,) + s.shape, dtype=s.dtype)

        reward = np.zeros(self._num_envs, np.float32)
        term = np.zeros(self._num_envs, bool)
        trunc = np.zeros(self._num_envs, bool)
        info = {"steps": np.zeros(self._num_envs, np.uint64)}
        
        self._state = RenderEnvState(data, obs, reward, term, trunc, info)
        self._reset_done_envs()
        self._state.validate()
        return self._state

    def apply_action(self, actions: np.ndarray, state) -> mtx.SceneData:
        self.robot.apply_action(state.data, actions, action_mode=self._cfg.action_mode)
        return state

    def update_state(self, state, obs_required: bool = True) -> mtx.SceneData:
        reward, info = self._compute_reward(state.data)
        
        all_done = np.all(self.completed_mask, axis=1)
        terminated = all_done.copy()

        if obs_required:
            state.obs = self._build_obs(state.data)

        state.reward = reward.astype(np.float32)
        state.terminated = terminated
        state.info.update(info)
        return state

    def _reset_done_envs(self):
        if self._state is None: return
        done = self._state.terminated | self._state.truncated
        if not np.any(done): return
        
        self.current_obj_idx[done] = 0
        self.is_grasped[done] = False
        self.completed_mask[done] = False

        self._apply_keyframe(self._state.data[done])
        forward_kinematic(self.model, self._state.data[done])
        self.robot.reset_envs(self._state.data, done)

        if self._state.obs is not None:
            new_obs = self._build_obs(self._state.data[done])
            for k, v in new_obs.items():
                self._state.obs[k][done] = v
            
        self._state.reward[done] = 0.0
        self._state.terminated[done] = False
        self._state.truncated[done] = False
        self._state.info["steps"][done] = 0

    def reset(self, data: SceneData = None, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        if data is not None:
            self._apply_keyframe(data)
            forward_kinematic(self.model, data)
            return self._build_obs(data), {}

        if self._state is None: self.init_state()
        if done is None: done = np.ones(self._num_envs, bool)
        else: done = np.asarray(done, bool)
        
        if not np.any(done): return self._state.obs, self._state.info

        self.current_obj_idx[done] = 0
        self.is_grasped[done] = False
        self.completed_mask[done] = False

        self._apply_keyframe(self._state.data[done])
        forward_kinematic(self.model, self._state.data[done])
        self.robot.reset_envs(self._state.data, done)
        
        self._state.obs = self._build_obs(self._state.data)
        self._state.reward[done] = 0.0
        self._state.terminated[done] = False
        self._state.truncated[done] = False
        self._state.info["steps"][done] = 0
        return self._state.obs, self._state.info

    # --------------------------------------------------------------------------
    # Dense Reward
    # --------------------------------------------------------------------------
    def _compute_reward(self, data: SceneData) -> Tuple[np.ndarray, Dict[str, Any]]:
        cfg = self._cfg
        B = self._num_envs
        
        ee_pos = self.robot.get_ee_pose(data)[:, :3]
        basket_pos = np.asarray(self.basket_site.get_pose(data), dtype=np.float32)[:, :3]
        
        cur_idx = np.clip(self.current_obj_idx, 0, len(FRUIT_NAMES)-1)
        
        all_fruit_poses = np.stack([
            np.asarray(b.get_pose(data), dtype=np.float32)[:, :3] for b in self.fruit_bodies
        ], axis=1)
        
        target_fruit_pos = all_fruit_poses[np.arange(B), cur_idx, :]

        d_ee_fruit = np.linalg.norm(ee_pos - target_fruit_pos, axis=1)
        d_fruit_basket = np.linalg.norm(target_fruit_pos - basket_pos, axis=1)

        # Touch Logic
        current_touch_val = np.zeros(B, dtype=np.float32)
        for i, fname in enumerate(FRUIT_NAMES):
            touch_name = TOUCH_NAMES[fname]
            val = read_touch_scalar_safe(self.model, data, touch_name, B)
            mask = (cur_idx == i)
            current_touch_val[mask] = val[mask]
        
        is_touching_fruit = current_touch_val > cfg.touch_threshold
        
        # Gates
        # Grasp
        newly_grasped = (~self.is_grasped) & is_touching_fruit & (d_ee_fruit < 0.05)
        self.is_grasped = self.is_grasped | newly_grasped
        
        # Place
        in_basket_xy = np.linalg.norm(target_fruit_pos[:, :2] - basket_pos[:, :2], axis=1) < 0.15
        in_basket_z = (target_fruit_pos[:, 2] - basket_pos[:, 2]) < 0.10
        in_basket = in_basket_xy & in_basket_z
        
        place_success_now = in_basket & (d_fruit_basket < 0.1)
        # print(FRUIT_NAMES,place_success_now)
        # print(self.is_grasped)
        # print(in_basket)
        # print((d_fruit_basket < 0.1))
        if np.any(place_success_now):
            done_envs = np.where(place_success_now)[0]
            self.completed_mask[done_envs, cur_idx[done_envs]] = True
            self.current_obj_idx[done_envs] += 1
            self.is_grasped[done_envs] = False

        # Rewards
        r_stage = np.sum(self.completed_mask, axis=1) * cfg.place_bonus
        
        active_mask = (self.current_obj_idx < len(FRUIT_NAMES))
        r_reach = np.zeros(B, dtype=np.float32)
        r_move = np.zeros(B, dtype=np.float32)
        
        if np.any(active_mask):
            r_reach[active_mask] = (1.0 - np.tanh(5.0 * d_ee_fruit[active_mask])) * (~self.is_grasped[active_mask])
            r_move[active_mask] = (1.0 - np.tanh(2.0 * d_fruit_basket[active_mask])) * self.is_grasped[active_mask]

        r_grasp = self.is_grasped.astype(np.float32) * cfg.grasp_bonus

        total_reward = r_stage + r_reach + r_move + r_grasp

        info = {
            "cur_idx": self.current_obj_idx,
            "d_ee_fruit": d_ee_fruit,
            "is_grasped": self.is_grasped,
            "completed": np.sum(self.completed_mask, axis=1)
        }
        return total_reward, info

    def _build_obs(self, data: SceneData) -> Dict[str, np.ndarray]:
        robot_obs = self.robot.get_obs(data)
        obs_pix = self._render_pixels(data)
        
        B = data.shape[0] if data is not None else self._num_envs
        cur_idx = np.clip(self.current_obj_idx, 0, len(FRUIT_NAMES)-1)
        
        all_fruit_poses = np.stack([
            np.asarray(b.get_pose(data), dtype=np.float32) for b in self.fruit_bodies
        ], axis=1)
        target_fruit_pose = all_fruit_poses[np.arange(B), cur_idx, :]
        
        basket_pose = np.asarray(self.basket_body.get_pose(data), dtype=np.float32)
        
        instruction = str(self._cfg.instruction)
        prompt = str(self._cfg.prompt_template).format(task_description=instruction)
        prompts = np.array([prompt] * B, dtype=object)

        return {
            **obs_pix,
            **robot_obs,
            "basket_pose": basket_pose,
            "target_fruit_pose": target_fruit_pose,
            "target_idx": cur_idx.reshape(B, 1),
            "prompt": prompts
        }