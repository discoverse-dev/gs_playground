from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

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
ASSETS_TASK_DIR = ROOT_PATH.parent / "models" / "tasks" / "table30" / "04_hang_toothbrush_cup"

# [映射修正] Key 必须与 XML 中的 body name 一致
TASK_GAUSSIANS = {
    "red_bottle": "3dgs/red_bottle.ply",
    "rack": "3dgs/rack.ply",
}


@envcfg("table30/hang_bottle")
@dataclass
class HangBottleEnvCfg(RenderEnvCfg):
    # model / sim
    model_file: str = str((ASSETS_UR5E_DIR / "xmls" / "04_hang_toothbrush_cup.xml").as_posix())
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02

    # control
    max_episode_steps: int = 800
    action_mode: str = "eef"

    # observation / prompt
    prompt_template: str = "What action should the robot take to {task_description}?"
    instruction: str = "Pick up the red bottle and hang it on the rack."

    # rendering
    img_width: int = 320
    img_height: int = 240
    cam_id: Tuple[int, ...] = (0,)

    # assets
    gs_background_ply: str = ""
    gs_robot_gaussians: Optional[Dict[str, str]] = None

    # [Entities from XML]
    # 对应 XML 中的 <body name="red_bottle"> 和 <body name="rack">
    bottle_name: str = "red_bottle"
    rack_name: str = "rack"
    
    # [Sensors from XML]
    # 对应 XML 中的 <touch name="...">
    sensor_grasp: str = "bottle_grasp_touch"
    sensor_hook: str = "rack_hook_touch"

    # reward params
    touch_threshold: float = 0.1
    grasp_reward_bonus: float = 2.0
    hang_reward_bonus: float = 5.0
    
    # reset
    reset_enabled: bool = True
    reset_keyframe: int | str = "home"


@env("table30/hang_bottle", "np")
class HangBottleEnv(NpRenderEnv):
    """
    Task: Hang a bottle on a rack using UR5e.
    """

    def __init__(self, cfg: HangBottleEnvCfg, num_envs: int = 32):
        cfg.cam_id = tuple(cfg.cam_id) if not isinstance(cfg.cam_id, tuple) else cfg.cam_id
        super().__init__(cfg, num_envs=num_envs)
        self._cfg: HangBottleEnvCfg = cfg

        # 1. Initialize Robot (UR5e)
        self.robot = UR5eRobotiq(self.model)

        # 2. Initialize Task Handles
        self.bottle_body = self.model.get_body(self.model.get_body_index(cfg.bottle_name))
        self.rack_body = self.model.get_body(self.model.get_body_index(cfg.rack_name))
        
        # Sites for logic (Defined in XML)
        self.hook_site = self.model.get_site("rack_hook_site")
        self.grasp_site = self.model.get_site("bottle_grasp_site")

        # [State Tracking]
        B = self._num_envs
        self.is_grasped = np.zeros(B, dtype=bool)
        self.is_hung = np.zeros(B, dtype=bool)

        # 3. Init Renderer
        gauss = UR5eRobotiq.robot_gaussians()
        # [映射修正] 加载 task 相关的 ply
        gauss.update(build_task_gaussians(ASSETS_TASK_DIR, {k: ASSETS_TASK_DIR / v for k, v in TASK_GAUSSIANS.items()}))
        
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
            "bottle_pose": gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
            "rack_pose": gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
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
        
        terminated = self.is_hung.copy() # 完成任务即终止

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
        
        self.is_grasped[done] = False
        self.is_hung[done] = False

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

        self.is_grasped[done] = False
        self.is_hung[done] = False

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
    # 稠密奖励逻辑 (Dense Reward)
    # --------------------------------------------------------------------------
    def _compute_reward(self, data: SceneData) -> Tuple[np.ndarray, Dict[str, Any]]:
        cfg = self._cfg
        B = self._num_envs
        
        ee_pos = self.robot.get_ee_pose(data)[:, :3]
        bottle_grasp_pos = np.asarray(self.grasp_site.get_pose(data), dtype=np.float32)[:, :3]
        hook_pos = np.asarray(self.hook_site.get_pose(data), dtype=np.float32)[:, :3]
        
        # 读取传感器 (使用 safe access 防止报错)
        grasp_touch = read_touch_scalar_safe(self.model, data, cfg.sensor_grasp, B)
        hook_touch = read_touch_scalar_safe(self.model, data, cfg.sensor_hook, B)
        
        touching_bottle = grasp_touch > cfg.touch_threshold
        touching_hook = hook_touch > cfg.touch_threshold

        d_ee_bottle = np.linalg.norm(ee_pos - bottle_grasp_pos, axis=1)
        d_bottle_hook = np.linalg.norm(bottle_grasp_pos - hook_pos, axis=1)

        # 1. Reach Reward
        r_reach = 1.0 - np.tanh(5.0 * d_ee_bottle)
        
        # 2. Grasp Gate
        # 如果足够近且触发了传感器 -> 视为抓取成功
        self.is_grasped = self.is_grasped | (touching_bottle & (d_ee_bottle < 0.05))
        r_grasped = self.is_grasped.astype(np.float32) * cfg.grasp_reward_bonus

        # 3. Move to Hook Reward (仅在抓取后激活)
        r_move = 0.0
        if np.any(self.is_grasped):
            r_move = (1.0 - np.tanh(2.0 * d_bottle_hook)) * self.is_grasped.astype(np.float32)
        
        # 4. Hang Gate
        # 如果瓶子在挂钩附近，且挂钩传感器触发，且瓶子高度合适 -> 视为挂载成功
        # (瓶子高度 > 挂钩高度 - 偏差)
        is_high_enough = bottle_grasp_pos[:, 2] > (hook_pos[:, 2] - 0.05)
        success_now = touching_hook & is_high_enough & self.is_grasped
        self.is_hung = self.is_hung | success_now
        
        r_success = self.is_hung.astype(np.float32) * cfg.hang_reward_bonus

        total_reward = r_reach + r_grasped + r_move + r_success

        info = {
            "d_ee_bottle": d_ee_bottle,
            "d_bottle_hook": d_bottle_hook,
            "is_grasped": self.is_grasped,
            "is_hung": self.is_hung
        }
        return total_reward, info

    def _build_obs(self, data: SceneData) -> Dict[str, np.ndarray]:
        robot_obs = self.robot.get_obs(data)
        obs_pix = self._render_pixels(data)
        
        bottle_pose = np.asarray(self.bottle_body.get_pose(data), dtype=np.float32)
        rack_pose = np.asarray(self.rack_body.get_pose(data), dtype=np.float32)
        
        instruction = str(self._cfg.instruction)
        prompt = str(self._cfg.prompt_template).format(task_description=instruction)
        prompts = np.array([prompt] * (data.shape[0]), dtype=object)

        return {
            **obs_pix,
            **robot_obs,
            "bottle_pose": bottle_pose,
            "rack_pose": rack_pose,
            "prompt": prompts
        }