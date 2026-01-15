from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import gymnasium as gym
import motrixsim as mtx
import numpy as np
from motrixsim import SceneData, forward_kinematic

from gs_playground import ROOT_PATH
from gs_playground.src.env.motrix_env.render_env import RenderEnvCfg, NpRenderEnv, RenderEnvState
from gs_playground.src.env.registry import envcfg, env

from gs_playground.src.manipulation.robots.franka_emika_panda_robotiq.franka_robotiq import FrankaRobotiq


# Task specific assets
from gs_playground.src.manipulation.tasks.table30.gaussian_assets import (
    build_task_gaussians,
)

ASSETS_FRANKA_DIR = ROOT_PATH.parent / "models" / "robots" / "manipulation" / "franka_robotiq"
ASSETS_TASK_DIR = ROOT_PATH.parent / "models" / "tasks" / "table30" / "_01_press_three_buttons"

TASK_GAUSSIANS = {
    "button_blue": "3dgs/button_blue.ply",
    "button_green": "3dgs/button_green.ply",
    "button_pink": "3dgs/button_pink.ply",
}


@envcfg("table30/press_three_buttons")
@dataclass
class PressThreeButtonsEnvCfg(RenderEnvCfg):
    # model / sim
    model_file: str = str((ASSETS_FRANKA_DIR / "xmls" / "01_press_three_buttons.xml").as_posix())
    sim_dt: float = 0.005
    ctrl_dt: float = 0.02

    # control
    max_episode_steps: int = 500
    action_mode: str = "joint"  # "joint" or "eef"

    # observation / prompt
    prompt_template: str = "What action should the robot take to {task_description}?"
    instruction: str = "Press the blue, green, and pink buttons in order."

    # rendering
    img_width: int = 320
    img_height: int = 240
    cam_id: Tuple[int, ...] = (0,)

    # assets
    gs_background_ply: str = ""
    gs_robot_gaussians: Optional[Dict[str, str]] = None

    # task entities
    button_names: Tuple[str, str, str] = ("button_blue", "button_green", "button_pink")
    button_touch_names: Tuple[str, str, str] = ("button_blue_touch", "button_green_touch", "button_pink_touch")

    # reward params
    touch_threshold: float = 1e-3  # 触碰传感器阈值
    dist_reward_scale: float = 1.0 # 距离奖励权重
    touch_reward_bonus: float = 0.5 # 触碰瞬间的额外奖励
    stage_complete_reward: float = 5.0 # 完成一个按钮的阶段奖励
    
    # reset
    reset_enabled: bool = True
    reset_keyframe: int | str = 0


@env("table30/press_three_buttons", "np")
class PressThreeButtonsEnv(NpRenderEnv):
    """
    Task: Press three buttons using Franka Emika Panda.
    Includes internal state tracking for dense reward calculation.
    """

    def __init__(self, cfg: PressThreeButtonsEnvCfg, num_envs: int = 32):
        cfg.cam_id = tuple(cfg.cam_id) if not isinstance(cfg.cam_id, tuple) else cfg.cam_id
        super().__init__(cfg, num_envs=num_envs)
        self._cfg: PressThreeButtonsEnvCfg = cfg

        # 1. Initialize Robot
        self.robot = FrankaRobotiq(self.model)

        # 2. Initialize Task Handles
        self.button_bodies = [self.model.get_body(self.model.get_body_index(n)) for n in cfg.button_names]
        
        # [State Tracking] 用于 Env 内部计算 Reward 的状态
        B = self._num_envs
        self.current_btn_idx = np.zeros(B, dtype=np.int32) # 当前需要按第几个按钮 (0, 1, 2)
        self.btn_pressed_mask = np.zeros((B, 3), dtype=bool) # 记录哪些已经按过了

        # 4. Init Renderer (3DGS)
        gauss = FrankaRobotiq.robot_gaussians()
        gauss.update(build_task_gaussians(ASSETS_TASK_DIR, {k: ASSETS_TASK_DIR / v for k, v in TASK_GAUSSIANS.items()}))
        if cfg.gs_robot_gaussians:
            gauss.update(cfg.gs_robot_gaussians)
        bg = cfg.gs_background_ply.strip() or FrankaRobotiq.robot_background_ply()
        self.init_renderer(body_gaussians=gauss, background_ply=bg, minibatch=self._num_envs)
        # 5. Spaces
        # self._obs_rgb_shape = (cfg.img_height, cfg.img_width, 3)

        # self.data: Optional[SceneData] = None
        self._state = None

    # ---- Gym Props ----
    @property
    def observation_space(self) -> gym.Space:
        cam_spaces = {}
        for i, _ in enumerate(self._cam_ids):
            cam_spaces[f"pixels/view_{i}"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self._img_h, self._img_w, 3),
                dtype=np.uint8,
            )
        obs_spaces = {
            **cam_spaces,
            "qpos": gym.spaces.Box(low=self.model.joint_limits[0, :7], high=self.model.joint_limits[1, :7], dtype=np.float32),
            "gripper": gym.spaces.Box(low=self.model.joint_limits[0, 7:8], high=self.model.joint_limits[1, 7:8], dtype=np.float32),
            "ee_pose": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            "target_btn_idx": gym.spaces.Box(low=0, high=3, shape=(1,), dtype=np.int32),
            "prompt": gym.spaces.Text(max_length=256),
        }
        return gym.spaces.Dict(obs_spaces)

    @property
    def action_space(self) -> gym.Space:
        return self.robot.action_space

    # ---- NpEnv required ----
    def init_state(self) -> RenderEnvState:
        data = SceneData(self._model, batch=[self._num_envs])
        
        obs_example = self.observation_space        
        obs = {}
        for k, space in obs_example.items():
            if isinstance(space, gym.spaces.Text):
                obs[k] = np.empty((self._num_envs,), dtype=object)
            else:
                shape = (self._num_envs,) + space.shape
                obs[k] = np.zeros(shape, dtype=space.dtype)

        reward = np.zeros((self._num_envs,), dtype=np.float32)
        terminated = np.ones((self._num_envs,), dtype=bool)
        truncated = np.zeros((self._num_envs,), dtype=bool)
        info = {"steps": np.zeros((self._num_envs,), dtype=np.uint64)}
        
        self._state = RenderEnvState(data, obs, reward, terminated, truncated, info)
        self._reset_done_envs()
        self._state.validate()
        return self._state

    def apply_action(self, actions: np.ndarray, state) -> mtx.SceneData:
        self.robot.apply_action(state.data, actions, action_mode=self._cfg.action_mode)
        return state
    
    def _before_chunk_step(self, data: mtx.SceneData):
        """Update robot reference state for relative control at the start of a chunk."""
        self.robot.update_reference(data)

    def update_state(self, state, obs_required: bool = True) -> mtx.SceneData:
        # [新增] 在每一步更新状态时计算 Reward
        reward, info = self._compute_reward(state.data)
        
        # 所有的都完成才算 Terminated (或者根据需求)
        all_done = np.all(self.btn_pressed_mask, axis=1)
        terminated = all_done.copy()

        if obs_required:
            obs = self._build_obs(state.data)
            state.obs = obs

        state.reward = reward.astype(np.float32)
        state.terminated = terminated
        # 将 reward 详情放入 info 方便调试
        state.info.update(info)
        
        return state

    def _reset_task_state(self, done: np.ndarray):
        """Reset internal reward tracking variables."""
        if np.any(done):
            self.current_btn_idx[done] = 0
            self.btn_pressed_mask[done] = False

    def _reset_done_envs(self):
        if self._state is None: return
        done = self._state.terminated | self._state.truncated
        if not np.any(done):
            return
        
        # Reset Logic
        self._reset_task_state(done)

        self._apply_keyframe(self._state.data[done])
        forward_kinematic(self.model, self._state.data[done])
        self.robot.reset_envs(self._state.data, done)

        obs_subset = self._build_obs(self._state.data[done])
        for k, v in obs_subset.items():
            self._state.obs[k][done] = v
            
        self._state.reward[done] = 0.0
        self._state.terminated[done] = False
        self._state.truncated[done] = False
        self._state.info["steps"][done] = 0

    def reset(self, data: SceneData = None, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        if data is not None:
            self._apply_keyframe(data)
            forward_kinematic(self.model, data)
            obs = self._build_obs(data)
            return obs, {}

        if self._state is None:
            self.init_state()
            
        if done is None:
            done_mask = np.ones((self._num_envs,), dtype=bool)
        else:
            done_mask = np.asarray(done, dtype=bool)
            
        if not np.any(done_mask):
             return self._state.obs, self._state.info

        self._reset_task_state(done_mask)
        self._apply_keyframe(self._state.data[done_mask])
        forward_kinematic(self.model, self._state.data[done_mask])
        self.robot.reset_envs(self._state.data, done_mask)
        
        obs_new_all = self._build_obs(self._state.data)
        self._state.obs = obs_new_all
        self._state.reward[done_mask] = 0.0
        self._state.terminated[done_mask] = False
        self._state.truncated[done_mask] = False
        self._state.info["steps"][done_mask] = 0

        return self._state.obs, self._state.info

    # --------------------------------------------------------------------------
    # [新增] 稠密奖励计算逻辑 (Dense Reward with Touch Gate)
    # --------------------------------------------------------------------------
    def _compute_reward(self, data: SceneData) -> Tuple[np.ndarray, Dict[str, Any]]:
        cfg = self._cfg
        B = self._num_envs
        
        # 1. 获取 EE 位置
        ee_pos = self.robot.get_ee_pose(data)[:, :3] # (B, 3)

        # 2. 获取所有按钮位置 (B, 3, 7) -> (B, 3, 3)
        btn_poses = np.stack([np.asarray(b.get_pose(data), dtype=np.float32) for b in self.button_bodies], axis=1)
        btn_pos_xyz = btn_poses[:, :, :3]

        # 3. 确定“当前目标” (Target)
        # 限制 idx 防止越界 (如果完成了就是最后一个)
        cur_idx_clamped = np.clip(self.current_btn_idx, 0, 2)
        
        # 使用 numpy 高级索引提取每个 env 对应的目标位置
        # range(B) 生成 [0, 1, ... B-1]
        # cur_idx_clamped 生成 [0, 0, 1, 2 ...]
        target_pos = btn_pos_xyz[np.arange(B), cur_idx_clamped, :] # (B, 3)

        # 4. 计算距离 (Distance)
        dist = np.linalg.norm(ee_pos - target_pos, axis=1) # (B,)

        # 5. 读取 Touch 传感器 (Touch Gate)
        # 我们只关心“当前目标”的传感器
        # 为了批处理效率，我们读取所有传感器然后 mask
        touch_vals = np.zeros((B,), dtype=np.float32)
        for i in range(3):
            s_name = cfg.button_touch_names[i]
            # read_touch_scalar_safe 返回 (B,)
            val = read_touch_scalar_safe(self.model, data, s_name, B)
            
            # 只取当前目标是 i 的那些 env 的值
            mask = (cur_idx_clamped == i)
            touch_vals[mask] = val[mask]

        is_touched = touch_vals > cfg.touch_threshold
        
        # 6. 更新状态逻辑 (State Transition)
        # 如果当前没有完成 + 距离足够近 + 触发了 Touch -> 视为该阶段完成
        # 注意：这里加入距离判定(dist < 0.05)是为了防止传感器噪声，
        # 或者是防止意外触发了其他按钮（虽然上面 mask 已经过滤了 ID，但双保险更好）
        is_success_now = is_touched & (dist < 0.05) & (~self.btn_pressed_mask[np.arange(B), cur_idx_clamped])
        
        # 更新 mask 和 index
        if np.any(is_success_now):
            # 标记完成
            env_ids = np.where(is_success_now)[0]
            # 更新 mask
            self.btn_pressed_mask[env_ids, cur_idx_clamped[env_ids]] = True
            # 推进 index
            self.current_btn_idx[env_ids] += 1

        # 7. 计算总奖励 (Composite Reward)
        
        # A. 进度奖励 (Stage Reward): 已经完成的按钮数量 * 固定分值
        # 这鼓励 Agent 保持在高完成度的状态
        num_completed = np.sum(self.btn_pressed_mask, axis=1)
        r_stage = num_completed * cfg.stage_complete_reward

        # B. 距离奖励 (Reach Reward): 仅针对“当前未完成”的目标
        # 使用 tanh 核函数将距离映射到 [0, 1]，距离越近奖励越高
        # 且仅当任务未全部完成时给予
        all_done = (num_completed >= 3)
        r_reach = cfg.dist_reward_scale * (1.0 - np.tanh(5.0 * dist))
        r_reach[all_done] = 1.0 # 如果全完成了，给满距离分（保持在终点）

        # C. 触碰奖励 (Touch Reward): 瞬间奖励
        r_touch = is_touched.astype(np.float32) * cfg.touch_reward_bonus

        total_reward = r_stage + r_reach + r_touch

        info = {
            "dist": dist,
            "touch_val": touch_vals,
            "cur_idx": self.current_btn_idx,
            "completed": num_completed
        }
        
        return total_reward, info

    def _build_obs(self, data: SceneData) -> Dict[str, np.ndarray]:
        robot_obs = self.robot.get_obs(data)
        obs_pix = self._render_pixels(data)

        instruction = str(self._cfg.instruction).strip()
        prompt = str(self._cfg.prompt_template).format(task_description=instruction)
        
        B = data.shape[0] if data is not None else self._num_envs
        prompts = np.array([prompt] * B, dtype=object)
        
        # [Obs] 加入 target_idx 方便 Agent 知道当前该按哪个
        # (B, 1)
        target_idx_obs = self.current_btn_idx.reshape(B, 1)

        obs_dict = {
            **obs_pix,
            **robot_obs,
            "target_btn_idx": target_idx_obs,
            "prompt": prompts,
        }
        return obs_dict