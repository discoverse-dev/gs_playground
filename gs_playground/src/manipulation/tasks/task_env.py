from __future__ import annotations

import abc

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import motrixsim as mtx
import numpy as np
import torch
from motrixsim import SceneData, forward_kinematic

from gs_playground import ROOT_PATH
from gs_playground.src.env.motrix_env.render_env import RenderEnvCfg, NpRenderEnv, RenderEnvState

from gs_playground.src.manipulation.robots.universal_robots_ur5e_robotiq.ur5e_robotiq import UR5eRobotiq
from gs_playground.src.manipulation.robots.franka_emika_panda_robotiq.franka_robotiq import FrankaRobotiq

ASSETS_UR5E_DIR = ROOT_PATH / "models" / "robots" / "manipulation" / "universal_robots_ur5e_robotiq"

@dataclass
class TaskEnvCfg(RenderEnvCfg):
    # control
    ctrl_dt: float = 0.02
    max_episode_steps: int = 500
    action_mode: str = "joint"  # "joint" or "eef"

    # observation / prompt
    instruction: str = ""

    # robot
    robot_name: str = ""

class TaskEnv(NpRenderEnv):
    def __init__(self, cfg: TaskEnvCfg, num_envs: int = 32):
        super().__init__(cfg, num_envs=num_envs)
        # Shared RNG for task-specific randomization
        self._rng = np.random.default_rng()

        # 1. Initialize Robot Helper
        if cfg.robot_name == "ur5e_robotiq":
            Robot = UR5eRobotiq
        elif cfg.robot_name == "franka_robotiq":
            Robot = FrankaRobotiq
        else:
            raise ValueError(f"Unknown robot_name: {cfg.robot_name}")

        self.robot = Robot(self.model)
        # 4. Init Renderer (3DGS)
        self.gauss_dict = Robot.robot_gaussians()
        if cfg.gs_robot_gaussians:
            self.gauss_dict.update(cfg.gs_robot_gaussians)
        # Allow task to supply extra 3DGS assets before renderer is initialized
        self.gauss_dict.update(self.task_gaussians())
        self.gauss_bg = cfg.gs_background_ply.strip() or Robot.robot_background_ply()
        self.init_renderer(body_gaussians=self.gauss_dict, background_ply=self.gauss_bg)

    # ---- Task extension hooks ----
    def task_gaussians(self) -> Dict[str, str]:
        """Optional: tasks can override to supply extra gaussian assets."""
        return {}

    def _randomize(self, data: SceneData, done_mask: np.ndarray, phase: str = "reset"):
        """
        Optional per-task randomization hook.

        Args:
            data: SceneData view corresponding to the environments being reset.
            done_mask: boolean array over ALL envs indicating which envs are being reset.
            phase: string hint ("reset", "auto_reset") for subclass logic.
        """
        return

    # ---- ABEnv props ----
    @property
    def num_dof_arm(self) -> int:
        return self.robot.num_dof_arm

    @property
    def observation_space(self) -> gym.Space:
        cam_spaces = {}
        for i, _ in enumerate(self.cam_ids):
            cam_spaces[f"pixels/view_{i}"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self._img_h, self._img_w, 3),
                dtype=np.uint8,
            )
        obs_spaces: Dict[str, gym.Space] = {
            **cam_spaces,
            "qpos": gym.spaces.Box(low=self.model.joint_limits[0, :self.num_dof_arm], high=self.model.joint_limits[1, :self.num_dof_arm], dtype=np.float32),
            "gripper": gym.spaces.Box(low=self.model.joint_limits[0, self.num_dof_arm:self.num_dof_arm+1], high=self.model.joint_limits[1, self.num_dof_arm:self.num_dof_arm+1], dtype=np.float32),
            "ee_pose": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        }
        return gym.spaces.Dict(obs_spaces)

    @property
    def action_space(self) -> gym.Space:
        return self.robot.action_space

    # ---- NpEnv required ----
    def init_state(self) -> RenderEnvState:
        data = SceneData(self._model, batch=[self._num_envs])

        # Initialize obs structure from observation_space (supports text/object spaces)
        obs_example = self.observation_space
        obs: Dict[str, np.ndarray] = {}
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

        # Initial reset
        self._reset_done_envs()

        self._state.validate()
        return self._state

    def update_state(self, state: RenderEnvState, obs_required: bool = True) -> RenderEnvState:
        # Compute reward and write info in-place to avoid extra dict copies
        reward = self._compute_reward(state)
        terminated = state.terminated.copy()
        # Tasks can latch success flags inside _compute_reward; prefer that over recomputation here
        if hasattr(self, "success_latched"):
            terminated = np.asarray(self.success_latched).copy()

        if obs_required:
            obs = self._build_obs(state.data)
            state.obs = obs

        state.reward = reward.astype(np.float32)
        state.terminated = terminated
        return state

    def apply_action(self, actions: np.ndarray, state) -> mtx.SceneData:
        self.robot.apply_action(state.data, actions, action_mode=self.cfg.action_mode)
        return state

    def _before_chunk_step(self, data: mtx.SceneData):
        """Update robot reference state for relative control at the start of a chunk."""
        self.robot.update_reference(data)

    @abc.abstractmethod
    def _reset_task_state(self, done: np.ndarray):
        """Reset internal task state variables for done environments."""

    def _reset_done_envs(self):
        """Automatic reset called by step() for DONE environments."""
        done = self._state.terminated | self._state.truncated
        if not np.any(done):
            return

        # 1. Reset Task State
        self._reset_task_state(done)
        
        # 2. Physics Reset (apply keyframe)
        self._apply_keyframe(self._state.data[done])
        forward_kinematic(self.model, self._state.data[done])

        # 2b. Task/domain randomization hook (after keyframe, before robot reset)
        self._randomize(self._state.data[done], done_mask=done, phase="auto_reset")

        # 2b. Update BG cache for done envs if renderer has background
        if self._bg_renderer is not None:
            data_subset = self._state.data[done]
            body_pos_sub, body_quat_sub = self._compute_body_arrays(data_subset)
            cam_xpos_sub, cam_xmat_sub, fovy_sub = self._compute_camera_arrays(data_subset)
            bg_gsb_sub = self._bg_renderer.batch_update_gaussians(body_pos_sub, body_quat_sub)
            bg_imgs_sub, _ = self._bg_renderer.batch_env_render(
                bg_gsb_sub, cam_xpos_sub, cam_xmat_sub, self._img_h, self._img_w, fovy_sub
            )

            if self._bg_imgs is None:
                B = int(self._num_envs)
                self._bg_imgs = torch.zeros(
                    (B,) + tuple(bg_imgs_sub.shape[1:]),
                    dtype=bg_imgs_sub.dtype,
                    device=bg_imgs_sub.device,
                )
            mask_t = torch.from_numpy(done.astype(np.bool_)).to(self._bg_imgs.device)
            self._bg_imgs[mask_t] = bg_imgs_sub

        # 3. Reset Robot Internal State (for relative/delta control)
        self.robot.reset_envs(self._state.data, done)

        # 4. Observation Update
        # Build obs for SUBSET
        obs_subset = self._build_obs(self._state.data[done])
        
        # Assign to state.obs (Element-wise for Dict)
        for k, v in obs_subset.items():
            self._state.obs[k][done] = v
            
        # 5. Reset flags
        self._state.reward[done] = 0.0
        self._state.terminated[done] = False
        self._state.truncated[done] = False
        self._state.info["steps"][done] = 0

    def reset(self, data: SceneData = None, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        """
        Public Reset.
        If data provided (internal use), just obs.
        If data None (external use), full reset logic.
        """
        # Case 1: Internal callback (subset) - Legacy support if super called
        if data is not None:
            self._apply_keyframe(data)
            forward_kinematic(self.model, data)
            obs = self._build_obs(data)
            return obs, {}

        # Case 2: Public API
        if self._state is None:
            self.init_state()
            
        if done is None:
            done_mask = np.ones((self._num_envs,), dtype=bool)
        else:
            done_mask = np.asarray(done, dtype=bool)
            
        if not np.any(done_mask):
             return self._state.obs, self._state.info

        # Logic for public reset
        self._reset_task_state(done_mask)
        
        self._apply_keyframe(self._state.data[done_mask])
        forward_kinematic(self.model, self._state.data[done_mask])

        # Task/domain randomization hook (after keyframe, before robot reset)
        self._randomize(self._state.data[done_mask], done_mask=done_mask, phase="reset")

        # Reset Robot Internal State
        self.robot.reset_envs(self._state.data, done_mask)
        
        # Update obs with subset
        # We need to update state.obs, not just return new obs
        obs_new_all = self._build_obs(self._state.data) # Simplest: rebuild all. Or rebuild subset.
        # Rebuilding all ensures consistency but is slower. 
        # For correctness of return value:
        
        self._state.obs = obs_new_all
        self._state.reward[done_mask] = 0.0
        self._state.terminated[done_mask] = False
        self._state.truncated[done_mask] = False
        self._state.info["steps"][done_mask] = 0

        return self._state.obs, self._state.info

    # ---- helpers ----
    @abc.abstractmethod
    def _compute_reward(self, state: RenderEnvState) -> np.ndarray:
        """Compute reward given full state; write aux metrics into state.info in-place."""

    def _build_obs(self, data: SceneData) -> Dict[str, np.ndarray]:
        obs_pix = self._render_pixels(data)
        robot_obs = self.robot.get_obs(data)
        obs_dict = {
            **obs_pix,
            **robot_obs,
        }
        return obs_dict
