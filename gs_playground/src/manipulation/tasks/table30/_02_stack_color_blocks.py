from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import motrixsim as mtx
import numpy as np
from motrixsim import SceneData, forward_kinematic

from gs_playground import ROOT_PATH
from gs_playground.src.env.motrix_env.render_env import RenderEnvCfg, NpRenderEnv, RenderEnvState
from gs_playground.src.env.registry import envcfg, env

# Robots
from gs_playground.src.manipulation.robots.universal_robots_ur5e_robotiq.ur5e_robotiq import UR5eRobotiq

# Task specific assets (kept relative simplicity)
from gs_playground.src.manipulation.tasks.table30.gaussian_assets import (
    build_task_gaussians,
)

ASSETS_UR5E_DIR = ROOT_PATH / "models" / "robots" / "manipulation" / "universal_robots_ur5e_robotiq"
ASSETS_TASK_DIR = ROOT_PATH / "models" / "tasks" / "table30" / "_02_stack_color_blocks"
TASK_GAUSSIANS = {
    "cube_blue": "3dgs/cube_blue.ply",
    "cube_yellow": "3dgs/cube_yellow.ply",
    "cube_orange": "3dgs/cube_orange.ply",
}
print("[StackColorBlocksEnv] TASK_GAUSSIANS paths:")
for name, rel in TASK_GAUSSIANS.items():
    path = ASSETS_TASK_DIR / rel
    print("  ", name, "->", path, "exists?" , path.exists())


@envcfg("table30/stack_color_blocks")
@dataclass
class StackColorBlocksEnvCfg(RenderEnvCfg):
    # model / sim
    model_file: str = str((ASSETS_UR5E_DIR / "xmls" / "table30_02_stack_color_blocks.xml").as_posix())
    sim_dt: float = 0.005
    ctrl_dt: float = 0.02

    # control
    max_episode_steps: int = 500
    action_mode: str = "joint"  # "joint" or "eef"

    # observation / prompt
    prompt_template: str = "What action should the robot take to {task_description}?"
    instruction: str = "Stack the blue cube on top of the yellow cube."

    # rendering
    img_width: int = 320
    img_height: int = 240
    cam_id: Tuple[int, ...] = (0,)

    # assets
    gs_background_ply: str = ""
    gs_robot_gaussians: Optional[Dict[str, str]] = None

    # task entities
    cube_names: Tuple[str, str, str] = ("cube_blue", "cube_yellow", "cube_orange")
    
    # task params
    success_dist_xy: float = 0.05
    success_delta_z_min: float = 0.02
    success_delta_z_max: float = 0.10
    gripper_close_thresh: float = 0.2
    grasp_dist_thresh: float = 0.03

    # reset
    reset_enabled: bool = True
    reset_keyframe: int | str = 0


@env("table30/stack_color_blocks", "np")
class StackColorBlocksEnv(NpRenderEnv):
    """
    Task: Stack color blocks.
    Robot: UR5e + Robotiq 2F-85.
    Backend: MotrixSim (np).
    """

    def __init__(self, cfg: StackColorBlocksEnvCfg, num_envs: int = 32):
        # ensure camera list aligns with cfg.cam_id
        cfg.cam_id = tuple(cfg.cam_id) if not isinstance(cfg.cam_id, tuple) else cfg.cam_id
        super().__init__(cfg, num_envs=num_envs)
        self._cfg: StackColorBlocksEnvCfg = cfg

        # 1. Initialize Robot Helper
        self.robot = UR5eRobotiq(self.model)

        # 2. Initialize Task Handles
        self.cube_bodies = [self.model.get_body(self.model.get_body_index(n)) for n in cfg.cube_names]

        # 3. State Trackers
        B = self._num_envs
        self.top_idx = np.zeros((B,), dtype=np.int32)
        self.base_idx = np.zeros((B,), dtype=np.int32)
        self.grasp_latched = np.zeros((B,), dtype=bool)
        self.success_latched = np.zeros((B,), dtype=bool)

        # 4. Init Renderer (3DGS)
        gauss = UR5eRobotiq.robot_gaussians()
        gauss.update(build_task_gaussians(ASSETS_TASK_DIR, {k: ASSETS_TASK_DIR / v for k, v in TASK_GAUSSIANS.items()}))
        if cfg.gs_robot_gaussians:
            gauss.update(cfg.gs_robot_gaussians)
        bg = cfg.gs_background_ply.strip() or UR5eRobotiq.robot_background_ply()
        self.init_renderer(body_gaussians=gauss, background_ply=bg, minibatch=self._num_envs)

        # 5. Spaces
        # self._obs_rgb_shape = (cfg.img_height, cfg.img_width, 3)

        # self.data: Optional[SceneData] = None
        self._state = None

    # ---- ABEnv props ----
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
            # Robot obs
            "qpos": gym.spaces.Box(low=self.model.joint_limits[0, :6], high=self.model.joint_limits[1, :6], dtype=np.float32),
            "gripper": gym.spaces.Box(low=self.model.joint_limits[0, 6:7], high=self.model.joint_limits[1, 6:7], dtype=np.float32),
            # EE Pose is now 6D: XYZ + RPY
            "ee_pose": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            # Task obs
            "cube_pose": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self._cfg.cube_names), 7), dtype=np.float32),
            "prompt": gym.spaces.Text(max_length=256),
        }
        return gym.spaces.Dict(obs_spaces)

    @property
    def action_space(self) -> gym.Space:
        return self.robot.action_space

    # ---- NpEnv required ----
    def init_state(self) -> RenderEnvState:
        data = SceneData(self._model, batch=[self._num_envs])
        # self.data = data
        
        # Initialize obs structure
        obs_example = self.observation_space        
        obs = {}
        for k, space in obs_example.items():
            # Prompt is special (Text space doesn't have shape in same way or produces strings)
            if isinstance(space, gym.spaces.Text):
                # Using object array for strings or handled separately
                obs[k] = np.empty((self._num_envs,), dtype=object)
            else:
                # Create zeros for each key based on space shape
                # Space shape is per-env, we need to add batch dim
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

    def apply_action(self, actions: np.ndarray, state) -> mtx.SceneData:
        # Delegate to Robot
        self.robot.apply_action(state.data, actions, action_mode=self._cfg.action_mode)
        return state

    def _before_chunk_step(self, data: mtx.SceneData):
        """Update robot reference state for relative control at the start of a chunk."""
        self.robot.update_reference(data)

    def update_state(self, state, obs_required: bool = True) -> mtx.SceneData:
        # Pass state.data to compute_reward
        reward, info = self._compute_reward(state.data)
        terminated = self.success_latched.copy()

        if obs_required:
            obs = self._build_obs(state.data)
            state.obs = obs

        state.reward = reward.astype(np.float32)
        state.terminated = terminated
        
        # Careful with updating dict info in place if structure differs?
        # NpEnvState definition implies info is dict.
        state.info.update(info)
        return state

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

    def _reset_done_envs(self):
        """Automatic reset called by step() for DONE environments."""
        if self._state is None: return

        done = self._state.terminated | self._state.truncated
        if not np.any(done):
            return

        # 1. Reset Task State
        self._reset_task_state(done)
        
        # 2. Physics Reset (apply keyframe)
        self._apply_keyframe(self._state.data[done])
        forward_kinematic(self.model, self._state.data[done])

        # 3. Reset Robot Internal State (for relative/delta control)
        self.robot.reset_envs(self._state.data, done)

        # 4. Render BG (optional/if needed by renderer cache)
        # NpRenderEnv logic: if we have BG renderer, we might need to update cache if we moved static objects?
        # For now, assuming static BG is fine. If implementation requires calling bg render, access _bg_renderer.
        # But we skip super() to avoid crash.

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
    def _compute_reward(self, data: SceneData) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        B = self._num_envs
        cube_pose = np.stack(
            [np.asarray(b.get_pose(data), dtype=np.float32) for b in self.cube_bodies],
            axis=1,
        )
        
        ee_pos = self.robot.get_ee_pose(data)[:, :3]

        idx = np.arange(B)
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
        return reward.astype(np.float32), {
            "is_success": self.success_latched.copy(),
            "is_grasped": self.grasp_latched.copy(),
            "reach_dist": dist_ee_obj,
            "stack_xy": dist_xy,
            "dz": dz,
        }

    def _build_obs(self, data: SceneData) -> Dict[str, np.ndarray]:
        # 1. Get Robot Obs
        robot_obs = self.robot.get_obs(data)
        
        # 2. Get Task Obs
        cube_pose = np.stack(
            [np.asarray(b.get_pose(data), dtype=np.float32) for b in self.cube_bodies],
            axis=1,
        )
        
        obs_pix = self._render_pixels(data)

        instruction = str(self._cfg.instruction).strip()
        prompt = str(self._cfg.prompt_template).format(task_description=instruction)
        
        # Batch prompt if needed? Text space expects per-env usually?
        # If Gym Text space, usually it's one string or list?
        # For simplicity, if batch size N, we might return list of N strings or object array.
        # But here we just return the string template (broadcasting handled by consumer?)
        # Better: return array of strings
        B = data.shape[0] if data is not None else self._num_envs
        prompts = np.array([prompt] * B, dtype=object)

        obs_dict = {
            **obs_pix,
            **robot_obs,
            "cube_pose": cube_pose,
            "prompt": prompts,
        }
        return obs_dict
