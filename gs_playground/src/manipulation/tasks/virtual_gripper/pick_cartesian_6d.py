from typing import Tuple

import numpy as np
import gymnasium as gym
import mujoco
from scipy.spatial.transform import Rotation as R

from dataclasses import dataclass, field
from gs_playground.src.env import registry
from gs_playground.src.env.mujoco_env.mj_env import MjNpEnv, MjNpEnvState
from gs_playground.src.manipulation.tasks.virtual_gripper.cfg import FrankaCartesianBaseCfg, RewardConfig

from gs_playground.src.manipulation.tasks.virtual_gripper.pick_cartesian_3d import FrankaPickCartesian3D

@registry.envcfg("franka-pick-cartesian-6d")
@dataclass
class FrankaPick6DCfg(FrankaCartesianBaseCfg):
    reward_config: RewardConfig = field(
        default_factory=lambda: RewardConfig(
            scales={
                "gripper_box": 4.0,
                "box_target": 8.0,
                "no_floor_collision": 5.0,
                "action_rate": -0.0005,
                "lifted_reward": 1.5,
                "success_reward": 20.0,
                "gripper_ctrl": 3.0,
                # New orientation rewards
                "box_target_rot": 2.0, # Alignment of box to target
            }
        )
    )

@registry.env("franka-pick-cartesian-6d", sim_backend="mujoco")
class FrankaPickCartesian6D(FrankaPickCartesian3D):
    def __init__(self, cfg: FrankaPick6DCfg, num_envs=1):
        super().__init__(cfg, num_envs)
        
        # We need to store target quats because the physics body might not support rotation
        self._target_quats = np.zeros((num_envs, 4))
        self._target_quats[:, 0] = 1.0 

    def _init_action_space(self):
        # Action: [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper_ctrl]
        # 7 dimensions
        self._action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

    def _init_obs_space(self):
        # Based on user's _compute_obs:
        # gripper_pos (3)
        # gripper_quat (4)
        # qvel (6)
        # gripper_state (1)
        # box_pos (3)
        # box_quat (4)
        # mocap_target_pos (3)
        # mocap_target_quat (4)
        
        # obs_dim = 3 + 4 + 6 + 1 + 3 + 4 + 3 + 4 # 28
        obs_dim = 3 + 4 + 6 + 1 + 3 + 4 + 3 + 4 + 3 + 4 # 35
        
        low = np.full((obs_dim,), -1.0, dtype=np.float32)
        high = np.full((obs_dim,), 1.0, dtype=np.float32)

        # Gripper state is now at index 3+4+6 = 13
        # 0-2 (3): pos
        # 3-6 (4): quat
        # 7-12 (6): vel
        # 13 (1): gripper state
        low[13] = 0.0
        high[13] = 0.9

        self._observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )

    def apply_action(self, actions: np.ndarray, state: MjNpEnvState) -> np.ndarray:
        # actions: (B, 7) -> [dx, dy, dz, d_ax, d_ay, d_az, gripper]
        
        # Pos Control (using indices from base classes)
        # We can use sensors for position (from Pick3D)
        curr_x = state.sensor_data[:, self.idx_root_x_pos].flatten()
        curr_y = state.sensor_data[:, self.idx_root_y_pos].flatten()
        curr_z = state.sensor_data[:, self.idx_root_z_pos].flatten()
        
        # Rotation Control:
        # We use physics state qpos directly for rotation angles as sensors might be missing
        # qpos structure: [time, qpos(nq), ...]
        # self._idx_qpos is start of qpos
        # Root Rotations indices: 3 (Z), 4 (Y), 5 (X)
        
        idx_qpos = self._idx_qpos
        # Note: qpos indices for root rot are 3,4,5.
        curr_rot_z = state.physics_state[:, idx_qpos + 3]
        curr_rot_y = state.physics_state[:, idx_qpos + 4]
        curr_rot_x = state.physics_state[:, idx_qpos + 5]
        
        scale = self._action_scale
        
        # Actions
        dx = actions[:, 0] * scale
        dy = actions[:, 1] * scale
        dz = actions[:, 2] * scale
        
        # Rotation actions (Assumed mapping: 3->Roll(X), 4->Pitch(Y), 5->Yaw(Z))
        # But actuators are Z(3), Y(4), X(5)
        d_rot_x = actions[:, 3] * scale
        d_rot_y = actions[:, 4] * scale
        d_rot_z = actions[:, 5] * scale
        
        gripper_raw = actions[:, 6]
        gripper_ctrl = (gripper_raw + 1) / 2 * 0.82
        gripper_ctrl = np.clip(gripper_ctrl, 0, 0.82)
        
        # Build Ctrl (B, 7)
        new_ctrl = np.zeros((self._num_envs, 7), dtype=np.float64)
        
        # Pos
        new_ctrl[:, 0] = curr_x + dx
        new_ctrl[:, 1] = curr_y + dy
        new_ctrl[:, 2] = curr_z + dz
        
        # Rot (Indices 3, 4, 5 correspond to Z, Y, X actuators)
        # Note: Actuator 3 drives Joint 3 (Rot Z)
        #       Actuator 4 drives Joint 4 (Rot Y)
        #       Actuator 5 drives Joint 5 (Rot X)
        
        new_ctrl[:, 3] = curr_rot_z + d_rot_z
        new_ctrl[:, 4] = curr_rot_y + d_rot_y
        new_ctrl[:, 5] = curr_rot_x + d_rot_x
        
        new_ctrl[:, 6] = gripper_ctrl
        
        return new_ctrl

    def _compute_obs(self, state: MjNpEnvState) -> np.ndarray:
        gripper_pos = state.sensor_data[:, self.idx_global_gripper_pos]
        gripper_quat = state.sensor_data[:, self.idx_global_gripper_quat]
        
        box_pos = state.sensor_data[:, self.idx_box_pos]
        box_quat = state.sensor_data[:, self.idx_box_quat]
        
        mocap_target_pos = state.sensor_data[:, self.idx_mocap_target_pos]
        mocap_target_quat = state.sensor_data[:, self.idx_mocap_target_quat]

        qvel = state.physics_state[:, self._idx_qvel : self._idx_qvel + 6] # Approx root linvel
        
        # Gripper state (finger joint)
        # qpos[6] is left_driver_joint
        gripper_state = state.physics_state[:, self._idx_qpos + 6 : self._idx_qpos + 7]

        obs_list = [
            gripper_pos,
            gripper_quat,
            qvel,
            gripper_state,
            box_pos,
            box_quat,
            mocap_target_pos,
            mocap_target_quat,
            box_pos - mocap_target_pos,
            box_quat - mocap_target_quat,
        ]
        
        return np.concatenate(obs_list, axis=1)

    def reset(self, env_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        num_resets = len(env_indices)
        if num_resets == 0:
            return None, None, {}

        # 1. Base Reset logic (Position)
        # We can reuse Pick3D logic or reimplement. Reimplementing is cleaner for 6D specific params.
        
        qpos = np.tile(self._init_qpos, (num_resets, 1))
        qvel = np.zeros((num_resets, self.nv))
        
        box_body = self._model.body("box")
        box_qpos_adr = self._model.jnt_qposadr[box_body.jntadr[0]]
        init_box_pos = self._init_qpos[box_qpos_adr : box_qpos_adr + 3] # (0.65, 0, 0.03)

        rng = np.random.default_rng()
        
        # --- Randomize Box ---
        # Pos: X/Y +/- 0.15
        box_random_min = np.array([-0.15, -0.2, 0.0])
        box_random_max = np.array([0.15, 0.2, 0.0])
        box_pos_noise = rng.uniform(box_random_min, box_random_max, size=(num_resets, 3))
        
        # Rot: Z Axis random (full range) -90 to 90 deg
        box_rot_z = rng.uniform(-90, 90, size=num_resets)
        box_euler = np.zeros((num_resets, 3))
        box_euler[:, 2] = box_rot_z
        
        box_quat_mj = R.from_euler('xyz', box_euler, degrees=True).as_quat(scalar_first=True) # wxyz
        
        # Set Box Qpos (Pos + Quat)
        # For free joint, qpos is 7: 3 pos + 4 quat
        qpos[:, box_qpos_adr : box_qpos_adr + 3] = init_box_pos + box_pos_noise
        qpos[:, box_qpos_adr + 3 : box_qpos_adr + 7] = box_quat_mj
        
        # --- Randomize Target ---
        # Angle Random: X/Y in +/- 30 deg, Z in +/- 90 deg
        target_rx = rng.uniform(-30, 30, size=num_resets)
        target_ry = rng.uniform(-30, 30, size=num_resets)
        target_rz = rng.uniform(-90, 90, size=num_resets)
        
        target_euler = np.stack([target_rx, target_ry, target_rz], axis=1)
        target_quat_mj = R.from_euler('zyx', target_euler, degrees=True).as_quat(scalar_first=True) # wxyz
        
        # Save for obs/reward (use xyzw for consistence with some tools, but obs usually wants wxyz or whatever env convention)
        # Mujoco convention is wxyz. Let's stick to wxyz for observation to match box_quat.
        self._target_quats[env_indices] = target_quat_mj
        
        # Pos Random: consistent with 3D
        target_pos_noise = rng.uniform([-0.15, -0.2, 0.2], [0.15, 0.2, 0.3], size=(num_resets, 3))
        target_pos = init_box_pos + target_pos_noise
        
        # Apply to Physics
        target_body = self._model.body("mocap_target")
        target_jnt_adr = target_body.jntadr[0]
        
        # Checking degrees of freedom
        # If target has only 3 joints (slide), we can only set pos.
        # If it has 6 or 7, we set pos+rot.
        # We'll try to set what we can.
        if target_jnt_adr != -1:
            # Check joint type/num
            # Assuming purely slide 3 sequence based on previous info
            # We set position
            target_qpos_adr = self._model.jnt_qposadr[target_jnt_adr]
            qpos[:, target_qpos_adr : target_qpos_adr + 3] = target_pos

        # Construct State
        new_states = np.zeros((num_resets, self.physics_state_dim))
        new_states[:, 0] = 0.0
        new_states[:, self._idx_qpos : self._idx_qpos + self.nq] = qpos
        new_states[:, self._idx_qvel : self._idx_qvel + self.nv] = qvel
        
        # Forward Kin
        sensor_batch = np.zeros((num_resets, self._model.nsensordata), dtype=np.float64)
        mj_data = self._worker_data[0]
        for i in range(num_resets):
            mj_data.time = 0.0
            mj_data.qpos[:] = qpos[i]
            mj_data.qvel[:] = qvel[i]
            mj_data.ctrl[:] = 0.0
            mj_data.qacc[:] = 0.0
            mj_data.qacc_warmstart[:] = 0.0
            mujoco.mj_forward(self._model, mj_data)
            sensor_batch[i] = mj_data.sensordata
            
        if hasattr(self, "_state") and self._state is not None:
             self._state.sensor_data[env_indices] = sensor_batch

        info_dummy = {
            "success": np.zeros(num_resets),
            "pos_err": np.zeros(num_resets),
        }

        obs_state = MjNpEnvState(
            physics_state=new_states,
            sensor_data=sensor_batch,
            obs=None,
            reward=None,
            terminated=None,
            truncated=None,
            ctrl=None,
            info=info_dummy,
        )
        
        # Compute Obs
        # We assume sensors have correct data (including target quat if xml supports it, or it will be fixed)
        obs = self._compute_obs(obs_state)
        
        return new_states, obs, {}

    def _update_cache(self, state: MjNpEnvState):
        super()._update_cache(state)
        
        # Add rotation error to cache
        box_quat = state.sensor_data[:, self.idx_box_quat] # wxyz
        target_quat = self._target_quats # wxyz
        
        dot = np.sum(box_quat * target_quat, axis=1)
        # rot_dist = 1 - abs(dot)  (0 to 1) roughly
        # angle = 2 * arccos(|dot|)
        
        state.info["rot_err_dot"] = np.abs(dot) # 1 is good, 0 is bad
        
    def _init_reward_functions(self):
        super()._init_reward_functions()
        self._reward_fns["box_target_rot"] = self._reward_box_target_rot

    def _reward_box_target_rot(self, state):
        # 1.0 if perfectly aligned, 0 if 90 deg off?
        # Using dot product | <q1, q2> |
        rd = state.info["rot_err_dot"] # [0, 1]
        d = state.info["box_target_dist"]
        
        return (rd * rd) * ((1.0 - np.tanh(5 * d))) * state.info["is_lifted"]

    def update_terminated(self, state: MjNpEnvState) -> MjNpEnvState:
        # Check Box Position (from base)
        base_state = super().update_terminated(state)
        done = base_state.terminated.astype(bool)
        
        # Update Success condition:
        # Must match position AND rotation
        pos_success = (state.info["box_target_dist"] < self._cfg.reward_config.success_threshold)
        
        # Rotation success: dot > 0.95 (approx 36 deg? cos(theta/2) = 0.95 -> theta/2 = 18deg -> theta=36)
        rot_success = (state.info["rot_err_dot"] > 0.9) 
        
        success = pos_success & rot_success
        
        # Update success info
        state.info["success"] = success.astype(float)
       
        # Override base done logic to remove "bad_orientation" generic check
        box_pos = state.info["box_pos"]
        out_of_bounds = (np.abs(box_pos[:, 0] - 0.65) > 0.2) | (np.abs(box_pos[:, 1]) > 0.2) | (box_pos[:, 2] < 0.0) | (box_pos[:, 2] > 0.5)
        
        return state.replace(terminated=out_of_bounds)

