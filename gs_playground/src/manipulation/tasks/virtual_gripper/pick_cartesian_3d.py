from typing import Tuple

import numpy as np
import gymnasium as gym
import mujoco

from dataclasses import dataclass, field
from gs_playground.src.env import registry
from gs_playground.src.env.mujoco_env.mj_env import MjNpEnv, MjNpEnvState
from gs_playground.src.manipulation.tasks.virtual_gripper.cfg import FrankaCartesianBaseCfg, RewardConfig

from gs_playground.src.manipulation.tasks.virtual_gripper.pick_cartesian import FrankaPickCartesian

@registry.envcfg("franka-pick-cartesian-3d")
@dataclass
class FrankaPick3DCfg(FrankaCartesianBaseCfg):
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
            }
        )
    )

@registry.env("franka-pick-cartesian-3d", sim_backend="mujoco")
class FrankaPickCartesian3D(FrankaPickCartesian):
    def __init__(self, cfg: FrankaPick3DCfg, num_envs=1):
        super().__init__(cfg, num_envs)
        
        # Override sensor indices for X axis
        self.idx_root_x_pos = self._get_sensor_slice("root_x_pos")
    
    def _init_action_space(self):
        # Action: [dx, dy, dz, gripper_ctrl]
        # We add one dimension for X control
        self._action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

    def apply_action(self, actions: np.ndarray, state: MjNpEnvState) -> np.ndarray:
        # actions: (num_envs, 4) -> [dx, dy, dz, gripper]
        
        # Extract current qpos from sensors
        qpos_x = state.sensor_data[:, self.idx_root_x_pos].flatten()
        qpos_y = state.sensor_data[:, self.idx_root_y_pos].flatten()
        qpos_z = state.sensor_data[:, self.idx_root_z_pos].flatten()
        
        # Scale actions
        dx = actions[:, 0] * self._action_scale
        dy = actions[:, 1] * self._action_scale
        dz = actions[:, 2] * self._action_scale
        gripper = actions[:, 3]
        
        # Map Gripper [-1, 1] -> [0, 0.82]
        gripper_ctrl = (gripper + 1) / 2 * 0.82
        gripper_ctrl = np.clip(gripper_ctrl, 0, 0.82)

        # Build full ctrl vector (B, 7)
        # Actuator Map:
        # act_root_x, act_root_y, act_root_z
        # act_root_rot_z, act_root_rot_y, act_root_rot_x
        # fingers_actuator
        new_ctrl = np.zeros((self._num_envs, 7), dtype=np.float64)
        
        # Update X, Y, Z
        target_x = qpos_x + dx
        target_y = qpos_y + dy
        target_z = qpos_z + dz
        
        # Set fixed values for Rotations (Indices 3, 4, 5)
        new_ctrl[:, 3] = 0
        new_ctrl[:, 4] = 0
        new_ctrl[:, 5] = 0
        
        # Assign targets
        new_ctrl[:, 0] = target_x
        new_ctrl[:, 1] = target_y
        new_ctrl[:, 2] = target_z
        new_ctrl[:, 6] = gripper_ctrl
        
        return new_ctrl

    def reset(self, env_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        # Call base reset to handle physics state structure helpers
        # But we need to reimplement randomization for 3D
        
        num_resets = len(env_indices)
        if num_resets == 0:
            return None, None, {}

        # Default qpos from home keyframe
        qpos = np.tile(self._init_qpos, (num_resets, 1))
        qvel = np.zeros((num_resets, self.nv))
        
        # Find addr
        box_body_id = self._model.body("box").id
        box_qpos_adr = self._model.jnt_qposadr[self._model.body("box").jntadr[0]]
        
        # Init box pos from self._init_qpos (0.65, 0, 0.03)
        init_box_pos = self._init_qpos[box_qpos_adr : box_qpos_adr + 3]

        rng = np.random.default_rng()
        
        # 1. Randomize Box (XYZ randomization)
        # Assuming robot can reach approx X in [0.4, 0.9], Y in [-0.4, 0.4]
        # Let's keep it safe:
        # X: +/- 0.15 around 0.65 (0.5 to 0.8)
        # Y: +/- 0.15 around 0.0
        box_random_min = np.array([-0.15, -0.15, 0.0])
        box_random_max = np.array([0.15, 0.15, 0.0])
        box_random = rng.uniform(box_random_min, box_random_max, size=(num_resets, 3))
        
        curr_box_pos = init_box_pos + box_random
        qpos[:, box_qpos_adr : box_qpos_adr + 3] = curr_box_pos
        
        # 2. Randomize Target (Relative to Box Base or Absolute)
        # Randomize offset from box init position, but with different distribution
        # Target should be above box potentially
        # X: +/- 0.15
        # Y: +/- 0.15 
        # Z: 0.2 to 0.3
        target_random_min = np.array([-0.15, -0.15, 0.2])
        target_random_max = np.array([0.15, 0.15, 0.3])
        target_random = rng.uniform(target_random_min, target_random_max, size=(num_resets, 3))
        
        # Target pos relative to center (init_box_pos)
        target_pos = init_box_pos + target_random

        # Update Target Body Joint
        target_body = self._model.body("mocap_target")
        target_jnt_adr = target_body.jntadr[0] # First joint (target_x)
        if target_jnt_adr != -1:
            target_qpos_adr = self._model.jnt_qposadr[target_jnt_adr]
            qpos[:, target_qpos_adr : target_qpos_adr + 3] = target_pos
        
        # Construct physics state
        nstate = self.physics_state_dim
        new_states = np.zeros((num_resets, nstate))
        
        # Time
        new_states[:, 0] = 0.0
        # Qpos
        new_states[:, self._idx_qpos : self._idx_qpos + self.nq] = qpos
        # Qvel
        new_states[:, self._idx_qvel : self._idx_qvel + self.nv] = qvel
        
        # Forward Kinematics for sensors
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
        
        obs_batch = self._compute_obs(obs_state)
        
        return new_states, obs_batch, {}
