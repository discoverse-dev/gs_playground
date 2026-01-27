from typing import Tuple

import numpy as np
import gymnasium as gym
import mujoco

from gs_playground.src.env import registry
from gs_playground.src.env.mujoco_env.mj_env import MjNpEnv, MjNpEnvState
from gs_playground.src.manipulation.tasks.eef.cfg import FrankaCartesianBaseCfg, RewardConfig

from dataclasses import dataclass, field

@registry.envcfg("franka-pick-cartesian")
@dataclass
class FrankaPickCartesianCfg(FrankaCartesianBaseCfg):
    reward_config: RewardConfig = field(
        default_factory=lambda: RewardConfig(
            scales={
                # Gripper goes to the box.
                "gripper_box": 4.0,
                # Box goes to the target mocap.
                "box_target": 8.0,
                # Do not collide the gripper with the floor.
                "no_floor_collision": 5.0,
                
                "action_rate": -0.0005,
                "lifted_reward": 1.5,
                "success_reward": 20.0,
                "gripper_ctrl": 3.0,
            }
        )
    )

@registry.env("franka-pick-cartesian", sim_backend="mujoco")
class FrankaPickCartesian(MjNpEnv):
    def __init__(self, cfg: FrankaPickCartesianCfg, num_envs=1):
        super().__init__(cfg, num_envs)
        
        self.nq = self._model.nq
        self.nv = self._model.nv
        
        # Offsets for qpos and qvel in physics_state
        # mjSTATE_FULLPHYSICS layout: time(1) | qpos(nq) | qvel(nv) | act(na) | plugin
        # Note: warmstart is passed separately in rollout, not part of state vector here.
        self._idx_qpos = 1 
        self._idx_qvel = 1 + self.nq
        
        # Initialize spaces
        self._init_action_space()
        self._num_action = self._action_space.shape[0]
        self._init_obs_space()
        self._num_observation = self._observation_space.shape[0]
        
        # Use keyframe for init
        key_id = self._model.key("home").id
        self._init_qpos = self._model.key_qpos[key_id].copy()
        
        # Identify key qpos indices based on model
        # Robotiq mocap model:  
        # Root Translate (0-2), Root Rotate (3-5), Gripper Joints (6-11), 
        # Box Free Joint (12-18), target joints (19-21)
        
        # Let's find body IDs
        self._obj_body_id = self._model.body("box").id
        self._gripper_site_id = self._model.site("gripper").id
        self._mocap_target_id = self._model.body("mocap_target").mocapid[0]
        
        self._gripper_site_name = "gripper"
        self._obj_body_name = "box"
        self._mocap_target_name = "mocap_target"

        # Qpos Map:
        # root x,y,z (slide): 0, 1, 2
        # root rot z,y,x (hinge): 3, 4, 5
        # left_driver_joint: 6
        # right_driver_joint: 7 (and other mimics)
        # box: 8...
        
        # Actuator Map:
        # act_root_x, act_root_y, act_root_z
        # act_root_rot_z, act_root_rot_y, act_root_rot_x
        # fingers_actuator
        
        # We only want to control Y, Z and gripper.
        # Fixed axes: X, Rot Z, Rot Y, Rot X.
        
        self._action_scale = cfg.control_config.action_scale

        self._init_sensor_indices()
        self._init_reward_functions()
        
        # Cache for action history if needed (not implemented here for simplicity unless required)

    def _init_reward_functions(self):
        self._reward_fns = {
            "gripper_box": self._reward_gripper_box,
            "box_target": self._reward_box_target,
            "no_floor_collision": self._reward_no_floor_collision,
            "lifted_reward": self._reward_lifted,
            "success_reward": self._reward_success,
            "gripper_ctrl": self._reward_gripper_ctrl,
        }

    def _init_sensor_indices(self):
        super()._init_sensor_indices()
        
        self.idx_root_y_pos = self._get_sensor_slice("root_y_pos")
        self.idx_root_z_pos = self._get_sensor_slice("root_z_pos")
        
        self.idx_global_gripper_pos = self._get_sensor_slice("global_gripper_pos")
        self.idx_global_gripper_quat = self._get_sensor_slice("global_gripper_quat")
        
        self.idx_box_pos = self._get_sensor_slice("box_pos")
        self.idx_box_quat = self._get_sensor_slice("box_quat")
        
        self.idx_mocap_target_pos = self._get_sensor_slice("mocap_target_pos")
        self.idx_mocap_target_quat = self._get_sensor_slice("mocap_target_quat")
        
        self.idx_left_finger_pad = self._get_sensor_slice("left_finger_pad_floor_found")
        self.idx_right_finger_pad = self._get_sensor_slice("right_finger_pad_floor_found")
        self.idx_box_zaxis = self._get_sensor_slice("box_zaxis")

    def _get_sensor_slice(self, name):
        idx = self.sensor_indices[name]
        adr = self._model.sensor_adr[idx]
        dim = self._model.sensor_dim[idx]
        return slice(adr, adr + dim)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def _init_action_space(self):
        # Action: [dy, dz, gripper_ctrl]
        self._action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

    def _init_obs_space(self):
        # We will expose:
        # P: 3 (gripper pos)
        # V: 3 (gripper lin vel)
        # G: 1 (open amount)
        # Box pos: 3
        # Box quat: 4
        # Target pos: 3 (from info/mocap)
        # Relative pos: 3 (gripper - box)
        # Relative pos: 3 (box - target)
        
        obs_dim = 3 + 3 + 1 + 3 + 4 + 3 + 3 + 3
        
        low = np.full((obs_dim,), -1.0, dtype=np.float32)
        high = np.full((obs_dim,), 1.0, dtype=np.float32)

        # Gripper strict limits [0, 0.9] from xml
        # Index breakdown:
        # 0-2: gripper_pos
        # 3-5: qvel
        # 6: gripper_state
        low[6] = 0.0
        high[6] = 0.9

        self._observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )

    def apply_action(self, actions: np.ndarray, state: MjNpEnvState) -> np.ndarray:
        # actions: (num_envs, 3) -> [dy, dz, gripper]
        
        # Current control from state?
        # In MjNpEnv, state.ctrl is what we write TO.
        # But we need "current" value to increment?
        # Or we treat action as delta to current pos?
        # The base class rollout logic is: initial_state -> rollout -> state_traj.
        # state.ctrl is applied for the step. 
        # If we want position control, we need to know where we ARE.
        
        # Extract current qpos from sensors because physics_state mapping is complex
        qpos_y = state.sensor_data[:, self.idx_root_y_pos].flatten()
        qpos_z = state.sensor_data[:, self.idx_root_z_pos].flatten()
        
        # Desired new positions
        # Scale actions
        dy = actions[:, 0] * self._action_scale
        dz = actions[:, 1] * self._action_scale
        gripper = actions[:, 2] # Usually 0-255 or 0-1.
        # Gripper in xml has ctrlrange="0 0.82"
        # Let's map [-1, 1] to [0, 0.82]
        gripper_ctrl = (gripper + 1) / 2 * 0.82
        
        # Constrain gripper
        gripper_ctrl = np.clip(gripper_ctrl, 0, 0.82)

        # Build full ctrl vector (B, 7)
        # Actuator Map:
        # act_root_x, act_root_y, act_root_z
        # act_root_rot_z, act_root_rot_y, act_root_rot_x
        # fingers_actuator
        new_ctrl = np.zeros((self._num_envs, 7), dtype=np.float64)
        
        # Set fixed values for X and Rotations
        # X target is 0.65 from xml
        new_ctrl[:, 0] = 0.65 
        # Rotations 0
        new_ctrl[:, 3] = 0
        new_ctrl[:, 4] = 0
        new_ctrl[:, 5] = 0
        
        # Updated Y, Z
        target_y = qpos_y + dy
        target_z = qpos_z + dz
        
        new_ctrl[:, 1] = target_y
        new_ctrl[:, 2] = target_z
        new_ctrl[:, 6] = gripper_ctrl
        
        return new_ctrl

    def update_state(self, state: MjNpEnvState, obs_required=True) -> MjNpEnvState:
        self._update_cache(state)
        
        # Obs
        if obs_required:
            obs = self._compute_obs(state)
            state.obs[:] = obs
        
        state = self.update_terminated(state)
        state = self.update_reward(state)
                
        return state
    
    def _update_cache(self, state: MjNpEnvState):
        gripper_pos = state.sensor_data[:, self.idx_global_gripper_pos]
        box_zaxis = state.sensor_data[:, self.idx_box_zaxis]
        box_pos = state.sensor_data[:, self.idx_box_pos]
        target_pos = state.sensor_data[:, self.idx_mocap_target_pos]
        
        # Distances
        box_target_dist = np.linalg.norm(target_pos - box_pos, axis=1)
        gripper_box_dist = np.linalg.norm(box_pos - gripper_pos, axis=1)
        
        state.info["gripper_pos"] = gripper_pos
        state.info["box_zaxis"] = box_zaxis
        state.info["box_pos"] = box_pos
        state.info["box_target_dist"] = box_target_dist
        state.info["gripper_box_dist"] = gripper_box_dist
        state.info["pos_err"] = box_target_dist
        state.info["is_lifted"] = (box_pos[:, 2] > 0.04)
        
        # Success
        success = box_target_dist < self._cfg.reward_config.success_threshold
        state.info["success"] = success.astype(float)

    def update_terminated(self, state: MjNpEnvState) -> MjNpEnvState:
        box_pos = state.info["box_pos"]
        box_zaxis = state.info["box_zaxis"]
        
        out_of_bounds = (np.abs(box_pos[:, 0] - 0.65) > 0.2) | (np.abs(box_pos[:, 1]) > 0.2) | (box_pos[:, 2] < 0.0)
        
        # Terminate if box z-axis (up vector) is less than 0.5 (tilted too much)
        bad_orientation = box_zaxis[:, 2] < 0.5
        
        done = out_of_bounds | bad_orientation
        
        return state.replace(terminated=done)

    def update_reward(self, state: MjNpEnvState) -> MjNpEnvState:
        total_reward = np.zeros(self._num_envs, dtype=np.float32)
        scales = self._cfg.reward_config.scales
        log = {}
        
        for name, scale in scales.items():
            if name in self._reward_fns:
                rew = self._reward_fns[name](state)
                total_reward += rew * scale
                log[name] = np.mean(rew * scale)
        
        # Extra logs
        log["pos_err"] = np.mean(state.info["pos_err"])
        log["success_rate"] = np.mean(state.info["success"])
        
        state.info["log"] = log
        return state.replace(reward=total_reward)

    def _reward_gripper_box(self, state):
        dist = state.info["gripper_box_dist"]
        return (1.0 - np.tanh(15.0 * dist))

    def _reward_box_target(self, state):
        dist = state.info["box_target_dist"]
        return (1.0 - np.tanh(10.0 * dist)) * state.info["is_lifted"]

    def _reward_no_floor_collision(self, state):
        c1 = state.sensor_data[:, self.idx_left_finger_pad]
        c2 = state.sensor_data[:, self.idx_right_finger_pad]
        contact_floor = (c1 > 0.5) | (c2 > 0.5)
        return -1.0 * contact_floor.astype(float).flatten()

    def _reward_success(self, state):
        return state.info["success"]

    def _reward_lifted(self, state):
        box_pos = state.info["box_pos"]
        target_pos = state.sensor_data[:, self.idx_mocap_target_pos]
        
        z_dist = np.abs(box_pos[:, 2] - target_pos[:, 2])
        
        return (1.0 - np.tanh(5.0 * z_dist)) * state.info["is_lifted"]

    def _reward_gripper_ctrl(self, state):
        gripper_box_dist = state.info["gripper_box_dist"]
        # gripper ctrl (actuator 6)
        gripper_ctrl = state.ctrl[:, 6]
        # reward if near box, proportional to gripper strength (closing)
        return (gripper_box_dist < 0.02) * (gripper_ctrl > 0.4)

    def _compute_obs(self, state: MjNpEnvState) -> np.ndarray:
        gripper_pos = state.sensor_data[:, self.idx_global_gripper_pos]
        # gripper_quat = state.sensor_data[:, self.idx_global_gripper_quat]
        
        box_pos = state.sensor_data[:, self.idx_box_pos]
        box_quat = state.sensor_data[:, self.idx_box_quat]
        
        mocap_target_pos = state.sensor_data[:, self.idx_mocap_target_pos]
        # mocap_target_quat = state.sensor_data[:, self.idx_mocap_target_quat]

        # Velocity? We have qvel in physics_state.
        # qvel 0-2 are root vels.
        qvel = state.physics_state[:, self._idx_qvel : self._idx_qvel + 3] # Approx root linvel
        
        # Gripper state (finger joint)
        # qpos[6] is left_driver_joint
        gripper_state = state.physics_state[:, self._idx_qpos + 6 : self._idx_qpos + 7]

        obs_list = [
            gripper_pos,
            qvel,
            gripper_state,
            box_pos,
            box_quat,
            mocap_target_pos,
            gripper_pos - box_pos,
            box_pos - mocap_target_pos
        ]
        
        return np.concatenate(obs_list, axis=1)

    def reset(self, env_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        # Reset physics state for these envs
        # We need to construct new state vectors.
        # self._init_qpos is (nq,)
        
        num_resets = len(env_indices)
        if num_resets == 0:
            return None, None, {}

        # Default qpos
        qpos = np.tile(self._init_qpos, (num_resets, 1))
        qvel = np.zeros((num_resets, self.nv))
        
        # Determine randomization base position
        # In pick.py, they add random noise to `self._init_obj_pos`. 
        # Here we use the box position from self._init_qpos as base.
        
        # Find addr
        box_body_id = self._model.body("box").id
        box_qpos_adr = self._model.jnt_qposadr[self._model.body("box").jntadr[0]]
        
        # Get init pos from qpos (assuming keyframe loaded it)
        # In XML, box pos is 0.65 0 0.03.
        init_box_pos = self._init_qpos[box_qpos_adr : box_qpos_adr + 3]

        rng = np.random.default_rng()
        
        # Randomize Box
        box_random_min = np.array([0.0, -0.15, 0.0])
        box_random_max = np.array([0.0, 0.15, 0.0])
        box_random = rng.uniform(box_random_min, box_random_max, size=(num_resets, 3))
        
        qpos[:, box_qpos_adr : box_qpos_adr + 3] = init_box_pos + box_random
        
        # Randomize Target
        # Note: pick.py uses init_obj_pos as base for TARGET too.
        # X axis is restricted to match gripper control
        target_random_min = np.array([0.0, -0.15, 0.2])
        target_random_max = np.array([0.0, 0.15, 0.3])
        target_random = rng.uniform(target_random_min, target_random_max, size=(num_resets, 3))
        target_pos = init_box_pos + target_random

        # We changed mocap_target to a regular body with slide joints!
        target_body = self._model.body("mocap_target")
        target_jnt_adr = target_body.jntadr[0] # First joint (target_x)
        if target_jnt_adr != -1:
            target_qpos_adr = self._model.jnt_qposadr[target_jnt_adr]
            # Set target pos
            qpos[:, target_qpos_adr : target_qpos_adr + 3] = target_pos
        
        # Construct physics state
        nstate = self.physics_state_dim
        new_states = np.zeros((num_resets, nstate))
        
        # Set qpos (offset 1)
        # mjSTATE_FULLPHYSICS: global time, qpos, qvel, act, plugin
        
        # Time
        new_states[:, 0] = 0.0
        # Qpos
        new_states[:, self._idx_qpos : self._idx_qpos + self.nq] = qpos
        # Qvel
        new_states[:, self._idx_qvel : self._idx_qvel + self.nv] = qvel
        
        # ---------------------------------------------------------
        # Compute valid sensors using a 0-step forward kinematics
        # ---------------------------------------------------------
        sensor_batch = np.zeros((num_resets, self._model.nsensordata), dtype=np.float64)
        mj_data = self._worker_data[0] # Use the first worker as scratchpad

        for i in range(num_resets):
            # Reset worker data
            mj_data.time = 0.0
            mj_data.qpos[:] = qpos[i]
            mj_data.qvel[:] = qvel[i]
            mj_data.ctrl[:] = 0.0 # reset ctrl
            mj_data.qacc[:] = 0.0
            mj_data.qacc_warmstart[:] = 0.0
            
            # Forward kinematics to update sensors (site positions, etc.)
            mujoco.mj_forward(self._model, mj_data)
            
            # Copy sensor data
            sensor_batch[i] = mj_data.sensordata

        # Update environment state with new sensors so other methods use fresh data
        if hasattr(self, "_state") and self._state is not None:
            self._state.sensor_data[env_indices] = sensor_batch

        # temporary state for obs computation
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
        
        # Compute Observation
        obs_batch = self._compute_obs(obs_state)
        
        return new_states, obs_batch, {}

