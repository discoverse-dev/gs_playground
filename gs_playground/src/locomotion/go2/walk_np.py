import gymnasium as gym
import mujoco
import numpy as np

from gs_playground.src.env import registry
from gs_playground.src.locomotion.go2.cfg import Go2WalkNpEnvCfg
from gs_playground.src.env.mujoco_env.mj_env import MjNpEnv, MjNpEnvState


## provide quat math utility from motrixsim.
def quat_rotate_inverse(quats, v):
    """
    Rotate a fixed vector v by a list of quaternions using a vectorized approach.
    Computes q^-1 * v * q (Inverse rotation).

    Parameters:
        quats (np.ndarray): Array of quaternions of shape (N, 4). Each quaternion is in [w, x, y, z] format (MuJoCo convention).
        v (np.ndarray): Fixed vector of shape (3,) to be rotated.

    Returns:
        np.ndarray: Array of rotated vectors of shape (N, 3).
    """
    # Normalize the quaternions to ensure they are unit quaternions

    # Extract the scalar (w) and vector (x, y, z) parts of the quaternions
    # MuJoCo uses [w, x, y, z]
    w = quats[:, 0]  # Shape (N,)
    
    # For inverse rotation, we use the conjugate quaternion: [w, -x, -y, -z]
    im = -quats[:, 1:]  # Shape (N, 3)

    # Compute the cross product between the imaginary part of each quaternion and the fixed vector v.
    # np.cross broadcasts v to match each row in im, resulting in an array of shape (N, 3)
    cross_im_v = np.cross(im, v)

    # Compute the intermediate terms for the rotation formula:
    term1 = w[:, np.newaxis] * cross_im_v  # w * cross(im, v)
    term2 = np.cross(im, cross_im_v)  # cross(im, cross(im, v))

    # Apply the rotation formula: v_rot = v + 2 * (term1 + term2)
    v_rotated = v + 2 * (term1 + term2)

    return v_rotated


@registry.env("go2-flat-terrain-walk", sim_backend="mujoco")
class Go2WalkTaskMj(MjNpEnv):
    def __init__(self, cfg: Go2WalkNpEnvCfg, num_envs=1):
        super().__init__(cfg, num_envs)

        self.nq = self._model.nq
        self.nv = self._model.nv
        # Offsets in physics_state (mjSTATE_FULLPHYSICS: time, qpos, qvel, act, qacc_warmstart)
        # assuming time (1), qpos(nq), qvel(nv) ...
        self._idx_qpos = 1
        self._idx_qvel = 1 + self.nq

        self._num_dof_pos = self.nq - 7 # Floating base 7
        self._num_dof_vel = self.nv - 6 # Floating base 6
        
        self._init_action_space()
        self._num_action = self._action_space.shape[0]
        self._init_obs_space()
        self._num_observation = self._observation_space.shape[0]
        
        self._init_dof_vel = np.zeros(
            (self._num_dof_vel,),
            dtype=np.float32,
        )
        # Compute init dof pos from keyframe 0 or qpos0
        # MjModel.qpos0 contains default position
        self._init_qpos = self._model.qpos0.copy()
        
        self._init_buffer()
        self._init_sensor_indices()
        self._init_reward_functions()

    def _init_reward_functions(self):
        """Register reward functions with standardized signature (state -> term)."""
        self._reward_fns = {
            "lin_vel_z": self._reward_lin_vel_z,
            "ang_vel_xy": self._reward_ang_vel_xy,
            "orientation": self._reward_orientation,
            "torques": self._reward_torques,
            "dof_vel": self._reward_dof_vel,
            "dof_acc": lambda s: self._reward_dof_acc(s, s.info),
            "action_rate": lambda s: self._reward_action_rate(s.info),
            "tracking_lin_vel": lambda s: self._reward_tracking_lin_vel(s, s.info["commands"]),
            "tracking_ang_vel": lambda s: self._reward_tracking_ang_vel(s, s.info["commands"]),
            "stand_still": lambda s: self._reward_stand_still(s, s.info["commands"]),
            "hip_pos": lambda s: self._reward_hip_pos(s, s.info["commands"]),
            "calf_pos": lambda s: self._reward_calf_pos(s, s.info["commands"]),
            "feet_air_time": lambda s: self._reward_feet_air_time(s.info["commands"], s.info),
            "termination": lambda s: self._reward_termination(s.terminated),
        }

    def _init_sensor_indices(self):
        super()._init_sensor_indices()
        
        # Strict match for foot contact sensors
        # We expect exactly 4 sensors named: FL_{foot}_contact, FR_{foot}_contact, RL_{foot}_contact, RR_{foot}_contact
        foot_name = self._cfg.asset.foot_name
        prefixes = ["FL", "FR", "RL", "RR"]
        expected_names = [f"{p}_{foot_name}_contact" for p in prefixes]
        
        self.contact_sensor_indices = []
        for name in expected_names:
            if name not in self.sensor_indices:
                raise ValueError(f"Required contact sensor '{name}' not found in model. Available sensors: {list(self.sensor_indices.keys())}")
            self.contact_sensor_indices.append(self.sensor_indices[name])
            
        print(f"Mapped contact sensors: {expected_names} -> {self.contact_sensor_indices}")

        # Resolve 'local_linvel' and 'gyro'
        # Assuming cfg.sensor.local_linvel is the sensor name string
        self.idx_linvel = self.sensor_indices.get(self._cfg.sensor.local_linvel, -1)
        self.idx_gyro = self.sensor_indices.get(self._cfg.sensor.gyro, -1)


    def _init_obs_space(self):
        # model = self.model
        num_dof_vel = self._num_dof_vel
        num_joint_angle = self._num_dof_pos
        num_linvel = 3
        num_gyro = 3
        num_gravity = 3
        num_actions = self._num_action
        num_command = 3

        num_obs = num_linvel + num_gyro + num_gravity + num_joint_angle + num_dof_vel + num_actions + num_command
        # 3 + 3 + 3 + 12 + 12 + 12 + 3 = 48

        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (num_obs,), dtype=np.float32)

    def _init_action_space(self):
        model = self.model
        # nu = number of actuators
        self._action_space = gym.spaces.Box(
            np.array(model.actuator_ctrlrange[:, 0]),
            np.array(model.actuator_ctrlrange[:, 1]),
            (model.nu,),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    def get_dof_pos(self, state: MjNpEnvState):
        # qpos[7:]
        # Extract qpos from physics_state
        return state.physics_state[:, self._idx_qpos + 7 : self._idx_qpos + self.nq]

    def get_dof_vel(self, state: MjNpEnvState):
        # qvel[6:]
        return state.physics_state[:, self._idx_qvel + 6 : self._idx_qvel + self.nv]

    def _init_buffer(self):
        cfg = self._cfg
        assert isinstance(cfg, Go2WalkNpEnvCfg)
        # init buffers

        self.reset_buf = np.ones(self._num_envs, dtype=bool)
        self.gravity_vec = np.array([0, 0, -1], dtype=np.float32)
        self.commands_scale = np.array(
            (
                [
                    cfg.normalization.lin_vel,
                    cfg.normalization.lin_vel,
                    cfg.normalization.ang_vel,
                ]
            ),
            dtype=np.float32,
        )

        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        self.hip_indices = []
        self.calf_indices = []
        
        # Map default angles from cfg to actuator order
        for i in range(self._model.nu):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if not name: continue
            
            for key, val in cfg.init_state.default_joint_angles.items():
                if key in name:
                    self.default_angles[i] = val
                    
            if "hip" in name:
                self.hip_indices.append(i)
            if "calf" in name:
                self.calf_indices.append(i)
        
        print("Default joint angles:", self.default_angles)
        
        # Initialize init_qpos for reset (update the joint part)
        self._init_qpos[7:] = self.default_angles

        # Foot stuff handle in _init_sensor_indices

    def apply_action(self, actions, state):
        # Update info for rewards
        state.info["last_dof_vel"] = self.get_dof_vel(state)
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions
        
        # Compute control
        ctrl = self._compute_target_jq(actions)
        return ctrl

    def _compute_target_jq(self, actions):
        # Compute target position from actions.
        target_jq = actions * self.cfg.control_config.action_scale + self.default_angles
        return target_jq

    def get_local_linvel(self, state: MjNpEnvState) -> np.ndarray:
        if self.idx_linvel != -1:
             # If using sensor, need to verify dim (3)
             # assuming sensors are contiguous? MjData.sensordata is flat.
             # We should use model.sensor_adr.
             adr = self._model.sensor_adr[self.idx_linvel]
             dim = self._model.sensor_dim[self.idx_linvel]
             return state.sensor_data[:, adr:adr+dim]
        return np.zeros((self._num_envs, 3))

    def get_gyro(self, state: MjNpEnvState) -> np.ndarray:
        if self.idx_gyro != -1:
             adr = self._model.sensor_adr[self.idx_gyro]
             dim = self._model.sensor_dim[self.idx_gyro]
             return state.sensor_data[:, adr:adr+dim]
        return np.zeros((self._num_envs, 3))

    def update_state(self, state, obs_required=True):
        # 1. Always update intermediate state info (sensors, math) needed for rewards/termination
        self._update_cache(state)
        
        # 2. Compute Observation if required (for agent)
        if obs_required:
            state = self.update_observation(state)
            
        # 3. Check termination and calculate reward
        state = self.update_terminated(state)
        state = self.update_reward(state)
        return state

    def _update_cache(self, state: MjNpEnvState):
        """Update cached info based on current physics/sensor state."""
        info = state.info
        
        # A. Update Local Gravity
        base_quat = state.physics_state[:, self._idx_qpos+3 : self._idx_qpos+7]
        local_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)
        info["local_gravity"] = local_gravity
        
        # B. Update Contacts
        if self.contact_sensor_indices:
             contact_vals = []
             for idx in self.contact_sensor_indices:
                  adr = self._model.sensor_adr[idx]
                  dim = self._model.sensor_dim[idx]
                  val = state.sensor_data[:, adr:adr+dim]
                  # Norm if dim > 1 (e.g. 3-axis force)
                  if dim > 1:
                       val = np.linalg.norm(val, axis=1)
                  else:
                       val = np.abs(val).flatten()
                  contact_vals.append(val)
             contact_vals = np.stack(contact_vals, axis=1)
             threshold = 1.0 
             info["contacts"] = contact_vals > threshold
        else:
             info["contacts"] = np.zeros((self._num_envs, 4), dtype=bool)

        # C. Update Air Time
        info["feet_air_time"] = self.update_feet_air_time(info)

    def _get_obs(self, state: MjNpEnvState, info: dict) -> np.ndarray:
        linear_vel = self.get_local_linvel(state)
        gyro = self.get_gyro(state)
        
        local_gravity = info["local_gravity"] # Use cached logic

        diff = self.get_dof_pos(state) - self.default_angles
        noisy_linvel = linear_vel * self.cfg.normalization.lin_vel
        noisy_gyro = gyro * self.cfg.normalization.ang_vel
        noisy_joint_angle = diff * self.cfg.normalization.dof_pos
        noisy_joint_vel = self.get_dof_vel(state) * self.cfg.normalization.dof_vel
        command = info["commands"] * self.commands_scale
        last_actions = info["current_actions"]

        obs = np.hstack(
            [
                noisy_linvel,
                noisy_gyro,
                local_gravity,
                noisy_joint_angle,
                noisy_joint_vel,
                last_actions,
                command,
            ]
        )
        return obs

    def update_observation(self, state: MjNpEnvState):
        obs = self._get_obs(state, state.info)
        return state.replace(obs=obs)

    def update_terminated(self, state: MjNpEnvState) -> MjNpEnvState:
        # Check termination based on base z or pitch/roll or body contact?
        # Check up direction of the base (projected Z)
        # local_gravity = R^T * [0,0,-1]. So local_gravity.z is roughly -1 if upright.
        # up_z (body Z in world) corresponds to -local_gravity.z
        
        local_gravity = state.info["local_gravity"]
        up_z = -local_gravity[:, 2]
        terminated = up_z <= 0.5
        
        return state.replace(
            terminated=terminated,
        )

    def update_feet_air_time(self, info: dict):
        feet_air_time = info["feet_air_time"]
        feet_air_time += self.cfg.ctrl_dt
        feet_air_time *= ~info["contacts"]
        return feet_air_time

    def resample_commands(self, num_envs: int):
        commands = np.random.uniform(
            low=self.cfg.commands.vel_limit[0],
            high=self.cfg.commands.vel_limit[1],
            size=(num_envs, 3),
        )
        return commands

    def update_reward(self, state: MjNpEnvState) -> MjNpEnvState:
        # Optimized: Calculate reward accumulatively using registered functions
        total_reward = np.zeros(self._num_envs, dtype=np.float32)
        scales = self.cfg.reward_config.scales
        
        # Logging dictionary for rsl_rl
        log = {}
        
        for name, scale in scales.items():
            if scale == 0.0:
                continue
                
            if name in self._reward_fns:
                # Call standardized lambda/method
                term = self._reward_fns[name](state)
                weighted_reward = term * scale
                total_reward += weighted_reward
                
                # Log average weighted reward per step
                log[f"reward/{name}"] = np.mean(weighted_reward)
        
        # Log other info metrics
        if "feet_air_time" in state.info:
            log["metrics/feet_air_time"] = np.mean(state.info["feet_air_time"])
        if "contacts" in state.info:
            log["metrics/contact_rate"] = np.mean(state.info["contacts"].astype(float))

        # Store log in info
        state.info["log"] = log
        
        # Clip reward
        total_reward = np.clip(total_reward, 0.0, 10000.0)
        
        # Apply termination masking
        total_reward = np.where(state.terminated, 0.0, total_reward)

        return state.replace(reward=total_reward)

    def reset(self, env_indices: np.ndarray) -> tuple[np.ndarray, dict]:
        num_reset = len(env_indices)

        # 1. Physics State Preparation (Vectorized)
        # Create default qpos/qvel for the batch
        qpos_batch = np.tile(self._init_qpos, (num_reset, 1))
        # Add noise here if configured (omitted for strict determinism)
        
        qvel_batch = np.zeros((num_reset, self.nv), dtype=np.float64)
        qvel_batch[:, 6:] = self._init_dof_vel
        
        # 2. Update Global State
        # We need to write back to self._state to ensure consistency
        if hasattr(self, '_state') and self._state is not None:
            # mjSTATE_FULLPHYSICS: time, qpos, qvel, act, qacc_warmstart
            # Indices: time=0 (1), qpos (nq), qvel (nv)
            # Reset time
            self._state.physics_state[env_indices, 0] = 0.0
            # Reset qpos
            self._state.physics_state[env_indices, self._idx_qpos : self._idx_qpos + self.nq] = qpos_batch
            # Reset qvel
            self._state.physics_state[env_indices, self._idx_qvel : self._idx_qvel + self.nv] = qvel_batch
            # Reset act/warmstart to 0 (simplified)
            # Assuming act is after qvel
            idx_act = self._idx_qvel + self.nv
            self._state.physics_state[env_indices, idx_act:] = 0.0

        # 3. Info & Command Initialization
        commands = self.resample_commands(num_reset)
        
        info = {
            "current_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "last_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "commands": commands,
            "last_dof_vel": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "feet_air_time": np.zeros((num_reset, 4), dtype=np.float32), 
            "contacts": np.zeros((num_reset, 4), dtype=bool),
        }

        # 4. Sensor Update (Sequential for reset)
        # We need to run mj_forward to populate sensors (contact, imu) for valid observations.
        sensor_batch = np.zeros((num_reset, self._model.nsensordata), dtype=np.float32)
        mj_data = self._worker_data[0]  # Use first worker for utility

        for i in range(num_reset):
            # Load state into worker data
            mj_data.time = 0.0
            mj_data.qpos[:] = qpos_batch[i]
            mj_data.qvel[:] = qvel_batch[i]
            mj_data.ctrl[:] = 0.0
            mj_data.qacc[:] = 0.0
            mj_data.qacc_warmstart[:] = 0.0

            # Execute Kinematics/Sensor update
            mujoco.mj_forward(self._model, mj_data)

            # Capture Sensor Data
            sensor_batch[i] = mj_data.sensordata

        # Update Global Sensor State
        if hasattr(self, "_state") and self._state is not None:
            self._state.sensor_data[env_indices] = sensor_batch

        # 5. Compute Observations (Vectorized - Single Call)
        # Update cache first for the initial state
        # Create batched state for obs and cache update
        # Initialize an empty info structure that _update_cache can populate
        # But we need to use the 'info' dict we already created
        
        # Reconstruct physics state
        obs_physics_state = np.zeros((num_reset, self.physics_state_dim), dtype=np.float64)
        obs_physics_state[:, self._idx_qpos : self._idx_qpos + self.nq] = qpos_batch
        obs_physics_state[:, self._idx_qvel : self._idx_qvel + self.nv] = qvel_batch

        obs_state = MjNpEnvState(
            physics_state=obs_physics_state,
            sensor_data=sensor_batch,
            obs=None,
            reward=None,
            terminated=None,
            truncated=None,
            ctrl=None,
            info=info,
        )
        
        # Manually call update_cache to populate local_gravity/contacts/etc.
        self._update_cache(obs_state)

        # Call _get_obs ONCE for the entire batch
        obs_batch = self._get_obs(obs_state, info)

        # MjNpEnv expects: new_physics_states, new_obs, info
        return obs_physics_state, obs_batch, info

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self, state):
        # Penalize z axis base linear velocity
        return np.square(self.get_local_linvel(state)[:, 2])

    def _reward_ang_vel_xy(self, state):
        # Penalize xy axes base angular velocity
        return np.sum(np.square(self.get_gyro(state)[:, :2]), axis=1)

    def _reward_orientation(self, state):
        # Penalize non flat base orientation
        # qpos 3:7 is quat
        gravity = state.info["local_gravity"]
        return np.sum(np.square(gravity[:, :2]), axis=1)

    def _reward_torques(self, state):
        # Penalize torques
        # In MjNpEnvState, we stored applied ctrl in state.ctrl. 
        # Torques might be proportional to ctrl if direct drive.
        # If we want realized force, we need sensors. 
        # But 'actuator_ctrls' in mtx was likely the input target.
        return np.sum(np.square(state.ctrl), axis=1)

    def _reward_dof_vel(self, state):
        # Penalize dof velocities
        return np.sum(np.square(self.get_dof_vel(state)), axis=1)

    def _reward_dof_acc(self, state, info):
        # Penalize dof accelerations
        return np.sum(
            np.square((info["last_dof_vel"] - self.get_dof_vel(state)) / self.cfg.ctrl_dt),
            axis=1,
        )

    def _reward_action_rate(self, info: dict):
        # Penalize changes in actions
        action_diff = info["current_actions"] - info["last_actions"]
        return np.sum(np.square(action_diff), axis=1)

    def _reward_termination(self, done):
        # Terminal reward / penalty
        return done

    def _reward_feet_air_time(self, commands: np.ndarray, info: dict):
        # Reward long steps
        feet_air_time = info["feet_air_time"]
        first_contact = (feet_air_time > 0.0) * info["contacts"]
        # reward only on first contact with the ground
        rew_airTime = np.sum((feet_air_time - 0.5) * first_contact, axis=1)
        # no reward for zero command
        rew_airTime *= np.linalg.norm(commands[:, :2], axis=1) > 0.1
        return rew_airTime

    def _reward_tracking_lin_vel(self, state, commands: np.ndarray):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = np.sum(np.square(commands[:, :2] - self.get_local_linvel(state)[:, :2]), axis=1)
        return np.exp(-lin_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(self, state, commands: np.ndarray):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = np.square(commands[:, 2] - self.get_gyro(state)[:, 2])
        return np.exp(-ang_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_stand_still(self, state, commands: np.ndarray):
        # Penalize motion at zero commands
        return np.sum(np.abs(self.get_dof_pos(state) - self.default_angles), axis=1) * (
            np.linalg.norm(commands, axis=1) < 0.1
        )

    def _reward_hip_pos(self, state, commands: np.ndarray):
        return (0.8 - np.abs(commands[:, 1])) * np.sum(
            np.square(self.get_dof_pos(state)[:, self.hip_indices] - self.default_angles[self.hip_indices]),
            axis=1,
        )

    def _reward_calf_pos(self, state, commands: np.ndarray):
        return (0.8 - np.abs(commands[:, 1])) * np.sum(
            np.square(self.get_dof_pos(state)[:, self.calf_indices] - self.default_angles[self.calf_indices]),
            axis=1,
        )
