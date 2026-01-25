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
            "collision": lambda s: self._reward_collision(s),
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
        
        # Mapped termination contact sensors
        self.termination_contact_indices = []
        term_contacts = self._cfg.asset.terminate_after_contacts_on
        if term_contacts:
            possible_parts = []
            if "base" in term_contacts:
                possible_parts.append("base")
            if "thigh" in term_contacts:
                possible_parts.extend([f"{p}_thigh" for p in prefixes])

            expected_term_sensors = [f"{part}_contact" for part in possible_parts]
            for name in expected_term_sensors:
                if name not in self.sensor_indices:
                     # Check if it was optional? The user said "check validness"
                     # We only added specific ones in XML. If an XML is updated but not code, or vice versa, this catches it.
                     raise ValueError(f"Required termination contact sensor '{name}' not found. Verify scene matching 'terminate_after_contacts_on'.")
                self.termination_contact_indices.append(self.sensor_indices[name])
            print(f"Mapped termination sensors: {expected_term_sensors}")

        # Mapped penalized contact sensors
        self.penalised_contact_indices = []
        penalized_contacts = getattr(self._cfg.asset, "penalize_contacts_on", [])
        if penalized_contacts:
             possible_parts = []
             if "thigh" in penalized_contacts:
                 prefixes = ["FL", "FR", "RL", "RR"]
                 possible_parts.extend([f"{p}_thigh" for p in prefixes])
             
             expected_pen_sensors = [f"{part}_contact" for part in possible_parts]
             for name in expected_pen_sensors:
                 if name not in self.sensor_indices:
                     # Warn or Error? Since we defined it in config, we expect it.
                     pass 
                 else:
                    self.penalised_contact_indices.append(self.sensor_indices[name])
             print(f"Mapped penalized sensors: {expected_pen_sensors}")

        # Resolve 'local_linvel' and 'gyro'
        if self._cfg.sensor.local_linvel not in self.sensor_indices:
            raise ValueError(f"Sensor '{self._cfg.sensor.local_linvel}' not found.")
        self.idx_linvel = self.sensor_indices[self._cfg.sensor.local_linvel]

        if self._cfg.sensor.gyro not in self.sensor_indices:
             raise ValueError(f"Sensor '{self._cfg.sensor.gyro}' not found.")
        self.idx_gyro = self.sensor_indices[self._cfg.sensor.gyro]


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
        adr = self._model.sensor_adr[self.idx_linvel]
        dim = self._model.sensor_dim[self.idx_linvel]
        return state.sensor_data[:, adr:adr+dim]

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
        contact_vals = []
        for idx in self.contact_sensor_indices:
            adr = self._model.sensor_adr[idx]
            dim = self._model.sensor_dim[idx]
            val = state.sensor_data[:, adr:adr+dim]
            # Sensor data [num_envs] (scalar) or [num_envs, 1] usually
            # But let's be robust. 
            # Note: in Mujoco sensor_data is flat if accessed raw, but here it is shaped (num_envs, nsensordata)
            # The contact sensor in XML "found" data returns a scalar (1 if found, 0 if not) if dimension is 1
            # Check scene_flatp.xml: <contact ... num="1" .../>
            # So val should be (num_envs, 1) or (num_envs,)
            
            # If we flatten, we handle shape issues
            val_flat = val.flatten()
            contact_vals.append(val_flat)
        
        # Stack to (num_envs, 4)
        if len(contact_vals) > 0:
            current_contacts = np.stack(contact_vals, axis=1)
            # Thresholding 0.5 because usually contact sensor returns force or boolean-like float
            # If reduce="netforce", it's a force value. It can be > 0.
            current_contacts = (current_contacts > 0.1) 
        else:
            current_contacts = np.zeros((self._num_envs, 4), dtype=bool)

        # C. Update Air Time
        # Logic: 
        # 1. Update feet_air_time += dt (for all feet)
        # 2. Reset feet_air_time = 0 WHERE contact is True
        # HOWEVER, the reward function relies on "first contact" logic:
        # It needs the air time BEFORE it is reset to 0.
        # But here we update the state cache.
        
        # The issue is the order of operations in `_reward_feet_air_time` vs `_update_cache`.
        # `_update_cache` is called at the beginning of `update_state`.
        # Then `update_reward` is called.
        
        # If we update (reset) the air time here, `_reward_feet_air_time` will see 0 air time for feet that just touched ground!
        # So we need to store the "last air time upon contact" or similar.
        
        # Let's see the implementation of `update_feet_air_time` (helper method):
        # feet_air_time += dt
        # feet_air_time *= ~contacts
        
        # Correct sequence for reward calculation:
        # 1. Increment air time for all feet.
        # 2. Identify feet that JUST touched ground (contact=True, prev_contact=False? Or just Contact=True and AirTime>0)
        # 3. Calculate reward for these feet using their current accumulated air time.
        # 4. THEN reset air time for contacting feet.
        
        # But `_update_cache` does both increment and reset.
        # So when `_reward_feet_air_time` is called later, `info["feet_air_time"]` is ALREADY 0 for contacting feet.
        
        # FIX: We need to calculate air_time reward logic INSIDE update_cache (or preserve the pre-reset value), 
        # or change the cache update logic.
        
        # Option A: Store `last_air_time` in info specifically for reward.
        prev_air_time = info.get("feet_air_time", np.zeros_like(current_contacts, dtype=np.float32)).copy()
        
        # Update logic logic reproduced here to match flow
        # Ensure initialization if missing
        if "feet_air_time" not in info:
             info["feet_air_time"] = np.zeros((self._num_envs, 4), dtype=np.float32)

        info["feet_air_time"] += self.cfg.ctrl_dt
        
        # Capture the air time for feet that are about to reset (contacting now)
        # We need this for the reward function which is called LATER
        info["air_time_at_contact"] = info["feet_air_time"] * current_contacts
        
        # Now apply reset
        info["feet_air_time"] *= ~current_contacts
        
        # Store contacts for next step / other logic
        info["contacts"] = current_contacts

    def _get_obs(self, state: MjNpEnvState, info: dict) -> np.ndarray:
        linear_vel = self.get_local_linvel(state)
        gyro = self.get_gyro(state)
        
        local_gravity = info["local_gravity"] # Use cached logic

        diff = self.get_dof_pos(state) - self.default_angles
        command = info["commands"] * self.commands_scale
        last_actions = info["current_actions"]

        obs = np.hstack(
            [
                linear_vel * self.cfg.normalization.lin_vel,
                gyro * self.cfg.normalization.ang_vel,
                local_gravity,
                diff * self.cfg.normalization.dof_pos,
                self.get_dof_vel(state) * self.cfg.normalization.dof_vel,
                last_actions,
                command,
            ]
        )
        return obs

    def update_observation(self, state: MjNpEnvState):
        obs = self._get_obs(state, state.info)
        return state.replace(obs=obs)

    def update_terminated(self, state: MjNpEnvState) -> MjNpEnvState:
        local_gravity = state.info["local_gravity"]
        up_z = -local_gravity[:, 2]
        
        # 1. Orientation termination
        is_fallen = up_z <= 0.5
        
        # 2. Contact termination (if configured via sensors)
        if hasattr(self, "termination_contact_indices") and self.termination_contact_indices:
             # Check if ANY of the termination sensors detected contact (> 0.5)
             # state.sensor_data shape is (num_envs, num_sensors)
             contact_values = state.sensor_data[:, self.termination_contact_indices]
             # If any sensor value > 0.5, we consider it a contact
             has_contact = np.any(contact_values > 0.5, axis=1)
             is_fallen = np.logical_or(is_fallen, has_contact)

        return state.replace(
            terminated=is_fallen,
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

        # Standard practice: set small percentage of commands to zero to train standing still
        # e.g. 5-10% chance
        mask = np.random.random(num_envs) < 0.05
        commands[mask] = 0.0
        
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
        
        # Remove termination masking that forces reward to 0. 
        # If we have a termination penalty (e.g. -100), we want the agent to receive it.
        # total_reward = np.where(state.terminated, 0.0, total_reward)

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
        # info["feet_air_time"] is reset to 0 for contacting feet in _update_cache
        # We use the snapshot "air_time_at_contact" taken just before reset
        air_time_at_contact = info.get("air_time_at_contact", np.zeros((self._num_envs, 4)))
        
        # Determine valid first contacts. 
        # Note: air_time_at_contact is > 0 only for contacting feet.
        # To avoid penalizing continuous contact (where air_time would be just dt),
        # we can optionaly filter for air_time > dt. 
        # Standard implementation creates 'first_contact' mask.
        # Here air_time_at_contact implies contact is True.
        
        # Standard logic: (time - 0.5) * contact
        # 0.5 is too large for walking/trotting (period ~0.5s total, air ~0.25s).
        # We want to encourage any air time > 0.0 or a small threshold like 0.2
        # Using a positive reward linear to air time is usually better for learning gait.
        rew_airTime = np.sum(air_time_at_contact - 0.3, axis=1)
        
        # no reward for zero command
        rew_airTime *= np.linalg.norm(commands[:, :3], axis=1) > 0.1
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

    def _reward_collision(self, state: MjNpEnvState):
        if not hasattr(self, "penalised_contact_indices") or not self.penalised_contact_indices:
            return 0.0
        # Check contact sensors
        contact_values = state.sensor_data[:, self.penalised_contact_indices]
        # Return 1.0 if any contact > 0.1
        return np.any(contact_values > 0.1, axis=1).astype(np.float32)