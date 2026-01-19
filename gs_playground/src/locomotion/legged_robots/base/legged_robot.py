import time
import numpy as np
import torch
from gs_playground.src.locomotion.legged_robots.base.legged_robot_cfg import (
    LeggedRobotTorchCfg,
)
from motrixsim import SceneData, load_model
from motrixsim.render import RenderApp


def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = (
        q_vec
        * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1)
        * 2.0
    )
    return a - b + c


def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(shape, device=device) + lower


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class Legged_Robot_Torch:
    def __init__(self, Cfg: LeggedRobotTorchCfg, headless=True):
        self.config = Cfg
        self.env_init()
        self.headless = headless
        if self.headless is False:
            self.render_init()
        else:
            self._render = None
        self.buffer_init()
        self.device = self.config.sim.device
        self.dt = self.config.control.decimation * self.config.sim.dt
        self.reward_scales = class_to_dict(self.config.rewards.scales)
        self.max_episode_length_s = self.max_episode_length * self.dt
        self._prepare_reward_function()

    def buffer_init(self):
        # init buffers
        k = False
        self.extras = {}
        self.device = self.config.sim.device
        self.dof_pos = torch.zeros(
            (self.config.env.num_envs, self.config.env.num_actions), dtype=torch.float32
        )
        self.dof_vel = torch.zeros(
            (self.config.env.num_envs, self.config.env.num_actions), dtype=torch.float32
        )
        self.commands = torch.zeros((self.config.env.num_envs, 3), dtype=torch.float32)
        self.obs = torch.zeros(
            (self.config.env.num_envs, self.config.env.num_observations),
            dtype=torch.float32,
        )
        self.kps = torch.ones(self.config.env.num_actions, dtype=torch.float32)
        self.kds = torch.ones(self.config.env.num_actions, dtype=torch.float32)
        self.episode_length_buf = torch.zeros(
            self.config.env.num_envs, dtype=torch.int32
        )
        self.linear_vel = torch.zeros(
            (self.config.env.num_envs, 3), dtype=torch.float32
        )
        self.gyro = torch.zeros((self.config.env.num_envs, 3), dtype=torch.float32)
        self.pose = torch.zeros((self.config.env.num_envs, 7), dtype=torch.float32)
        self.actions = torch.zeros(
            (self.config.env.num_envs, self.config.env.num_actions), dtype=torch.float32
        )
        self.gravity = torch.zeros((self.config.env.num_envs, 3), dtype=torch.float32)
        self.rew_buf = torch.zeros(
            self.config.env.num_envs, device=self.device, dtype=torch.float32
        )
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.gravity_vec = torch.tensor(
            [0, 0, -1], device=self.device, dtype=torch.float32
        ).repeat((self.config.env.num_envs, 1))
        self.commands_scale = torch.tensor(
            (
                [
                    self.config.normalization.obs_scales.lin_vel,
                    self.config.normalization.obs_scales.lin_vel,
                    self.config.normalization.obs_scales.ang_vel,
                ]
            ),
            dtype=torch.float32,
        )
        if self.config.env.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(
                self.config.env.num_envs,
                self.config.env.num_privileged_obs,
                device=self.device,
                dtype=torch.float,
            )
        else:
            self.privileged_obs_buf = None
        self.joints = []
        names = self.model.joint_names
        for i in range(len(names)):
            self.joints.append(self.model.get_joint(names[i]))
        if type(self.config.control.stiffness) is int:
            self.kps = self.kps * self.config.control.stiffness
            self.kds = self.kds * self.config.control.damping
            k = True
        self.max_episode_length = self.config.sim.max_episode_length
        self.reset_buf = torch.ones(self.config.env.num_envs, dtype=torch.bool)
        self.default_angles = torch.zeros(
            self.config.env.num_actions, dtype=torch.float32
        )
        found = False
        for i in range(self.model.num_actuators):
            for name in self.config.init_state.default_joint_angles.keys():
                if name in self.model.actuator_names[i]:
                    self.default_angles[i] = (
                        self.config.init_state.default_joint_angles[name]
                    )
                    found = True
            if not found:
                self.default_angles[i] = self.config.init_state.default_joint_angles[
                    "default"
                ]
            if not k:
                found = False
                for name in self.config.control.stiffness.keys():
                    if name in self.model.actuator_names[i]:
                        self.kps[i] = self.config.control.stiffness[name]
                        self.kds[i] = self.config.control.damping[name]
                        found = True
                    if found:
                        break
                if not found:
                    print("no found")
                    raise ValueError("PD gain of joint  were not defined")
        self.common_step_counter = 0
        self.last_actions = torch.zeros(
            self.config.env.num_envs, self.config.env.num_actions, dtype=torch.float32
        )
        self.torque_limits = self.config.control.torque_limits
        self.num_privileged_obs = self.config.env.num_privileged_obs
        self.num_obs = self.config.env.num_observations
        self.num_actions = self.config.env.num_actions
        self.num_envs = self.config.env.num_envs
        self.ground = self.model.get_geom_index(self.config.asset.ground)
        self.termination_contact = None
        self.cquerys = self.model.get_contact_query(self.datas)
        self.termination_contact_name = []
        self.geom_names = [item for item in self.model.geom_names if item is not None]
        for i in self.geom_names:
            if i == None:
                break
            for j in self.config.asset.terminate_after_contacts_on:

                if j in i:
                    self.termination_contact_name.append(i)
                    break
        print(self.termination_contact_name)
        for name in self.termination_contact_name:
            if self.termination_contact is None:
                self.termination_contact = np.array(
                    [[self.model.get_geom_index(name), self.ground]], dtype=np.uint32
                )
            else:
                self.termination_contact = np.append(
                    self.termination_contact,
                    np.array(
                        [[self.model.get_geom_index(name), self.ground]],
                        dtype=np.uint32,
                    ),
                    axis=0,
                )
        self.foot = None
        # self.model.geom_names =
        # filtered_list = [item for item in original_list if item is not None]

        print(self.geom_names)
        for i in self.geom_names:
            # print('###############')
            # print(i)
            if self.config.asset.foot_name in i:
                if self.foot is None:
                    self.foot = np.array(
                        [[self.model.get_geom_index(i), self.ground]], dtype=np.uint32
                    )
                else:
                    self.foot = np.append(
                        self.foot,
                        np.array(
                            [[self.model.get_geom_index(i), self.ground]],
                            dtype=np.uint32,
                        ),
                        axis=0,
                    )
        print(self.foot)
        print(self.default_angles)
        self.num_check = self.termination_contact.shape[0]
        self.foot_check_num = self.foot.shape[0]
        batch = (self.config.env.num_envs,)
        self.termination_check = (
            self.termination_contact
        )  # np.tile(self.termination_contact, (*batch, 1))
        # self.termination_check = self.termination_check.reshape(self.num_envs, self.num_check, 2)
        print("################")
        print(self.termination_check)
        self.foot_check = self.foot  # np.tile(self.foot, (*batch, 1))
        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.foot_check_num,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            self.foot_check_num,
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.foot_force = torch.zeros(
            self.num_envs,
            self.foot_check_num,
            3,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.floating_base = self.body.floatingbase
        if self.config.sensor.contact_sensor:
            self.foot_sensor_names = []
            self.contact_data = torch.zeros(
                self.num_envs,
                len(self.config.sensor.foot_name),
                1 + 12 * self.config.sensor.num_contact,
                dtype=torch.float32,
                device=self.device,
                requires_grad=False,
            )
            for i in range(len(self.config.sensor.foot_name)):
                # sensor_name = self.config.sensor.foot_name[i]
                self.foot_sensor_names.append(self.config.sensor.foot_name[i])
                self.contact_data[:, i, :] = torch.tensor(
                    self.model.get_sensor_value(
                        self.config.sensor.foot_name[i], self.datas
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
            # self.model.get_sensor_value(self.config.sensor.foot_name[0])
            print(self.contact_data)

        # self.model.

    def get_observations(self):
        return self.obs

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def buffer_reset(self):
        pass

    def render_init(self):
        # Create render window for visualization
        self._render = RenderApp()

        # Launch the render with the model

        # self._render.launch(self.model, batch=self.config.env.num_envs, render_offset=self.render_offset)
        self._render.launch(self.model, batch=self.config.env.num_envs)
        self.render_dt = self.config.sim.dt * self.config.control.decimation
        # print(self.render_offset)

        # self.render_offset[:] += np.array(self.config.init_state.pos)
        # print(torch.tensor(self.render_offset))

    def env_init(self):
        # The scene description file
        path = self.config.asset.file
        # Load the scene model
        self.model = load_model(path)
        self.model.options.timestep = self.config.sim.dt
        # Create the physics data of the model

        self.body = self.model.get_body(self.config.asset.body_name)
        self.datas = SceneData(self.model, batch=(self.config.env.num_envs,))
        row = int(np.sqrt(self.config.env.num_envs))
        l = self.config.env.space
        x = self.config.init_state.pos[0]
        y = self.config.init_state.pos[1]
        z = self.config.init_state.pos[2]
        self.render_offset = []
        for i in range(self.config.env.num_envs):
            self.render_offset.append([int(i / row) * l + x, (i % row) * l + y, z])
        self.render_offset = np.array(self.render_offset)

    def step(self, actions):
        # Apply actuations, simulate, call self.post_physics_step()
        self.render()
        self.actions = actions.to(self.config.sim.device)
        self.actions = torch.clip(
            self.actions,
            -self.config.control.clip_actions,
            self.config.control.clip_actions,
        )
        for _ in range(self.config.control.decimation):
            self.torques = self._compute_torques(self.actions)
            self.torques_numpy = self.torques.detach().numpy()
            self.datas.actuator_ctrls = self.torques_numpy
            self.model.step(self.datas)
        self.post_physics_step()
        self.compute_obs()
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_actions[:] = self.actions[:]
        return (
            self.obs,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def render(self):
        """Render the scene"""
        if self._render is not None:
            t_0 = time.time()
            self._render.sync(self.datas)
            passed = time.time() - t_0
            if passed < self.render_dt:
                time.sleep(self.render_dt - passed)

    def post_physics_step(self):  #################
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_heightmap_vis() if needed
        """
        self.episode_length_buf[:] += 1
        self.common_step_counter += 1

        # prepare quantities
        self.get_data()
        self._sync_dof_data()

        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        self.reset_ids(env_ids)
        self._post_physics_step_callback()

    def get_data(self):
        self.linear_vel = torch.from_numpy(
            self.model.get_sensor_value(self.config.sensor.local_linvel, self.datas)
        )
        self.gyro = torch.from_numpy(
            self.model.get_sensor_value(self.config.sensor.gyro, self.datas)
        )
        self.pose = torch.from_numpy(self.body.get_pose(self.datas))
        self.base_quat = self.pose[:, 3:7]
        self.gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading,
        compute measured terrain heights and randomly push robots
        """

        env_ids = (
            (self.episode_length_buf % 100 == 0).nonzero(as_tuple=False).flatten()
        )  ##### resample time in config
        self.resample_commands(env_ids)

    def check_termination(self):
        """Check if environments need to be reset"""
        check = self.cquerys.is_colliding(self.termination_check)
        check.reshape((self.num_envs, self.num_check))
        self.reset_buf = check.any(axis=1)
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        self.reset_buf = self.time_out_buf | self.reset_buf

    def reset_dofs(self, env_ids):
        bool_mask = self.reset_buf.numpy()
        bool_mask.dtype = np.bool
        for i in range(self.config.env.num_actions):
            self.joints[i].set_dof_pos(
                self.datas[bool_mask], len(env_ids) * [self.default_angles[i]]
            )
        #### add noise

    def reset_states(self, env_ids):
        bool_mask = self.reset_buf.numpy()
        bool_mask.dtype = np.bool
        # vel = np.zeros((len(env_ids), 3))
        vel = np.random.rand(len(env_ids), 3) * 2 - 1  ##(-1, 1)
        # ang_v = np.zeros((len(env_ids), 3))
        ang_v = np.random.rand(len(env_ids), 3) * 1 - 0.5
        self.floating_base.set_translation(
            self.datas[bool_mask], self.render_offset[env_ids]
        )
        self.floating_base.set_global_linear_velocity(self.datas[bool_mask], vel)
        self.floating_base.set_local_angular_velocity(self.datas[bool_mask], ang_v)

    def reset_ids(self, env_ids):
        if len(env_ids) == 0:
            return
        bool_mask = self.reset_buf.numpy()
        bool_mask.dtype = np.bool
        self.env_reset(env_ids)
        self.resample_commands(env_ids)
        self.reset_dofs(env_ids)
        self.reset_states(env_ids)
        self.model.forward_kinematic(self.datas[bool_mask])
        # buffer reset
        self.last_actions[env_ids] = 0
        self.last_dof_vel[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        self.feet_air_time[env_ids] = 0
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0

    def reset(self):
        env_ids = torch.arange(self.config.env.num_envs)
        self.reset_ids(env_ids)
        obs, privileged_obs, _, _, _ = self.step(
            torch.zeros(
                self.num_envs, self.num_actions, device=self.device, requires_grad=False
            )
        )
        return obs, privileged_obs

    def env_reset(self, env_ids):
        if len(env_ids) == 0:
            return
        bool_mask = self.reset_buf.numpy()
        bool_mask.dtype = np.bool
        self.datas[bool_mask].reset(self.model)
        self.model.forward_kinematic(self.datas[bool_mask])

    def get_observation(self):
        return self.obs

    def compute_obs(self):
        # raise NotImplementedError("compute_obs() must be implemented for env")
        diff = self.dof_pos - self.default_angles
        self.obs[:, :3] = self.linear_vel * self.config.normalization.obs_scales.lin_vel
        self.obs[:, 3:6] = self.gyro * self.config.normalization.obs_scales.ang_vel
        self.obs[:, 6:9] = self.gravity
        self.obs[:, 9:21] = diff * self.config.normalization.obs_scales.dof_pos
        self.obs[:, 21:33] = self.dof_vel * self.config.normalization.obs_scales.dof_vel
        self.obs[:, 33:45] = self.actions
        self.obs[:, 45:48] = self.commands * self.commands_scale

    def resample_commands(self, env_ids):
        commands = self.commands

        self.commands[env_ids, 0] = torch_rand_float(
            self.config.commands.ranges.lin_vel_x[0],
            self.config.commands.ranges.lin_vel_x[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.config.commands.ranges.lin_vel_y[0],
            self.config.commands.ranges.lin_vel_y[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.commands[env_ids, 2] = torch_rand_float(
            self.config.commands.ranges.ang_vel_yaw[0],
            self.config.commands.ranges.ang_vel_yaw[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        # set small commands to zero
        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1) > 0.1
        ).unsqueeze(1)

    def _compute_torques(self, actions):
        # Compute torques from actions.
        # pd controller
        actions_scaled = actions * self.config.control.action_scale
        control_type = self.config.control.control_type
        self._sync_dof_data()
        if control_type == "P":
            torques = (
                self.kps * (actions_scaled + self.default_angles - self.dof_pos)
                - self.kds * self.dof_vel
            )
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def process_contact_data(self):
        self.foot_force[:, :, :] = 0  ###(法向力， 切向力0, 切向力1)
        for i in range(self.config.sensor.num_contact):
            offset = 1 + i * 12
            self.foot_force += self.contact_data[:, :, offset : offset + 3]

    def _sync_dof_data(self):
        if self.config.sensor.contact_sensor:
            for i in range(len(self.foot_sensor_names)):
                self.contact_data[:, i, :] = torch.tensor(
                    self.model.get_sensor_value(self.foot_sensor_names[i], self.datas),
                    device=self.device,
                    dtype=torch.float32,
                )
        self.process_contact_data()
        self.dof_pos = torch.from_numpy(self.body.get_joint_dof_pos(self.datas))
        self.dof_vel = torch.from_numpy(self.body.get_joint_dof_vel(self.datas))

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))
        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(
                self.config.env.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for name in self.reward_scales.keys()
        }

    def compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.config.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.linear_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.gyro[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.gravity[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(
            torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1
        )

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.linear_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.config.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.gyro[:, 2])
        return torch.exp(-ang_vel_error / self.config.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        # contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact = self.foot_force[:, :, 0] > 1.0
        print(contact)
        contact_filt = torch.logical_or(contact, self.last_contacts)
        # contact_filt = torch.logical_or(self.contacts, self.last_contacts)
        self.last_contacts = contacts
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.5) * first_contact, dim=1
        )  # reward only on first contact with the ground
        rew_airTime *= (
            torch.norm(self.commands[:, :2], dim=1) > 0.1
        )  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_angles), dim=1) * (
            torch.norm(self.commands[:, :2], dim=1) < 0.1
        )

    def _reward_hip_pos(self):
        # return torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)
        return (0.8 - torch.abs(self.commands[:, 1])) * torch.sum(
            torch.square(
                self.dof_pos[:, [0, 3, 6, 9]] - self.default_angles[[0, 3, 6, 9]]
            ),
            dim=1,
        )

    def _reward_calf_pos(self):
        # return torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)
        return (0.8 - torch.abs(self.commands[:, 1])) * torch.sum(
            torch.square(
                self.dof_pos[:, [2, 5, 8, 11]] - self.default_angles[[2, 5, 8, 11]]
            ),
            dim=1,
        )
