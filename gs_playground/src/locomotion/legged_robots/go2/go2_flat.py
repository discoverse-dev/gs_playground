import torch
from gs_playground.src.locomotion.legged_robots.base.legged_robot import (
    Legged_Robot_Torch,
)

def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


class Go2_train_env(Legged_Robot_Torch):
    def __init__(self, Cfg, headless):
        super().__init__(Cfg, headless)
        self.feet_site_init()
        self.feet_pos = torch.zeros(
            self.num_envs,
            len(self.config.asset.foot_site),
            3,
            device=self.device,
            requires_grad=False,
        )
        self.feet_phase = torch.zeros(
            self.num_envs,
            len(self.config.asset.foot_site),
            device=self.device,
            requires_grad=False,
        )

    def feet_site_init(self):
        self.feet_site = []
        for j in self.config.asset.foot_site:
            for i in self.model.site_names:
                print(j)
                if j in i:
                    self.feet_site.append(self.model.get_site(i))
                    break

    def _sync_dof_data(self):
        super()._sync_dof_data()
        for i in range(len(self.config.asset.foot_site)):
            self.feet_pos[:, i] = torch.from_numpy(
                self.feet_site[i].get_pose(self.datas)
            )[:, :3]

    def post_physics_step(self):
        period = 0.6
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.feet_phase[:, 0] = self.phase
        self.feet_phase[:, 3] = self.phase

        self.feet_phase[:, 1] = (self.phase + 0.5) % 1
        self.feet_phase[:, 2] = (self.phase + 0.5) % 1
        super().post_physics_step()
    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros(self.config.env.num_observations, device=self.device, dtype=torch.float32)
        self.add_noise = self.config.noise.add_noise#self.cfg.noise.add_noise
        noise_scales = self.config.noise.noise_scales
        noise_level = self.config.noise.noise_level
        self.obs_scales = self.config.normalization.obs_scales
        # noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        # noise_vec[9:12] = 0. # commands
        noise_vec[6:6+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[6+self.num_actions:6+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[6+2*self.num_actions:6+3*self.num_actions] = 0. # previous actions
        
        return noise_vec

    def compute_obs(self):
        # raise NotImplementedError("compute_obs() must be implemented for env")
        diff = self.dof_pos - self.default_angles
        self.obs[:, :3] = self.gyro * self.config.normalization.obs_scales.ang_vel
        self.obs[:, 3:6] = self.gravity
        self.obs[:, 6:18] = diff * self.config.normalization.obs_scales.dof_pos
        self.obs[:, 18:30] = self.dof_vel * self.config.normalization.obs_scales.dof_vel
        self.obs[:, 30:42] = self.actions
        self.obs[:, 42:45] = self.commands * self.commands_scale
        self.obs[:, 45:49] = self.feet_phase
        if self.config.env.num_privileged_obs is not None:
            self.privileged_obs_buf[:, :3] = (
                self.linear_vel * self.config.normalization.obs_scales.lin_vel
            )
            self.privileged_obs_buf[:, 3:6] = (
                self.gyro * self.config.normalization.obs_scales.ang_vel
            )
            self.privileged_obs_buf[:, 6:9] = self.gravity
            self.privileged_obs_buf[:, 9:21] = (
                diff * self.config.normalization.obs_scales.dof_pos
            )
            self.privileged_obs_buf[:, 21:33] = (
                self.dof_vel * self.config.normalization.obs_scales.dof_vel
            )
            self.privileged_obs_buf[:, 33:45] = self.actions
            self.privileged_obs_buf[:, 45:48] = self.commands * self.commands_scale
            self.privileged_obs_buf[:, 48:52] = self.feet_phase
        if self.add_noise:
            self.obs += (2 * torch.rand_like(self.obs) - 1) * self.noise_vec
           
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.foot_force[:, :, 0] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.25) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.25) * first_contact, dim=1
        )  # reward only on first contact with the ground
        rew_airTime *= (
            torch.norm(self.commands[:, :2], dim=1) > 0.1
        )  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_contact(self):
        contact = self.foot_force[:, :, 0] > 1.0
        res = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False
        )
        for i in range(len(self.feet_site)):
            is_contact = (self.feet_phase[:, i] < 0.6) | (
                torch.norm(self.commands[:, :2], dim=1) < 0.1
            )
            res += ~(contact[:, i] ^ is_contact)
        return res

    def _reward_swing_feet_z(self):
        contact = self.foot_force[:, :, 0] > 1.0
        pos_error = torch.square((self.feet_pos[:, :, 2] - 0.08)) * ~contact
        return torch.sum(pos_error, dim=1)