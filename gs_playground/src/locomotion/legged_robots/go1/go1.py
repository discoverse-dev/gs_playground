import torch
import numpy as np
from gs_playground.src.locomotion.legged_robots.base.legged_robot import (
    Legged_Robot_Torch,
)
from gs_playground.addr import GS_GYM_ENVS_DIR
from motrixsim.render import Color


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


def load_hfield_with_header(file_path):
    """
    读取包含行列头信息的高度场二进制文件
    :param file_path: 文件路径
    :return: (nrows, ncols, height_data)
    """
    with open(file_path, "rb") as f:
        # 读取前8字节（nrows和ncols，每个占4字节）
        nrows = np.fromfile(f, dtype=np.int32, count=1)[0]
        ncols = np.fromfile(f, dtype=np.int32, count=1)[0]

        # 读取剩余的高度数据（float32格式）
        height_data = np.fromfile(f, dtype=np.float32).reshape(nrows, ncols)

    return nrows, ncols, height_data


class Go1_train_env(Legged_Robot_Torch):
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
        if self.config.terrain.measure_heights:
            self.nrows, self.ncols, self.height_data = load_hfield_with_header(
                GS_GYM_ENVS_DIR + self.config.terrain.hfield_path
            )
            min = np.min(self.height_data)
            max = np.max(self.height_data)
            # self.height_data = self.height_data - min
            # self.height_data = self.height_data / (max - min)
            # self.height_data = torch.from_numpy(self.height_data, device=self.device, dtype=torch.float32)
            self.height_data = torch.from_numpy(
                self.height_data
            )  # 从numpy数组创建tensor
            self.height_data = self.height_data.to(
                device=self.device, dtype=torch.float32
            )  # 单独设置设备和类型
            measured_points_x = [
                -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
            ]  # 1mx1.6m rectangle (without center line)
            measured_points_y = [
                -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,  0.1,  0.2,  0.3,  0.4, 0.5,
            ]
            y = torch.tensor(measured_points_y, device=self.device, requires_grad=False)
            x = torch.tensor(measured_points_x, device=self.device, requires_grad=False)
            grid_x, grid_y = torch.meshgrid(x, y)

            self.num_height_points = grid_x.numel()
            self.points = torch.zeros(
                self.num_envs,
                self.num_height_points,
                3,
                device=self.device,
                requires_grad=False,
            )
            self.points[:, :, 0] = grid_x.flatten()
            self.points[:, :, 1] = grid_y.flatten()

    def draw_point(self, a, b):
        self._render.gizmos.draw_sphere(
            0.5,
            ([a, b, (self.height_data[int(self.nrows / 2 - 10 * a), int(self.ncols / 2 + 10 * b)]- 0.5) * 3.78,]),
            color=Color.rgb(0, 0, 1),
        )

    def get_heights(self):
        points = quat_apply_yaw(
            self.base_quat.repeat(1, self.num_height_points), self.points
        ) + (self.pose[:, :3]).unsqueeze(1)
        points1 = (10 * points).to(torch.int)
        px = points1[:, :, 0].view(-1)
        py = points1[:, :, 1].view(-1)
        heights2 = self.height_data[int(self.nrows / 2) - py, int(self.ncols / 2) + px]
        heights = (
            self.height_data[int(self.nrows / 2) - py, int(self.ncols / 2) + px]
        ) * 0.005
        heights = heights.view(self.num_envs, -1)
        heights2 = heights2.view(self.num_envs, -1)
        print(heights[0].min())
        print(heights2[0].min())
        print(self.feet_pos[0, 0, 2])
        for i in range(170):
            self._render.gizmos.draw_sphere(
                0.01,
                np.array([points[0, i, 0], points[0, i, 1], heights[0, i]]),
                color=Color.rgb(0, 0, 1),
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

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        # contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
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
