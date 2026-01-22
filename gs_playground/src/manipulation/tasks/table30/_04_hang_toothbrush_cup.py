from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from motrixsim import SceneData

from gs_playground import ROOT_PATH
from gs_playground.src.env.registry import envcfg, env
from gs_playground.src.manipulation.tasks.task_env import TaskEnvCfg, TaskEnv
from gs_playground.src.env.motrix_env.render_env import RenderEnvState
from scipy.spatial.transform import Rotation  # 确保头部引入
ASSETS_TASK_DIR = ROOT_PATH / "models" / "tasks" / "table30" / "_04_hang_toothbrush_cup" / "3dgs"
TASK_GAUSSIANS = {
    "toothbrush_cup": ASSETS_TASK_DIR / "toothbrush_cup.ply",
    "rack": ASSETS_TASK_DIR / "rack.ply",
}


@envcfg("table30/hang_toothbrush_cup")
@dataclass
class HangToothbrushCupEnvCfg(TaskEnvCfg):
    # model / sim
    robot_name: str = "franka_robotiq"
    model_file: str = str((ROOT_PATH / "models" / "robots" / 
                           "manipulation" / "franka_emika_panda_robotiq" / 
                           "xmls" / "table30_04_hang_toothbrush_cup.xml").as_posix())

    # control
    action_mode: str = "eef_relative"  # "joint" or "eef"

    # rendering
    img_width: int = 640
    img_height: int = 480
 
    # instruction
    instruction: str = "Hang the orange toothbrush cup on the cup holder"

    # entities (XML names)
    cup_name: str = "toothbrush_cup"
    rack_name: str = "rack"

    # sites (XML names)
    grasp_site_name: str = "bottle_grasp_site"
    hook_site_name: str = "rack_hook_site"

    # sensors (XML names)
    sensor_grasp: str = "bottle_grasp_touch"
    sensor_hook: str = "rack_hook_touch"

    # reward params
    touch_threshold: float = 1e-3
    grasp_dist_threshold: float = 0.05

    reach_scale: float = 3.0
    move_scale: float = 2.0

    # Ratios: grasp:pre_hang:hang:reset = 3:2:4:1
    grasp_reward_bonus: float = 3.0          # latch 后固化
    pre_hang_reward_bonus: float = 2.0       # latch 后固化（到达 pre_hang）
    hang_reward_bonus: float = 4.0           # 稀疏奖励（仅首次“挂上”事件给）
    reset_reward_bonus: float = 1.0          # 稀疏奖励（仅首次“复位完成”事件给，且需已挂上）

    hang_height_margin: float = 0.05         # cup_z > hook_z - margin

    pre_hang_offset: Tuple[float, float, float] = (-0.04, -0.15, 0.03)
    hang_offset: Tuple[float, float, float] = (-0.047, -0.02, -0.03)
    
    pre_hang_dist_threshold: float = 0.03
    hang_dist_threshold: float = 0.05

    reset_pos_target: Tuple[float, float, float] = (0.32 , 0.00084485 ,0.31)
    reset_dist_threshold: float = 0.10

    # randomization
    xy_jitter: float = 0.10  # uniform[-xy_jitter, xy_jitter] (meters)

    reset_enabled: bool = True
    reset_keyframe: int | str = "home"


@env("table30/hang_toothbrush_cup", "np")
class HangToothbrushCupEnv(TaskEnv):
    """
    Task: Hang the toothbrush cup on the rack, then reset to home to finish.
    Robot: Franka + Robotiq.
    Backend: MotrixSim (np).
    """

    def __init__(self, cfg: HangToothbrushCupEnvCfg, num_envs: int = 32):
        super().__init__(cfg, num_envs=num_envs)
        self._cfg: HangToothbrushCupEnvCfg = cfg

        # bodies
        self.cup_body = self.model.get_body(self.model.get_body_index(cfg.cup_name))
        self.rack_body = self.model.get_body(self.model.get_body_index(cfg.rack_name))

        # sites
        self.grasp_site = self.model.get_site(cfg.grasp_site_name)
        self.hook_site = self.model.get_site(cfg.hook_site_name)
        self.rand_yaw = np.zeros((self.num_envs,), dtype=np.float32)  # rad


        # task latch state
        self.is_grasped = np.zeros((self.num_envs,), dtype=bool)
        self.is_pre_hang = np.zeros((self.num_envs,), dtype=bool)
        self.is_hung = np.zeros((self.num_envs,), dtype=bool)          # “挂上”里程碑（一次）
        self.is_reset = np.zeros((self.num_envs,), dtype=bool)         # “复位完成”里程碑（一次）
        self.success_latched = np.zeros((self.num_envs,), dtype=bool)  # 最终成功：hung + reset

    # ---- Task hooks ----
    def task_gaussians(self) -> Dict[str, str]:
        return TASK_GAUSSIANS
    


    # ... Inside your class ...
    def _randomize(self, data: SceneData, done_mask: np.ndarray, phase: str = "reset"):
            """
            Randomization: 
            1. Sample cup XY positions strictly within the XML range geom. 
            XML: <geom name="range" pos="0.45 0.0 0.055" size="0.14 0.14 0.001" .../>
            2. Randomize cup Yaw angle (-45 to +45 degrees).
            Rack remains fixed.
            """
            if data.shape[0] == 0:
                return

            import numpy as np
            from scipy.spatial.transform import Rotation

            # 获取 Batch Size
            B = data.shape[0]

            # 1. 获取 Cup 当前 Pose (用于保留 Z 高度和基础姿态)
            # shape: (B, 7) -> [x, y, z, qx, qy, qz, qw]
            cup_poses = np.asarray(self.cup_body.get_pose(data), dtype=np.float32)
            new_cup_poses = cup_poses.copy()

            # ---------------------------
            # A. 位置随机化 (Absolute Range)
            # ---------------------------
            # 对应 XML: pos="0.45 0.0 ..." size="0.14 0.14 ..."
            range_center = np.array([0.43, -0.05], dtype=np.float32)
            range_half = np.array([0.14, 0.14], dtype=np.float32)

            lower_bound = range_center - range_half
            upper_bound = range_center + range_half

            # 在矩形边界内均匀采样
            # shape: (B, 2)
            rand_xy = self._rng.uniform(lower_bound, upper_bound, size=(B, 2)).astype(np.float32)
            
            # 覆盖 XY，保留 Z (0.085959)
            new_cup_poses[:, :2] = rand_xy

            # ---------------------------
            # B. 旋转随机化 (Yaw Rotation)
            # ---------------------------
            # 随机范围: -45 到 +45 度 (即 -pi/4 到 pi/4)
            yaw_noise = self._rng.uniform(-np.pi/4, np.pi/4, size=(B,))
            
            # 记录随机角度到 info (用于 debug)
            if done_mask is not None and done_mask.shape[0] == B:
                self.rand_yaw[done_mask.astype(bool)] = yaw_noise[done_mask.astype(bool)]
            else:
                if B == self.num_envs:
                    self.rand_yaw[:] = yaw_noise

            # 计算旋转
            q_curr = cup_poses[:, 3:7]
            r_curr = Rotation.from_quat(q_curr)
            r_noise = Rotation.from_euler('z', yaw_noise)

            # 合成新旋转: R_new = R_noise * R_curr (左乘表示绕全局 Z 轴旋转)
            r_new = r_noise * r_curr

            # 填回新的四元数
            new_cup_poses[:, 3:7] = r_new.as_quat().astype(np.float32)

            # ---------------------------
            # C. 写回物理引擎
            # ---------------------------
            for env_idx in range(B):
                self.cup_body.set_dof_pos(
                    data[env_idx],
                    new_cup_poses[env_idx],
                    include_floatingbase=True,
                )

    def _reset_task_state(self, done: np.ndarray):
        done = np.asarray(done, dtype=bool)
        if done.size == 0 or not np.any(done):
            return
        self.is_grasped[done] = False
        self.is_pre_hang[done] = False
        self.is_hung[done] = False
        self.is_reset[done] = False
        self.success_latched[done] = False

    def _compute_reward(self, state: RenderEnvState) -> np.ndarray:
        data: SceneData = state.data
        info = state.info
        cfg = self._cfg
        B = self.num_envs

        # poses
        ee_pos = self.robot.get_ee_pose(data)[:, :3]
        cup_grasp_pos = np.asarray(self.grasp_site.get_pose(data), dtype=np.float32)[:, :3]
        hook_pos = np.asarray(self.hook_site.get_pose(data), dtype=np.float32)[:, :3]

        # sensors
        grasp_touch = (
            np.asarray(self.model.get_sensor_value(cfg.sensor_grasp, data), dtype=np.float32).reshape(B, -1)[:, 0]
        )
        hook_touch = (
            np.asarray(self.model.get_sensor_value(cfg.sensor_hook, data), dtype=np.float32).reshape(B, -1)[:, 0]
        )

        touching_cup = grasp_touch > cfg.touch_threshold
        touching_hook = hook_touch > cfg.touch_threshold

        # distances
        d_ee_cup = np.linalg.norm(ee_pos - cup_grasp_pos, axis=1)

        # stage targets
        pre_hang_tgt = hook_pos + np.asarray(cfg.pre_hang_offset, dtype=np.float32).reshape(1, 3)
        hang_tgt = hook_pos + np.asarray(cfg.hang_offset, dtype=np.float32).reshape(1, 3)

        d_cup_pre_hang = np.linalg.norm(cup_grasp_pos - pre_hang_tgt, axis=1)
        dxz = cup_grasp_pos[:, [0, 2]] - hang_tgt[:, [0, 2]]
        d_cup_hang = np.linalg.norm(dxz, axis=1)

        # 1) Dense reach-to-cup
        #    你要求：到达指定位置后 r_reach 不再给。这里用 is_grasped 作为“reach 阶段结束”。
        r_reach_raw = 1.0 - np.tanh(cfg.reach_scale * d_ee_cup)
        r_reach = r_reach_raw * (~self.is_grasped).astype(np.float32)

        # 2) Grasp latch + fixed grasp reward
        grasp_now = touching_cup & (d_ee_cup < cfg.grasp_dist_threshold)
        self.is_grasped = self.is_grasped | grasp_now
        r_grasp_fixed = self.is_grasped.astype(np.float32) * cfg.grasp_reward_bonus

        # 3) Two-stage move shaping + pre_hang fixed bonus
        pre_hang_reached_now = self.is_grasped & (~self.is_pre_hang) & (d_cup_pre_hang < cfg.pre_hang_dist_threshold)
        self.is_pre_hang = self.is_pre_hang | pre_hang_reached_now
        r_pre_hang_fixed = self.is_pre_hang.astype(np.float32) * cfg.pre_hang_reward_bonus
        r_move_pre = (1.0 - np.tanh(cfg.move_scale * d_cup_pre_hang)) * (
            self.is_grasped & (~self.is_pre_hang)
        ).astype(np.float32)

        r_move_hang = (1.0 - np.tanh(cfg.move_scale * d_cup_hang)) * (
            self.is_grasped & self.is_pre_hang & (~self.is_hung)
        ).astype(np.float32)

        r_move = r_move_pre 

        # 4) Stage 1 sparse: “hung” milestone (only once)
        high_enough = cup_grasp_pos[:, 2] > (hang_tgt[:, 2] - cfg.hang_height_margin)

        hung_now = (
            touching_hook
            & high_enough
            & self.is_grasped
            & self.is_pre_hang
            & (d_cup_hang < cfg.hang_dist_threshold)
            & (~self.is_hung)
        )


        self.is_hung = self.is_hung | hung_now
        r_hang_sparse = self.is_hung.astype(np.float32) * cfg.hang_reward_bonus

        # 5) Stage 2 sparse: reset-after-hung (final success)
        reset_pos = np.asarray(cfg.reset_pos_target, dtype=np.float32).reshape(1, 3)
        d_ee_reset = np.linalg.norm(ee_pos - reset_pos, axis=1)

        reset_now = self.is_hung & (~self.is_reset) & (d_ee_reset < cfg.reset_dist_threshold)
        self.is_reset = self.is_reset | reset_now

        success_now = reset_now & (~self.success_latched)
        self.success_latched = (self.success_latched | success_now) & touching_hook

        r_reset_sparse = reset_now.astype(np.float32) * cfg.reset_reward_bonus
        # print(r_reset_sparse)
        total_reward = r_reach + r_grasp_fixed + r_pre_hang_fixed + r_move + r_hang_sparse + r_reset_sparse
        # print(total_reward)
        # print(r_reach,r_grasp_fixed,r_pre_hang_fixed,r_move,r_hang_sparse,r_reset_sparse)
        # if self.is_pre_hang.any() :
        #     print("d_cup_pre_hang",d_cup_pre_hang)
        #     print("d_cup_hang",d_cup_hang)
        #     print("hang_tgt",hang_tgt)
        #     print("cup_grasp_pos",cup_grasp_pos)
        # print("is_hung",self.is_hung)
        # ---- stash info ----
        info["d_ee_cup"] = d_ee_cup
        info["d_cup_pre_hang"] = d_cup_pre_hang
        info["d_cup_hang"] = d_cup_hang
        info["d_ee_reset"] = d_ee_reset

        info["grasp_touch"] = grasp_touch
        info["hook_touch"] = hook_touch

        info["is_grasped"] = self.is_grasped.copy()
        info["is_pre_hang"] = self.is_pre_hang.copy()
        info["is_hung"] = self.is_hung.copy()
        info["is_reset"] = self.is_reset.copy()

        # final success (only after hung + reset)
        info["is_success"] = self.success_latched.copy()
        info["hung_now"] = hung_now
        info["reset_now"] = reset_now
        info["success_now"] = success_now
        info["rand_yaw"] = self.rand_yaw.copy()                 # rad
        info["rand_yaw_deg"] = (self.rand_yaw * 180.0 / np.pi)  # deg


        return total_reward.astype(np.float32)
    
    # def update_state(self, state: RenderEnvState, obs_required: bool = True) -> RenderEnvState:
    #     state = super().update_state(state, obs_required=obs_required)

    #     # -------------------------
    #     # Fail-fast: workspace bounds check (no new cfg params)
    #     # -------------------------
    #     data: SceneData = state.data

    #     # Workspace bounds (world frame). Adjust to your table if needed.
    #     # Here we only check XY; optionally you can add a Z floor as well.
    #     x_min, x_max = -0.75, 0.10
    #     y_min, y_max = -0.35, 0.35

    #     # Optional Z floor (comment out if you don't want it)
    #     z_min = -0.20

    #     # Choose bodies to check: cup is usually sufficient; rack can be included if desired.
    #     # If you only want cup: keep [self.cup_body]
    #     bodies = [self.cup_body]  # or: [self.cup_body, self.rack_body]

    #     # poses: (B, K, 7) where K = len(bodies)
    #     poses = np.stack(
    #         [np.asarray(b.get_pose(data), dtype=np.float32) for b in bodies],
    #         axis=1,
    #     )
    #     pos = poses[..., :3]   # (B, K, 3)
    #     xy = pos[..., :2]      # (B, K, 2)

    #     x_ok = (xy[..., 0] >= x_min) & (xy[..., 0] <= x_max)
    #     y_ok = (xy[..., 1] >= y_min) & (xy[..., 1] <= y_max)
    #     z_ok = pos[..., 2] >= z_min

    #     in_bounds_each = x_ok & y_ok & z_ok          # (B, K)
    #     out_of_bounds = ~np.all(in_bounds_each, axis=1)  # (B,)

    #     if np.any(out_of_bounds):
    #         terminated = np.asarray(state.terminated, dtype=bool).copy()
    #         terminated[out_of_bounds] = True
    #         state.terminated = terminated

    #         # Debug info (optional but useful)
    #         info = state.info
    #         info["out_of_bounds"] = out_of_bounds
    #         info["ws_in_bounds_each"] = in_bounds_each
    #         info["ws_pos"] = pos  # (B, K, 3)

    #     return state

