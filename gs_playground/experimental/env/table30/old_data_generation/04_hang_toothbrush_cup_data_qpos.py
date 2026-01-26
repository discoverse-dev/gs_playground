from __future__ import annotations

import os
import json
import time
import shutil
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np
import mediapy

from scipy.spatial.transform import Rotation

from gs_playground.src.manipulation.tasks.table30._04_hang_toothbrush_cup import (
    HangToothbrushCupEnv,
    HangToothbrushCupEnvCfg,
)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def smooth_step_pos(curr: np.ndarray, tgt: np.ndarray, max_dp: Any) -> np.ndarray:
    dp = tgt - curr
    n = np.linalg.norm(dp, axis=1, keepdims=True)
    s = np.minimum(1.0, max_dp / (n + 1e-9)) 
    return curr + dp * s

def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    sym = np.pi
    return (a + sym / 2.0) % sym - sym / 2.0


def closest_yaw(target: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """
    Map target yaw to the equivalent angle (2π-periodic) that is closest to curr yaw.
    """
    d = wrap_to_pi(target - curr)
    return curr + d


def to_xyzw_from_possible_wxyz(q: np.ndarray, assume: str = "xyzw") -> np.ndarray:
    if assume == "xyzw":
        return q
    if assume == "wxyz":
        return q[..., [1, 2, 3, 0]]
    raise ValueError(f"Unknown assume={assume}")


# -----------------------------------------------------------------------------
# Video Writer
# -----------------------------------------------------------------------------
class EpisodeVideoWriter:
    """
    Stream mp4 writer based on mediapy.VideoWriter (ffmpeg).
    Expects RGB frames (H, W, 3). No BGR conversion needed.
    """
    def __init__(self, path: str, fps: int, size_wh: Tuple[int, int], *, codec: str = "h264", qp: int | None = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        w, h = int(size_wh[0]), int(size_wh[1])
        self.shape_hw = (h, w)  # mediapy expects (height, width)

        # Note: mediapy.VideoWriter is a context manager; we open it here for streaming writes.
        # Bitrate control: specify at most one of bps/qp/crf. If none, it uses defaults. :contentReference[oaicite:3]{index=3}
        kwargs = {"codec": codec}
        if qp is not None:
            kwargs["qp"] = int(qp)

        self._ctx = mediapy.VideoWriter(path, shape=self.shape_hw, fps=float(fps), **kwargs)
        self._writer = self._ctx.__enter__()

        self._closed = False

    def write(self, rgb: Optional[np.ndarray]) -> None:
        if rgb is None or self._closed:
            return

        img = np.asarray(rgb)

        # Ensure RGB uint8
        if img.dtype != np.uint8:
            img = mediapy.to_uint8(img)

        # Ensure shape matches (H, W, 3)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H,W,3), got shape={img.shape}")

        if tuple(img.shape[:2]) != tuple(self.shape_hw):
            # mediapypy.resize_image expects shape (H,W)
            img = mediapy.resize_image(img, self.shape_hw)
            if img.dtype != np.uint8:
                img = mediapy.to_uint8(img)

        # Write one frame :contentReference[oaicite:4]{index=4}
        self._writer.add_image(img)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None
            self._writer = None



# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CollectorCfg:
    # dataset
    data_size: int = 10
    num_envs: int = 2


    seed: int = 1500
    save_dir: str = "./data/table30_hang_toothbrush_cup_dual_view"

    # env control
    max_ctrl_steps: int = 1200

    # motion position
    max_dp: float = 0.005
    pos_tol: float = 0.001

    # rotation control
    rot_gain: float = 0.6
    max_dr: float = 0.08
    yaw_tol: float = 0.02

    # yaw offsets
    yaw_offset_grasp: float = 0.0

    # keypoints offsets (World Frame)
    grasp_offset: Tuple[float, float, float] = (0.0, 0.0, 0.02)
    pre_grasp_z: float = 0.05

    pre_hang_offset: Tuple[float, float, float] = (-0.046, -0.15, 0.03)
    hang_offset: Tuple[float, float, float] = (-0.046, -0.02, 0.03)
    retreat_dx: float = 0.10

    # gripper
    gripper_open: float = 0.0
    gripper_close: float = 0.8

    # timing / dwell
    close_hold_steps: int = 25
    release_hold_steps: int = 25
    waypoint_dwell_steps: int = 20

    # sampling / render
    sample_every_steps: int = 1
    save_video: bool = True
    render_every_steps: int = 1
    video_fps: int = 30
    video_w: int = 640
    video_h: int = 480
    
    # [MODIFIED] Changed to list of strings for multiple cameras
    cam_view_key: Sequence[str] = field(default_factory=lambda: ["pixels/view_1", "pixels/view_0"])

    # quat convention for sites/poses
    site_quat_convention: str = "xyzw"

    # text fields
    subtask: Optional[str] = None
    prompt: Optional[str] = None


# -----------------------------------------------------------------------------
# Collector
# -----------------------------------------------------------------------------
class HangToothbrushCupCollector:
    # Phase 1: Approach (Manhattan)
    ST_APP_LIFT_Z = 0
    ST_APP_ALIGN_X = 1
    ST_APP_ALIGN_Y = 2
    ST_APP_ALIGN_YAW = 3
    ST_APP_DESCEND = 4

    # Phase 2: Grasp
    ST_CLOSE = 5

    # Phase 3: Transport (Manhattan)
    ST_TRP_LIFT_Z = 6
    ST_TRP_UNYAW = 7
    ST_TRP_ALIGN_X = 8
    ST_TRP_ALIGN_Y = 9
    ST_HANG_DOWN = 10

    # Phase 4: Release & End
    ST_RELEASE = 11
    ST_RETREAT = 12
    ST_GO_RESET = 13
    ST_DONE = 14

    def __init__(self, cfg: CollectorCfg, env_cfg: Optional[HangToothbrushCupEnvCfg] = None):
        self.cfg = cfg
        os.makedirs(cfg.save_dir, exist_ok=True)
        self.videos_dir = os.path.join(cfg.save_dir, "videos")
        os.makedirs(self.videos_dir, exist_ok=True)

        self.env_cfg = env_cfg if env_cfg is not None else HangToothbrushCupEnvCfg()
        self.env = HangToothbrushCupEnv(self.env_cfg, num_envs=int(cfg.num_envs))
        self.env.reset()

        self.model = self.env.model
        self.B = int(cfg.num_envs)
        
        # Determine camera keys
        self.cam_keys = list(cfg.cam_view_key)

        instruction = str(getattr(self.env._cfg, "instruction", "") or "")
        self.ep_subtask = np.array([cfg.subtask or instruction] * self.B, dtype=object)
        self.ep_prompt = np.array([cfg.prompt or (instruction if instruction else "hang toothbrush cup")] * self.B, dtype=object)

        B = self.B
        self.active = np.zeros(B, dtype=bool)
        self.done = np.zeros(B, dtype=bool)
        self.success = np.zeros(B, dtype=bool)
        self.ctrl_step = np.zeros(B, dtype=np.int32)
        self._attempt_id = np.zeros(B, dtype=np.int64)

        self.states = np.zeros(B, dtype=np.int32)
        self.state_enter_step = np.zeros(B, dtype=np.int32)
        self.state_reach_step = np.full(B, -1, dtype=np.int32)

        # Controls
        self.exec_pos = np.zeros((B, 3), dtype=np.float32)
        self.exec_quat = np.zeros((B, 4), dtype=np.float32)  # xyzw
        self.home_pos = np.zeros((B, 3), dtype=np.float32)

        # Latch positions
        self.latched_start_pos = np.zeros((B, 3), dtype=np.float32)
        self.latched_grasp_pos = np.zeros((B, 3), dtype=np.float32)
        self.latched_hook_pos = np.zeros((B, 3), dtype=np.float32)

        # Latch yaw info
        self.latched_start_quat = np.zeros((B, 4), dtype=np.float32)
        self.latched_start_yaw = np.zeros((B,), dtype=np.float32)
        self.latched_cup_yaw = np.zeros((B,), dtype=np.float32)

        self.buffers: List[Dict[str, Any]] = [self._new_buffer() for _ in range(B)]
        
        # [MODIFIED] Video writers are now a List of Dicts: List[Dict[str, EpisodeVideoWriter]]
        # Structure: self.video_writers[env_id][camera_key]
        self.video_writers: List[Dict[str, EpisodeVideoWriter]] = [{} for _ in range(B)]
        
        # [MODIFIED] Temp paths are also keyed by camera name
        # We replace '/' with '_' in filenames to avoid path errors
        self._tmp_video_paths: List[Dict[str, str]] = []
        for i in range(B):
            paths = {}
            for key in self.cam_keys:
                safe_key = key.replace("/", "_")
                paths[key] = os.path.join(self.videos_dir, f"_tmp_env{i}_{safe_key}.mp4")
            self._tmp_video_paths.append(paths)

        self.saved_success = 0
        self.attempted = 0
        self._last_log_t = time.perf_counter()
        self._last_action = np.zeros((B, 7), dtype=np.float32)

        self.grasp_site = self.env.grasp_site
        self.hook_site = self.env.hook_site

    @staticmethod
    def _new_buffer() -> Dict[str, Any]:
        return {
            "times": [],
            "logic_states": [],
            "qpos": [],
            "ee_pose": [],
            "gripper": [],
            "ctrl": [],
            "reward": [],
            "is_success": [],
            "video_frames": 0,
        }

    def start_episodes(self, env_ids: np.ndarray, seed: int) -> None:
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if env_ids.size == 0:
            return

        try:
            self.env._rng = np.random.default_rng(int(seed))
        except Exception:
            pass

        done_mask = np.zeros((self.B,), dtype=bool)
        done_mask[env_ids] = True
        self.env.reset(done=done_mask)

        self.active[env_ids] = True
        self.done[env_ids] = False
        self.success[env_ids] = False
        self.ctrl_step[env_ids] = 0
        self.state_enter_step[env_ids] = 0
        self.state_reach_step[env_ids] = -1

        data = self.env._state.data
        self.env.robot.reset_envs(data, done_mask)
        self.env.robot.update_reference(data)

        # --- Robot start pose ---
        ee_pose_raw = self.env.robot.get_ee_pose(data)
        self.exec_pos[env_ids] = ee_pose_raw[env_ids, :3]
        self.latched_start_pos[env_ids] = ee_pose_raw[env_ids, :3]
        self.home_pos[env_ids] = ee_pose_raw[env_ids, :3]

        if ee_pose_raw.shape[1] == 6:
            r = Rotation.from_euler("xyz", ee_pose_raw[env_ids, 3:6], degrees=False)
            self.exec_quat[env_ids] = r.as_quat().astype(np.float32)
        else:
            self.exec_quat[env_ids] = ee_pose_raw[env_ids, 3:7].astype(np.float32)

        self.latched_start_quat[env_ids] = self.exec_quat[env_ids].copy()
        self.latched_start_yaw[env_ids] = Rotation.from_quat(self.exec_quat[env_ids]).as_euler("xyz", degrees=False)[:, 2].astype(
            np.float32
        )

        # --- Site poses (grasp/hook) ---
        grasp_pose7 = np.asarray(self.grasp_site.get_pose(data), dtype=np.float32).reshape(self.B, -1)
        hook_pose7 = np.asarray(self.hook_site.get_pose(data), dtype=np.float32).reshape(self.B, -1)

        self.latched_grasp_pos[env_ids] = grasp_pose7[env_ids, :3]
        self.latched_hook_pos[env_ids] = hook_pose7[env_ids, :3]

        # Cup yaw from grasp site quaternion
        grasp_q = grasp_pose7[env_ids, 3:7]
        grasp_q = to_xyzw_from_possible_wxyz(grasp_q, assume=str(self.cfg.site_quat_convention))
        cup_yaw = Rotation.from_quat(grasp_q).as_euler("xyz", degrees=False)[:, 2]
        self.latched_cup_yaw[env_ids] = cup_yaw.astype(np.float32)

        # Start state
        self.states[env_ids] = self.ST_APP_LIFT_Z

        for env_id in env_ids.tolist():
            self.buffers[env_id] = self._new_buffer()
            # Clear old writers
            self._close_writers_for_env(env_id)
            if self.cfg.save_video:
                self._reset_video_writer(env_id)
            self._attempt_id[env_id] += 1
            self.attempted += 1

    def _close_writers_for_env(self, env_id: int):
        writers = self.video_writers[env_id]
        for key in list(writers.keys()):
            writers[key].close()
        self.video_writers[env_id] = {}

    def _reset_video_writer(self, env_id: int):
        # [MODIFIED] Initialize writers for all configured keys
        for key in self.cam_keys:
            tmp_path = self._tmp_video_paths[env_id][key]
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            
            self.video_writers[env_id][key] = EpisodeVideoWriter(
                tmp_path, 
                int(self.cfg.video_fps), 
                (int(self.cfg.video_w), int(self.cfg.video_h))
            )

    def _capture_step(self, env_id: int) -> None:
        obs = self.env._state.obs
        info = self.env._state.info

        buf = self.buffers[env_id]
        buf["times"].append(float(self.ctrl_step[env_id] * 0.02))
        buf["logic_states"].append(int(self.states[env_id]))
        buf["qpos"].append(obs["qpos"][env_id].tolist())
        buf["ee_pose"].append(obs["ee_pose"][env_id].tolist())
        buf["gripper"].append(obs["gripper"][env_id].tolist())
        buf["ctrl"].append(self._last_action[env_id].tolist())
        buf["reward"].append(float(np.asarray(self.env._state.reward).reshape(-1)[env_id]))

        is_success = bool(np.asarray(info.get("is_success", False)).reshape(-1)[env_id])
        buf["is_success"].append(is_success)

    def _write_video_frame(self, env_id: int) -> None:
        writers = self.video_writers[env_id]
        if not writers:
            return
            
        obs = self.env._state.obs
        frame_written = False
        
        # [MODIFIED] Loop through all keys and write to respective video files
        for key in self.cam_keys:
            if key in obs and key in writers:
                rgb = obs[key][env_id]
                if rgb is not None:
                    writers[key].write(rgb.copy())
                    frame_written = True
        
        # Only increment frame count if we actually wrote something
        if frame_written:
            self.buffers[env_id]["video_frames"] += 1

    def _flush_jsonl(self, env_id: int, ep_idx: int, vid_paths_map: Dict[str, str]):
            path = os.path.join(self.cfg.save_dir, f"episode_{ep_idx:05d}.jsonl")
            buf = self.buffers[env_id]
            n = len(buf["times"])
            prompt = str(self.ep_prompt[env_id])

            with open(path, "w", encoding="utf-8") as f:
                for i in range(n):
                    # 1. 获取原始数据
                    raw_pose = buf["ee_pose"][i]   # 仿真原始数据 (通常是 [x,y,z, qx,qy,qz,qw])
                    raw_gripper = buf["gripper"][i] # 夹爪数据 (通常是 [width])
                    
                    # 2. 解析位置和旋转
                    # 假设 raw_pose 是标准的 7维 [x, y, z, qx, qy, qz, qw]
                    x, y, z = raw_pose[0], raw_pose[1], raw_pose[2]
                    

                    euler = raw_pose[3:6]

                    # 3. 应用你的修正公式
                    # (1) Z轴高度补偿
                    new_z = z + 0.1525
                    
                    # (2) Yaw角修正: -1/2 * pi - value
                    # euler[0]=roll, euler[1]=pitch, euler[2]=yaw
                    old_yaw = euler[2]
                    new_yaw = -0.5 * np.pi - old_yaw

                    # 4. 构造你要求的 7维 ee_pose: [x, y, z, r, p, y, gripper]
                    # 提取夹爪数值 (如果它是列表)
                    g_val = raw_gripper[0] if isinstance(raw_gripper, (list, np.ndarray)) else float(raw_gripper)
                    
                    # 组合最终的 list
                    custom_ee_pose_7d = [
                        x,          # 0: x
                        y,          # 1: y
                        new_z,      # 2: z (已修正)
                        euler[0],   # 3: roll
                        euler[1],   # 4: pitch
                        new_yaw,    # 5: yaw (已修正)
                        g_val       # 6: gripper
                    ]

                    rec = {
                        "prompt": prompt,
                        "qpos": buf["qpos"][i],
                        "ee_pose": custom_ee_pose_7d,  # <--- 使用修正后的自定义7维格式
                        "gripper": buf["gripper"][i],  # 这里可以保留原始夹爪数据作为备份
                        "ctrl": buf["ctrl"][i],
                        "is_robot": True,
                    }
                    
                    # 处理多视角视频
                    for k_idx, key in enumerate(self.cam_keys):
                        json_key = f"images_{k_idx + 1}"
                        if key in vid_paths_map:
                            rec[json_key] = {
                                "url": vid_paths_map[key],
                                "type": "video", 
                                "frame_idx": i
                            }

                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _finalize_episode(self, env_id: int) -> None:
        # Close all writers for this env
        self._close_writers_for_env(env_id)

        if self.success[env_id] and (self.saved_success < int(self.cfg.data_size)):
            ep_idx = int(self.saved_success)
            saved_paths_map = {}

            if self.cfg.save_video:
                # [MODIFIED] Move all temp videos to final paths
                for key in self.cam_keys:
                    tmp_path = self._tmp_video_paths[env_id][key]
                    if os.path.exists(tmp_path):
                        safe_key = key.replace("/", "_")
                        final_rel_path = f"videos/episode_{ep_idx:05d}_{safe_key}.mp4"
                        abs_video_path = os.path.join(self.cfg.save_dir, final_rel_path)
                        
                        try:
                            shutil.move(tmp_path, abs_video_path)
                            saved_paths_map[key] = final_rel_path
                        except Exception as e:
                            print(f"Error moving video file: {e}")

            self._flush_jsonl(env_id, ep_idx, saved_paths_map)
            self.saved_success += 1
            print(f"[Saved] episode {ep_idx}. Total saved: {self.saved_success}")

        # Clean up any remaining temp files
        for key in self.cam_keys:
            tmp_path = self._tmp_video_paths[env_id][key]
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        
        self.buffers[env_id] = self._new_buffer()

    # ----------------------------
    # Core Logic (Unchanged)
    # ----------------------------
    def _enter_state(self, mask: np.ndarray, new_state: int) -> None:
        if not np.any(mask):
            return
        self.states[mask] = int(new_state)
        self.state_enter_step[mask] = self.ctrl_step[mask].copy()
        self.state_reach_step[mask] = -1

    def _step_logic(self) -> None:
        cfg = self.cfg
        B = self.B

        running = self.active & (~self.done)
        if not np.any(running):
            return

        # 1) Manhattan keypoints
        start_p = self.latched_start_pos
        grasp_p = self.latched_grasp_pos + np.asarray(cfg.grasp_offset, dtype=np.float32).reshape(1, 3)
        hook_p = self.latched_hook_pos

        safe_z = grasp_p[:, 2] + float(cfg.pre_grasp_z)

        p_app_lift_z = start_p.copy()
        p_app_lift_z[:, 2] = safe_z

        p_app_align_x = p_app_lift_z.copy()
        p_app_align_x[:, 0] = grasp_p[:, 0]

        p_app_align_y = grasp_p.copy()
        p_app_align_y[:, 2] = safe_z

        p_app_descend = grasp_p

        pre_hang_p = hook_p + np.asarray(cfg.pre_hang_offset, dtype=np.float32).reshape(1, 3)
        hang_p = hook_p + np.asarray(cfg.hang_offset, dtype=np.float32).reshape(1, 3)

        p_trp_lift_z = grasp_p.copy()
        p_trp_lift_z[:, 2] = pre_hang_p[:, 2]

        p_trp_align_x = p_trp_lift_z.copy()
        p_trp_align_x[:, 0] = pre_hang_p[:, 0]

        p_trp_align_y = pre_hang_p

        retreat_p = hang_p.copy()
        retreat_p[:, 0] -= float(cfg.retreat_dx)

        reset_p = self.home_pos

        # 2) Target assignment
        s = self.states

        tgt_pos = self.exec_pos.copy()
        tgt_yaw = np.full((B,), np.nan, dtype=np.float32)
        grip_cmd = np.full((B,), float(cfg.gripper_open), dtype=np.float32)

        def set_target(state_id, pos, grip):
            mask = running & (s == state_id)
            if np.any(mask):
                tgt_pos[mask] = pos[mask]
                grip_cmd[mask] = float(grip)

        def set_yaw_target(state_id, yaw_arr):
            mask = running & (s == state_id)
            if np.any(mask):
                tgt_yaw[mask] = yaw_arr[mask]

        # Approach
        set_target(self.ST_APP_LIFT_Z, p_app_lift_z, cfg.gripper_open)
        set_target(self.ST_APP_ALIGN_X, p_app_align_x, cfg.gripper_open)
        set_target(self.ST_APP_ALIGN_Y, p_app_align_y, cfg.gripper_open)

        # Yaw align to cup
        set_target(self.ST_APP_ALIGN_YAW, p_app_align_y, cfg.gripper_open)
        yaw_grasp = (self.latched_cup_yaw + float(cfg.yaw_offset_grasp)).astype(np.float32) * -1
        set_yaw_target(self.ST_APP_ALIGN_YAW, yaw_grasp)

        set_target(self.ST_APP_DESCEND, p_app_descend, cfg.gripper_open)
        set_target(self.ST_CLOSE, p_app_descend, cfg.gripper_close)

        # Transport
        set_target(self.ST_TRP_LIFT_Z, p_trp_lift_z, cfg.gripper_close)
        set_target(self.ST_TRP_UNYAW, p_trp_lift_z, cfg.gripper_close)
        yaw_back = self.latched_start_yaw.astype(np.float32) * -1
        set_yaw_target(self.ST_TRP_UNYAW, yaw_back)

        set_target(self.ST_TRP_ALIGN_X, p_trp_align_x, cfg.gripper_close)
        set_target(self.ST_TRP_ALIGN_Y, p_trp_align_y, cfg.gripper_close)
        set_target(self.ST_HANG_DOWN, hang_p, cfg.gripper_close)

        # Release & End
        set_target(self.ST_RELEASE, hang_p, cfg.gripper_open)
        set_target(self.ST_RETREAT, retreat_p, cfg.gripper_open)
        set_target(self.ST_GO_RESET, reset_p, cfg.gripper_open)

        # 3) Compute control
        limit = np.where(self.states[:, None] == self.ST_HANG_DOWN, cfg.max_dp * 0.333, cfg.max_dp)

        self.exec_pos = smooth_step_pos(self.exec_pos, tgt_pos, limit)

        ref_pose = self.env.robot.ref_ee_pose
        ref_pos = ref_pose[:, :3]

        if ref_pose.shape[1] == 6:
            ref_quat = Rotation.from_euler("xyz", ref_pose[:, 3:6], degrees=False).as_quat().astype(np.float32)
        else:
            ref_quat = ref_pose[:, 3:7].astype(np.float32)

        want_rot = running & (~np.isnan(tgt_yaw))
        desired_quat = self.exec_quat.copy()

        obs = self.env._state.obs
        ee_pose = obs.get("ee_pose", None)
        if ee_pose is not None and ee_pose.shape[1] == 7:
            ee_quat = ee_pose[:, 3:7].astype(np.float32)
        elif ee_pose is not None and ee_pose.shape[1] == 6:
            ee_quat = Rotation.from_euler("xyz", ee_pose[:, 3:6], degrees=False).as_quat().astype(np.float32)
        else:
            ee_quat = ref_quat.copy()

        curr_yaw = Rotation.from_quat(ee_quat).as_euler("xyz", degrees=False)[:, 2].astype(np.float32)

        if np.any(want_rot):
            target_y_raw = tgt_yaw[want_rot].astype(np.float32)
            target_y = closest_yaw(target_y_raw, curr_yaw[want_rot]).astype(np.float32)

            start_y = self.latched_start_yaw[want_rot].astype(np.float32)
            delta_y = wrap_to_pi(target_y - start_y).astype(np.float32)

            r_delta = Rotation.from_euler("z", delta_y)
            r_start = Rotation.from_quat(self.latched_start_quat[want_rot])
            r_target = r_delta * r_start

            desired_quat[want_rot] = r_target.as_quat().astype(np.float32)
            self.exec_quat[want_rot] = desired_quat[want_rot]

        rotvec_cmd = np.zeros((B, 3), dtype=np.float32)
        if np.any(want_rot):
            r_des = Rotation.from_quat(desired_quat[want_rot])
            r_ref = Rotation.from_quat(ref_quat[want_rot])
            r_err = r_des * r_ref.inv()
            rv = r_err.as_rotvec().astype(np.float32)

            mag = np.linalg.norm(rv, axis=1, keepdims=True) + 1e-9
            scale = np.minimum(1.0, float(cfg.max_dr) / mag)
            rv = rv * scale

            rotvec_cmd[want_rot] = rv * float(cfg.rot_gain)

        action = np.zeros((B, 7), dtype=np.float32)
        action[:, :3] = (self.exec_pos - ref_pos) * 0.5
        action[:, 3:6] = rotvec_cmd
        action[:, 6] = grip_cmd

        self._last_action[:] = action
        self.env.step(action)

        # 4) Checks & FSM transitions
        def is_reached(p: np.ndarray) -> np.ndarray:
            return np.linalg.norm(self.exec_pos - p, axis=1) < float(cfg.pos_tol)

        def _check_reach_dwell(state_idx, p):
            in_s = running & (self.states == state_idx)
            ok = is_reached(p)
            just = in_s & ok & (self.state_reach_step == -1)
            self.state_reach_step[just] = self.ctrl_step[just]
            dwell = (self.ctrl_step - self.state_reach_step) >= int(cfg.waypoint_dwell_steps)
            return in_s & (self.state_reach_step != -1) & dwell & ok

        def _check_yaw_dwell(state_idx, y):
            in_s = running & (self.states == state_idx)
            dy = wrap_to_pi(curr_yaw - y)
            ok = np.abs(dy) < float(cfg.yaw_tol)
            just = in_s & ok & (self.state_reach_step == -1)
            self.state_reach_step[just] = self.ctrl_step[just]
            dwell = (self.ctrl_step - self.state_reach_step) >= int(cfg.waypoint_dwell_steps)
            return in_s & (self.state_reach_step != -1) & dwell & ok

        self._enter_state(_check_reach_dwell(self.ST_APP_LIFT_Z, p_app_lift_z), self.ST_APP_ALIGN_X)
        self._enter_state(_check_reach_dwell(self.ST_APP_ALIGN_X, p_app_align_x), self.ST_APP_ALIGN_Y)
        self._enter_state(_check_reach_dwell(self.ST_APP_ALIGN_Y, p_app_align_y), self.ST_APP_ALIGN_YAW)
        self._enter_state(_check_yaw_dwell(self.ST_APP_ALIGN_YAW, yaw_grasp), self.ST_APP_DESCEND)
        self._enter_state(_check_reach_dwell(self.ST_APP_DESCEND, p_app_descend), self.ST_CLOSE)

        mask_close = running & (self.states == self.ST_CLOSE)
        if np.any(mask_close):
            done_close = (self.ctrl_step - self.state_enter_step) >= int(cfg.close_hold_steps)
            self._enter_state(mask_close & done_close, self.ST_TRP_LIFT_Z)

        self._enter_state(_check_reach_dwell(self.ST_TRP_LIFT_Z, p_trp_lift_z), self.ST_TRP_UNYAW)
        self._enter_state(_check_yaw_dwell(self.ST_TRP_UNYAW, yaw_back), self.ST_TRP_ALIGN_X)
        self._enter_state(_check_reach_dwell(self.ST_TRP_ALIGN_X, p_trp_align_x), self.ST_TRP_ALIGN_Y)
        self._enter_state(_check_reach_dwell(self.ST_TRP_ALIGN_Y, p_trp_align_y), self.ST_HANG_DOWN)
        self._enter_state(_check_reach_dwell(self.ST_HANG_DOWN, hang_p), self.ST_RELEASE)

        in_rel = running & (self.states == self.ST_RELEASE)
        done_rel = in_rel & ((self.ctrl_step - self.state_enter_step) >= int(cfg.release_hold_steps))
        self._enter_state(done_rel, self.ST_RETREAT)

        done_ret = is_reached(retreat_p)
        self._enter_state(running & (self.states == self.ST_RETREAT) & done_ret, self.ST_GO_RESET)

        done_rst = is_reached(reset_p)
        self._enter_state(running & (self.states == self.ST_GO_RESET) & done_rst, self.ST_DONE)

        info = self.env._state.info
        is_success = np.asarray(
            info.get("is_success", info.get("success", np.zeros((B,), dtype=np.bool_))),
            dtype=np.bool_,
        ).reshape(-1)

        finished = running & is_success
        if np.any(finished):
            self.states[finished] = self.ST_DONE

    def collect(self) -> None:
        cfg = self.cfg
        target = int(cfg.data_size)

        all_ids = np.arange(self.B, dtype=np.int64)
        self.start_episodes(all_ids, seed=int(cfg.seed))

        while self.saved_success < target:
            self._step_logic()

            running = self.active & (~self.done)
            sample_mask = running & ((self.ctrl_step % int(cfg.sample_every_steps)) == 0)
            for env_id in np.where(sample_mask)[0].tolist():
                self._capture_step(env_id)

            if cfg.save_video:
                render_mask = running & ((self.ctrl_step % int(cfg.render_every_steps)) == 0)
                for env_id in np.where(render_mask)[0].tolist():
                    self._write_video_frame(env_id)

            self.ctrl_step[running] += 1

            info = self.env._state.info
            is_success = np.asarray(
                info.get("is_success", info.get("success", np.zeros((self.B,), dtype=np.bool_))),
                dtype=np.bool_,
            ).reshape(-1)

            for i in range(self.B):
                if (not self.active[i]) or self.done[i]:
                    continue
                timeout = int(self.ctrl_step[i]) >= int(cfg.max_ctrl_steps)
                finished = bool(is_success[i]) or (int(self.states[i]) == int(self.ST_DONE))
                if finished or timeout:
                    self.done[i] = True
                    self.success[i] = bool(is_success[i])

            done_ids = np.where(self.active & self.done)[0]
            for env_id in done_ids.tolist():
                self._finalize_episode(int(env_id))

            if self.saved_success >= target:
                self.active[done_ids] = False
            else:
                restart_ids = done_ids[self.active[done_ids]]
                if restart_ids.size > 0:
                    batch_seed = int(cfg.seed + self.attempted)
                    self.done[restart_ids] = False
                    self.start_episodes(restart_ids, seed=batch_seed)

            now = time.perf_counter()
            if (now - self._last_log_t) >= 2.0:
                print(
                    f"[collect] active={int(np.sum(self.active))}/{self.B} "
                    f"saved_success={int(self.saved_success)}/{target} attempted={int(self.attempted)}"
                )
                self._last_log_t = now

        print(f"[DONE] saved_success={self.saved_success}/{target}")
        print(f"Saved to: {cfg.save_dir}")

    def close(self) -> None:
        # [MODIFIED] Close all writers in the dictionary structure
        for env_writers in self.video_writers:
            for vw in env_writers.values():
                if vw is not None:
                    vw.close()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--data_size", type=int, default=None)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no_video", action="store_true")
    p.add_argument("--max_ctrl_steps", type=int, default=None)
    p.add_argument("--action_mode", type=str, default=None)
    p.add_argument("--site_quat_convention", type=str, default=None, choices=["xyzw", "wxyz"])
    args = p.parse_args()

    env_cfg = HangToothbrushCupEnvCfg()
    if args.action_mode is not None:
        env_cfg.action_mode = str(args.action_mode)

    cfg = CollectorCfg(
        save_dir=args.save_dir if args.save_dir is not None else CollectorCfg.save_dir,
        data_size=args.data_size if args.data_size is not None else CollectorCfg.data_size,
        num_envs=args.num_envs if args.num_envs is not None else CollectorCfg.num_envs,
        seed=args.seed if args.seed is not None else CollectorCfg.seed,
        save_video=(not args.no_video),
        max_ctrl_steps=args.max_ctrl_steps if args.max_ctrl_steps is not None else CollectorCfg.max_ctrl_steps,
        site_quat_convention=args.site_quat_convention if args.site_quat_convention else CollectorCfg.site_quat_convention,
    )

    runner = HangToothbrushCupCollector(cfg, env_cfg=env_cfg)
    try:
        runner.collect()
    finally:
        runner.close()


if __name__ == "__main__":
    main()