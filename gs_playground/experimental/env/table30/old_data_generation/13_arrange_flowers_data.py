# =============================================================================
# File: collect_arrange_flowers.py
# =============================================================================
from __future__ import annotations

import os
import json
import time
import shutil
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Sequence
import mediapy
import numpy as np

from scipy.spatial.transform import Rotation

from gs_playground.src.manipulation.tasks.table30._13_arrange_flowers import (
    ArrangeFlowersEnv,
    ArrangeFlowersEnvCfg,
)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def smooth_step_pos(curr: np.ndarray, tgt: np.ndarray, max_dp: float) -> np.ndarray:
    dp = tgt - curr
    n = np.linalg.norm(dp, axis=1, keepdims=True)
    s = np.minimum(1.0, float(max_dp) / (n + 1e-9))
    return curr + dp * s


def normalize_quat(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / (n + 1e-9)


def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def quat_to_yaw(q_xyzw: np.ndarray) -> np.ndarray:
    # q shape (..., 4), xyzw
    return Rotation.from_quat(q_xyzw).as_euler("xyz", degrees=False)[..., 2].astype(np.float32)


def obs_ee_quat_xyzw(obs_ee_pose_row: np.ndarray) -> np.ndarray:
    """
    Robustly interpret obs['ee_pose'][i] rotation part:
      - if len==7: xyz + quat(xyzw)
      - if len==6: xyz + euler(xyz)
    Return quat(xyzw).
    """
    if obs_ee_pose_row.shape[0] == 7:
        q = obs_ee_pose_row[3:7].astype(np.float32)
        return normalize_quat(q[None, :])[0]
    if obs_ee_pose_row.shape[0] == 6:
        e = obs_ee_pose_row[3:6].astype(np.float32)
        q = Rotation.from_euler("xyz", e, degrees=False).as_quat().astype(np.float32)
        return normalize_quat(q[None, :])[0]
    # fallback
    return Rotation.identity().as_quat().astype(np.float32)


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
    data_size: int = 1
    num_envs: int = 1
    seed: int = 0
    save_dir: str = "./data/table30_arrange_flowers_vase_swap_test"

    # env control
    max_ctrl_steps: int = 1700

    # motion params
    max_dp: float = 0.005
    pos_tol: float = 0.002

    # rotation control (yaw-only task but we still command rotvec)
    rot_gain: float = 1.0
    max_dr: float = 0.01
    yaw_tol: float = 0.08              # rad, for yaw-only reach check (replaces angle_tol for rotation states)
    rot_dwell_steps: int = 15
    rot_state_timeout_steps: int = 250 # avoid dead loop in rotation states

    # task offsets (base grasp offset)
    grasp_offset: Tuple[float, float, float] = (-0.03, -0.02, 0.0)
    lift_height_z: float = 0.45

    # insertion
    insert_depth: float = 0.25
    align_z_offset: float = 0.10

    # pregrasp margin (avoid pushing flower before grasp)
    pregrasp_z_margin: float = 0.03

    # gripper
    gripper_open: float = 0.0
    gripper_close: float = 0.82

    # timing
    close_hold_steps: int = 20
    waypoint_dwell_steps: int = 15

    # retreat
    retreat_dx: float = 0.08

    # yaw interfaces (degrees)
    pick1_yaw_deg: float = -30.0
    pick2_yaw_deg: float = -30.0

    # -----------------------------
    # Beautified offsets (replacing your hard-coded tweaks)
    # -----------------------------
    # pick1 coarse approach extra offset (applied on top of flower+grasp_offset for coarse X/Y)
    pick1_coarse_xy_offset: Tuple[float, float] = (-0.03, 0.0)

    # place1 hover offset around dst vase center (XY)
    place1_dst_hover_xy_offset: Tuple[float, float] = (0.03, 0.02)

    # pick2 grasp extra Z (on top of flower_z + grasp_offset_z)
    pick2_grasp_z_extra: float = 0.05

    # place2 hover offset around src vase center (XY)
    place2_src_hover_xy_offset: Tuple[float, float] = (0.05, 0.03)

    # place2 insert extra Z (negative => deeper) in addition to insert_depth
    place2_insert_z_extra: float = -0.08

    # action scaling (you手动写过 pos_err*0.5, grip*0.98 这类)
    action_pos_scale: float = 0.5
    action_grip_scale: float = 1.0

    # -----------------------------
    # Video / multi-view saving
    # -----------------------------
    sample_every_steps: int = 1
    save_video: bool = True
    render_every_steps: int = 1
    video_fps: int = 30
    video_w: int = 640
    video_h: int = 480
    cam_view_key: Sequence[str] = field(default_factory=lambda: ["pixels/view_0", "pixels/view_1"])

    # MotrixSim renderer video (DEBUG) - keep your previous behavior
    enable_motrix_video: bool = False
    motrix_video_fps: int = 30
    motrix_video_width: int = 640
    motrix_video_height: int = 480

    instruction: str = (
        "Pick up the flower from the source vase, place it into the target vase, "
        "then pick it up again and place it back into the source vase. "
        "Each motion first aligns XY, then performs Z motion. Rotation only changes yaw. "
        "Before grasp, do a pregrasp descend then fine-align (Z->Y->X), then final descend and grasp."
    )


# -----------------------------------------------------------------------------
# Collector
# -----------------------------------------------------------------------------
class ArrangeFlowersCollector:
    # Pick from source vase (pick1)
    ST_APP1_LIFT_Z = 0
    ST_APP1_ALIGN_X = 1
    ST_APP1_ALIGN_Y = 2
    ST_ROT_PICK1 = 3
    ST_PRE1_Z = 4
    ST_PRE1_ALIGN_Y = 5
    ST_PRE1_ALIGN_X = 6
    ST_DESCEND1 = 7
    ST_CLOSE1 = 8
    ST_LIFT1 = 9

    # Place into target vase (place1)
    ST_TRP1_ALIGN_X = 10
    ST_TRP1_ALIGN_Y = 11
    ST_PLACE1_DESCEND = 12
    ST_OPEN1 = 13
    ST_RETREAT1 = 14

    # Pick again (pick2) from target vase
    ST_APP2_LIFT_Z = 15
    ST_APP2_ALIGN_X = 16
    ST_APP2_ALIGN_Y = 17
    ST_ROT_PICK2 = 18
    ST_PRE2_Z = 19
    ST_PRE2_ALIGN_Y = 20
    ST_PRE2_ALIGN_X = 21
    ST_DESCEND2 = 22
    ST_CLOSE2 = 23
    ST_LIFT2 = 24

    # Place back into source vase (place2)
    ST_TRP2_ALIGN_X = 25
    ST_TRP2_ALIGN_Y = 26
    ST_PLACE2_DESCEND = 27
    ST_OPEN2 = 28
    ST_RETREAT2 = 29

    ST_DONE = 30

    def __init__(self, cfg: CollectorCfg, env_cfg: Optional[ArrangeFlowersEnvCfg] = None):
        self.cfg = cfg
        os.makedirs(cfg.save_dir, exist_ok=True)
        self.videos_dir = os.path.join(cfg.save_dir, "videos")
        os.makedirs(self.videos_dir, exist_ok=True)

        # Env cfg
        self.env_cfg = env_cfg if env_cfg is not None else ArrangeFlowersEnvCfg()

        # Pass MotrixSim recorder config into env cfg (best-effort)
        setattr(self.env_cfg, "enable_motrix_video", bool(cfg.enable_motrix_video))
        setattr(self.env_cfg, "motrix_video_fps", int(cfg.motrix_video_fps))
        setattr(self.env_cfg, "motrix_video_width", int(cfg.motrix_video_width))
        setattr(self.env_cfg, "motrix_video_height", int(cfg.motrix_video_height))
        setattr(self.env_cfg, "motrix_video_output_dir", self.videos_dir)

        self.env = ArrangeFlowersEnv(self.env_cfg, num_envs=int(cfg.num_envs))
        self.env.reset()

        self.B = int(cfg.num_envs)

        self.flower_body = self.env.flower_body
        self.vase_src_body = getattr(self.env, "vase_src_body", self.env.vase_body)
        self.vase_dst_body = getattr(self.env, "vase_dst_body", None)
        if self.vase_dst_body is None:
            raise RuntimeError("Env missing vase_dst_body (vase2). Please update env to include vase2.")

        # cameras
        self.cam_keys = list(cfg.cam_view_key)

        # lifecycle
        B = self.B
        self.active = np.zeros(B, dtype=bool)
        self.done = np.zeros(B, dtype=bool)
        self.success = np.zeros(B, dtype=bool)
        self.ctrl_step = np.zeros(B, dtype=np.int32)

        # FSM
        self.states = np.zeros(B, dtype=np.int32)
        self.state_enter_step = np.zeros(B, dtype=np.int32)
        self.state_reach_step = np.full(B, -1, dtype=np.int32)
        self._attempt_id = np.zeros(B, dtype=np.int64)

        # executor
        self.exec_pos = np.zeros((B, 3), dtype=np.float32)
        self.exec_quat = np.zeros((B, 4), dtype=np.float32)  # xyzw

        # rotation lock pos (critical)
        self.rot_lock_pos = np.zeros((B, 3), dtype=np.float32)

        # latches
        self.latched_start_pos = np.zeros((B, 3), dtype=np.float32)
        self.latched_start_quat = np.zeros((B, 4), dtype=np.float32)
        self.hold_quat = np.zeros((B, 4), dtype=np.float32)

        # buffers + multi-view video
        self.buffers: List[Dict[str, Any]] = [self._new_buffer() for _ in range(B)]
        self.video_writers: List[Dict[str, EpisodeVideoWriter]] = [{} for _ in range(B)]
        self._tmp_video_paths: List[Dict[str, str]] = []
        for i in range(B):
            per_cam = {}
            for key in self.cam_keys:
                safe_key = key.replace("/", "_")
                per_cam[key] = os.path.join(self.videos_dir, f"_tmp_env{i}_{safe_key}.mp4")
            self._tmp_video_paths.append(per_cam)

        self.saved_count = 0
        self.attempted = 0
        self._last_log_t = time.perf_counter()
        self._last_action = np.zeros((B, 7), dtype=np.float32)

        # motrix recorder sanity flag
        self._warned_no_motrix = False

    # -----------------------------
    # Buffer / Video
    # -----------------------------
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
            "video_frames": 0,     # number of frames already written
            "frame_idxs": [],      # per sample record -> intended frame index
        }

    def _close_writers_for_env(self, env_id: int) -> None:
        writers = self.video_writers[env_id]
        for k in list(writers.keys()):
            try:
                writers[k].close()
            except Exception:
                pass
        self.video_writers[env_id] = {}

    def _reset_video_writer(self, env_id: int) -> None:
        self._close_writers_for_env(env_id)
        for key in self.cam_keys:
            tmp_path = self._tmp_video_paths[env_id][key]
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            self.video_writers[env_id][key] = EpisodeVideoWriter(
                tmp_path, int(self.cfg.video_fps), (int(self.cfg.video_w), int(self.cfg.video_h))
            )

    def _write_video_frame(self, env_id: int) -> None:
        writers = self.video_writers[env_id]
        if not writers:
            return

        obs = self.env._state.obs
        wrote_any = False
        for key in self.cam_keys:
            if key in obs and key in writers:
                rgb = obs[key][env_id]
                if rgb is not None:
                    writers[key].write(rgb.copy())

                    wrote_any = True

        if wrote_any:
            self.buffers[env_id]["video_frames"] += 1

    def _capture_step(self, env_id: int) -> None:
        obs = self.env._state.obs
        buf = self.buffers[env_id]

        # frame index alignment: because your loop does capture then render,
        # the frame for this record will be written after capture => current video_frames is the correct frame_idx.
        frame_idx: Optional[int] = None
        if self.cfg.save_video:
            if (int(self.ctrl_step[env_id]) % int(self.cfg.render_every_steps)) == 0:
                frame_idx = int(buf["video_frames"])

        buf["frame_idxs"].append(frame_idx)

        buf["times"].append(float(self.ctrl_step[env_id] * 0.02))
        buf["logic_states"].append(int(self.states[env_id]))
        buf["qpos"].append(obs["qpos"][env_id].tolist())
        buf["ee_pose"].append(obs["ee_pose"][env_id].tolist())
        buf["gripper"].append(obs["gripper"][env_id].tolist())
        buf["ctrl"].append(self._last_action[env_id].tolist())

        # keep the original flower script convention
        buf["is_success"].append(bool(self.success[env_id]))
        buf["reward"].append(float(1.0 if self.success[env_id] else 0.0))

    def _flush_jsonl(self, env_id: int, ep_idx: int, vid_paths_map: Dict[str, str]) -> None:
        path = os.path.join(self.cfg.save_dir, f"episode_{ep_idx:05d}.jsonl")
        buf = self.buffers[env_id]
        n = len(buf["times"])
        prompt = str(self.cfg.instruction)

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
        # close writers
        self._close_writers_for_env(env_id)

        if self.success[env_id] and  self.saved_count < int(self.cfg.data_size):
            ep_idx = int(self.saved_count)
            saved_paths_map: Dict[str, str] = {}

            if self.cfg.save_video:
                for key in self.cam_keys:
                    tmp_path = self._tmp_video_paths[env_id][key]
                    if os.path.exists(tmp_path):
                        safe_key = key.replace("/", "_")
                        final_rel_path = f"videos/episode_{ep_idx:05d}_{safe_key}.mp4"
                        abs_path = os.path.join(self.cfg.save_dir, final_rel_path)
                        try:
                            shutil.move(tmp_path, abs_path)
                            saved_paths_map[key] = final_rel_path
                        except Exception as e:
                            print(f"[WARN] move video failed ({key}): {e}")

            self._flush_jsonl(env_id, ep_idx, saved_paths_map)
            self.saved_count += 1
            print(f"[Saved] Episode {ep_idx}. Total saved: {self.saved_count}")

        # cleanup temp leftovers
        for key in self.cam_keys:
            tmp_path = self._tmp_video_paths[env_id][key]
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        self.buffers[env_id] = self._new_buffer()

    # -----------------------------
    # Episode control
    # -----------------------------
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

        # init EE
        all_poses = self.env.robot.get_ee_pose(data)
        for idx in env_ids.tolist():
            pose = all_poses[idx]
            self.exec_pos[idx] = pose[:3]
            self.latched_start_pos[idx] = pose[:3]

            if len(pose) == 7:
                self.exec_quat[idx] = pose[3:]
            elif len(pose) == 6:
                euler = pose[3:]
                self.exec_quat[idx] = Rotation.from_euler("xyz", euler, degrees=False).as_quat()
            else:
                self.exec_quat[idx] = Rotation.identity().as_quat()

            self.exec_quat[idx] = normalize_quat(self.exec_quat[idx][None, :])[0].astype(np.float32)
            self.latched_start_quat[idx] = self.exec_quat[idx].copy()

        self.hold_quat[env_ids] = self.latched_start_quat[env_ids].copy()
        self.rot_lock_pos[env_ids] = self.latched_start_pos[env_ids].copy()

        # start state
        self.states[env_ids] = self.ST_APP1_LIFT_Z

        for env_id in env_ids.tolist():
            self.buffers[env_id] = self._new_buffer()

            if self.cfg.save_video:
                self._reset_video_writer(env_id)

            # restart MotrixSim recorder for env0
            if env_id == 0 and bool(self.cfg.enable_motrix_video):
                rec = getattr(self.env, "_motrix_recorder", None)
                if rec is not None:
                    rec.restart_episode(self.videos_dir, int(self.saved_count))
                else:
                    if not self._warned_no_motrix:
                        print(
                            "[WARN] enable_motrix_video=True but env._motrix_recorder is None. "
                            "Please implement Motrix recorder in ArrangeFlowersEnv like ArrangeFruitsEnv."
                        )
                        self._warned_no_motrix = True

            self._attempt_id[env_id] += 1
            self.attempted += 1

    # -----------------------------
    # FSM helper
    # -----------------------------
    def _enter_state(self, mask: np.ndarray, new_state: int) -> None:
        if not np.any(mask):
            return
        self.states[mask] = int(new_state)
        self.state_enter_step[mask] = self.ctrl_step[mask].copy()
        self.state_reach_step[mask] = -1

    # -----------------------------
    # Core logic
    # -----------------------------
    def _step_logic(self) -> None:
        cfg = self.cfg
        B = self.B

        data = self.env._state.data
        obs = self.env._state.obs

        running = self.active & (~self.done)
        if not np.any(running):
            return

        s = self.states

        # --- current object poses ---
        flower_pose = np.asarray(self.flower_body.get_pose(data), dtype=np.float32)
        flower_p = flower_pose[:, :3]

        vase_src_pose = np.asarray(self.vase_src_body.get_pose(data), dtype=np.float32)
        vase_dst_pose = np.asarray(self.vase_dst_body.get_pose(data), dtype=np.float32)
        vase_src_p = vase_src_pose[:, :3]
        vase_dst_p = vase_dst_pose[:, :3]

        g_off = np.array(cfg.grasp_offset, dtype=np.float32)

        # --- Base orientation (lock roll/pitch via base quat, yaw only) ---
        q_base = self.latched_start_quat  # (B,4) xyzw
        r_base = Rotation.from_quat(q_base)

        r_pick1 = Rotation.from_euler("z", float(cfg.pick1_yaw_deg), degrees=True)
        r_pick2 = Rotation.from_euler("z", float(cfg.pick2_yaw_deg), degrees=True)
        q_pick1 = normalize_quat((r_pick1 * r_base).as_quat().astype(np.float32))
        q_pick2 = normalize_quat((r_pick2 * r_base).as_quat().astype(np.float32))

        # yaw targets for robust reach-check
        yaw_tgt_pick1 = quat_to_yaw(q_pick1)
        yaw_tgt_pick2 = quat_to_yaw(q_pick2)

        def build_above(p_obj: np.ndarray, z: float) -> np.ndarray:
            out = p_obj.copy()
            out[:, 2] = float(z)
            return out

        # Safe height
        p_safe = obs["ee_pose"][:, :3].copy()
        p_safe[:, 2] = np.maximum(p_safe[:, 2], 0.20)

        # ---------------- pick1: coarse align ----------------
        p1_above = build_above(flower_p, cfg.lift_height_z)

        pick1_dx, pick1_dy = cfg.pick1_coarse_xy_offset
        p1_x = p1_above.copy()
        p1_x[:, 0] = flower_p[:, 0] + g_off[0] + float(pick1_dx)
        p1_y = p1_x.copy()
        p1_y[:, 1] = flower_p[:, 1] + g_off[1] + float(pick1_dy)

        p1_pre = p1_y.copy()
        p1_pre[:, 2] = flower_p[:, 2] + g_off[2] + float(cfg.pregrasp_z_margin)

        p1_fine_y = p1_pre.copy()
        p1_fine_y[:, 1] = flower_p[:, 1] + g_off[1]
        p1_fine_x = p1_fine_y.copy()
        p1_fine_x[:, 0] = flower_p[:, 0] + g_off[0]

        p1_grasp = p1_fine_x.copy()
        p1_grasp[:, 2] = flower_p[:, 2] + g_off[2]

        p1_lift = p1_grasp.copy()
        p1_lift[:, 2] = float(cfg.lift_height_z)

        # ---------------- place1 to dst ----------------
        dst_dx, dst_dy = cfg.place1_dst_hover_xy_offset
        p_dst_hover = p1_lift.copy()
        p_dst_hover[:, 0] = vase_dst_p[:, 0] + float(dst_dx)
        p_dst_hover[:, 1] = vase_dst_p[:, 1] + float(dst_dy)
        p_dst_hover[:, 2] = float(cfg.lift_height_z)

        p_dst_align = p_dst_hover.copy()
        p_dst_align[:, 2] += float(cfg.align_z_offset)

        p_dst_insert = p_dst_align.copy()
        p_dst_insert[:, 2] -= float(cfg.insert_depth)

        p_retreat1 = p_dst_align.copy()
        p_retreat1[:, 0] -= float(cfg.retreat_dx)

        # ---------------- pick2: recompute from current flower pose ----------------
        p2_above = build_above(flower_p, cfg.lift_height_z)

        p2_x = p2_above.copy()
        p2_x[:, 0] = flower_p[:, 0] + g_off[0]
        p2_y = p2_x.copy()
        p2_y[:, 1] = flower_p[:, 1] + g_off[1]

        p2_pre = p2_y.copy()
        p2_pre[:, 2] = flower_p[:, 2] + g_off[2] + float(cfg.pregrasp_z_margin)

        p2_fine_y = p2_pre.copy()
        p2_fine_y[:, 1] = flower_p[:, 1] + g_off[1]
        p2_fine_x = p2_fine_y.copy()
        p2_fine_x[:, 0] = flower_p[:, 0] + g_off[0]

        p2_grasp = p2_fine_x.copy()
        p2_grasp[:, 2] = flower_p[:, 2] + g_off[2] + float(cfg.pick2_grasp_z_extra)

        p2_lift = p2_grasp.copy()
        p2_lift[:, 2] = float(cfg.lift_height_z)

        # ---------------- place2 back to src ----------------
        src_dx, src_dy = cfg.place2_src_hover_xy_offset

        p_src_hover = p2_lift.copy()
        p_src_hover[:, 0] = vase_src_p[:, 0] + float(src_dx)
        p_src_hover[:, 1] = vase_src_p[:, 1] + float(src_dy)
        p_src_hover[:, 2] = float(cfg.lift_height_z)

        p_src_align = p_src_hover.copy()
        p_src_align[:, 2] += float(cfg.align_z_offset)

        p_src_insert = p_src_align.copy()
        p_src_insert[:, 2] -= float(cfg.insert_depth)
        p_src_insert[:, 2] += float(cfg.place2_insert_z_extra)

        p_retreat2 = p_src_align.copy()
        p_retreat2[:, 0] -= float(cfg.retreat_dx)

        # --- target assignment ---
        tgt_pos = self.exec_pos.copy()
        tgt_quat = self.exec_quat.copy()
        grip_cmd = np.full((B,), float(cfg.gripper_open), dtype=np.float32)

        def set_target(state_id: int, pos: np.ndarray, quat: np.ndarray, grip: float):
            mask = running & (s == state_id)
            if np.any(mask):
                tgt_pos[mask] = pos[mask]
                tgt_quat[mask] = quat[mask]
                grip_cmd[mask] = float(grip)

        q_hold = normalize_quat(self.hold_quat.copy()).astype(np.float32)

        # pick1
        set_target(self.ST_APP1_LIFT_Z, p_safe, q_hold, cfg.gripper_open)
        set_target(self.ST_APP1_ALIGN_X, p1_x, q_hold, cfg.gripper_open)
        set_target(self.ST_APP1_ALIGN_Y, p1_y, q_hold, cfg.gripper_open)
        set_target(self.ST_ROT_PICK1, self.rot_lock_pos, q_pick1, cfg.gripper_open)
        set_target(self.ST_PRE1_Z, p1_pre, q_pick1, cfg.gripper_open)
        set_target(self.ST_PRE1_ALIGN_Y, p1_fine_y, q_pick1, cfg.gripper_open)
        set_target(self.ST_PRE1_ALIGN_X, p1_fine_x, q_pick1, cfg.gripper_open)
        set_target(self.ST_DESCEND1, p1_grasp, q_pick1, cfg.gripper_open)
        set_target(self.ST_CLOSE1, p1_grasp, q_pick1, cfg.gripper_close)
        set_target(self.ST_LIFT1, p1_lift, q_pick1, cfg.gripper_close)

        # place1
        set_target(self.ST_TRP1_ALIGN_X, p_dst_hover, q_hold, cfg.gripper_close)
        set_target(self.ST_TRP1_ALIGN_Y, p_dst_hover, q_hold, cfg.gripper_close)
        set_target(self.ST_PLACE1_DESCEND, p_dst_insert, q_hold, cfg.gripper_close)
        set_target(self.ST_OPEN1, p_dst_insert, q_hold, cfg.gripper_open)
        set_target(self.ST_RETREAT1, p_retreat1, q_hold, cfg.gripper_open)

        # pick2
        set_target(self.ST_APP2_LIFT_Z, p_safe, q_hold, cfg.gripper_open)
        set_target(self.ST_APP2_ALIGN_X, p2_x, q_hold, cfg.gripper_open)
        set_target(self.ST_APP2_ALIGN_Y, p2_y, q_hold, cfg.gripper_open)
        set_target(self.ST_ROT_PICK2, self.rot_lock_pos, q_pick2, cfg.gripper_open)
        set_target(self.ST_PRE2_Z, p2_pre, q_pick2, cfg.gripper_open)
        set_target(self.ST_PRE2_ALIGN_Y, p2_fine_y, q_pick2, cfg.gripper_open)
        set_target(self.ST_PRE2_ALIGN_X, p2_fine_x, q_pick2, cfg.gripper_open)
        set_target(self.ST_DESCEND2, p2_grasp, q_pick2, cfg.gripper_open)
        set_target(self.ST_CLOSE2, p2_grasp, q_pick2, cfg.gripper_close)
        set_target(self.ST_LIFT2, p2_lift, q_pick2, cfg.gripper_close)

        # place2
        set_target(self.ST_TRP2_ALIGN_X, p_src_hover, q_hold, cfg.gripper_close)
        set_target(self.ST_TRP2_ALIGN_Y, p_src_hover, q_hold, cfg.gripper_close)
        set_target(self.ST_PLACE2_DESCEND, p_src_insert, q_hold, cfg.gripper_close)
        set_target(self.ST_OPEN2, p_src_insert, q_hold, cfg.gripper_open)
        set_target(self.ST_RETREAT2, p_retreat2, q_hold, cfg.gripper_open)

        # --- execute position ---
        self.exec_pos = smooth_step_pos(self.exec_pos, tgt_pos, cfg.max_dp)

        # --- execute rotation: keep your incremental approach but make it robust for yaw-only states ---
        rot_states = np.array([self.ST_ROT_PICK1, self.ST_ROT_PICK2], dtype=np.int32)
        rot_mask = running & np.isin(s, rot_states)

        if np.any(rot_mask):
            # read actual ee yaw from obs for shortest-path yaw stepping
            ee_pose = obs["ee_pose"]
            ee_quat = np.stack([obs_ee_quat_xyzw(ee_pose[i]) for i in range(B)], axis=0)
            curr_yaw = quat_to_yaw(ee_quat)

            # target yaw per env depends on state
            yaw_tgt = curr_yaw.copy()
            yaw_tgt[s == self.ST_ROT_PICK1] = yaw_tgt_pick1[s == self.ST_ROT_PICK1]
            yaw_tgt[s == self.ST_ROT_PICK2] = yaw_tgt_pick2[s == self.ST_ROT_PICK2]

            dyaw = wrap_to_pi(yaw_tgt - curr_yaw)
            dyaw_step = np.clip(dyaw, -float(cfg.max_dr), float(cfg.max_dr))

            # build new quat by updating yaw on top of base roll/pitch (latched_start_quat)
            base_rpy = Rotation.from_quat(self.latched_start_quat).as_euler("xyz", degrees=False).astype(np.float32)
            new_rpy = base_rpy.copy()
            new_rpy[:, 2] = (curr_yaw + dyaw_step).astype(np.float32)
            new_q = Rotation.from_euler("xyz", new_rpy, degrees=False).as_quat().astype(np.float32)

            self.exec_quat[rot_mask] = normalize_quat(new_q[rot_mask])
        else:
            # for non-rotation states, keep quaternion as-is (no active rotation command)
            pass

        self.exec_quat = normalize_quat(self.exec_quat).astype(np.float32)

        # --- action (eef_relative) ---
        ref_pose_6d = self.env.robot.ref_ee_pose
        ref_pos = ref_pose_6d[:, :3]
        ref_euler = ref_pose_6d[:, 3:6]
        ref_quat = Rotation.from_euler("xyz", ref_euler, degrees=False).as_quat().astype(np.float32)

        # position cmd
        action = np.zeros((B, 7), dtype=np.float32)
        action[:, :3] = (self.exec_pos - ref_pos) * float(cfg.action_pos_scale)

        # rotation cmd: only in rotation states
        rotvec_cmd = np.zeros((B, 3), dtype=np.float32)
        if np.any(rot_mask):
            idxs = np.where(rot_mask)[0]
            for i in idxs:
                r_des = Rotation.from_quat(self.exec_quat[i])
                r_ref = Rotation.from_quat(ref_quat[i])
                r_e = r_des * r_ref.inv()
                rv = r_e.as_rotvec().astype(np.float32)
                mag = np.linalg.norm(rv) + 1e-9
                scale = min(1.0, float(cfg.max_dr) / float(mag))
                rotvec_cmd[i] = rv * scale * float(cfg.rot_gain)

        action[:, 3:6] = rotvec_cmd
        action[:, 6] = grip_cmd * float(cfg.action_grip_scale)

        self._last_action[:] = action
        self.env.step(action)

        # refresh obs after step for checks
        obs = self.env._state.obs

        # ---------------- transitions ----------------
        def is_pos_reached(target: np.ndarray) -> np.ndarray:
            return np.linalg.norm(self.exec_pos - target, axis=1) < float(cfg.pos_tol)

        def rot_state_done(mask_in_state: np.ndarray, yaw_target: np.ndarray) -> np.ndarray:
            # yaw-only reach check using real obs
            ee_pose = obs["ee_pose"]
            ee_quat = np.stack([obs_ee_quat_xyzw(ee_pose[i]) for i in range(B)], axis=0)
            curr_yaw = quat_to_yaw(ee_quat)
            dyaw = wrap_to_pi(curr_yaw - yaw_target)
            ok = np.abs(dyaw) < float(cfg.yaw_tol)

            just = mask_in_state & ok & (self.state_reach_step == -1)
            if np.any(just):
                self.state_reach_step[just] = self.ctrl_step[just]
            dwell = (self.ctrl_step - self.state_reach_step) >= int(cfg.rot_dwell_steps)

            # timeout fallback
            too_long = (self.ctrl_step - self.state_enter_step) >= int(cfg.rot_state_timeout_steps)
            return mask_in_state & ((self.state_reach_step != -1) & dwell & ok | too_long)

        def _check_and_dwell(state_id: int, pos_tgt: np.ndarray) -> np.ndarray:
            in_state = running & (s == state_id)
            ok = is_pos_reached(pos_tgt)
            just = in_state & ok & (self.state_reach_step == -1)
            if np.any(just):
                self.state_reach_step[just] = self.ctrl_step[just]
            dwell = (self.ctrl_step - self.state_reach_step) >= int(cfg.waypoint_dwell_steps)
            return in_state & (self.state_reach_step != -1) & dwell & ok

        # pick1
        self._enter_state(_check_and_dwell(self.ST_APP1_LIFT_Z, p_safe), self.ST_APP1_ALIGN_X)
        self._enter_state(_check_and_dwell(self.ST_APP1_ALIGN_X, p1_x), self.ST_APP1_ALIGN_Y)

        to_rot1 = _check_and_dwell(self.ST_APP1_ALIGN_Y, p1_y)
        if np.any(to_rot1):
            self.rot_lock_pos[to_rot1] = self.exec_pos[to_rot1].copy()
            self._enter_state(to_rot1, self.ST_ROT_PICK1)

        done_rot1 = rot_state_done(running & (s == self.ST_ROT_PICK1), yaw_tgt_pick1)
        if np.any(done_rot1):
            self.hold_quat[done_rot1] = q_pick1[done_rot1].copy()
            self._enter_state(done_rot1, self.ST_PRE1_Z)

        self._enter_state(_check_and_dwell(self.ST_PRE1_Z, p1_pre), self.ST_PRE1_ALIGN_Y)
        self._enter_state(_check_and_dwell(self.ST_PRE1_ALIGN_Y, p1_fine_y), self.ST_PRE1_ALIGN_X)
        self._enter_state(_check_and_dwell(self.ST_PRE1_ALIGN_X, p1_fine_x), self.ST_DESCEND1)
        self._enter_state(_check_and_dwell(self.ST_DESCEND1, p1_grasp), self.ST_CLOSE1)

        mask_close1 = running & (s == self.ST_CLOSE1)
        if np.any(mask_close1):
            t_in = self.ctrl_step - self.state_enter_step
            done_close = t_in >= int(cfg.close_hold_steps)
            self._enter_state(mask_close1 & done_close, self.ST_LIFT1)

        self._enter_state(_check_and_dwell(self.ST_LIFT1, p1_lift), self.ST_TRP1_ALIGN_X)

        # place1
        self._enter_state(_check_and_dwell(self.ST_TRP1_ALIGN_X, p_dst_hover), self.ST_TRP1_ALIGN_Y)
        self._enter_state(_check_and_dwell(self.ST_TRP1_ALIGN_Y, p_dst_hover), self.ST_PLACE1_DESCEND)
        self._enter_state(_check_and_dwell(self.ST_PLACE1_DESCEND, p_dst_insert), self.ST_OPEN1)

        mask_open1 = running & (s == self.ST_OPEN1)
        if np.any(mask_open1):
            t_in = self.ctrl_step - self.state_enter_step
            done_open = t_in >= int(cfg.close_hold_steps)
            self._enter_state(mask_open1 & done_open, self.ST_RETREAT1)

        self._enter_state(_check_and_dwell(self.ST_RETREAT1, p_retreat1), self.ST_APP2_LIFT_Z)

        # pick2
        self._enter_state(_check_and_dwell(self.ST_APP2_LIFT_Z, p_safe), self.ST_APP2_ALIGN_X)
        self._enter_state(_check_and_dwell(self.ST_APP2_ALIGN_X, p2_x), self.ST_APP2_ALIGN_Y)

        to_rot2 = _check_and_dwell(self.ST_APP2_ALIGN_Y, p2_y)
        if np.any(to_rot2):
            self.rot_lock_pos[to_rot2] = self.exec_pos[to_rot2].copy()
            self._enter_state(to_rot2, self.ST_ROT_PICK2)

        done_rot2 = rot_state_done(running & (s == self.ST_ROT_PICK2), yaw_tgt_pick2)
        if np.any(done_rot2):
            self.hold_quat[done_rot2] = q_pick2[done_rot2].copy()
            self._enter_state(done_rot2, self.ST_PRE2_Z)

        self._enter_state(_check_and_dwell(self.ST_PRE2_Z, p2_pre), self.ST_PRE2_ALIGN_Y)
        self._enter_state(_check_and_dwell(self.ST_PRE2_ALIGN_Y, p2_fine_y), self.ST_PRE2_ALIGN_X)
        self._enter_state(_check_and_dwell(self.ST_PRE2_ALIGN_X, p2_fine_x), self.ST_DESCEND2)
        self._enter_state(_check_and_dwell(self.ST_DESCEND2, p2_grasp), self.ST_CLOSE2)

        mask_close2 = running & (s == self.ST_CLOSE2)
        if np.any(mask_close2):
            t_in = self.ctrl_step - self.state_enter_step
            done_close = t_in >= int(cfg.close_hold_steps)
            self._enter_state(mask_close2 & done_close, self.ST_LIFT2)

        self._enter_state(_check_and_dwell(self.ST_LIFT2, p2_lift), self.ST_TRP2_ALIGN_X)

        # place2
        self._enter_state(_check_and_dwell(self.ST_TRP2_ALIGN_X, p_src_hover), self.ST_TRP2_ALIGN_Y)
        self._enter_state(_check_and_dwell(self.ST_TRP2_ALIGN_Y, p_src_hover), self.ST_PLACE2_DESCEND)
        self._enter_state(_check_and_dwell(self.ST_PLACE2_DESCEND, p_src_insert), self.ST_OPEN2)

        mask_open2 = running & (s == self.ST_OPEN2)
        if np.any(mask_open2):
            t_in = self.ctrl_step - self.state_enter_step
            done_open = t_in >= int(cfg.close_hold_steps)
            self._enter_state(mask_open2 & done_open, self.ST_RETREAT2)

        self._enter_state(_check_and_dwell(self.ST_RETREAT2, p_retreat2), self.ST_DONE)

    # -----------------------------
    # Collection loop
    # -----------------------------
    def collect(self) -> None:
        cfg = self.cfg
        target_n = int(cfg.data_size)

        all_ids = np.arange(self.B, dtype=np.int64)
        self.start_episodes(all_ids, seed=int(cfg.seed))

        print(f"Starting ArrangeFlowers Collection (multi-view). Target: {target_n}")

        while self.saved_count < target_n:
            self._step_logic()

            running = self.active & (~self.done)

            # capture (before render) so frame_idx = current video_frames is correct
            sample_mask = running & ((self.ctrl_step % int(cfg.sample_every_steps)) == 0)
            for env_id in np.where(sample_mask)[0].tolist():
                self._capture_step(int(env_id))

            if cfg.save_video:
                render_mask = running & ((self.ctrl_step % int(cfg.render_every_steps)) == 0)
                for env_id in np.where(render_mask)[0].tolist():
                    self._write_video_frame(int(env_id))

            self.ctrl_step[running] += 1

            # termination conditions
            for i in range(self.B):
                if not running[i]:
                    continue
                fsm_done = (int(self.states[i]) == int(self.ST_DONE))
                timeout = (int(self.ctrl_step[i]) >= int(cfg.max_ctrl_steps))
                env_success = bool(getattr(self.env, "success_latched", np.zeros((self.B,), dtype=bool))[i])
                if fsm_done or timeout or env_success:
                    self.done[i] = True
                    self.success[i] = bool(env_success) or bool(fsm_done)

            # finalize & restart
            for i in range(self.B):
                if self.active[i] and self.done[i]:
                    self._finalize_episode(i)
                    if self.saved_count < target_n:
                        self.active[i] = False
                        new_seed = int(cfg.seed + self.attempted)
                        self.start_episodes(np.array([i]), seed=new_seed)
                    else:
                        self.active[i] = False

            now = time.perf_counter()
            if (now - self._last_log_t) > 2.0:
                print(f"[Collect] Saved: {self.saved_count}/{target_n} | Active: {int(self.active.sum())}")
                self._last_log_t = now

        print(f"Done. Saved to {cfg.save_dir}")
        self.close()

    def close(self) -> None:
        # close env (motrix recorder etc.)
        if hasattr(self.env, "close"):
            try:
                self.env.close()
            except Exception:
                pass

        # close all writers
        for env_id in range(self.B):
            self._close_writers_for_env(env_id)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()

    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--data_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no_video", action="store_true")

    p.add_argument("--pick1_yaw_deg", type=float, default=None)
    p.add_argument("--pick2_yaw_deg", type=float, default=None)
    p.add_argument("--pregrasp_z_margin", type=float, default=None)

    # offsets knobs
    p.add_argument("--pick1_coarse_dx", type=float, default=None)
    p.add_argument("--pick1_coarse_dy", type=float, default=None)
    p.add_argument("--dst_hover_dx", type=float, default=None)
    p.add_argument("--dst_hover_dy", type=float, default=None)
    p.add_argument("--pick2_grasp_z_extra", type=float, default=None)
    p.add_argument("--src_hover_dx", type=float, default=None)
    p.add_argument("--src_hover_dy", type=float, default=None)
    p.add_argument("--src_insert_z_extra", type=float, default=None)

    # action scaling
    p.add_argument("--action_pos_scale", type=float, default=None)
    p.add_argument("--action_grip_scale", type=float, default=None)

    # MotrixSim toggles
    p.add_argument("--no_motrix_video", action="store_true", help="Disable MotrixSim renderer video recording.")
    p.add_argument("--motrix_video_fps", type=int, default=None)
    p.add_argument("--motrix_video_width", type=int, default=None)
    p.add_argument("--motrix_video_height", type=int, default=None)

    args = p.parse_args()

    # build cfg with overrides
    cfg = CollectorCfg(
        save_dir=args.save_dir if args.save_dir else CollectorCfg.save_dir,
        num_envs=args.num_envs if args.num_envs else CollectorCfg.num_envs,
        data_size=args.data_size if args.data_size is not None else CollectorCfg.data_size,
        seed=args.seed if args.seed is not None else CollectorCfg.seed,
        save_video=(not args.no_video),

        pick1_yaw_deg=(args.pick1_yaw_deg if args.pick1_yaw_deg is not None else CollectorCfg.pick1_yaw_deg),
        pick2_yaw_deg=(args.pick2_yaw_deg if args.pick2_yaw_deg is not None else CollectorCfg.pick2_yaw_deg),
        pregrasp_z_margin=(args.pregrasp_z_margin if args.pregrasp_z_margin is not None else CollectorCfg.pregrasp_z_margin),

        pick1_coarse_xy_offset=(
            args.pick1_coarse_dx if args.pick1_coarse_dx is not None else CollectorCfg.pick1_coarse_xy_offset[0],
            args.pick1_coarse_dy if args.pick1_coarse_dy is not None else CollectorCfg.pick1_coarse_xy_offset[1],
        ),
        place1_dst_hover_xy_offset=(
            args.dst_hover_dx if args.dst_hover_dx is not None else CollectorCfg.place1_dst_hover_xy_offset[0],
            args.dst_hover_dy if args.dst_hover_dy is not None else CollectorCfg.place1_dst_hover_xy_offset[1],
        ),
        pick2_grasp_z_extra=(args.pick2_grasp_z_extra if args.pick2_grasp_z_extra is not None else CollectorCfg.pick2_grasp_z_extra),
        place2_src_hover_xy_offset=(
            args.src_hover_dx if args.src_hover_dx is not None else CollectorCfg.place2_src_hover_xy_offset[0],
            args.src_hover_dy if args.src_hover_dy is not None else CollectorCfg.place2_src_hover_xy_offset[1],
        ),
        place2_insert_z_extra=(args.src_insert_z_extra if args.src_insert_z_extra is not None else CollectorCfg.place2_insert_z_extra),

        action_pos_scale=(args.action_pos_scale if args.action_pos_scale is not None else CollectorCfg.action_pos_scale),
        action_grip_scale=(args.action_grip_scale if args.action_grip_scale is not None else CollectorCfg.action_grip_scale),

        enable_motrix_video=(not args.no_motrix_video),
        motrix_video_fps=(args.motrix_video_fps if args.motrix_video_fps is not None else CollectorCfg.motrix_video_fps),
        motrix_video_width=(args.motrix_video_width if args.motrix_video_width is not None else CollectorCfg.motrix_video_width),
        motrix_video_height=(args.motrix_video_height if args.motrix_video_height is not None else CollectorCfg.motrix_video_height),
    )

    runner = ArrangeFlowersCollector(cfg)
    try:
        runner.collect()
    except KeyboardInterrupt:
        pass
    finally:
        runner.close()


if __name__ == "__main__":
    main()
