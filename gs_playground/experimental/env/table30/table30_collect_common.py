
# =============================================================================
# File: table30_collect_common.py
# =============================================================================
from __future__ import annotations

import os
import json
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import mediapy
from scipy.spatial.transform import Rotation


# -----------------------------------------------------------------------------
# Math / pose utilities
# -----------------------------------------------------------------------------
def smooth_step_pos(curr: np.ndarray, tgt: np.ndarray, max_dp: float) -> np.ndarray:
    """Smooth step with max displacement per step. curr/tgt: (B,3)."""
    dp = tgt - curr
    n = np.linalg.norm(dp, axis=1, keepdims=True)
    s = np.minimum(1.0, float(max_dp) / (n + 1e-9))
    return curr + dp * s


def normalize_quat(q_xyzw: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q_xyzw, axis=-1, keepdims=True)
    return q_xyzw / (n + 1e-9)


def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def quat_to_yaw(q_xyzw: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(q_xyzw).as_euler("xyz", degrees=False)[..., 2].astype(np.float32)


def ee_pose_to_rpy_xyzw(ee_pose_row: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robustly interpret ee pose from env obs:
      - len==7: [x,y,z,qx,qy,qz,qw] or [x,y,z,quat(xyzw)] in xyzw
      - len==6: [x,y,z,roll,pitch,yaw]
    Returns:
      xyz (3,), rpy (3,) in radians.
    """
    row = np.asarray(ee_pose_row, dtype=np.float32).reshape(-1)
    if row.shape[0] == 6:
        xyz = row[:3]
        rpy = row[3:6]
        return xyz, rpy
    if row.shape[0] == 7:
        xyz = row[:3]
        q = row[3:7]
        q = normalize_quat(q[None, :])[0]
        rpy = Rotation.from_quat(q).as_euler("xyz", degrees=False).astype(np.float32)
        return xyz, rpy
    # fallback: best effort
    xyz = row[:3] if row.shape[0] >= 3 else np.zeros((3,), dtype=np.float32)
    rpy = np.zeros((3,), dtype=np.float32)
    return xyz.astype(np.float32), rpy.astype(np.float32)


@dataclass(frozen=True)
class PoseFixCfg:
    """
    For JSONL export only (does NOT affect control):
      z'   = z + z_offset
      yaw' = yaw_offset + yaw_sign * yaw
    """
    z_offset: float = 0.0
    yaw_offset: float = 0.0
    yaw_sign: float = 1.0
    wrap_yaw: bool = True


def apply_pose_fix_to_rpy(rpy: np.ndarray, pose_fix: PoseFixCfg) -> np.ndarray:
    out = np.asarray(rpy, dtype=np.float32).copy()
    out[2] = float(pose_fix.yaw_offset) + float(pose_fix.yaw_sign) * float(out[2])
    if pose_fix.wrap_yaw:
        out[2] = float(wrap_to_pi(np.array([out[2]], dtype=np.float32))[0])
    return out


def build_export_ee_pose7(ee_pose_row: np.ndarray, gripper_row: Any, pose_fix: PoseFixCfg) -> List[float]:
    """
    Export format (7d): [x, y, z_fixed, roll, pitch, yaw_fixed, gripper]
    """
    xyz, rpy = ee_pose_to_rpy_xyzw(np.asarray(ee_pose_row))
    xyz = xyz.astype(np.float32)
    rpy = rpy.astype(np.float32)

    xyz[2] = xyz[2] + float(pose_fix.z_offset)
    rpy = apply_pose_fix_to_rpy(rpy, pose_fix)

    if isinstance(gripper_row, (list, tuple, np.ndarray)):
        g_val = float(np.asarray(gripper_row).reshape(-1)[0])
    else:
        g_val = float(gripper_row)

    return [float(xyz[0]), float(xyz[1]), float(xyz[2]), float(rpy[0]), float(rpy[1]), float(rpy[2]), float(g_val)]


# -----------------------------------------------------------------------------
# Video (mediapy) - multi-view
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class VideoCfg:
    fps: int = 30
    width: int = 640
    height: int = 480
    codec: str = "h264"
    crf: Optional[int] = None  # if None, use mediapy default


class _MediapyStreamWriter:
    """
    Streaming MP4 writer using mediapy.VideoWriter.

    Notes:
      - Expects RGB uint8 frames.
      - mediapy.VideoWriter expects frames shaped (H, W, 3).
    """
    def __init__(self, path: str, cfg: VideoCfg):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.cfg = cfg
        self.shape_hw = (int(cfg.height), int(cfg.width))
        kwargs = {"fps": int(cfg.fps), "codec": str(cfg.codec)}
        if cfg.crf is not None:
            # mediapy uses ffmpeg args; 'crf' is a common setting for libx264/265
            kwargs["ffmpeg_args"] = ["-crf", str(int(cfg.crf))]
        self._vw = mediapy.VideoWriter(path, shape=self.shape_hw, **kwargs)
        self._entered = False

    def _ensure_open(self) -> None:
        if not self._entered:
            self._vw.__enter__()
            self._entered = True

    def add(self, rgb: np.ndarray) -> None:
        self._ensure_open()
        img = np.asarray(rgb)
        if img.dtype != np.uint8:
            img = mediapy.to_uint8(img)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected RGB (H,W,3), got {img.shape}")
        if tuple(img.shape[:2]) != tuple(self.shape_hw):
            img = mediapy.resize_image(img, self.shape_hw)
            if img.dtype != np.uint8:
                img = mediapy.to_uint8(img)
        self._vw.add_image(img)

    def close(self) -> None:
        if self._vw is None:
            return
        try:
            if not self._entered:
                # open/close anyway to flush
                self._vw.__enter__()
                self._entered = True
            self._vw.__exit__(None, None, None)
        except Exception:
            pass
        self._vw = None


def _safe_key(k: str) -> str:
    return str(k).replace("/", "_").replace(" ", "_")


class MultiViewVideoManager:
    def __init__(self, save_dir: str, cam_keys: Sequence[str], video_cfg: VideoCfg):
        self.save_dir = str(save_dir)
        self.videos_dir = os.path.join(self.save_dir, "videos")
        os.makedirs(self.videos_dir, exist_ok=True)

        self.cam_keys = list(cam_keys)
        self.video_cfg = video_cfg

        # per env: cam_key -> writer
        self._writers: List[Dict[str, _MediapyStreamWriter]] = []
        self._tmp_paths: List[Dict[str, str]] = []
        self._frame_counts: List[int] = []

    def init_envs(self, num_envs: int) -> None:
        self._writers = [{} for _ in range(int(num_envs))]
        self._tmp_paths = []
        for i in range(int(num_envs)):
            mp: Dict[str, str] = {}
            for k in self.cam_keys:
                mp[k] = os.path.join(self.videos_dir, f"_tmp_env{i}_{_safe_key(k)}.mp4")
            self._tmp_paths.append(mp)
        self._frame_counts = [0 for _ in range(int(num_envs))]

    def reset_env(self, env_id: int) -> None:
        env_id = int(env_id)
        self.close_env(env_id)
        for k in self.cam_keys:
            tmp = self._tmp_paths[env_id][k]
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
            self._writers[env_id][k] = _MediapyStreamWriter(tmp, self.video_cfg)
        self._frame_counts[env_id] = 0

    def close_env(self, env_id: int) -> None:
        env_id = int(env_id)
        writers = self._writers[env_id]
        for k in list(writers.keys()):
            try:
                writers[k].close()
            except Exception:
                pass
        self._writers[env_id] = {}

    def maybe_write(self, obs: Dict[str, Any], env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            env_id = int(env_id)
            writers = self._writers[env_id]
            if not writers:
                continue
            wrote_any = False
            for k in self.cam_keys:
                if k in obs and k in writers:
                    rgb = obs[k][env_id]
                    if rgb is not None:
                        writers[k].add(rgb)
                        wrote_any = True
            if wrote_any:
                self._frame_counts[env_id] += 1

    def frame_count(self, env_id: int) -> int:
        return int(self._frame_counts[int(env_id)])

    def finalize_env(self, env_id: int, ep_idx: int) -> Dict[str, str]:
        """
        Close writers, move temp mp4 into save_dir/videos.
        Returns: cam_key -> relative path (videos/episode_XXXXX_<cam>.mp4)
        """
        env_id = int(env_id)
        ep_idx = int(ep_idx)
        self.close_env(env_id)

        out: Dict[str, str] = {}
        for k in self.cam_keys:
            tmp = self._tmp_paths[env_id][k]
            if os.path.exists(tmp):
                rel = f"videos/episode_{ep_idx:05d}_{_safe_key(k)}.mp4"
                dst = os.path.join(self.save_dir, rel)
                try:
                    shutil.move(tmp, dst)
                    out[k] = rel
                except Exception:
                    # best effort cleanup
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass
        return out

    def cleanup_env(self, env_id: int) -> None:
        env_id = int(env_id)
        self.close_env(env_id)
        for k in self.cam_keys:
            tmp = self._tmp_paths[env_id][k]
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
        self._frame_counts[env_id] = 0


# -----------------------------------------------------------------------------
# Buffer + JSONL export
# -----------------------------------------------------------------------------
@dataclass
class EpisodeBuffer:
    times: List[float] = field(default_factory=list)
    logic_states: List[int] = field(default_factory=list)
    qpos: List[Any] = field(default_factory=list)
    ee_pose_raw: List[Any] = field(default_factory=list)
    gripper_raw: List[Any] = field(default_factory=list)
    ctrl: List[Any] = field(default_factory=list)
    reward: List[float] = field(default_factory=list)
    is_success: List[bool] = field(default_factory=list)
    frame_idxs: List[Optional[int]] = field(default_factory=list)
    extra: Dict[str, List[Any]] = field(default_factory=dict)

    def add_extra_field(self, k: str, v: Any) -> None:
        if k not in self.extra:
            self.extra[k] = []
        self.extra[k].append(v)


class Table30CollectorIO:
    """
    IO layer shared across Table30 collectors:
      - per-env buffering
      - multi-view video writing via mediapy
      - jsonl writing with ee_pose fix (yaw/z)
    """
    def __init__(
        self,
        save_dir: str,
        num_envs: int,
        cam_keys: Sequence[str],
        *,
        save_video: bool,
        sample_every_steps: int,
        render_every_steps: int,
        video_cfg: VideoCfg,
        pose_fix: PoseFixCfg,
        dt: float = 0.02,
    ):
        self.save_dir = str(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        self.num_envs = int(num_envs)
        self.cam_keys = list(cam_keys)

        self.save_video = bool(save_video)
        self.sample_every_steps = int(sample_every_steps)
        self.render_every_steps = int(render_every_steps)

        self.video_cfg = video_cfg
        self.pose_fix = pose_fix
        self.dt = float(dt)

        self.buffers: List[EpisodeBuffer] = [EpisodeBuffer() for _ in range(self.num_envs)]

        self.video_mgr: Optional[MultiViewVideoManager] = None
        if self.save_video:
            self.video_mgr = MultiViewVideoManager(self.save_dir, self.cam_keys, self.video_cfg)
            self.video_mgr.init_envs(self.num_envs)

    def reset_env(self, env_id: int) -> None:
        env_id = int(env_id)
        self.buffers[env_id] = EpisodeBuffer()
        if self.save_video and self.video_mgr is not None:
            self.video_mgr.reset_env(env_id)

    def capture_step(
        self,
        env_id: int,
        ctrl_step: int,
        state: int,
        obs: Dict[str, Any],
        last_action: np.ndarray,
        *,
        success: bool,
        reward: float,
        extra_step: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Capture one logical sample for a single env.
        """
        env_id = int(env_id)
        buf = self.buffers[env_id]

        # frame index alignment:
        # capture happens BEFORE rendering in caller loop (recommended),
        # so "current video_frames" corresponds to the frame that will be written next.
        frame_idx: Optional[int] = None
        if self.save_video and self.video_mgr is not None:
            if (int(ctrl_step) % int(self.render_every_steps)) == 0:
                frame_idx = int(self.video_mgr.frame_count(env_id))
        buf.frame_idxs.append(frame_idx)

        buf.times.append(float(ctrl_step) * self.dt)
        buf.logic_states.append(int(state))
        buf.qpos.append(obs["qpos"][env_id].tolist() if isinstance(obs["qpos"][env_id], np.ndarray) else obs["qpos"][env_id])
        buf.ee_pose_raw.append(obs["ee_pose"][env_id].tolist() if isinstance(obs["ee_pose"][env_id], np.ndarray) else obs["ee_pose"][env_id])
        buf.gripper_raw.append(obs["gripper"][env_id].tolist() if isinstance(obs["gripper"][env_id], np.ndarray) else obs["gripper"][env_id])
        buf.ctrl.append(last_action[env_id].tolist())
        buf.is_success.append(bool(success))
        buf.reward.append(float(reward))

        if extra_step:
            for k, v in extra_step.items():
                buf.add_extra_field(str(k), v)

    def maybe_write_video(
        self,
        obs: Dict[str, Any],
        env_ids: Optional[Sequence[int]] = None,
        ctrl_step: Optional[int] = None,
        *,
        env_id: Optional[int] = None,
    ) -> None:
        """Write video frames for the provided envs if this step is a render step.

        Backward/forward compatible shim:
          - preferred: maybe_write_video(obs, env_ids=[...], ctrl_step=step)
          - also accepts: maybe_write_video(obs, env_id=..., ctrl_step=step)
        """
        if not self.save_video or self.video_mgr is None:
            return

        # normalize env selection
        if env_ids is None:
            if env_id is None:
                return
            env_ids = [int(env_id)]

        # if caller already filtered render steps, allow ctrl_step=None
        if ctrl_step is not None:
            if (int(ctrl_step) % int(self.render_every_steps)) != 0:
                return

        self.video_mgr.maybe_write(obs, env_ids)

    def finalize_episode(
        self,
        env_id: int,
        ep_idx: int,
        *,
        prompt: str,
        success: bool,
        extra_episode: Optional[Dict[str, Any]] = None,
        images_prefix: str = "images_",
        include_raw_gripper: bool = True,
    ) -> None:
        """
        If success: move videos + write jsonl.
        In all cases: cleanup tmp videos + reset buffer.
        """
        env_id = int(env_id)
        ep_idx = int(ep_idx)

        vid_map: Dict[str, str] = {}
        if self.save_video and self.video_mgr is not None:
            if bool(success):
                vid_map = self.video_mgr.finalize_env(env_id, ep_idx)
            else:
                self.video_mgr.cleanup_env(env_id)

        if bool(success):
            # write jsonl
            out_path = os.path.join(self.save_dir, f"episode_{ep_idx:05d}.jsonl")
            buf = self.buffers[env_id]
            n = len(buf.times)

            with open(out_path, "w", encoding="utf-8") as f:
                for i in range(n):
                    rec: Dict[str, Any] = {
                        "prompt": str(prompt),
                        "qpos": buf.qpos[i],
                        "ee_pose": build_export_ee_pose7(buf.ee_pose_raw[i], buf.gripper_raw[i], self.pose_fix),
                        "ctrl": buf.ctrl[i],
                        "reward": float(buf.reward[i]),
                        "is_success": bool(buf.is_success[i]),
                        "is_robot": True,
                    }
                    if include_raw_gripper:
                        rec["gripper"] = buf.gripper_raw[i]

                    # extra per-step fields
                    for k, series in buf.extra.items():
                        if i < len(series):
                            rec[k] = series[i]

                    # videos
                    if self.save_video:
                        for k_idx, cam_k in enumerate(self.cam_keys):
                            json_key = f"{images_prefix}{k_idx + 1}"
                            if cam_k in vid_map:
                                frame_idx = buf.frame_idxs[i]
                                if frame_idx is None:
                                    frame_idx = i
                                rec[json_key] = {"url": vid_map[cam_k], "type": "video", "frame_idx": int(frame_idx)}

                    if extra_episode:
                        for k, v in extra_episode.items():
                            # only set if not clobbering per-step values
                            if k not in rec:
                                rec[k] = v

                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # reset buffer after finalize
        self.buffers[env_id] = EpisodeBuffer()

    def close(self) -> None:
        if self.video_mgr is not None:
            for env_id in range(self.num_envs):
                self.video_mgr.cleanup_env(env_id)
