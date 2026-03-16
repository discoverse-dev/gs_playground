import os
import json
import numpy as np
import mediapy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class VideoCfg:
    fps: int = 30
    width: int = 480
    height: int = 360
    codec: str = "h264"


class VideoWriter:
    def __init__(self, path: str, cfg: VideoCfg):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.cfg = cfg
        kwargs = {"fps": cfg.fps, "codec": cfg.codec}
        self._vw = mediapy.VideoWriter(path, shape=(cfg.height, cfg.width), **kwargs)
        self._entered = False

    def add(self, rgb: np.ndarray) -> None:
        if not self._entered:
            self._vw.__enter__()
            self._entered = True
        img = np.asarray(rgb, dtype=np.uint8) if rgb.dtype == np.uint8 else mediapy.to_uint8(rgb)
        if tuple(img.shape[:2]) != (self.cfg.height, self.cfg.width):
            img = mediapy.resize_image(img, (self.cfg.height, self.cfg.width))
            if img.dtype != np.uint8:
                img = mediapy.to_uint8(img)
        self._vw.add_image(img)

    def close(self) -> None:
        if self._vw and not self._entered:
            self._vw.__enter__()
            self._entered = True
        if self._vw:
            self._vw.__exit__(None, None, None)
        self._vw = None


@dataclass
class EpisodeBuffer:
    times: List[float] = field(default_factory=list)
    base_pose: List[List[float]] = field(default_factory=list)
    ctrl: List[List[float]] = field(default_factory=list)
    frame_idxs: List[int] = field(default_factory=list)


class NavDataCollector:
    def __init__(self, save_dir: str, cam_keys: Sequence[str], video_cfg: VideoCfg, dt: float = 0.02):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "videos"), exist_ok=True)

        self.cam_keys = list(cam_keys)
        self.video_cfg = video_cfg
        self.dt = dt

        self.buffer = EpisodeBuffer()
        self.writers: Dict[str, VideoWriter] = {}
        self.frame_count = 0
        self.ep_idx = 0

    def start_episode(self) -> None:
        self.buffer = EpisodeBuffer()
        self.frame_count = 0
        for k in self.cam_keys:
            path = os.path.join(self.save_dir, "videos", f"_tmp_{k.replace('/', '_')}.mp4")
            self.writers[k] = VideoWriter(path, self.video_cfg)

    def add_step(self, step: int, base_pose: np.ndarray, ctrl: np.ndarray) -> None:
        self.buffer.times.append(step * self.dt)
        self.buffer.base_pose.append(base_pose.tolist())
        self.buffer.ctrl.append(ctrl.tolist())
        self.buffer.frame_idxs.append(self.frame_count)

    def add_frame(self, cam_data: Dict[str, np.ndarray]) -> None:
        for k in self.cam_keys:
            if k in cam_data and k in self.writers:
                self.writers[k].add(cam_data[k])
        self.frame_count += 1

    def save_episode(self, prompt: str) -> None:
        for w in self.writers.values():
            w.close()

        vid_map = {}
        for k in self.cam_keys:
            tmp = os.path.join(self.save_dir, "videos", f"_tmp_{k.replace('/', '_')}.mp4")
            if os.path.exists(tmp):
                rel = f"videos/episode_{self.ep_idx:05d}_{k.replace('/', '_')}.mp4"
                dst = os.path.join(self.save_dir, rel)
                os.rename(tmp, dst)
                vid_map[k] = rel

        jsonl_path = os.path.join(self.save_dir, f"episode_{self.ep_idx:05d}.jsonl")
        with open(jsonl_path, "w") as f:
            for i in range(len(self.buffer.times)):
                rec = {
                    "prompt": prompt,
                    "base_pose": self.buffer.base_pose[i],
                    "ctrl": self.buffer.ctrl[i],
                    "is_robot": True,
                }
                for idx, k in enumerate(self.cam_keys):
                    if k in vid_map:
                        rec[f"images_{idx+1}"] = {
                            "url": vid_map[k],
                            "type": "video",
                            "frame_idx": self.buffer.frame_idxs[i]
                        }
                f.write(json.dumps(rec) + "\n")

        self.ep_idx += 1
        self.writers = {}

    def cleanup(self) -> None:
        for w in self.writers.values():
            w.close()
        for k in self.cam_keys:
            tmp = os.path.join(self.save_dir, "videos", f"_tmp_{k.replace('/', '_')}.mp4")
            if os.path.exists(tmp):
                os.remove(tmp)
        self.writers = {}
