from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation
import motrixsim as mtx

from ..base import EnvCfg
from .mtx_env import NpEnv, NpEnvState

from gaussian_renderer import BatchSplatConfig, MtxBatchSplatRenderer


@dataclass
class RenderEnvCfg(EnvCfg):
    """Rendering configuration extending EnvCfg.

    Adds image dimensions and camera selection for batch rendering.
    cam_id: list/tuple of camera indices (>=1).
    """

    img_width: int = 320
    img_height: int = 240
    cam_id: Sequence[int] = field(default_factory=lambda: [0])

    # assets
    # Optional rendering assets (provide defaults for compatibility)
    gs_background_ply: str = ""
    gs_robot_gaussians: Dict[str, str] = field(default_factory=dict)

    # MotrixSim video recording configuration
    enable_motrix_video: bool = False  # Whether to enable MotrixSim video recording
    motrix_video_fps: int = 30  # Video frame rate
    motrix_video_output_dir: str = "motrix_videos"  # Video output directory
    motrix_video_width: int = 320  # Video width
    motrix_video_height: int = 240  # Video height


@dataclass
class RenderEnvState(NpEnvState):
    """Extend NpEnvState to hold structured obs with pixels."""

    obs: Dict[str, torch.Tensor]


class NpRenderEnv(NpEnv):
    """A NumPy-based env with gaussian-splat rendering and BG cache.

    - Uses separate foreground (FG) and background (BG) renderers similar to
        `MjxBatchGSRendererWithBGCache`.
    - Background images are rendered once and cached in `self._bg_imgs` until
        `reset_renderer_bg_cache()` is called.
    - Observation/reward logic remains minimal; override for task-specific use.
    """

    def __init__(self, cfg: EnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)
        self._fg_renderer = None
        self._bg_renderer = None
        self._bg_imgs = None
        # Render config (derived from EnvCfg with sensible defaults)
        self._img_w: int = getattr(cfg, "img_width", 320)
        self._img_h: int = getattr(cfg, "img_height", 240)
        cam_ids_raw = getattr(cfg, "cam_id", [0])
        if isinstance(cam_ids_raw, (int, np.integer)):
            self._cam_ids = [int(cam_ids_raw)]
        else:
            self._cam_ids = [int(c) for c in cam_ids_raw]
        if len(self._cam_ids) == 0:
            raise ValueError("cam_id must contain at least one camera index")

        # MotrixSim video recorder (optional)
        self._motrix_recorder = None
        self._motrix_video_enabled = getattr(cfg, "enable_motrix_video", False)
        if self._motrix_video_enabled:
            self._init_motrix_recorder()

    @property
    def cam_ids(self) -> Sequence[int]:
        """Get configured camera IDs for rendering."""
        return self._cam_ids

    # -------------------- MotrixSim video recording --------------------
    def _init_motrix_recorder(self):
        """Initialize MotrixSim video recorder with multi-camera support."""
        if self._motrix_recorder is not None:
            return

        from pathlib import Path

        cfg = self._cfg
        output_dir = Path(getattr(cfg, "motrix_video_output_dir", "motrix_videos"))
        output_dir.mkdir(parents=True, exist_ok=True)

        from .motrix_video_recorder import MotrixVideoRecorder

        self._motrix_recorder = MotrixVideoRecorder(
            model=self._model,
            output_path=str(output_dir),  # Directory for multi-camera files
            fps=getattr(cfg, "motrix_video_fps", 30),
            img_width=getattr(cfg, "motrix_video_width", 640),
            img_height=getattr(cfg, "motrix_video_height", 480),
            cam_ids=self._cam_ids,  # Pass camera IDs
        )
        self._motrix_recorder.start()

    def _capture_motrix_frame(self):
        """Capture current frame to MotrixSim video."""
        if self._motrix_recorder is None:
            return
        self._motrix_recorder.capture_frame(self._state.data)

    # -------------------- Renderer helpers --------------------
    def init_renderer(
        self,
        body_gaussians: Dict[str, str],
        background_ply: Optional[str] = None,
        minibatch: int = 32,
    ):
        """Initialize separate FG/BG renderers for batch rendering with BG cache."""
        fg_cfg = BatchSplatConfig(
            body_gaussians=dict(body_gaussians),
            background_ply=None,
            minibatch=int(minibatch),
        )
        self._fg_renderer = MtxBatchSplatRenderer(fg_cfg, self._model)

        if background_ply:
            bg_cfg = BatchSplatConfig(
                body_gaussians={},
                background_ply=str(background_ply),
                minibatch=int(minibatch),
            )
            self._bg_renderer = MtxBatchSplatRenderer(bg_cfg, self._model)
        else:
            self._bg_renderer = None

        # Clear cache whenever renderer is re-initialized
        self._bg_imgs = None

    # -------------------- Camera/body utilities --------------------
    def _compute_camera_arrays(self, data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute cam_xpos(B,C,3), cam_xmat(B,C,3,3), fovy(B,C) for configured cam_ids."""
        poses = []
        rmats = []
        fovys = []
        for cid in self.cam_ids:
            cam = self._model.cameras[cid]
            cam_pose = cam.get_pose(data)  # (B,7) xyzw
            poses.append(cam_pose[..., :3])
            quat_xyzw = cam_pose[..., 3:7]
            rmats.append(Rotation.from_quat(quat_xyzw).as_matrix())
            fovys.append(
                np.full((cam_pose.shape[0],), float(cam.fovy), dtype=np.float32)
            )

        cam_xpos = np.stack(poses, axis=1)  # (B,C,3)
        cam_xmat = np.stack(rmats, axis=1)  # (B,C,3,3)
        fovy = np.stack(fovys, axis=1)  # (B,C)
        return cam_xpos, cam_xmat, fovy

    def _compute_body_arrays(self, data) -> Tuple[np.ndarray, np.ndarray]:
        """Compute body_pos(B,nbody,3) and body_quat(B,nbody,4) in xyzw order."""
        link_poses = self._model.get_link_poses(
            data
        )  # (B, nbody, 7): pos(3)+quat(4 xyzw)
        body_pos = link_poses[..., :3]
        body_quat = link_poses[..., 3:7]
        return body_pos, body_quat

    def batch_render(self, data) -> Tuple[np.ndarray, np.ndarray]:
        """Render using current state (or provided data) for all configured cameras."""
        if self._fg_renderer is None:
            raise RuntimeError("Renderer not initialized. Call `init_renderer` first.")

        cam_xpos, cam_xmat, fovy = self._compute_camera_arrays(data)
        body_pos, body_quat = self._compute_body_arrays(data)
        H = int(self._img_h)
        W = int(self._img_w)

        # Background cached once (similar to MjxBatchGSRendererWithBGCache)
        # if self._bg_renderer is not None and self._bg_imgs is None:
        if True :
            bg_gsb = self._bg_renderer.batch_update_gaussians(body_pos, body_quat)
            self._bg_imgs, _ = self._bg_renderer.batch_env_render(
                bg_gsb, cam_xpos, cam_xmat, H, W, fovy
            )

        # Update FG gaussians and render with cached BG
        gsb = self._fg_renderer.batch_update_gaussians(body_pos, body_quat)

        effective_bg = self._bg_imgs
        rgb_t, depth_t = self._fg_renderer.batch_env_render(
            gsb, cam_xpos, cam_xmat, H, W, fovy, bg_imgs=effective_bg
        )

        return rgb_t, depth_t

    # -------------------- Obs helpers --------------------
    def _render_pixels(self, data) -> Dict[str, torch.Tensor]:
        """Render pixel observations for all configured cameras."""
        rgb_t, _ = self.batch_render(data)
        rgb_np_u8 = (255 * torch.clamp(rgb_t, 0.0, 1.0)).to(torch.uint8).cpu().numpy()

        obs_pix: Dict[str, torch.Tensor] = {}
        for i, _ in enumerate(self.cam_ids):
            obs_pix[f"pixels/view_{i}"] = rgb_np_u8[:, i]
        return obs_pix

    def _init_obs_dict(self, n: int, obs_dim: int) -> Dict[str, torch.Tensor]:
        obs = {"state": torch.zeros((n, max(1, obs_dim)), dtype=torch.float32)}
        for i, _ in enumerate(self.cam_ids):
            # Use uint8 for pixel placeholders to match rendered uint8 output
            obs[f"pixels/view_{i}"] = torch.zeros(
                (n, self._img_h, self._img_w, 3), dtype=torch.uint8
            )
        return obs

    # -------------------- Override: state/init/step with pixel obs --------------------
    def init_state(self) -> RenderEnvState:
        # `observation_space` may be a Dict (pixel envs). Don't assume a single
        # shape attribute; `_init_obs_dict` builds a dict-based obs, so pass
        # a placeholder obs_dim (0) which is handled above.
        obs_dim = 0
        obs = self._init_obs_dict(self._num_envs, obs_dim)
        reward = np.zeros((self._num_envs,), dtype=np.float32)
        terminated = np.ones((self._num_envs,), dtype=bool)
        truncated = np.zeros((self._num_envs,), dtype=bool)
        info = {"steps": np.zeros((self._num_envs,), dtype=np.uint64)}
        data = mtx.SceneData(self._model, batch=[self._num_envs])
        self._state = RenderEnvState(data, obs, reward, terminated, truncated, info)
        self._reset_done_envs()
        self._state.validate()

    def step(self, actions: np.ndarray) -> RenderEnvState:
        """Execute one environment step with optional MotrixSim video recording."""
        state = super().step(actions)

        # Capture frame to MotrixSim video (if enabled)
        if self._motrix_video_enabled:
            self._capture_motrix_frame()

        return state

    def _reset_done_envs(self):
        state: RenderEnvState = self._state
        done = state.done
        data_subset = state.data[done]

        if self._bg_renderer is not None:
            body_pos_sub, body_quat_sub = self._compute_body_arrays(data_subset)
            cam_xpos_sub, cam_xmat_sub, fovy_sub = self._compute_camera_arrays(
                data_subset
            )
            bg_gsb_sub = self._bg_renderer.batch_update_gaussians(
                body_pos_sub, body_quat_sub
            )
            bg_imgs_sub, _ = self._bg_renderer.batch_env_render(
                bg_gsb_sub,
                cam_xpos_sub,
                cam_xmat_sub,
                self._img_h,
                self._img_w,
                fovy_sub,
            )

            if self._bg_imgs is None:
                B = int(self._num_envs)
                self._bg_imgs = torch.zeros(
                    (B,) + tuple(bg_imgs_sub.shape[1:]),
                    dtype=bg_imgs_sub.dtype,
                    device=bg_imgs_sub.device,
                )
            mask_t = torch.from_numpy(done.astype(np.bool_)).to(self._bg_imgs.device)
            self._bg_imgs[mask_t] = bg_imgs_sub

        super()._reset_done_envs()

    def close(self):
        """Clean up resources including MotrixSim video recorder."""
        # Stop MotrixSim video recorder
        if self._motrix_recorder is not None:
            self._motrix_recorder.stop()
            self._motrix_recorder = None


__all__ = ["RenderEnvCfg", "RenderEnvState", "NpRenderEnv"]
