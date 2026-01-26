from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Sequence

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .mj_env import MjNpEnv, MjNpEnvState, EnvCfg
from gaussian_renderer import BatchSplatConfig, MjxBatchSplatRenderer, GSRendererMuJoCo

@dataclass
class RenderEnvCfg(EnvCfg):
    """
    Rendering configuration for MuJoCo environments.
    """
    img_width: int = 320
    img_height: int = 240
    cam_id: Sequence[int] = field(default_factory=lambda: [0])
    
    # 3DGS assets
    gs_background_ply: str = ""
    gs_robot_gaussians: Dict[str, str] = field(default_factory=dict)
    
@dataclass
class RenderEnvState(MjNpEnvState):
    """
    State with pixel observations.
    """
    obs: Dict[str, torch.Tensor]


class MujocoRenderEnv(MjNpEnv):
    """
    A MuJoCo-based environment with efficient batch Gaussian Splatting rendering.
    Uses sensor injection and compiled rollout to extract visualization data efficiently.
    """
    
    def __init__(self, cfg: RenderEnvCfg, num_envs: int = 1):
        self._render_cfg = cfg # convenience alias
        
        # 1. Patch XML to add sensors for efficient pose extraction
        # This must be done BEFORE calling super().__init__ which loads the model
        model_path = cfg.model_file
        
        # We need to know which bodies and cameras to track
        # Cameras are defined by config
        # Bodies are defined by gs_robot_gaussians keys
        self._body_names = list(cfg.gs_robot_gaussians.keys())
        
        # We need to temporarily load the model to resolve camera IDs to names if they are indices
        # or verify existence. 
        # However, to avoid double loading, we can parse XML directly or just load once.
        # Loading once is safer for name resolution.
        temp_model = mujoco.MjModel.from_xml_path(model_path)
        
        cam_ids = cfg.cam_id if isinstance(cfg.cam_id, (list, tuple)) else [cfg.cam_id]
        self._cam_ids = [int(c) for c in cam_ids]
        
        self._cam_names = []
        for cid in self._cam_ids:
            name = mujoco.mj_id2name(temp_model, mujoco.mjtObj.mjOBJ_CAMERA, cid)
            # If camera has no name, we might need to give it one or rely on ID? 
            # Sensor framepos/quat usually works with objname. 
            if not name:
                # MuJoCo XML requires names for sensor binding by name.
                # If using index binding, we need to handle that. 
                # For simplicity, assume cameras have names or we generated them?
                # Actually, if we use temp file, we can ADD names if missing!
                pass 
            self._cam_names.append(name)
            
        del temp_model

        # Inject sensors
        patched_xml_path = self._inject_sensors(model_path, self._body_names, self._cam_names)
        
        # 2. Update config to use patched model
        original_model_file = cfg.model_file
        cfg.model_file = patched_xml_path
        
        try:
            super().__init__(cfg, num_envs)
        finally:
            # Restore config and cleanup temp file
            # cfg.model_file = original_model_file # Keep custom file? No, we loaded it into self._model.
            # MJModel loads file content. We can delete file now.
            if os.path.exists(patched_xml_path):
               os.remove(patched_xml_path)
               
        # 3. Setup Renderer and caching
        self._init_renderer()
        self._cache_sensor_indices()
        
        # BG caching
        self._bg_renderer = None
        self._bg_imgs = None
        if cfg.gs_background_ply:
            bg_cfg = BatchSplatConfig(
                body_gaussians={},
                background_ply=cfg.gs_background_ply,
                minibatch=512
            )
            self._bg_renderer = MjxBatchSplatRenderer(bg_cfg, self._model)
            
        self._img_w = cfg.img_width
        self._img_h = cfg.img_height

    def _inject_sensors(self, xml_path: str, body_names: list, cam_names: list) -> str:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        sensor_sec = root.find("sensor")
        if sensor_sec is None:
            sensor_sec = ET.SubElement(root, "sensor")
            
        # Add body sensors
        for bn in body_names:
            ET.SubElement(sensor_sec, "framepos", name=f"gs_pos_b_{bn}", objtype="body", objname=bn)
            ET.SubElement(sensor_sec, "framequat", name=f"gs_quat_b_{bn}", objtype="body", objname=bn)
            
        # Add camera sensors
        for i, cn in enumerate(cam_names):
            if cn:
                ET.SubElement(sensor_sec, "framepos", name=f"gs_pos_c_{cn}", objtype="camera", objname=cn)
                ET.SubElement(sensor_sec, "framequat", name=f"gs_quat_c_{cn}", objtype="camera", objname=cn)
            else:
                 # Handle unnamed camera? 
                 # We can use index referencing if modifying XML allows 'objid' but standard XML uses names.
                 # Fallback: We might not support unnamed cameras for now or need to name them in XML.
                 # For now assuming names exist.
                 pass

        # Write to temp file
        fd, tmp_path = tempfile.mkstemp(suffix=".xml", dir=os.path.dirname(xml_path))
        os.close(fd)
        tree.write(tmp_path)
        return tmp_path

    def _cache_sensor_indices(self):
        # We need to find where our sensors live in the sensor array
        self._b_pos_idxs = np.array([self._get_sensor_range(f"gs_pos_b_{bn}", 3) for bn in self._body_names])
        self._b_quat_idxs = np.array([self._get_sensor_range(f"gs_quat_b_{bn}", 4) for bn in self._body_names])
        
        self._c_pos_idxs = []
        self._c_quat_idxs = []
        
        # Only track valid named cameras
        self._valid_cam_indices = [] # indices in self._cam_ids/names that actully have sensors
        
        for i, cn in enumerate(self._cam_names):
            if cn:
                self._c_pos_idxs.append(self._get_sensor_range(f"gs_pos_c_{cn}", 3))
                self._c_quat_idxs.append(self._get_sensor_range(f"gs_quat_c_{cn}", 4))
                self._valid_cam_indices.append(i)
                
        self._c_pos_idxs = np.array(self._c_pos_idxs)
        self._c_quat_idxs = np.array(self._c_quat_idxs)

    def _init_renderer(self):
        cfg = self._render_cfg
        batch_cfg = BatchSplatConfig(
            body_gaussians=cfg.gs_robot_gaussians,
            background_ply=None,
            minibatch=512
        )
        self._renderer = MjxBatchSplatRenderer(batch_cfg, self._model)

    def init_state(self) -> RenderEnvState:
        # Override to setup dictionary obs
        obs_dim = 0 # Placeholder, we use dict
        
        obs = self._init_obs_dict(self._num_envs)
        reward = np.zeros((self._num_envs,), dtype=np.float32)
        terminated = np.ones((self._num_envs,), dtype=bool)
        truncated = np.zeros((self._num_envs,), dtype=bool)
        info = {"steps": np.zeros((self._num_envs,), dtype=np.uint64)}
        
        data = [mujoco.MjData(self._model) for _ in range(self._num_envs)]
        
        self._state = RenderEnvState(data, obs, reward, terminated, truncated, info)
        self._reset_done_envs()
        self._state.validate()
        return self._state

    def _init_obs_dict(self, n: int) -> Dict[str, torch.Tensor]:
        obs = {}
        # We can add state obs here if needed, but primary focus is pixels
        # obs["state"] = ...
        
        for i, cam_id in enumerate(self._cam_ids):
             obs[f"pixels/view_{i}"] = torch.zeros(
                 (n, self._img_h, self._img_w, 3), dtype=torch.uint8
             )
        return obs

    def update_state(self, state: RenderEnvState, obs_required: bool = True) -> RenderEnvState:
        # This calls user implementation which update state.obs, etc.
        # But for RenderEnv, we might want to automatically render pixels if obs_required is True.
        # Since MujocoEnv.update_state is abstract, user implements it.
        # User Implementation should call self.render_pixels() to get images.
        return state

    def render_pixels(self) -> Dict[str, torch.Tensor]:
        """
        Extract pose data from the last sensor trajectory and render images.
        """
        if self._last_sensor_traj is None:
            # Maybe called before first step? force a forward to get sensors?
            # Or just return zeros
            return self._init_obs_dict(self._num_envs)
            
        final_sensors = self._last_sensor_traj # (num_env, nsensordata)
        
        # 1. Extract Body Poses
        # (num_env, nbody, 3)
        body_pos = final_sensors[:, self._b_pos_idxs.flatten()].reshape(self._num_envs, len(self._body_names), 3)
        body_quat_wxyz = final_sensors[:, self._b_quat_idxs.flatten()].reshape(self._num_envs, len(self._body_names), 4)
        body_quat = np.roll(body_quat_wxyz, -1, axis=-1) # wxyz -> xyzw
        
        # 2. Extract Camera Poses
        # Only for valid named cameras
        # (num_env, n_valid_cam, 3)
        n_cam = len(self._valid_cam_indices)
        if n_cam == 0:
            return {}
            
        cam_pos = final_sensors[:, self._c_pos_idxs.flatten()].reshape(self._num_envs, n_cam, 3)
        cam_quat_wxyz = final_sensors[:, self._c_quat_idxs.flatten()].reshape(self._num_envs, n_cam, 4)
        cam_quat_xyzw = np.roll(cam_quat_wxyz, -1, axis=-1)
        
        # Convert Quat to Matrix
        # Flatten for efficient batched rotation
        r = Rotation.from_quat(cam_quat_xyzw.reshape(-1, 4))
        cam_xmat = r.as_matrix().reshape(self._num_envs, n_cam, 9)
        
        # FOV (static)
        # We should cache this maybe
        fovy = np.zeros((self._num_envs, n_cam), dtype=np.float32)
        for i, idx in enumerate(self._valid_cam_indices):
            fovy[:, i] = self._model.cam_fovy[self._cam_ids[idx]]
            
        # 3. Render
        H, W = self._img_h, self._img_w
        
        # Background Cache
        if self._bg_renderer is not None and self._bg_imgs is None:
             bg_gsb = self._bg_renderer.batch_update_gaussians(body_pos, body_quat)
             self._bg_imgs, _ = self._bg_renderer.batch_env_render(
                 bg_gsb, cam_pos, cam_xmat, H, W, fovy
             )
             
        # Foreground
        gsb = self._renderer.batch_update_gaussians(body_pos, body_quat)
        rgb_t, depth_t = self._renderer.batch_env_render(
            gsb, cam_pos, cam_xmat, H, W, fovy, bg_imgs=self._bg_imgs
        )
        
        # 4. Pack into Dict
        # (num_env, n_valid_cam, H, W, 3) -> Transpose to (num_env, n_valid_cam, H, W, 3) ?
        # batch_env_render returns (num_env, n_cam, H, W, 3) usually?
        # Check mj_batch.ipynb: rgb.shape
        # It seems returns (num_env * n_cam, H, W, 3) or (num_env, n_cam, ...)
        # mj_batch says: rgb[:16,0,...], implies (num_env, n_cam, H, W, 3)
        
        rgb_np_u8 = (255 * torch.clamp(rgb_t, 0.0, 1.0)).to(torch.uint8).cpu().numpy()
        
        obs_pix = {}
        for i, cam_idx in enumerate(self._valid_cam_indices):
             # Original cam index for view_i naming?
             # If we skipped some cameras because unnamed, indices shift. 
             # Assuming linear mapping for valid ones.
             obs_pix[f"pixels/view_{i}"] = rgb_t[:, i] # Keep Tensor or Numpy? 
             # mtx_env RenderEnv returns tensor in _render_pixels? 
             # "torch.clamp(rgb_t...)" implies tensor.
             # Wait, mtx_env converts to numpy u8 in _render_pixels.
             obs_pix[f"pixels/view_{i}"] = torch.from_numpy(rgb_np_u8[:, i])
             
        return obs_pix

    def _reset_done_envs(self):
         # Standard reset
         super()._reset_done_envs()
         
         # If we reset, we might want to update pixels for those envs immediately if obs required?
         # super()._reset_done_envs() calls reset() -> gets obs.
         # For RenderEnv, reset() impl should call render_pixels() if it wants image obs for reset states.
         # However, render_pixels() relies on _last_sensor_traj which comes from rollout.
         # Newly reset envs don't have rollouts yet. They have init state.
         
         # Handling "Render on Reset":
         # If we need obs for reset envs, we need to run forward to get sensors for them.
         # But _last_sensor_traj is global for all envs.
         # We could selectively update _last_sensor_traj for reset indices using mj_forward.
         pass
