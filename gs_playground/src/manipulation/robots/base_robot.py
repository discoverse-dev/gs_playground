from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import motrixsim as mtx
import gymnasium as gym
from gs_playground.src.utils.path_utils import as_posix_if_exists

class BaseRobot(ABC):
    """
    Abstract base class for robot manipulation helpers.
    Defines the interface for applying actions and retrieving robot-specific observations.
    """
    num_dof_arm = 7
    ee_site_name = None
    gripper_actuator_name = None

    GAUSSIANS: Dict[str, Path] = {}
    BACKGROUND_PLY: Path = None

    def __init__(self, mx_model: mtx.SceneModel):
        self.mx_model = mx_model
        # self.num_envs = num_envs
        self._setup_robot()

        # State memory for relative control
        self.ref_qpos = None
        self.ref_ee_pose = None
        
        # State memory for delta control (last command)
        self.last_cmd_qpos = None
        self.last_cmd_ee_pose = None

    def _setup_robot(self):
        self.ee_site = self.mx_model.get_site(self.ee_site_name)
        self.gripper_act_id = self.mx_model.get_actuator_index(self.gripper_actuator_name)

    def reset_envs(self, data: mtx.SceneData, env_mask: np.ndarray):
        """
        Partially resets state for specific environments (based on mask).
        Syncs last command and reference to current physical state for those envs.
        """
        if not np.any(env_mask):
            return

        # 1. Sync Joint State
        curr_q = np.asarray(data.dof_pos[:, :self.num_dof_arm], dtype=np.float32)
        if self.last_cmd_qpos is not None:
            self.last_cmd_qpos[env_mask] = curr_q[env_mask]
        
        if self.ref_qpos is not None:
            self.ref_qpos[env_mask] = curr_q[env_mask]

        # 2. Sync EE Pose
        # Check if we need to compute pose (expensive if not needed)
        if self.last_cmd_ee_pose is not None or self.ref_ee_pose is not None:
             curr_ee = self.get_ee_pose(data)
             
             if self.last_cmd_ee_pose is not None:
                 self.last_cmd_ee_pose[env_mask] = curr_ee[env_mask]
            
             if self.ref_ee_pose is not None:
                 self.ref_ee_pose[env_mask] = curr_ee[env_mask]

    def update_reference(self, data: mtx.SceneData):
        """Updates the reference state for relative control (e.g. at start of inference chunk)."""
        # If last command exists, use it to prevent jump due to tracking error
        if self.last_cmd_qpos is not None:
            self.ref_qpos = self.last_cmd_qpos.copy()
        else:
            self.ref_qpos = np.asarray(data.dof_pos[:, :self.num_dof_arm], dtype=np.float32).copy()
            
        if self.last_cmd_ee_pose is not None:
             self.ref_ee_pose = self.last_cmd_ee_pose.copy()
        else:
             self.ref_ee_pose = self.get_ee_pose(data).copy()

    def apply_action(self, data: mtx.SceneData, action: np.ndarray, action_mode: str = "joint"):
        # Lazy initialization of reference if not set
        if self.ref_qpos is None:
            self.update_reference(data)

        # Lazy initialization of Last Command (if None, sync to current state)
        if self.last_cmd_qpos is None:
             self.last_cmd_qpos = np.asarray(data.dof_pos[:, :self.num_dof_arm], dtype=np.float32).copy()
        if self.last_cmd_ee_pose is None:
             self.last_cmd_ee_pose = self.get_ee_pose(data).copy()

        act = np.asarray(action, dtype=np.float32)
        
        # Parse mode
        parts = action_mode.split('_')
        main_mode = parts[0]
        sub_mode = parts[1] if len(parts) > 1 else 'absolute'

        if main_mode == "eef":
            # Action: 7 dims (3 pos + 3 rpy + 1 gripper)
            act_pose = act[:, :6]
            act_gripper = act[:, 6]
            
            target_pose_6d = None
            
            if sub_mode == 'absolute':
                target_pose_6d = act_pose
            elif sub_mode == 'delta':
                # Delta is relative to LAST COMMAND
                target_pose_6d = self.last_cmd_ee_pose + act_pose
            elif sub_mode == 'relative':
                # Relative is relative to REFERENCE pose (e.g. start of chunk)
                target_pose_6d = self.ref_ee_pose + act_pose
            else:
                raise ValueError(f"Unknown sub_mode: {sub_mode}")

            self.last_cmd_qpos = None # Invalidate joint history to force sync on mode switch

            # Convert Target (XYZ+RPY) to XYZ+Quat (xyzw)
            t_pos = target_pose_6d[:, :3]
            t_rpy = target_pose_6d[:, 3:] # Roll, Pitch, Yaw
            
            # rpy -> zyx input: [yaw, pitch, roll]
            t_euler_zyx = np.flip(t_rpy, axis=-1)
            r = Rotation.from_euler('zyx', t_euler_zyx, degrees=False)
            t_quat_xyzw = r.as_quat()
            
            # motrixsim uses xyzw, so direct concatenation
            desired_pose = np.concatenate([t_pos, t_quat_xyzw], axis=-1).astype(np.float32)
            
            res = self.solver.solve(self.chain, data, desired_pose)
            
            residual = np.asarray(res[..., 1], dtype=np.float32)
            desired_j = np.asarray(res[..., 2:], dtype=np.float32)
            
            ctrl = np.array(data.actuator_ctrls, copy=False) 
            
            good = np.isfinite(residual) & (residual < 2e-2) 
            if np.any(good):
                ctrl[good, :self.num_dof_arm] = desired_j[good, :self.num_dof_arm]
                # Update Last Command only for successful IK solves
                self.last_cmd_ee_pose[good] = target_pose_6d[good]

            # Gripper
            ctrl[:, self.gripper_act_id] = act_gripper
            
            data.actuator_ctrls = ctrl

        elif main_mode == "joint":
            # Action: self.num_dof_arm joints + 1 gripper
            act_joints = act[:, :self.num_dof_arm]
            act_gripper = act[:, self.num_dof_arm]

            current_joints = np.asarray(data.dof_pos[:, :self.num_dof_arm], dtype=np.float32)
            target_joints = None
            
            if sub_mode == 'absolute':
                target_joints = act_joints
            elif sub_mode == 'delta':
                # Delta is relative to LAST COMMAND
                target_joints = self.last_cmd_qpos + act_joints
            elif sub_mode == 'relative':
                target_joints = self.ref_qpos + act_joints
            else:
                raise ValueError(f"Unknown sub_mode: {sub_mode}")
            
            # Update Last Command
            self.last_cmd_qpos = target_joints
            self.last_cmd_ee_pose = None # Invalidate eef history to force sync on mode switch

            new_ctrl = np.zeros_like(data.actuator_ctrls)
            new_ctrl[:, :self.num_dof_arm] = target_joints
            
            # Preserve existing if correct, but here we overwrite
            # Since we construct full state
            new_ctrl[:, self.gripper_act_id] = act_gripper
            
            data.actuator_ctrls = new_ctrl

        else:
            raise ValueError(f"Unknown action_mode: {action_mode}")

    def get_obs(self, data: mtx.SceneData) -> Dict[str, np.ndarray]:
        qpos = np.asarray(data.dof_pos[:, :self.num_dof_arm], dtype=np.float32)
        ee_pose = self.get_ee_pose(data)
        gripper_ctrl = np.asarray(data.actuator_ctrls)[:, self.gripper_act_id : self.gripper_act_id + 1]
        
        return {
            "qpos": qpos,
            "ee_pose": ee_pose,
            "gripper": gripper_ctrl.astype(np.float32)
        }

    def get_ee_pose(self, data: mtx.SceneData) -> np.ndarray:
        # Return 6D pose: XYZ + RPY (Roll, Pitch, Yaw)
        # motrixsim EE pose is [x, y, z, qx, qy, qz, qw] (xyzw)
        pose_7d = np.asarray(self.ee_site.get_pose(data), dtype=np.float32)
        pos = pose_7d[..., :3]
        quat_xyzw = pose_7d[..., 3:]
        
        # scipy expects xyzw, so we can use directly
        r = Rotation.from_quat(quat_xyzw)
        # zyx -> [yaw, pitch, roll]
        euler_zyx = r.as_euler('zyx', degrees=False)
        # flip to [roll, pitch, yaw]
        rpy = np.flip(euler_zyx, axis=-1)
        
        return np.concatenate([pos, rpy], axis=-1).astype(np.float32)
        
    @property
    def action_space(self) -> gym.Space:
        limits = self.mx_model.actuator_ctrl_limits
        return gym.spaces.Box(low=limits[0], high=limits[1], dtype=np.float32)

    @classmethod
    def robot_gaussians(cls) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for name, path in cls.GAUSSIANS.items():
            posix = as_posix_if_exists(path)
            if posix:
                out[name] = posix
        return out

    @classmethod
    def robot_background_ply(cls) -> Optional[str]:
        return as_posix_if_exists(cls.BACKGROUND_PLY)
