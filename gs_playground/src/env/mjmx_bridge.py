from typing import Tuple, List, Union, Dict, Optional, Any, TYPE_CHECKING

import numpy as np

import mujoco

if TYPE_CHECKING:
    # only import motrixsim for type checking to avoid hard dependency at runtime
    import motrixsim

class MjMxBridge:
    def __init__(self, mjcf_path: str, assets: Optional[Dict[str, Any]] = None):
        if assets:
            self._mj_model = mujoco.MjModel.from_xml_path(mjcf_path, assets=assets)
        else:
            self._mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self._mj_data = mujoco.MjData(self._mj_model)

        self.map_qpos_idx_mjmx = np.arange(self._mj_model.nq, dtype=np.int32)
        for i in range(self._mj_model.nbody):
            body = self._mj_model.body(i)
            if body.dofnum == 6 and self._mj_model.body_jntadr[i] > -1 and self._mj_model.jnt_type[self._mj_model.body_jntadr[i]] == int(mujoco.mjtJoint.mjJNT_FREE):
                # xyz + quat[wxyz] -> xyz + quat[xyzw]
                qpos_adr = self._mj_model.jnt_qposadr[body.jntadr[0]]
                self.map_qpos_idx_mjmx[qpos_adr+3:qpos_adr+7] = [
                    qpos_adr+4, qpos_adr+5, qpos_adr+6, qpos_adr+3
                ]

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model
    
    @property
    def mj_data(self) -> mujoco.MjData:
        return self._mj_data

    def reset(self) -> None:
        mujoco.mj_resetData(self._mj_model, self._mj_data)

    def forward(self) -> None:
        mujoco.mj_forward(self._mj_model, self._mj_data)

    def update(self, mx_data: "motrixsim.SceneData") -> mujoco.MjData:
        assert mx_data.dof_pos.shape[-1] == self._mj_model.nq, "DOF position size mismatch, {} vs {}".format(mx_data.dof_pos.shape[-1], self._mj_model.nq)
        if len(mx_data.dof_pos.shape) == 2:
            self._mj_data.qpos[self.map_qpos_idx_mjmx] = mx_data.dof_pos[0, :]
        else:
            self._mj_data.qpos[self.map_qpos_idx_mjmx] = mx_data.dof_pos[:]
        mujoco.mj_forward(self._mj_model, self._mj_data)
        return self._mj_data

    def load_keyframe(self, mx_data: "motrixsim.SceneData", mx_model: "motrixsim.SceneModel", keyframe_idx: Union[int, str]) -> None:
        mujoco.mj_resetData(self._mj_model, self._mj_data)
        mujoco.mj_resetDataKeyframe(self._mj_model, self._mj_data, self._mj_model.key(keyframe_idx).id)
        mujoco.mj_forward(self._mj_model, self._mj_data)
        if len(mx_data.dof_pos.shape) == 2:
            num_env = mx_data.dof_pos.shape[0]
            mx_data.set_dof_pos(np.repeat(self._mj_data.qpos[self.map_qpos_idx_mjmx][np.newaxis, :], num_env, axis=0), mx_model)
            mx_data.set_dof_vel(np.repeat(self._mj_data.qvel[np.newaxis, :], num_env, axis=0))
            mx_data.actuator_ctrls = np.repeat(self._mj_data.ctrl[np.newaxis, :], num_env, axis=0)
        else:
            mx_data.set_dof_pos(self._mj_data.qpos[self.map_qpos_idx_mjmx], mx_model)
            mx_data.set_dof_vel(self._mj_data.qvel)
            mx_data.actuator_ctrls = self._mj_data.ctrl.copy()
