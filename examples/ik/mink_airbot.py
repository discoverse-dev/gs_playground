from typing import Any, Dict, Optional, Union

import time
import mink
import mujoco
import mujoco.viewer
import torch
import numpy as np
from etils import epath
from gaussian_renderer import GSRendererMuJoCo
from gs_playground.src.utils.mink_arm_ik import MinkIK

from gs_playground import ROOT_PATH
from gs_playground.src.manipulation.robots.airbot_play.airbot_play import AirbotPlay

_USE_MOCAP_IK = True
_SYNC = True

H = 300; W = 400
_ASSETS_AIRBOT_PLAY_DIR = ROOT_PATH / "models" / "robots" / "manipulation" / "airbot_play"

def update_assets(
    assets: Dict[str, Any],
    path: Union[str, epath.Path],
    glob: str = "*",
    recursive: bool = False,
):
  for f in epath.Path(path).glob(glob):
    if f.is_file():
      assets[f.name] = f.read_bytes()
    elif f.is_dir() and recursive:
      update_assets(assets, f, glob, recursive)

def get_assets() -> Dict[str, bytes]:
    assets = {}
    path = _ASSETS_AIRBOT_PLAY_DIR / "xmls"
    update_assets(assets, path, "*.xml")
    update_assets(assets, path / "assets", recursive=True)
    return assets

class AirbotPlayCfg:
    mjcf_file_path = "xmls/single_cube.xml"
    decimation     = 8
    timestep       = 0.005
    gaussians = AirbotPlay.robot_gaussians()

class AirbotPlayBase:
    def __init__(self, config: AirbotPlayCfg):
        self.config = config
        self.free_camera = None

        xml_path = _ASSETS_AIRBOT_PLAY_DIR / self.config.mjcf_file_path
        self.mjcf_xml = xml_path.read_text()
        self._model_assets = get_assets()
        self.mj_model = mujoco.MjModel.from_xml_string(self.mjcf_xml, assets=self._model_assets)
        self.mj_model.opt.timestep = self.config.timestep
        self.mj_data = mujoco.MjData(self.mj_model)

        self.renderer = GSRendererMuJoCo(self.config.gaussians, self.mj_model)

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.mj_model.key("home").id)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return self.getObservation()

    def step(self, action: np.ndarray = None):
        if action is not None:
            self.mj_data.ctrl[:] = action

        for _ in range(self.config.decimation):
            mujoco.mj_step(self.mj_model, self.mj_data)

    def checkSuccess(self):
        return False

    def getObservation(self):
        self.renderer.update_gaussians(self.mj_data)
        results_tensor = self.renderer.render(
            self.mj_model,
            self.mj_data,
            [-1, 0] if self.free_camera is not None else [0],  # camera id list
            W,
            H,
            self.free_camera
        )
        rgb_np = (255. * torch.clamp(results_tensor[0][0], 0.0, 1.0)).to(torch.uint8).cpu().numpy()
        
        observation_dict = {
            "state"       : self.mj_data.sensordata[:7].copy(),
            "rgb"         : rgb_np,
        }
        
        if self.free_camera is not None and -1 in results_tensor:
            rgb_free = (255. * torch.clamp(results_tensor[-1][0], 0.0, 1.0)).to(torch.uint8).cpu().numpy()
            observation_dict["free_camera"] = rgb_free
        
        return observation_dict

    def set_mocap_target(self, target_name, target_pos, target_quat, box_color=(0,1,0,0.1)):
        """设置Mocap目标位置和姿态"""
        mocap_id = self.mj_model.body(target_name).mocapid
        if mocap_id >= 0:
            self.mj_data.mocap_pos[mocap_id] = target_pos
            self.mj_data.mocap_quat[mocap_id] = target_quat
            self.mj_model.geom(f'{target_name}_box').rgba = box_color

if __name__ == "__main__":
    cfg = AirbotPlayCfg()
    cfg.gaussians["background"] = AirbotPlay.robot_background_ply()
    # cfg.gaussians["cube_blue"] = (_ASSETS_TASK_DIR / "3dgs" / "cube_blue.ply").as_posix()
    # cfg.gaussians["cube_orange"] = (_ASSETS_TASK_DIR / "3dgs" / "cube_orange.ply").as_posix()
    # cfg.gaussians["cube_yellow"] = (_ASSETS_TASK_DIR / "3dgs" / "cube_yellow.ply").as_posix()

    exec_node = AirbotPlayBase(cfg)
    obs = exec_node.reset()

    if _USE_MOCAP_IK:
        ik_model = mujoco.MjModel.from_xml_string(exec_node.mjcf_xml, assets=exec_node._model_assets)
        ik_solver = MinkIK(ik_model, 6, frame_name="endpoint")

        mocap_name = "mocap_target"
        mocap_box_name = mocap_name + "_box"
        mocap_id = exec_node.mj_model.body(mocap_name).mocapid[0]

        mink.move_mocap_to_frame(exec_node.mj_model, exec_node.mj_data, mocap_name, "endpoint", "site")
        ik_solver.configuration.update(exec_node.mj_data.qpos)
        ik_solver.posture_task.set_target_from_configuration(ik_solver.configuration)

    _last_time = -1.
    with mujoco.viewer.launch_passive(exec_node.mj_model, exec_node.mj_data) as viewer:
        exec_node.free_camera = viewer.cam
        while viewer.is_running():
            if exec_node.mj_data.time < _last_time:
                _last_time = -1.
                exec_node.reset()
                if _USE_MOCAP_IK:
                    mink.move_mocap_to_frame(exec_node.mj_model, exec_node.mj_data, mocap_name, "endpoint", "site")
                    ik_solver.configuration.update(exec_node.mj_data.qpos)
                    ik_solver.posture_task.set_target_from_configuration(ik_solver.configuration)
            _last_time = exec_node.mj_data.time

            step_time = time.time()

            if _USE_MOCAP_IK:
                mink_target_se3 = mink.SE3.from_mocap_name(exec_node.mj_model, exec_node.mj_data, mocap_name)
                ik_solver.end_effector_task.set_target(mink_target_se3)
                res = ik_solver.converge_ik()
                if res:
                    # 设置目标框为绿色（表示IK计算成功）
                    exec_node.mj_model.geom(mocap_box_name).rgba = (0.3, 0.6, 0.3, 0.2)
                else:
                    # 设置目标框为红色（表示IK计算失败）
                    exec_node.mj_model.geom(mocap_box_name).rgba = (0.6, 0.3, 0.3, 0.2)
                solution = exec_node.mj_data.ctrl.copy()
                solution[:ik_solver.ndof_arm] = ik_solver.configuration.data.qpos[:ik_solver.ndof_arm]
            else:
                solution = None

            exec_node.step(solution)
            obs = exec_node.getObservation()
            
            if "free_camera" in obs:
                viewport = mujoco.MjrRect(viewer.viewport.left + viewer.viewport.width - W, 0, W, H * 2)
                viewer.set_images([(viewport, np.vstack([obs["free_camera"], obs["rgb"]]))])
            else:
                viewport = mujoco.MjrRect(viewer.viewport.left + viewer.viewport.width - W, 0, W, H)
                viewer.set_images([(viewport, obs["rgb"])])

            viewer.sync()
            if _SYNC:
                time.sleep(max(0, exec_node.mj_model.opt.timestep * cfg.decimation - (time.time() - step_time)))
