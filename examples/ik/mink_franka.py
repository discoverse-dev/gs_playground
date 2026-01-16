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
from gs_playground.src.manipulation.robots.franka_emika_panda_robotiq.franka_robotiq import FrankaRobotiq


_TEST_SPEED = False
_USE_MOCAP_IK = True
_SYNC = True

H = 300; W = 400
_ARM_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]
_FINGER_JOINTS = ["right_driver_joint", "left_driver_joint"]

_ASSETS_FRANKA_DIR = ROOT_PATH / "models" / "robots" / "manipulation" / "franka_emika_panda_robotiq"
_ASSETS_TASK_DIR = ROOT_PATH / "models" / "tasks" / "table30" / "_01_press_three_buttons"

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
    path = _ASSETS_FRANKA_DIR / "xmls"
    update_assets(assets, path, "*.xml")
    update_assets(assets, path / "assets", recursive=True)
    update_assets(assets, _ASSETS_TASK_DIR / "meshes", recursive=True)
    return assets

class FrankaCfg:
    mjcf_file_path = "xmls/table30_01_press_three_buttons.xml"
    decimation     = 8
    timestep       = 0.005

    gaussians = FrankaRobotiq.robot_gaussians()

class FrankaBase:
    def __init__(self, config: FrankaCfg):
        self.config = config
        self.free_camera = None

        xml_path = _ASSETS_FRANKA_DIR / self.config.mjcf_file_path
        self.mjcf_xml = xml_path.read_text()
        self._model_assets = get_assets()
        self.mj_model = mujoco.MjModel.from_xml_string(self.mjcf_xml, assets=self._model_assets)
        self.mj_model.opt.timestep = self.config.timestep
        self.mj_data = mujoco.MjData(self.mj_model)

        self._robot_arm_qposadr = np.array([
            self.mj_model.jnt_qposadr[self.mj_model.joint(j).id] for j in _ARM_JOINTS
        ])
        self._robot_qposadr = np.array([
            self.mj_model.jnt_qposadr[self.mj_model.joint(j).id] for j in _ARM_JOINTS + _FINGER_JOINTS
        ])
        self.renderer = GSRendererMuJoCo(self.config.gaussians, self.mj_model)
        self.timing_stats = []

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
        evt_start = torch.cuda.Event(enable_timing=True)
        evt_update = torch.cuda.Event(enable_timing=True)
        evt_render = torch.cuda.Event(enable_timing=True)
        evt_end = torch.cuda.Event(enable_timing=True)

        evt_start.record()
        self.renderer.update_gaussians(self.mj_data)
        evt_update.record()
        
        results_tensor = self.renderer.render(
            self.mj_model,
            self.mj_data,
            [-1, 0] if self.free_camera is not None else [0],  # camera id list
            W,
            H,
            self.free_camera
        )
        evt_render.record()
        
        rgb_np = (255. * torch.clamp(results_tensor[0][0], 0.0, 1.0)).to(torch.uint8).cpu().numpy()
        evt_end.record()

        if _TEST_SPEED:
            torch.cuda.synchronize()
            t_update = evt_start.elapsed_time(evt_update)
            t_render = evt_update.elapsed_time(evt_render)
            t_post = evt_render.elapsed_time(evt_end)
            t_total = evt_start.elapsed_time(evt_end)
            
          
            self.timing_stats.append({
                'update': t_update,
                'render': t_render,
                'post': t_post,
                'total': t_total,
            })

        observation_dict = {
            "state" : self.mj_data.qpos[self._robot_qposadr].copy(),
            "rgb"   : rgb_np,
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
    cfg = FrankaCfg()
    cfg.gaussians["background"] = FrankaRobotiq.robot_background_ply()
    # cfg.gaussians["cube_blue"] = (_ASSETS_TASK_DIR / "3dgs" / "cube_blue.ply").as_posix()
    # cfg.gaussians["cube_orange"] = (_ASSETS_TASK_DIR / "3dgs" / "cube_orange.ply").as_posix()
    # cfg.gaussians["cube_yellow"] = (_ASSETS_TASK_DIR / "3dgs" / "cube_yellow.ply").as_posix()

    exec_node = FrankaBase(cfg)
    obs = exec_node.reset()
    exec_node.timing_stats = [] # Clear warmup stats


    if _USE_MOCAP_IK:
        ik_model = mujoco.MjModel.from_xml_string(exec_node.mjcf_xml, assets=exec_node._model_assets)
        ik_solver = MinkIK(ik_model, len(_ARM_JOINTS), frame_name="gripper")

        mocap_name = "mocap_target"
        mocap_box_name = mocap_name + "_box"
        mocap_id = exec_node.mj_model.body(mocap_name).mocapid[0]

        mink.move_mocap_to_frame(exec_node.mj_model, exec_node.mj_data, mocap_name, "gripper", "site")
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
                    mink.move_mocap_to_frame(exec_node.mj_model, exec_node.mj_data, mocap_name, "gripper", "site")
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
            
            if _TEST_SPEED and len(exec_node.timing_stats) >= 100:
                print("=" * 60)
                print(f"Average Timing over {len(exec_node.timing_stats)} steps:")
                
                avg_update = np.mean([s['update'] for s in exec_node.timing_stats])
                avg_render = np.mean([s['render'] for s in exec_node.timing_stats])
                avg_post = np.mean([s['post'] for s in exec_node.timing_stats])
                avg_total = np.mean([s['total'] for s in exec_node.timing_stats])
                
                print(f"  Update Gaussians: {avg_update:.2f} ms")
                print(f"  Render:           {avg_render:.2f} ms")
                print(f"  Post-process:     {avg_post:.2f} ms")
                print(f"  Total:            {avg_total:.2f} ms")
                
                if len(exec_node.timing_stats) > 0 and 'detailed' in exec_node.timing_stats[0]:
                    print(f"  Detailed CUDA Timing (Average):")
                    first_detailed = exec_node.timing_stats[0]['detailed']
                    for name in first_detailed.keys():
                        avg_val = np.mean([s['detailed'][name] for s in exec_node.timing_stats])
                        print(f"    {name}: {avg_val:.2f} ms")
                break

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
