from etils import epath
import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt
from gs_playground import ROOT_PATH

# import queue as pyqueue
# import multiprocessing as mp
# from gamepad_reader import joystick_process_main

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE.parent / "onnx"
_MJCF_PATH = ROOT_PATH / "models" / "robots" / "locomotion" / "go1" / "scene_mjx_feetonly_flat_terrain.xml"


_JOINT_NUM = 12

# Default angles from keyframe "home" in scene_mjx_feetonly_flat_terrain.xml
# FR, FL, RR, RL
default_joint_pos = np.array([
    0.1, 0.9, -1.8,   # FR
    -0.1, 0.9, -1.8,  # FL
    0.1, 0.9, -1.8,   # RR
    -0.1, 0.9, -1.8   # RL
])

action_scale = 0.5
lin_vel_scale = 2.0
ang_vel_scale = 0.25

class OnnxController:
    """ONNX controller for the Go1 robot."""

    def __init__(
        self,
        policy_path: str,
        default_angles: np.ndarray,
        n_substeps: int,
        action_scale: float = 0.5,
    ):
        self._policy = rt.InferenceSession(
            policy_path, providers=["CPUExecutionProvider"]
        )
        self._output_names = [self._policy.get_outputs()[0].name]

        self._action_scale = action_scale
        self._default_angles = default_angles
        self._last_action = np.zeros_like(default_angles, dtype=np.float32)

        self._counter = 0
        self._n_substeps = n_substeps

        # self.joy_queue = mp.Queue(maxsize=1)
        # joy_stop_event = mp.Event()
        # self.joy_process = mp.Process(target=joystick_process_main, args=(self.joy_queue, joy_stop_event), daemon=True)
        # self.joy_process.start()
        # self.latest_axes, self.latest_buttons = None, None
        
        # Command state
        self.command = np.zeros(3, dtype=np.float32)

    def get_obs(self, model, data) -> np.ndarray:
        # Match observation space from Go1WalkTaskMj._get_obs
        # 1. Linear Vel (Local)
        linvel = data.sensor("local_linvel").data
        # 2. Gyro
        gyro = data.sensor("gyro").data
        # 3. Local Gravity (Projected)
        imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0, 0, -1])
        # 4. Joint Diff
        joint_angles = (data.qpos[7:7+_JOINT_NUM] - self._default_angles)
        # 5. Joint Vel
        joint_velocities = data.qvel[6:6+_JOINT_NUM]
        
        # Handle commands (Joystick or Fixed)
        # if not self.joy_queue is None:
        #     try:
        #         self.latest_axes, self.latest_buttons = self.joy_queue.get_nowait()
        #         # Using scale variables
        #         self.command[0] = -self.latest_axes[1] * lin_vel_scale
        #         self.command[1] = -self.latest_axes[0] * lin_vel_scale * 0.5
        #         self.command[2] = -self.latest_axes[3] * ang_vel_scale * 2.0 
                
        #     except pyqueue.Empty:
        #         pass

        # For simple test, set command to walk forward
        self.command[0] = 1.0 # 1.0 m/s forward

        obs = np.hstack([
            linvel,
            gyro,
            gravity,
            joint_angles,
            joint_velocities,
            self._last_action,
            self.command
        ])
        return obs.astype(np.float32)

    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._counter += 1
        if self._counter % self._n_substeps == 0:
            obs = self.get_obs(model, data)
            onnx_input = {self._policy.get_inputs()[0].name: obs.reshape(1, -1)}
            onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
            self._last_action = onnx_pred.copy()
            data.ctrl[:] = onnx_pred * self._action_scale + self._default_angles

def load_callback(model=None, data=None):
    mujoco.set_mjcb_control(None)

    model = mujoco.MjModel.from_xml_path(
        _MJCF_PATH.as_posix()
    )
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Go1 cfg: sim_dt=0.004, ctrl_dt=0.02
    ctrl_dt = 0.02
    sim_dt = 0.004
    n_substeps = int(round(ctrl_dt / sim_dt))
    model.opt.timestep = sim_dt

    print(f"Loading ONNX policy from: {_ONNX_DIR / 'go1_policy.onnx'}")

    policy = OnnxController(
        policy_path=(_ONNX_DIR / "go1_policy.onnx").as_posix(),
        default_angles=default_joint_pos,
        n_substeps=n_substeps,
        action_scale=action_scale,
    )

    mujoco.set_mjcb_control(policy.get_control)

    return model, data

if __name__ == "__main__":
    viewer.launch(loader=load_callback)
