import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation
from motrixsim import SceneData


class Go2LocomotionPolicy:
    _DEFAULT_ANGLES = np.array([0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8])
    onnx_path = "gs_playground/src/nav_demo/policies/go2_policy.onnx"

    def __init__(self, robot, action_scale=0.5, lin_vel_scale=1.0, ang_vel_scale=1.5):
        self._robot = robot
        self.default_angles = self._DEFAULT_ANGLES.copy()
        self.action_scale = action_scale
        self.lin_vel_scale = lin_vel_scale
        self.ang_vel_scale = ang_vel_scale
        self.last_action = np.zeros_like(self.default_angles, dtype=np.float32)
        self._policy_session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
        self._input_name = self._policy_session.get_inputs()[0].name
        self._output_name = self._policy_session.get_outputs()[0].name

    def scale_command(self, raw_command):
        scales = np.array([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale])
        return raw_command * scales

    def get_observation(self, data, command):
        lin_vel = self._robot.local_linear_vel(data)
        gyro = self._robot.gyro(data)
        gravity = self._robot.gravity(data)
        dof_pos = self._robot.dof_pos(data)
        dof_vel = self._robot.dof_vel(data)
        obs = np.hstack([lin_vel, gyro, gravity, dof_pos - self.default_angles, dof_vel, self.last_action, command])
        return obs.astype(np.float32)

    def compute_action(self, observation):
        outputs = self._policy_session.run([self._output_name], {self._input_name: observation.reshape(1, -1)})
        return outputs[0][0]

    def step(self, data, command):
        command = self.scale_command(command)
        obs = self.get_observation(data, command)
        action = self.compute_action(obs)
        self.apply_action(data, action)
        return self.is_fallen(data)

    def apply_action(self, data, action):
        ctrl = action * self.action_scale + self.default_angles
        self._robot.set_actuator_ctrls(data, ctrl)
        self.last_action = action.copy()

    def is_fallen(self, data):
        pose = self._robot.base_pose(data)
        rotation = Rotation.from_quat(pose[3:7])
        rotated_z_axis = rotation.apply(np.array([0.0, 0.0, 1.0]))
        return np.dot(rotated_z_axis, np.array([0.0, 0.0, 1.0])) < 0.3


class Go1LocomotionPolicy:
    _DEFAULT_ANGLES = np.array([0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8])
    onnx_path = "gs_playground/src/nav_demo/policies/go1_policy.onnx"

    def __init__(self, robot, action_scale=0.5, lin_vel_scale=0.7, ang_vel_scale=1.5):
        self._robot = robot
        self.default_angles = self._DEFAULT_ANGLES.copy()
        self.action_scale = action_scale
        self.lin_vel_scale = lin_vel_scale
        self.ang_vel_scale = ang_vel_scale
        self.last_action = np.zeros_like(self.default_angles, dtype=np.float32)
        self._policy_session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
        self._input_name = self._policy_session.get_inputs()[0].name
        self._output_name = self._policy_session.get_outputs()[0].name

    def scale_command(self, raw_command):
        scales = np.array([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale])
        return raw_command * scales

    def get_observation(self, data, command):
        lin_vel = self._robot.local_linear_vel(data)
        gyro = self._robot.gyro(data)
        gravity = self._robot.gravity(data)
        dof_pos = self._robot.dof_pos(data)
        dof_vel = self._robot.dof_vel(data)
        obs = np.hstack([lin_vel, gyro, gravity, dof_pos - self.default_angles, dof_vel, self.last_action, command])
        return obs.astype(np.float32)

    def compute_action(self, observation):
        outputs = self._policy_session.run([self._output_name], {self._input_name: observation.reshape(1, -1)})
        return outputs[0][0]

    def step(self, data, command):
        command = self.scale_command(command)
        obs = self.get_observation(data, command)
        action = self.compute_action(obs)
        self.apply_action(data, action)
        return self.is_fallen(data)

    def apply_action(self, data, action):
        ctrl = action * self.action_scale + self.default_angles
        self._robot.set_actuator_ctrls(data, ctrl)
        self.last_action = action.copy()

    def is_fallen(self, data):
        pose = self._robot.base_pose(data)
        rotation = Rotation.from_quat(pose[3:7])
        rotated_z_axis = rotation.apply(np.array([0.0, 0.0, 1.0]))
        return np.dot(rotated_z_axis, np.array([0.0, 0.0, 1.0])) < 0.3


class G1LocomotionPolicy:
    _DEFAULT_ANGLES = np.array([-0.312, 0, 0, 0.669, -0.363, 0, -0.312, 0, 0, 0.669, -0.363, 0, 0, 0, 0.073, 0.2, 0.2, 0, 0.6, 0, 0, 0, 0.2, -0.2, 0, 0.6, 0, 0, 0])
    onnx_path = "gs_playground/src/nav_demo/policies/g1_policy.onnx"

    def __init__(self, robot, action_scale=0.5, lin_vel_scale=1.0, ang_vel_scale=1.0, ctrl_dt=0.02):
        self._robot = robot
        self.default_angles = self._DEFAULT_ANGLES.copy()
        self.action_scale = action_scale
        self.lin_vel_scale = lin_vel_scale
        self.ang_vel_scale = ang_vel_scale
        self.last_action = np.zeros_like(self.default_angles, dtype=np.float32)
        self.ctrl_dt = ctrl_dt
        self._phase = np.array([0.0, np.pi])
        self._gait_freq = 1.5
        self._policy_session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
        self._input_name = self._policy_session.get_inputs()[0].name
        self._output_name = self._policy_session.get_outputs()[0].name

    def scale_command(self, raw_command):
        scales = np.array([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale])
        return raw_command * scales

    def get_observation(self, data, command):
        lin_vel = self._robot.local_linear_vel(data)
        gyro = self._robot.gyro(data)
        gravity = self._robot.gravity(data)
        dof_pos = self._robot.dof_pos(data)
        dof_vel = self._robot.dof_vel(data)
        phase = np.concatenate([np.cos(self._phase), np.sin(self._phase)])
        obs = np.hstack([lin_vel, gyro, gravity, command, dof_pos - self.default_angles, dof_vel, self.last_action, phase])
        return obs.astype(np.float32)

    def compute_action(self, observation):
        outputs = self._policy_session.run([self._output_name], {self._input_name: observation.reshape(1, -1)})
        return outputs[0][0]

    def step(self, data, command):
        scaled_command = self.scale_command(command)
        obs = self.get_observation(data, scaled_command)
        action = self.compute_action(obs)
        self.apply_action(data, action)
        phase_dt = 2 * np.pi * self._gait_freq * self.ctrl_dt
        self._phase = np.fmod(self._phase + phase_dt + np.pi, 2 * np.pi) - np.pi
        return self.is_fallen(data)

    def apply_action(self, data, action):
        ctrl = action * self.action_scale + self.default_angles
        self._robot.set_actuator_ctrls(data, ctrl)
        self.last_action = action.copy()

    def is_fallen(self, data):
        pose = self._robot.base_pose(data)
        rotation = Rotation.from_quat(pose[3:7])
        rotated_z_axis = rotation.apply(np.array([0.0, 0.0, 1.0]))
        return np.dot(rotated_z_axis, np.array([0.0, 0.0, 1.0])) < 0.3
