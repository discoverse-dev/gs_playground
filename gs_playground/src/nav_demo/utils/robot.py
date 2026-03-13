import numpy as np
import motrixsim as mx
from motrixsim import Body, SceneData


class RobotBase:
    def __init__(self, body: Body):
        self._body = body
        self._model = body.model
        self._base_link = body.base_link

    @property
    def body(self) -> Body:
        return self._body

    @property
    def model(self) -> mx.SceneModel:
        return self._model

    @property
    def base_link(self) -> mx.Link:
        return self._base_link

    @property
    def num_actuators(self) -> int:
        return self._body.num_actuators

    def dof_pos(self, data: SceneData) -> np.ndarray:
        return self._body.get_joint_dof_pos(data)

    def dof_vel(self, data: SceneData) -> np.ndarray:
        return self._body.get_joint_dof_vel(data)

    def base_pose(self, data: SceneData) -> np.ndarray:
        return self._base_link.get_pose(data)

    def set_actuator_ctrls(self, data: SceneData, ctrls: np.ndarray) -> None:
        self._body.set_actuator_ctrls(data, ctrls)

    def gravity(self, data: SceneData) -> np.ndarray:
        rot = self._body.get_rotation_mat(data)
        return rot.T @ np.array([0.0, 0.0, -1.0])


class Go2Robot(RobotBase):
    mjcf_path = "gs_playground/models/robots/navigation/go2/go2_mjx.xml"
    base_link_name = "base"

    def local_linear_vel(self, data: SceneData) -> np.ndarray:
        return self._model.get_sensor_value("local_linvel", data)

    def gyro(self, data: SceneData) -> np.ndarray:
        return self._model.get_sensor_value("gyro", data)


class Go1Robot(RobotBase):
    mjcf_path = "gs_playground/models/robots/navigation/go1/go1_mjx_fullcollisions.xml"
    base_link_name = "trunk"

    def local_linear_vel(self, data: SceneData) -> np.ndarray:
        return self._model.get_sensor_value("local_linvel", data)

    def gyro(self, data: SceneData) -> np.ndarray:
        return self._model.get_sensor_value("gyro", data)


class G1Robot(RobotBase):
    mjcf_path = "gs_playground/models/robots/navigation/g1/g1.xml"
    base_link_name = "pelvis"

    def local_linear_vel(self, data: SceneData) -> np.ndarray:
        return self._model.get_sensor_value("local_linvel_pelvis", data)

    def gyro(self, data: SceneData) -> np.ndarray:
        return self._model.get_sensor_value("gyro_pelvis", data)
