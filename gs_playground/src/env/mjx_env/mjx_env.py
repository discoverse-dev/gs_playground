"""Core classes for GS Playground."""

import abc
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from etils import epath
from flax import struct
import jax
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

Observation = Union[jax.Array, Mapping[str, jax.Array]]
ObservationSize = Union[int, Mapping[str, Union[Tuple[int, ...], int]]]


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


def make_data(
    model: mujoco.MjModel,
    qpos: Optional[jax.Array] = None,
    qvel: Optional[jax.Array] = None,
    ctrl: Optional[jax.Array] = None,
    act: Optional[jax.Array] = None,
    mocap_pos: Optional[jax.Array] = None,
    mocap_quat: Optional[jax.Array] = None,
    impl: Optional[str] = None,
    nconmax: Optional[int] = None,
    njmax: Optional[int] = None,
    device: Optional[jax.Device] = None,
) -> mjx.Data:
    """Initialize MJX Data."""
    data = mjx.make_data(model, impl=impl, nconmax=nconmax, njmax=njmax, device=device)
    if qpos is not None:
        data = data.replace(qpos=qpos)
    if qvel is not None:
        data = data.replace(qvel=qvel)
    if ctrl is not None:
        data = data.replace(ctrl=ctrl)
    if act is not None:
        data = data.replace(act=act)
    if mocap_pos is not None:
        data = data.replace(mocap_pos=mocap_pos.reshape(model.nmocap, -1))
    if mocap_quat is not None:
        data = data.replace(mocap_quat=mocap_quat.reshape(model.nmocap, -1))
    return data


def step(
    model: mjx.Model,
    data: mjx.Data,
    action: jax.Array,
    n_substeps: int = 1,
) -> mjx.Data:
    def single_step(data, _):
        data = data.replace(ctrl=action)
        data = mjx.step(model, data)
        return data, None
    data = jax.lax.scan(single_step, data, (), n_substeps)[0]
    return mjx.forward(model, data)

@struct.dataclass
class State:
    """Environment state for training and inference."""

    data: mjx.Data
    obs: Observation
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array]
    info: Dict[str, Any]

    def tree_replace(
        self, params: Dict[str, Optional[jax.typing.ArrayLike]]
    ) -> "State":
        new = self
        for k, v in params.items():
            new = _tree_replace(new, k.split("."), v)
        return new


def _tree_replace(
    base: Any,
    attr: Sequence[str],
    val: Optional[jax.typing.ArrayLike],
) -> Any:
    """Sets attributes in a struct.dataclass with values."""
    if not attr:
        return base

    # special case for List attribute
    if len(attr) > 1 and isinstance(getattr(base, attr[0]), list):
        raise NotImplementedError("List attributes are not supported.")

    if len(attr) == 1:
        return base.replace(**{attr[0]: val})

    return base.replace(
        **{attr[0]: _tree_replace(getattr(base, attr[0]), attr[1:], val)}
    )


class MjxEnv(abc.ABC):
    """Base class for playground environments."""

    def __init__(
        self,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        self._config = config.lock()
        if config_overrides:
            self._config.update_from_flattened_dict(config_overrides)

        self._ctrl_dt = config.ctrl_dt
        self._sim_dt = config.sim_dt

    @abc.abstractmethod
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

    @abc.abstractmethod
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""

    @property
    @abc.abstractmethod
    def xml_path(self) -> str:
        """Path to the xml file for the environment."""

    @property
    @abc.abstractmethod
    def action_size(self) -> int:
        """Size of the action space."""

    @property
    @abc.abstractmethod
    def mj_model(self) -> mujoco.MjModel:
        """Mujoco model for the environment."""

    @property
    @abc.abstractmethod
    def mjx_model(self) -> mjx.Model:
        """Mjx model for the environment."""

    @property
    def dt(self) -> float:
        """Control timestep for the environment."""
        return self._ctrl_dt

    @property
    def sim_dt(self) -> float:
        """Simulation timestep for the environment."""
        return self._sim_dt

    @property
    def n_substeps(self) -> int:
        """Number of sim steps per control step."""
        return int(round(self.dt / self.sim_dt))

    @property
    def observation_size(self) -> ObservationSize:
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        obs = abstract_state.obs
        if isinstance(obs, Mapping):
            return jax.tree_util.tree_map(lambda x: x.shape, obs)
        return obs.shape[-1]

    @property
    def model_assets(self) -> Dict[str, Any]:
        """Dictionary of model assets to use with MjModel.from_xml_path."""
        if hasattr(self, "_model_assets"):
            return self._model_assets
        raise NotImplementedError(
            "_model_assets not defined for this environment"
            "see cartpole.py for an example."
        )

    @property
    def unwrapped(self) -> "MjxEnv":
        return self


def get_sensor_data(
    model: mujoco.MjModel, data: mjx.Data, sensor_name: str
) -> jax.Array:
    """Gets sensor data given sensor name."""
    sensor_id = model.sensor(sensor_name).id
    sensor_adr = model.sensor_adr[sensor_id]
    sensor_dim = model.sensor_dim[sensor_id]
    return data.sensordata[sensor_adr : sensor_adr + sensor_dim]


def dof_width(joint_type: Union[int, mujoco.mjtJoint]) -> int:
    """Get the dimensionality of the joint in qvel."""
    if isinstance(joint_type, mujoco.mjtJoint):
        joint_type = joint_type.value
    return {0: 6, 1: 3, 2: 1, 3: 1}[joint_type]


def qpos_width(joint_type: Union[int, mujoco.mjtJoint]) -> int:
    """Get the dimensionality of the joint in qpos."""
    if isinstance(joint_type, mujoco.mjtJoint):
        joint_type = joint_type.value
    return {0: 7, 1: 4, 2: 1, 3: 1}[joint_type]


def get_qpos_ids(model: mujoco.MjModel, joint_names: Sequence[str]) -> np.ndarray:
    index_list: list[int] = []
    for jnt_name in joint_names:
        jnt = model.joint(jnt_name).id
        jnt_type = model.jnt_type[jnt]
        qadr = model.jnt_qposadr[jnt]
        qdim = qpos_width(jnt_type)
        index_list.extend(range(qadr, qadr + qdim))
    return np.array(index_list)


def get_qvel_ids(model: mujoco.MjModel, joint_names: Sequence[str]) -> np.ndarray:
    index_list: list[int] = []
    for jnt_name in joint_names:
        jnt = model.joint(jnt_name).id
        jnt_type = model.jnt_type[jnt]
        vadr = model.jnt_dofadr[jnt]
        vdim = dof_width(jnt_type)
        index_list.extend(range(vadr, vadr + vdim))
    return np.array(index_list)
