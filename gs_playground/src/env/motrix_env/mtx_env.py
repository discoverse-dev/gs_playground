import abc
import dataclasses
from dataclasses import dataclass

import motrixsim as mtx
import numpy as np

from .base import ABEnv, EnvCfg


@dataclass
class NpEnvState:
    data: mtx.SceneData
    obs: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict

    @property
    def done(self) -> np.ndarray:
        """
        Check if the environment is done.
        """
        return np.logical_or(self.terminated, self.truncated)

    def replace(self, **updates) -> "NpEnvState":
        return dataclasses.replace(self, **updates)

    def validate(self):
        num_envs = self.data.shape[0]
        assert self.reward.shape == (num_envs,), self.reward.shape
        assert self.terminated.shape == (num_envs,), self.terminated.shape
        assert self.truncated.shape == (num_envs,), self.truncated.shape


class NpEnv(ABEnv):
    _model: mtx.SceneModel
    _cfg: EnvCfg
    _state: NpEnvState = None
    _render_spacing: float

    def __init__(self, cfg: EnvCfg, num_envs: int = 1):
        self._cfg = cfg
        self._num_envs = num_envs
        self._model = mtx.load_model(cfg.model_file)
        self._model.options.timestep = cfg.sim_dt
        self._render_spacing = cfg.render_spacing

    @property
    def model(self) -> mtx.SceneModel:
        """
        Get the scene model
        """
        return self._model

    @property
    def state(self) -> NpEnvState:
        """
        Get the current environment state
        """
        return self._state

    @property
    def cfg(self) -> EnvCfg:
        """
        Get the environment configuration
        """
        return self._cfg

    @property
    def render_spacing(self) -> bool:
        """
        Get the render spacing, with which the multi-envs will be rendered seperately in grid
        """
        return self._render_spacing

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def init_state(self) -> NpEnvState:
        """
        Create a new environment state
        """
        obs = np.zeros((self._num_envs, self.observation_space.shape[0]), dtype=np.float32)
        reward = np.zeros((self._num_envs,), dtype=np.float32)
        terminated = np.ones((self._num_envs,), dtype=bool)
        truncated = np.zeros((self._num_envs,), dtype=bool)
        info = {"steps": np.zeros((self._num_envs,), dtype=np.uint64)}
        data = mtx.SceneData(self._model, batch=[self._num_envs])
        self._state = NpEnvState(data, obs, reward, terminated, truncated, info)
        self._reset_done_envs()
        self._state.validate()
        return self._state

    def _reset_done_envs(self):
        """
        Reset the environments that are done
        """
        state = self._state
        done = state.done
        assert done.shape == (self._num_envs,)
        if not np.any(done):
            return

        np.putmask(state.info["steps"], done, 0)
        data = state.data[done]
        obs, info1 = self.reset(data)
        state.obs[done] = obs
        if info1:

            def replace_dict_values(dst, new_values, mask):
                for key, value in new_values.items():
                    if key not in dst:
                        dst[key] = value
                    else:
                        if isinstance(value, np.ndarray):
                            dst[key][mask] = value
                        elif isinstance(value, dict):
                            assert isinstance(dst[key], dict)
                            replace_dict_values(dst[key], value, mask)

            replace_dict_values(state.info, info1, done)

    def _update_truncate(self):
        """
        Truncate the environments that have reached max episode length
        """
        if not self._cfg.max_episode_steps:
            return
        self._state.truncated = self._state.info["steps"] >= self._cfg.max_episode_steps

    @abc.abstractmethod
    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        """
        Apply the action to the environment

        Args:
            actions (np.ndarray): The actions to apply
            state (NpEnvState): The environment state to apply the actions.
        """

    @abc.abstractmethod
    def update_state(self, state: NpEnvState, obs_required: bool = True) -> NpEnvState:
        """
        Update the environment state after physics step

        Args:
            state (NpEnvState): The environment state to update
            obs_required (bool): Whether to compute observations. Defaults to True.
        """

    @abc.abstractmethod
    def reset(
        self,
        data: mtx.SceneData,
        done: np.ndarray = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment for the done envs

        Args:
            data (mtx.SceneData): The scene data to reset
            done (Optional[np.ndarray]): A boolean array indicating which envs to reset. If None, reset all envs.

        Returns:
            tuple[np.ndarray, dict]: The initial observations and info after reset
        """
        pass

    def physics_step(self):
        for _ in range(self._cfg.sim_substeps):
            self._model.step(self._state.data)

    def _prev_physics_step(self):
        state = self._state
        state.reward.fill(0.0)
        state.terminated.fill(False)
        state.truncated.fill(False)

    def _before_chunk_step(self, data: mtx.SceneData):
        """
        Hook called before executing a chunk of actions.
        Override this to update robot reference states for relative control.
        
        Args:
            data (mtx.SceneData): Current state data (usually matches the observation).
        """
        pass

    def step(self, actions: np.ndarray) -> NpEnvState:
        if self._state is None:
            self.init_state()

        # Handle action dimensions
        # 1. auto crop if input action dim > action_space dim
        if actions.shape[-1] > self.action_space.shape[0]:
            actions = actions[..., :self.action_space.shape[0]]

        # 2. handle chunk action (B, T, D) vs single action (B, D)
        if actions.ndim == 2:
            # (B, D) -> (B, 1, D)
            actions = actions[:, None, :]
        
        # Now actions is (B, T, D)
        num_steps = actions.shape[1]
        
        # Hook for chunk start (e.g. update relative control reference)
        self._before_chunk_step(self._state.data)

        cumulative_reward = np.zeros(self._num_envs, dtype=np.float32)
        chunk_terminated = np.zeros(self._num_envs, dtype=bool)
        chunk_truncated = np.zeros(self._num_envs, dtype=bool)

        for t in range(num_steps):
            self._prev_physics_step()
            self._state = self.apply_action(actions[:, t], self._state)
            assert self._state is not None, "apply_action must return a valid NpEnvState"
            self.physics_step()
            
            # Optimization: only compute obs on last step
            is_last_step = (t == num_steps - 1)
            self._state = self.update_state(self._state, obs_required=is_last_step)
                
            self._state.info["steps"] += 1
            
            # Accumulate reward before reset might clear it
            cumulative_reward += self._state.reward

            self._update_truncate()
            
            # Accumulate done flags (since we defer reset)
            chunk_terminated |= self._state.terminated
            chunk_truncated |= self._state.truncated
        
        # Apply accumulated flags to state
        self._state.terminated = chunk_terminated
        self._state.truncated = chunk_truncated
        self._state.reward = cumulative_reward
        
        # Reset done envs at the very end of the chunk
        self._reset_done_envs()
        
        return self._state

    # -------------------- helpers --------------------
    def _apply_keyframe(self, data: mtx.SceneData):
        """
        Apply keyframe if available; supports int index or string name.
        Uses cfg.reset_enabled / cfg.reset_keyframe if present, else defaults to on/0.
        """
        enabled = getattr(self._cfg, "reset_enabled", True)
        keyframe = getattr(self._cfg, "reset_keyframe", 0)
        if not enabled:
            return
        kfs = getattr(self._model, "keyframes", None)
        if kfs is None:
            return
        try:
            if isinstance(keyframe, (int, np.integer)):
                idx = int(keyframe)
                if 0 <= idx < len(kfs):
                    kfs[idx].apply(data)
                    return
            elif isinstance(keyframe, str):
                for k in kfs:
                    if getattr(k, "name", None) == keyframe:
                        k.apply(data)
                        return
                kfs[0].apply(data)
            else:
                kfs[0].apply(data)
        except BaseException:
            pass

    def _get_actuator_index_or_raise(self, name: str) -> int:
        idx = self._model.get_actuator_index(name)
        if idx is None or idx < 0:
            raise ValueError(f"Actuator '{name}' not found in model")
        return int(idx)
