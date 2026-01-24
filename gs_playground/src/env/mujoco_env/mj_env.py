
import abc
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

import gymnasium as gym
import mujoco
from mujoco import rollout
import numpy as np

# Reuse EnvCfg and ABEnv from motrix_env to maintain consistency and interface
# Assuming they are generic enough
from gs_playground.src.env.motrix_env.base import ABEnv, EnvCfg


@dataclass
class MujocoEnvState:
    data: List[mujoco.MjData]
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

    def replace(self, **updates) -> "MujocoEnvState":
        return dataclasses.replace(self, **updates)

    def validate(self):
        num_envs = len(self.data)
        assert self.reward.shape == (num_envs,), self.reward.shape
        assert self.terminated.shape == (num_envs,), self.terminated.shape
        assert self.truncated.shape == (num_envs,), self.truncated.shape


class MujocoEnv(ABEnv):
    _model: mujoco.MjModel
    _cfg: EnvCfg
    _state: MujocoEnvState = None
    _num_envs: int

    def __init__(self, cfg: EnvCfg, num_envs: int = 1):
        self._cfg = cfg
        self._num_envs = num_envs
        self._model = mujoco.MjModel.from_xml_path(cfg.model_file)
        self._model.opt.timestep = cfg.sim_dt
        
        # MjData is not thread-safe for write access, so we need one per env for parallel stepping
        # But MjModel is thread-safe for read access.
        
        # Validate that model timestep matches config
        # (Usually we set it, but good to check if xml overrides)
        # self._model.opt.timestep = cfg.sim_dt # Already set

    @property
    def model(self) -> mujoco.MjModel:
        """
        Get the mujoco model
        """
        return self._model

    @property
    def state(self) -> MujocoEnvState:
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
    def num_envs(self) -> int:
        return self._num_envs

    def init_state(self) -> MujocoEnvState:
        """
        Create a new environment state
        """
        obs = np.zeros((self._num_envs, self.observation_space.shape[0]), dtype=np.float32)
        reward = np.zeros((self._num_envs,), dtype=np.float32)
        terminated = np.ones((self._num_envs,), dtype=bool)
        truncated = np.zeros((self._num_envs,), dtype=bool)
        info = {"steps": np.zeros((self._num_envs,), dtype=np.uint64)}
        
        # Create a list of MjData, one for each environment
        data = [mujoco.MjData(self._model) for _ in range(self._num_envs)]
        
        self._state = MujocoEnvState(data, obs, reward, terminated, truncated, info)
        self._reset_done_envs()
        self._state.validate()
        return self._state

    def _reset_done_envs(self):
        """
        Reset the environments that are done. 
        Note: logic copied/adapted from motrix_env.NpEnv
        """
        state = self._state
        done = state.done
        assert done.shape == (self._num_envs,)
        if not np.any(done):
            return

        np.putmask(state.info["steps"], done, 0)
        
        # We need to collect the data objects that need reset
        indices = np.where(done)[0]
        data_to_reset = [state.data[i] for i in indices]
        
        obs, info1 = self.reset(data_to_reset, indices)
        
        # Update observation for reset envs
        if obs is not None:
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
    def apply_action(self, actions: np.ndarray, state: MujocoEnvState) -> MujocoEnvState:
        """
        Apply the action to the environment. 
        The implementation should set properties in state.data (e.g. data.ctrl)
        """

    @abc.abstractmethod
    def update_state(self, state: MujocoEnvState, obs_required: bool = True) -> MujocoEnvState:
        """
        Update the environment state after physics step (e.g. compute obs, rewards)
        """

    @abc.abstractmethod
    def reset(
        self,
        data: List[mujoco.MjData],
        env_indices: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment for the done envs

        Args:
            data (List[mujoco.MjData]): The list of mjData to reset
            env_indices (np.ndarray): The indices of the envs being reset

        Returns:
            tuple[np.ndarray, dict]: The initial observations and info after reset
        """
        pass

    def physics_step(self):
        """
        Step the physics simulation for all environments in parallel using mujoco.rollout.
        """
        nsubsteps = self._cfg.sim_substeps
        nbatch = self._num_envs
        model = self._model
        
        # Prepare initial states for rollout from current data
        # mujoco.rollout requires initial state array
        state_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        
        # We need a contiguous array for rollout
        # TODO: Optimize by keeping a persistent state buffer if possible
        current_states = np.zeros((nbatch, state_size), dtype=np.float64)
        
        # Gather current state
        # When nbatch=1, data is a list of 1 MjData
        for i in range(nbatch):
            mujoco.mj_getState(model, self._state.data[i], current_states[i], mujoco.mjtState.mjSTATE_FULLPHYSICS)
            
        # Execute rollout
        # rollout takes list of MjData for multithreading
        # It returns (state_traj, sensor_traj)
        # state_traj shape: (nbatch, nstep, nstate)
        state_traj, _ = rollout.rollout(
            model, 
            self._state.data, 
            initial_state=current_states, 
            nstep=nsubsteps
        )
        
        # Apply the final state back to MjData to ensure consistency
        # rollout might not leave data in the final state, or we want to be explicit
        final_states = state_traj[:, -1, :]
        for i in range(nbatch):
            mujoco.mj_setState(model, self._state.data[i], final_states[i], mujoco.mjtState.mjSTATE_FULLPHYSICS)


    def _prev_physics_step(self):
        state = self._state
        state.reward.fill(0.0)
        state.terminated.fill(False)
        state.truncated.fill(False)

    def _before_chunk_step(self, data: List[mujoco.MjData]):
        """
        Hook called before executing a chunk of actions.
        """
        pass

    def step(self, actions: np.ndarray) -> MujocoEnvState:
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
        
        # Hook for chunk start
        self._before_chunk_step(self._state.data)

        cumulative_reward = np.zeros(self._num_envs, dtype=np.float32)
        chunk_terminated = np.zeros(self._num_envs, dtype=bool)
        chunk_truncated = np.zeros(self._num_envs, dtype=bool)

        for t in range(num_steps):
            self._prev_physics_step()
            self._state = self.apply_action(actions[:, t], self._state)
            assert self._state is not None, "apply_action must return a valid MujocoEnvState"
            
            self.physics_step()
            
            # Optimization: only compute obs on last step
            is_last_step = (t == num_steps - 1)
            self._state = self.update_state(self._state, obs_required=is_last_step)
                
            self._state.info["steps"] += 1
            
            # Accumulate reward before reset might clear it
            cumulative_reward += self._state.reward

            self._update_truncate()
            
            # Accumulate done flags
            chunk_terminated |= self._state.terminated
            chunk_truncated |= self._state.truncated
        
        # Apply accumulated flags to state
        self._state.terminated = chunk_terminated
        self._state.truncated = chunk_truncated
        self._state.reward = cumulative_reward
        
        # Reset done envs at the very end of the chunk
        self._reset_done_envs()
        
        return self._state
