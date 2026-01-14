import numpy as np
from motrixsim.render import RenderApp, RenderSettings

from .mtx_env import NpEnv

class NpRenderer:
    """
    The renderer for Np sim environments.
    """

    _env: NpEnv

    def __init__(self, env: NpEnv):
        num_envs = env.num_envs
        num_envs = 1 if num_envs is None else num_envs
        spacing = env.render_spacing
        cols = int(np.ceil(np.sqrt(num_envs)))
        offsets = []
        for i in range(num_envs):
            row = i // cols
            col = i % cols
            x = col * spacing
            y = row * spacing
            z = 0.0
            offsets.append([x, y, z])

        self._env = env
        self._render = RenderApp()
        settings = RenderSettings.performance()
        settings.enable_shadow = True  # disable shadow for better performance
        self._render.launch(
            env.model,
            batch=num_envs,
            render_offset=offsets,
            render_settings=settings,
        )
        self._sync_render_data = True
        self._render.system_camera.active = self._sync_render_data

    def render(self) -> None:
        """
        render the env
        """

        self._render.sync(data=self._env.state.data if self._sync_render_data else None)
        if self._render.input.is_key_just_pressed("space"):
            self._sync_render_data = not self._sync_render_data
            self._render.system_camera.active = self._sync_render_data
