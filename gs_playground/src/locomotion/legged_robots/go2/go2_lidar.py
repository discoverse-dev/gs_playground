import os

import motrixsim as mtx
import numpy as np

from .go2_flat import Go2_train_env
from gs_playground.src.env.mjmx_bridge import MjMxBridge

def print_lidar_note():
    print("Visit https://github.com/TATP-233/MuJoCo-LiDAR for installation instructions.")
    print("\nInstallation steps:")
    print("  git clone https://github.com/TATP-233/MuJoCo-LiDAR.git")
    print("  cd MuJoCo-LiDAR")
    print("  pip install -e \".[jax]\"")
    print("\nVerify JAX installation:")
    print("  python -c \"import jax; print(jax.default_backend())\"")
    print("  (Should print 'gpu')")

    print("\nNote: ")
    print("(1) 这一版的 batch-mujoco-lidar 仅支持使用 JAX 后端")
    print("(2) Batch-JAX 后端目前只支持基础几何体，包括：")
    print("    - PLANE : 平面")
    print("    - HFIELD : 高场")
    print("    - SPHERE : 球体")
    print("    - CAPSULE : 胶囊体")
    print("    - ELLIPSOID : 椭球体")
    print("    - CYLINDER  : 圆柱体")
    print("    - BOX : 长方体")
    print("    不支持网格面片（MESH）")
    print("(3) motrixsim 目前没有batch_array的geom_pos和geom_mat接口，因此不支持动态修改场景，需要一开始通过 MjMxBridge.forward() 计算初始的geom_xpos和geom_xmat（很快会支持）")

    print("\nExample code:")
    print("使用 Go2 机器人进行 LiDAR 扫描的示例代码：")
    print("    https://github.com/TATP-233/motrixsim-docs/blob/dev/lidar/examples/go2_lidar.py")
    print("使用批量 LiDAR 进行扫描的示例代码：")
    print("    https://github.com/TATP-233/motrixsim-docs/blob/dev/lidar/examples/batch_lidar.py")

print_lidar_note()

try:

    backend = os.environ.get("LIDAR_BACKEND", "jax")
    import mujoco_lidar
    if backend == "jax":
        assert mujoco_lidar.__version__ >= "0.2.3", "Please upgrade mujoco-lidar to version 0.2.3 or higher."
        import jax
        import jax.numpy as jnp
        from mujoco_lidar import scan_gen
        from mujoco_lidar.core_jax import MjLidarJax
    elif backend == "taichi":
        assert mujoco_lidar.__version__ >= "0.2.5", "Please upgrade mujoco-lidar to version 0.2.3 or higher."
        import taichi as ti
        if not hasattr(ti, '_is_initialized') or not ti._is_initialized:
            ti.init(arch=ti.gpu)
        from mujoco_lidar.core_ti import MjLidarTi

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3" # 如果显存充足，可以调大一些
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

except ImportError:
    print("[ERROR] mujoco_lidar package not found. Please install mujoco-lidar[jax] to run this example.")
    print_lidar_note()
    exit(0)

class Go2_lidar_train_env(Go2_train_env):
    def __init__(self, Cfg, headless):
        super().__init__(Cfg, headless)
        self.bridge = MjMxBridge(Cfg.asset.file)
        self.bridge.forward()

        # 注意，这里的 geomgroup 设置为只检测group=2的几何体，在xml中设置地形环境的group=2，忽略机器人本体的碰撞几何体
        geomgroup = np.array([0, 0, 1, 0, 0, 0], dtype=np.ubyte)
        if backend == "jax":
            self.lidar_wrapper = MjLidarJax(
                self.bridge.mj_model,
                geomgroup=geomgroup,
                bodyexclude=self.bridge.mj_model.body("base").id
            )
        elif backend == "taichi":
            self.lidar_wrapper = MjLidarTi(
                self.bridge.mj_model,
                geomgroup=geomgroup,
                bodyexclude=self.bridge.mj_model.body("base").id,
                max_candidates=64
            )
            # Important: must call update to sync the mj_data before tracing rays at first time
            self.lidar_wrapper.update(self.bridge.mj_data)

        self.lidar_site = self.model.get_site("lidar")
        self.dynamic_lidar = Cfg.lidarcfg.dynamic_lidar
        self.downsample = Cfg.lidarcfg.downsample
        self.geom_xpos_batch_jax = jnp.repeat(jnp.expand_dims(jnp.array(self.bridge.mj_data.geom_xpos), axis=0), self.num_envs, axis=0)
        self.geom_xmat_batch_jax = jnp.repeat(jnp.expand_dims(jnp.array(self.bridge.mj_data.geom_xmat), axis=0), self.num_envs, axis=0)

        lidartype = Cfg.lidarcfg.lidartype
        if lidartype in {"avia", "mid40", "mid70", "mid360", "tele"}:
            self.livox_generator = scan_gen.LivoxGenerator(lidartype)
            self.rays_theta, self.rays_phi = self.livox_generator.sample_ray_angles()
            self.use_livox_lidar = True
        elif lidartype == "airy":
            self.rays_theta, self.rays_phi = scan_gen.generate_airy96()
        elif lidartype == "HDL64":
            self.rays_theta, self.rays_phi = scan_gen.generate_HDL64()
        elif lidartype == "vlp32":
            self.rays_theta, self.rays_phi = scan_gen.generate_vlp32()
        elif lidartype == "os128":
            self.rays_theta, self.rays_phi = scan_gen.generate_os128()
        elif lidartype == "custom":
            self.rays_theta, self.rays_phi = scan_gen.generate_grid_scan_pattern(360, 64, phi_range=(0., np.pi/2.))
        else:
            raise ValueError(f"不支持的LiDAR型号: {lidartype}")
        if self.downsample > 1:
            self.rays_theta = self.rays_theta[::self.downsample]
            self.rays_phi = self.rays_phi[::self.downsample]
        self.rays_theta = jnp.array(np.ascontiguousarray(self.rays_theta).astype(np.float32))
        self.rays_phi = jnp.array(np.ascontiguousarray(self.rays_phi).astype(np.float32))

    def get_lidar_scan(self, data: mtx.SceneData) -> jnp.ndarray:
        """
        获取当前环境中所有机器人的LiDAR扫描数据
        Args:
            data (mtx.SceneData): 当前场景数据
        Returns:
            distances (jnp.ndarray): 形状为 (num_envs, num_rays) 的LiDAR距离数据
            local_rays_batch (jnp.ndarray): 形状为 (num_envs, num_rays, 3) 的局部射线方向数据
        """
        if self.dynamic_lidar:
            self.rays_theta, self.rays_phi = self.livox_generator.sample_ray_angles(downsample=self.downsample)

        if backend == "jax":
            distances, local_rays_batch = self.lidar_wrapper.trace_rays_batch(
                self.geom_xpos_batch_jax,
                self.geom_xmat_batch_jax,
                self.lidar_site.get_position(data),
                self.lidar_site.get_rotation_mat(data),
                self.rays_theta,
                self.rays_phi
            )
            local_points = local_rays_batch * distances[..., jnp.newaxis]
        elif backend == "taichi":
            distances_ti, local_points_ti = self.lidar_wrapper.trace_rays_batch(
                self.lidar_site.get_position(data),
                self.lidar_site.get_rotation_mat(data),
                self.rays_theta,
                self.rays_phi
            )
            distances = distances_ti.to_numpy()
            local_points = local_points_ti.to_numpy()

        # batch_size = self.num_envs
        # num_rays = self.rays_theta.shape[0]
        # world_points = jnp.einsum('bij,bkj->bki', self.lidar_site.get_rotation_mat(data), local_points) + self.lidar_site.get_position(data)[:, jnp.newaxis, :]

        # assert distances.shape == (batch_size, num_rays), f"Expected distances shape {(batch_size, num_rays)}, but got {distances.shape}"
        # assert local_rays_batch.shape == (batch_size, num_rays, 3), f"Expected local_rays_batch shape {(batch_size, num_rays, 3)}, but got {local_rays_batch.shape}"
        # assert local_points.shape == (batch_size, num_rays, 3), f"Expected local_points shape {(batch_size, num_rays, 3)}, but got {local_points.shape}"
        # assert world_points.shape == (batch_size, num_rays, 3), f"Expected world_points shape {(batch_size, num_rays, 3)}, but got {world_points.shape}"

        return distances, local_points

    def _get_obs(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        raw_obs = super()._get_obs(data, info)
        lidar_points = self.get_lidar_scan(data)
        obs = {
            "lidar_points": np.array(lidar_points),
            "state_obs" : raw_obs,
        }
        return obs
    

