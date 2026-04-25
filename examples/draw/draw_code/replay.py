from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import motrixsim as mx
from motrixsim import forward_kinematic
from motrixsim.render import RenderApp

from gaussian_renderer import BatchSplatConfig, MtxBatchSplatRenderer
from replay_task_config import PACK_LOADERS, REPLAY_INIT_APPLIERS, get_task_spec

# NOTE: This script requires gs_playground HangToothbrushCupEnv for episode replay.
# For standalone testing without gs_playground, use capture_steps.py instead.
try:
    from gs_playground import ROOT_PATH
except ImportError:
    raise ImportError(
        "replay_batch_capture_gs_uv_key_v_run_04.py requires gs_playground for HangToothbrushCupEnv.\n"
        "For standalone usage, use capture_steps.py instead."
    )

# -----------------------------
# Assets
# -----------------------------
ASSETS_FRANKA_DIR = ROOT_PATH / "models" / "robots" / "manipulation" / "franka_emika_panda_robotiq"
TASK_SPEC = None


def find_root_body_by_name(world, name: str):
    for body in world.hierarchy.bodies:
        if getattr(body.link, "name", None) == name:
            return body
    return None


def add_z_to_xyz(xyz, z_offset: float) -> np.ndarray:
    out = np.asarray(xyz, dtype=np.float32).copy()
    out[2] += float(z_offset)
    return out


def ensure_pedestal_from_test_world(world, test_world) -> None:
    if find_root_body_by_name(world, "base") is not None:
        return

    world.attach(test_world, other_link_name="base")


def ensure_table_legs_from_test_world(world, test_world) -> None:
    if find_root_body_by_name(world, "table_legs") is not None:
        return

    task_table_geom = None
    for geom in world.hierarchy.geoms:
        if getattr(geom, "name", None) == "table":
            task_table_geom = geom
            break
    if task_table_geom is None:
        return

    world.attach(test_world, other_link_name="table")

    attached_table = find_root_body_by_name(world, "table")
    if attached_table is None:
        return

    attached_table.link.local_translation = np.asarray(
        task_table_geom.position, dtype=np.float32
    ).copy()
    attached_table.link.local_rotation = np.asarray(
        task_table_geom.orientation, dtype=np.float32
    ).copy()
    attached_table.link.name = "table_legs"
    attached_table.link.geoms = [
        geom
        for geom in attached_table.link.geoms
        if str(getattr(geom, "name", "")).startswith("table_leg_")
    ]


def apply_msd_replay_scene_overrides(
    world,
    *,
    link0_pos_csv: Optional[str],
    base_pos_csv: Optional[str],
    scene_z_offset: float,
    inject_pedestal_from_test: bool,
    keep_base_body_unlifted: bool,
    replay_body_names: Tuple[str, ...],
) -> None:
    if inject_pedestal_from_test:
        test_xml = (
            ROOT_PATH
            / "models"
            / "robots"
            / "manipulation"
            / "franka_emika_panda_robotiq"
            / "xmls"
            / "test.xml"
        )
        test_world = mx.msd.from_file(test_xml.as_posix())
        ensure_pedestal_from_test_world(world, test_world)
        ensure_table_legs_from_test_world(world, test_world)

    if base_pos_csv is not None:
        base_body = find_root_body_by_name(world, "base")
        if base_body is None:
            print(
                f"[info] no root body 'base' found in msd scene {getattr(world, 'source_path', None)}, skip base override"
            )
        else:
            base_body.link.local_translation = np.asarray(
                parse_xyz_csv(base_pos_csv, "--temp_base_pos"),
                dtype=np.float32,
            )
            print(f"[info] override base pos -> {base_pos_csv}")

    if link0_pos_csv is not None:
        link0_body = find_root_body_by_name(world, "link0")
        if link0_body is None:
            raise RuntimeError("Could not find root body 'link0' in msd scene")
        link0_body.link.local_translation = np.asarray(
            parse_xyz_csv(link0_pos_csv, "--temp_link0_pos"),
            dtype=np.float32,
        )
        print(f"[info] override link0 pos -> {link0_pos_csv}")

    if float(scene_z_offset) == 0.0:
        return

    for camera in world.hierarchy.cameras:
        camera.position = add_z_to_xyz(camera.position, scene_z_offset)

    for light in world.hierarchy.lights:
        light.position = add_z_to_xyz(light.position, scene_z_offset)

    for geom in world.hierarchy.geoms:
        if getattr(geom, "name", None) == "floor":
            continue
        geom.position = add_z_to_xyz(geom.position, scene_z_offset)

    keep_names = set(replay_body_names)
    if keep_base_body_unlifted:
        keep_names.add("base")
    if link0_pos_csv is not None:
        # `link0` is explicitly overridden above, so do not lift it again here.
        keep_names.add("link0")
    for body in world.hierarchy.bodies:
        if getattr(body.link, "name", None) in keep_names:
            continue
        body.link.local_translation = add_z_to_xyz(
            body.link.local_translation, scene_z_offset
        )

    print(f"[info] lifted top-level scene z by {scene_z_offset}")

# -----------------------------
# Frustum MJCF builder (dynamic texture screen)
# -----------------------------
def build_frustum_mjcf(
    cam_pos,
    cam_x,
    cam_y,
    cam_fwd,
    fovy_deg,
    dist,
    aspect,
    tex_w,
    tex_h,
    edge_radius=0.002,
):
    """Build MJCF XML for a camera frustum with a dynamic texture screen."""
    half_h = dist * np.tan(np.deg2rad(fovy_deg) * 0.5)
    half_w = half_h * aspect

    center = cam_pos + cam_fwd * dist
    c0 = center + (-cam_x * half_w) + (cam_y * half_h)
    c1 = center + (cam_x * half_w) + (cam_y * half_h)
    c2 = center + (cam_x * half_w) + (-cam_y * half_h)
    c3 = center + (-cam_x * half_w) + (-cam_y * half_h)

    # Screen orientation: flip so texture faces inward (toward apex)
    R = np.column_stack([-cam_x, cam_y, cam_fwd])
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    quat_wxyz = f"{q[3]:.8f} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f}"

    def v3(a):
        return f"{a[0]:.6f} {a[1]:.6f} {a[2]:.6f}"

    apex = v3(cam_pos)
    sc = v3(center)
    tl, tr, br, bl = v3(c0), v3(c1), v3(c2), v3(c3)
    R_s = edge_radius

    return f"""<mujoco>
  <asset>
    <texture name="gs_screen_tex" type="2d" builtin="dynamic"
             width="{tex_w}" height="{tex_h}" _perinstance="true"/>
    <material name="gs_screen_mat" texture="gs_screen_tex" castshadow="false"/>
    <material name="frustum_edge_mat" rgba="1 1 1 1" emission="0.6 0.6 0.6 1" castshadow="false"/>
  </asset>
  <worldbody>
    <geom name="gs_screen" type="box" size="{half_w:.6f} {half_h:.6f} 0.005"
          pos="{sc}" quat="{quat_wxyz}"
          material="gs_screen_mat" contype="0" conaffinity="0"/>
    <geom type="capsule" size="{R_s}" fromto="{apex} {tl}" material="frustum_edge_mat" contype="0" conaffinity="0"/>
    <geom type="capsule" size="{R_s}" fromto="{apex} {tr}" material="frustum_edge_mat" contype="0" conaffinity="0"/>
    <geom type="capsule" size="{R_s}" fromto="{apex} {br}" material="frustum_edge_mat" contype="0" conaffinity="0"/>
    <geom type="capsule" size="{R_s}" fromto="{apex} {bl}" material="frustum_edge_mat" contype="0" conaffinity="0"/>
    <geom type="capsule" size="{R_s}" fromto="{tl} {tr}" material="frustum_edge_mat" contype="0" conaffinity="0"/>
    <geom type="capsule" size="{R_s}" fromto="{tr} {br}" material="frustum_edge_mat" contype="0" conaffinity="0"/>
    <geom type="capsule" size="{R_s}" fromto="{br} {bl}" material="frustum_edge_mat" contype="0" conaffinity="0"/>
    <geom type="capsule" size="{R_s}" fromto="{bl} {tl}" material="frustum_edge_mat" contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>"""

# -----------------------------
# Model camera pose getter (batched)
# -----------------------------
def get_model_camera_pose_xyzw_fovy_batched(
    model, data, cam_id: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    cam = model.cameras[int(cam_id)]
    pose = np.asarray(cam.get_pose(data), dtype=np.float32)

    if pose.ndim == 1:
        pose = pose[None, :]
    if pose.ndim >= 3:
        pose = pose.reshape(pose.shape[0], -1, pose.shape[-1])[:, 0, :]

    pos = pose[:, :3].astype(np.float32)
    quat_xyzw = pose[:, 3:7].astype(np.float32)

    n = np.linalg.norm(quat_xyzw, axis=1, keepdims=True) + 1e-12
    quat_xyzw = quat_xyzw / n

    fovy = float(getattr(cam, "fovy", 45.0))
    return pos, quat_xyzw, fovy


# -----------------------------
def make_grid_offsets(batch_size: int, cols: int, spacing: float = 2.0) -> List[List[float]]:
    """
    Row-major grid centered around the origin.

    `spacing` is interpreted as y-spacing; x-spacing keeps the historical 1.25 ratio.
    """
    batch_size = int(batch_size)
    cols = int(cols)
    if batch_size <= 0 or cols <= 0:
        return []

    sy = float(spacing)
    sx = 1.25 * sy
    rows = int(np.ceil(batch_size / cols))
    x0 = 0.5 * (cols - 1) * sx
    y0 = 0.5 * (rows - 1) * sy

    offsets: List[List[float]] = []
    for idx in range(batch_size):
        row = idx // cols
        col = idx % cols
        offsets.append([col * sx - x0, y0 - row * sy, 0.0])
    return offsets


def infer_grid_cols(batch_size: int) -> int:
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return int(np.ceil(np.sqrt(batch_size)))


# -----------------------------
# GS renderer
# -----------------------------
def build_gs_renderer(model, batch_size: int) -> MtxBatchSplatRenderer:
    if TASK_SPEC is None:
        raise RuntimeError("TASK_SPEC is not initialized")
    GS_BODY_GAUSSIANS: Dict[str, str] = {
        # franka
        "link1": (ASSETS_FRANKA_DIR / "3dgs" / "franka" / "link1.ply").as_posix(),
        "link2": (ASSETS_FRANKA_DIR / "3dgs" / "franka" / "link2.ply").as_posix(),
        "link3": (ASSETS_FRANKA_DIR / "3dgs" / "franka" / "link3.ply").as_posix(),
        "link4": (ASSETS_FRANKA_DIR / "3dgs" / "franka" / "link4.ply").as_posix(),
        "link5": (ASSETS_FRANKA_DIR / "3dgs" / "franka" / "link5.ply").as_posix(),
        "link6": (ASSETS_FRANKA_DIR / "3dgs" / "franka" / "link6.ply").as_posix(),
        "link7": (ASSETS_FRANKA_DIR / "3dgs" / "franka" / "link7.ply").as_posix(),
        # robotiq
        "robotiq_base": (
            ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "robotiq_base.ply"
        ).as_posix(),
        "left_driver": (
            ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "left_driver.ply"
        ).as_posix(),
        "left_coupler": (
            ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "left_coupler.ply"
        ).as_posix(),
        "left_spring_link": (
            ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "left_spring_link.ply"
        ).as_posix(),
        "left_follower": (
            ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "left_follower.ply"
        ).as_posix(),
        "right_driver": (
            ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "right_driver.ply"
        ).as_posix(),
        "right_coupler": (
            ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "right_coupler.ply"
        ).as_posix(),
        "right_spring_link": (
            ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "right_spring_link.ply"
        ).as_posix(),
        "right_follower": (
            ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "right_follower.ply"
        ).as_posix(),
        # task
        **TASK_SPEC.task_body_gaussians,
    }

    cfg = BatchSplatConfig(
        body_gaussians=dict(GS_BODY_GAUSSIANS),
        background_ply=None,  # foreground renderer does NOT include background
        minibatch=int(batch_size),
    )
    return MtxBatchSplatRenderer(cfg, model)


def parse_xyz_csv(csv_text: str, arg_name: str) -> Tuple[float, float, float]:
    parts = [float(x.strip()) for x in str(csv_text).split(",")]
    if len(parts) != 3:
        raise ValueError(f"{arg_name} must be 'x,y,z'")
    return float(parts[0]), float(parts[1]), float(parts[2])


def resolve_background_ply(args) -> str:
    if TASK_SPEC is None:
        raise RuntimeError("TASK_SPEC is not initialized")
    if args.background_ply:
        return str(args.background_ply)
    return TASK_SPEC.default_background_ply


def build_bg_renderer(
    model, batch_size: int, background_ply: str
) -> Optional[MtxBatchSplatRenderer]:
    if not (isinstance(background_ply, str) and background_ply.strip()):
        return None
    bg_cfg = BatchSplatConfig(
        body_gaussians={},
        background_ply=background_ply,
        minibatch=int(batch_size),
    )
    return MtxBatchSplatRenderer(bg_cfg, model)


def render_gs_batch_rgb_u8(
    *,
    gs_renderer: MtxBatchSplatRenderer,
    bg_renderer: Optional[MtxBatchSplatRenderer],
    bg_imgs,
    model,
    data,
    batch_size: int,
    gs_cam_id: int,
    H: int,
    W: int,
) -> Tuple[np.ndarray, object]:
    forward_kinematic(model, data)

    link_poses = model.get_link_poses(data)
    body_pos = link_poses[..., :3]
    body_quat = link_poses[..., 3:7]

    cam_pos_b3, cam_quat_b4, fovy = get_model_camera_pose_xyzw_fovy_batched(
        model, data, cam_id=gs_cam_id
    )

    cam_pos_np = cam_pos_b3[:, None, :]  # (B,1,3)

    r = Rotation.from_quat(cam_quat_b4.reshape(-1, 4))
    cam_xmat = r.as_matrix().astype(np.float32).reshape(batch_size, 3, 3)
    cam_xmat_np = cam_xmat[:, None, :, :]  # (B,1,3,3)

    fovy_np = np.full((batch_size, 1), float(fovy), dtype=np.float32)

    device = gs_renderer.device
    cam_pos_t = torch.from_numpy(cam_pos_np).to(device=device, dtype=torch.float32)
    cam_xmat_t = torch.from_numpy(cam_xmat_np).to(device=device, dtype=torch.float32)

    # background (cache once)
    if bg_renderer is not None and bg_imgs is None:
        bg_gsb = bg_renderer.batch_update_gaussians(body_pos, body_quat)
        bg_imgs, _ = bg_renderer.batch_env_render(
            bg_gsb, cam_pos_t, cam_xmat_t, int(H), int(W), fovy_np
        )

    gsb = gs_renderer.batch_update_gaussians(body_pos, body_quat)
    rgb_t, _ = gs_renderer.batch_env_render(
        gsb, cam_pos_t, cam_xmat_t, int(H), int(W), fovy_np, bg_imgs=bg_imgs
    )

    rgb = (
        rgb_t.detach().cpu().numpy()
        if isinstance(rgb_t, torch.Tensor)
        else np.asarray(rgb_t)
    )

    # normalize to (B,H,W,3)
    if rgb.ndim == 5 and rgb.shape[2] == 3:
        rgb = np.transpose(rgb, (0, 1, 3, 4, 2))
    if rgb.ndim == 5 and rgb.shape[1] == 1:
        rgb = rgb.squeeze(1)
    if rgb.ndim == 4 and rgb.shape[1] == 3:
        rgb = np.transpose(rgb, (0, 2, 3, 1))
    if rgb.ndim == 5:
        rgb = rgb[:, 0, ...]

    rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    if rgb_u8.ndim != 4 or rgb_u8.shape[-1] != 3:
        raise RuntimeError(f"Unexpected GS rgb shape: {rgb_u8.shape}")
    return rgb_u8, bg_imgs


# -----------------------------
# Replay pack loading
# -----------------------------
def load_replay_npz(path: Path) -> Dict[str, np.ndarray]:
    if TASK_SPEC is None:
        raise RuntimeError("TASK_SPEC is not initialized")
    return PACK_LOADERS[TASK_SPEC.task_id](path)


def list_npz_in_dir(d: Path) -> List[Path]:
    # common naming: ep_XXXXXX.npz; but we accept all *.npz
    files = sorted(d.glob("*.npz"))
    return files


def lightweight_replay_step(env, actions: np.ndarray) -> object:
    if env._state is None:
        env.init_state()
    actions = np.asarray(actions)
    if actions.shape[-1] > env.action_space.shape[0]:
        actions = actions[..., : env.action_space.shape[0]]
    if actions.ndim == 2:
        actions = actions[:, None, :]

    env._before_chunk_step(env._state.data)
    for t in range(actions.shape[1]):
        env._state = env.apply_action(actions[:, t], env._state)
        assert env._state is not None, "apply_action must return a valid NpEnvState"
        env.physics_step()
    return env._state


def main() -> None:
    global TASK_SPEC
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, required=True, choices=["04", "13"])

    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--replay_npz",
        type=str,
        default=None,
        help="Single episode (replicate across the batch)",
    )
    grp.add_argument(
        "--replay_dir",
        type=str,
        default=None,
        help="Directory containing a batch of *.npz episodes",
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Replay batch size. Required for --replay_npz; optional for --replay_dir.",
    )
    ap.add_argument(
        "--grid_cols",
        type=int,
        default=None,
        help="Number of columns in the grid layout. Default: ceil(sqrt(batch_size)).",
    )

    ap.add_argument("--num_steps", type=int, default=None)

    # GS render settings
    ap.add_argument("--gs_cam_id", type=int, default=0)
    ap.add_argument("--gs_w", type=int, default=320)
    ap.add_argument("--gs_h", type=int, default=240)

    ap.add_argument("--frustum_dist", type=float, default=0.5)
    ap.add_argument(
        "--spacing", type=float, default=2.0
    )  # recommend 2.0 to match your old (2.5,2.0)
    ap.add_argument(
        "--auto_start",
        action="store_true",
        help="Start replay immediately without waiting for key 'v'",
    )
    ap.add_argument(
        "--temp_base_pos",
        type=str,
        default=None,
        help="Optional override for robot base body pos. Format: x,y,z",
    )
    ap.add_argument(
        "--temp_link0_pos",
        type=str,
        default=None,
        help="Optional override for robot link0 body pos. Format: x,y,z",
    )
    ap.add_argument(
        "--background_ply",
        type=str,
        default=None,
        help="Optional override for background ply. Default: high replay background_085.ply.",
    )

    args = ap.parse_args()
    TASK_SPEC = get_task_spec(str(args.task))

    packs: List[Dict[str, np.ndarray]] = []
    batch_size: Optional[int] = int(args.batch_size) if args.batch_size is not None else None

    # -----------------------------
    # Resolve mode + packs + batch size
    # -----------------------------
    if args.replay_npz is not None:
        if batch_size is None:
            raise RuntimeError(
                "--batch_size is required when using --replay_npz."
            )
        B = int(batch_size)
        if B <= 0:
            raise RuntimeError("--batch_size must be > 0.")
        p = load_replay_npz(Path(args.replay_npz))
        packs = [p for _ in range(B)]
    else:
        d = Path(args.replay_dir)
        if (not d.exists()) or (not d.is_dir()):
            raise RuntimeError(f"--replay_dir is not a directory: {d}")

        paths = list_npz_in_dir(d)
        if not paths:
            raise RuntimeError(f"No .npz files found in: {d}")

        if batch_size is None:
            B = len(paths)
        else:
            B = int(batch_size)
            if B <= 0:
                raise RuntimeError("--batch_size must be > 0.")
        if len(paths) < B:
            raise RuntimeError(
                f"--batch_size={B}, but replay_dir only has {len(paths)} npz files."
            )
        packs = [load_replay_npz(p) for p in paths[:B]]

    # -----------------------------
    # Replay length
    lengths = [int(p["actions"].shape[0]) for p in packs]
    if args.num_steps is not None:
        T = int(args.num_steps)
        if any(L < T for L in lengths):
            raise RuntimeError(
                f"--num_steps={T} exceeds at least one episode length: {lengths}"
            )
    else:
        T = min(lengths)

    # -----------------------------
    # Env setup (batch) - ONLY init pose, no stepping
    # -----------------------------
    env_cfg = TASK_SPEC.env_cfg_cls()
    if getattr(TASK_SPEC, "model_file", None):
        env_cfg.model_file = str(TASK_SPEC.model_file)
    if hasattr(env_cfg, "replay_z_offset"):
        setattr(env_cfg, "replay_z_offset", TASK_SPEC.default_replay_z_offset)

    base_pos_csv = args.temp_base_pos if args.temp_base_pos is not None else TASK_SPEC.default_base_pos
    link0_pos_csv = args.temp_link0_pos if args.temp_link0_pos is not None else TASK_SPEC.default_link0_pos

    frustum_dist = float(args.frustum_dist)
    frustum_aspect = float(args.gs_w) / float(args.gs_h)

    # Monkey-patch load_model so replay keeps the remote frustum-screen feature.
    _orig_load_model = mx.load_model

    def _patched_load_model(path):
        msd_scene = mx.msd.from_file(path)
        apply_msd_replay_scene_overrides(
            msd_scene,
            link0_pos_csv=link0_pos_csv,
            base_pos_csv=base_pos_csv,
            scene_z_offset=float(TASK_SPEC.scene_z_offset),
            inject_pedestal_from_test=bool(TASK_SPEC.inject_pedestal_from_test),
            keep_base_body_unlifted=bool(TASK_SPEC.keep_base_body_unlifted),
            replay_body_names=tuple(TASK_SPEC.replay_body_names),
        )

        msd_cam = msd_scene.hierarchy.cameras[int(args.gs_cam_id)]
        cam_pos = np.asarray(msd_cam.position, dtype=np.float32)
        cam_fovy_deg = float(msd_cam.fovy)
        q_xyzw = np.asarray(msd_cam.orientation, dtype=np.float32)
        R_wc = Rotation.from_quat(q_xyzw).as_matrix().astype(np.float32)
        cam_x = R_wc[:, 0]
        cam_y = R_wc[:, 1]
        cam_fwd = -R_wc[:, 2]

        frustum_xml = build_frustum_mjcf(
            cam_pos=cam_pos,
            cam_x=cam_x,
            cam_y=cam_y,
            cam_fwd=cam_fwd,
            fovy_deg=cam_fovy_deg,
            dist=frustum_dist,
            aspect=frustum_aspect,
            tex_w=int(args.gs_w),
            tex_h=int(args.gs_h),
        )
        msd_frustum = mx.msd.from_str(frustum_xml)
        msd_scene.attach(msd_frustum)
        return msd_scene.build()

    mx.load_model = _patched_load_model
    try:
        env = TASK_SPEC.env_cls(env_cfg, num_envs=B)
    finally:
        mx.load_model = _orig_load_model

    env.model.options.timestep = env_cfg.sim_dt

    for i in range(B):
        REPLAY_INIT_APPLIERS[TASK_SPEC.task_id](env, int(i), packs[i])

    env.reset(done=np.ones((B,), dtype=bool))  # init pose applied here

    model = env.model
    data = env._state.data

    # -----------------------------
    # Render + GS renderer
    # -----------------------------
    grid_cols = int(args.grid_cols) if args.grid_cols is not None else infer_grid_cols(B)
    render_offset = make_grid_offsets(B, cols=grid_cols, spacing=float(args.spacing))
    if len(render_offset) != B:
        raise RuntimeError(
            f"internal: layout offsets length mismatch: len={len(render_offset)} B={B}"
        )

    gs_renderer = build_gs_renderer(model, batch_size=B)
    bg_ply = resolve_background_ply(args)
    print(f"[info] using background ply: {bg_ply}")
    bg_renderer = build_bg_renderer(model, batch_size=B, background_ply=bg_ply)
    bg_imgs = None

    # runtime states
    started = bool(args.auto_start)
    t_idx = 0

    print("========================================")
    print("Interactive Replay:")
    print(f"  layout=grid batch_size={B} grid_cols={grid_cols}")
    print("  - Before start: you can hand-tune system camera freely (NO stepping).")
    print("  - Press 'v' once: start running continuously, until end / --num_steps.")
    print("========================================")

    with RenderApp() as render:
        render.launch(model, batch=B, render_offset=render_offset)
        render.sync(data)

        # Dynamic texture handle for GS frustum screen
        gs_screen_img = render.get_texture_image("gs_screen_tex")
        while not render.is_closed and t_idx < T:
            inp = render.input

            if started:
                a_batch = np.zeros((B, 7), dtype=np.float32)
                for i in range(B):
                    a_batch[i] = packs[i]["actions"][t_idx]

                lightweight_replay_step(env, a_batch)

                # If your GS camera moves over time and bg appears wrong, uncomment next line:
                # bg_imgs = None

                # Render GS (with background) and update the frustum screen texture.
                gs_u8, bg_imgs = render_gs_batch_rgb_u8(
                    gs_renderer=gs_renderer,
                    bg_renderer=bg_renderer,
                    bg_imgs=bg_imgs,
                    model=model,
                    data=data,
                    batch_size=B,
                    gs_cam_id=int(args.gs_cam_id),
                    H=int(args.gs_h),
                    W=int(args.gs_w),
                )

                # The frustum screen texture is marked per-instance, so batch replay
                # must upload the full (B, H, W, 3) tensor here.
                gs_screen_img.pixels = gs_u8

                t_idx += 1
            elif inp.is_key_just_pressed("v"):
                started = True
                print(f"[start] running from step {t_idx} / {T}")

            render.sync(data)

    if hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()
