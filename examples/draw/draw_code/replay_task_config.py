from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np

from gs_playground import ROOT_PATH
from gs_playground.src.manipulation.tasks.table30._04_hang_toothbrush_cup import (
    HangToothbrushCupEnv,
    HangToothbrushCupEnvCfg,
)
from gs_playground.src.manipulation.tasks.table30._13_arrange_flowers import (
    ArrangeFlowersEnv,
    ArrangeFlowersEnvCfg,
)


ASSETS_FRANKA_DIR = ROOT_PATH / "models" / "robots" / "manipulation" / "franka_emika_panda_robotiq"


@dataclass(frozen=True)
class TaskReplaySpec:
    task_id: str
    task_name: str
    env_cls: type
    env_cfg_cls: type
    model_file: str | None
    task_assets_dir: Path
    task_body_gaussians: Dict[str, str]
    default_background_ply: str
    default_replay_z_offset: float
    default_link0_pos: str | None
    default_base_pos: str | None
    scene_z_offset: float
    inject_pedestal_from_test: bool
    keep_base_body_unlifted: bool
    replay_body_names: Tuple[str, ...]


def load_pack_13(path: Path) -> Dict[str, np.ndarray]:
    pack = np.load(path.as_posix(), allow_pickle=True)
    return {
        "flower_pose_wxyz": pack["flower_pose_wxyz"].astype(np.float32).reshape(7),
        "vase_src_pose_wxyz": pack["vase_src_pose_wxyz"].astype(np.float32).reshape(7),
        "vase_dst_pose_wxyz": pack["vase_dst_pose_wxyz"].astype(np.float32).reshape(7),
        "actions": pack["actions"].astype(np.float32),
        "ep_idx": np.asarray(pack["ep_idx"]).astype(np.int32) if "ep_idx" in pack else np.int32(-1),
    }


def load_pack_04(path: Path) -> Dict[str, np.ndarray]:
    pack = np.load(path.as_posix(), allow_pickle=True)
    return {
        "cup_pose_xyzw": pack["cup_pose_xyzw"].astype(np.float32).reshape(7),
        "actions": pack["actions"].astype(np.float32),
        "ep_idx": np.asarray(pack["ep_idx"]).astype(np.int32) if "ep_idx" in pack else np.int32(-1),
    }


def apply_replay_init_13(env, env_id: int, pack: Dict[str, np.ndarray]) -> None:
    env.set_replay_init(
        int(env_id),
        flower_pose_wxyz=pack["flower_pose_wxyz"],
        vase_src_pose_wxyz=pack["vase_src_pose_wxyz"],
        vase_dst_pose_wxyz=pack["vase_dst_pose_wxyz"],
    )


def apply_replay_init_04(env, env_id: int, pack: Dict[str, np.ndarray]) -> None:
    env.set_replay_init(
        int(env_id),
        cup_pose_xyzw=pack["cup_pose_xyzw"],
    )


TASK_SPECS: Dict[str, TaskReplaySpec] = {
    "13": TaskReplaySpec(
        task_id="13",
        task_name="arrange_flowers",
        env_cls=ArrangeFlowersEnv,
        env_cfg_cls=ArrangeFlowersEnvCfg,
        model_file=str(
            (
                ROOT_PATH
                / "models"
                / "robots"
                / "manipulation"
                / "franka_emika_panda_robotiq"
                / "xmls"
                / "table30_13_arrange_flower.xml"
            ).as_posix()
        ),
        task_assets_dir=ROOT_PATH / "models" / "tasks" / "table30" / "_13_arrange_flowers",
        task_body_gaussians={
            "flower": str((ROOT_PATH / "models" / "tasks" / "table30" / "_13_arrange_flowers" / "3dgs" / "flower1.ply").as_posix()),
            "vase": str((ROOT_PATH / "models" / "tasks" / "table30" / "_13_arrange_flowers" / "3dgs" / "vase.ply").as_posix()),
            "vase2": str((ROOT_PATH / "models" / "tasks" / "table30" / "_13_arrange_flowers" / "3dgs" / "vase2.ply").as_posix()),
        },
        default_background_ply=str((ASSETS_FRANKA_DIR / "3dgs" / "background_085.ply").as_posix()),
        default_replay_z_offset=0.85,
        default_link0_pos="0,0,0.85",
        default_base_pos=None,
        scene_z_offset=0.85,
        inject_pedestal_from_test=True,
        keep_base_body_unlifted=True,
        replay_body_names=("flower", "vase", "vase2"),
    ),
    "04": TaskReplaySpec(
        task_id="04",
        task_name="hang_toothbrush_cup",
        env_cls=HangToothbrushCupEnv,
        env_cfg_cls=HangToothbrushCupEnvCfg,
        model_file=None,
        task_assets_dir=ROOT_PATH / "models" / "tasks" / "table30" / "_04_hang_toothbrush_cup",
        task_body_gaussians={
            "toothbrush_cup": str((ROOT_PATH / "models" / "tasks" / "table30" / "_04_hang_toothbrush_cup" / "3dgs" / "toothbrush_cup.ply").as_posix()),
            "rack": str((ROOT_PATH / "models" / "tasks" / "table30" / "_04_hang_toothbrush_cup" / "3dgs" / "rack.ply").as_posix()),
        },
        default_background_ply=str((ASSETS_FRANKA_DIR / "3dgs" / "background_085.ply").as_posix()),
        default_replay_z_offset=0.85,
        default_link0_pos="0,0,0.85",
        default_base_pos="0,0,0.85",
        scene_z_offset=0.85,
        inject_pedestal_from_test=True,
        keep_base_body_unlifted=True,
        replay_body_names=("toothbrush_cup",),
    ),
}


PACK_LOADERS: Dict[str, Callable[[Path], Dict[str, np.ndarray]]] = {
    "13": load_pack_13,
    "04": load_pack_04,
}


REPLAY_INIT_APPLIERS: Dict[str, Callable[[object, int, Dict[str, np.ndarray]], None]] = {
    "13": apply_replay_init_13,
    "04": apply_replay_init_04,
}


def get_task_spec(task_id: str) -> TaskReplaySpec:
    if task_id not in TASK_SPECS:
        raise KeyError(f"Unknown task_id={task_id!r}. Available: {sorted(TASK_SPECS)}")
    return TASK_SPECS[task_id]
