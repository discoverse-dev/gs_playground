[![English](https://img.shields.io/badge/lang-English-brightgreen)](README.md) [![中文](https://img.shields.io/badge/lang-%E4%B8%AD%E6%96%87-blue)](README_zh.md)
# GS Playground

GS Playground is a high-performance robotics simulation and learning platform focused on:

- **Locomotion (legged robots)**
- **Manipulation (robot arms)**

It supports an end-to-end workflow from **expert data generation**, **large-scale RL training**, to **Sim-to-Real** deployment.

---

## Quick Links

- [Installation](#installation)
- [Quick Start (5-minute sanity check)](#quick-start-5-minute-sanity-check)
- [Locomotion: Training & Playback](#locomotion-training--playback)
- [Manipulation: Table30 Data & Validation](#manipulation-table30-data--validation)
- [Benchmarks](#benchmarks)
- [FAQ](#faq)
- [Project Notes (Features / Architecture / Directory Structure)](#project-notes-features--architecture--directory-structure)

---

## Installation

Recommended: **Python 3.10+**. Dependencies are managed via **uv**.

### 1) Sync dependencies

For unstable networks, you may use a mirror:

```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --all-extras --reinstall-package motrixsim
```

### 2) Activate environment

```bash
source .venv/bin/activate
```

### 3) Install rsl_rl (RL core)

This project depends on a specific `rsl_rl` version:

```bash
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl
git checkout v1.0.2
uv pip install -e .
cd ..
```

---

## Quick Start (5-minute sanity check)

### A) Validate Manipulation env & action execution

```bash
python gs_playground/experimental/env/table30/test_action.py
```

(Optional) Validate Gym registry:

```bash
python gs_playground/experimental/env/table30/test_registry.py
```

### B) Minimal Locomotion training run

```bash
python gs_playground/src/locomotion/scripts/train.py
```

Logs are saved under `logs/`.

---

## Locomotion: Training & Playback

Built on `rsl_rl (PPO)`, supporting Unitree Go1 / Go2 / G1, etc.

### 1) Train

```bash
python gs_playground/src/locomotion/scripts/train.py
```

Configs and rewards can be adjusted under:

- `gs_playground/src/locomotion/legged_robots/`

### 2) Play

```bash
python gs_playground/src/locomotion/scripts/play.py \
  --resume_path "logs/your_experiment_name/model_latest.pt" \
  --num_envs 10
```

- `--resume_path`: checkpoint path (e.g., `model_latest.pt`)
- `--num_envs`: number of parallel envs for robustness checks

---

## Manipulation: Table30 Data & Validation

Table30 provides a desktop manipulation task suite (stacking, grasping, buttons, etc.), supporting Airbot / Franka / UR5e.

### 1) Expert data generation

```bash
python gs_playground/experimental/env/table30/02_stack_color_blocks_data.py
python gs_playground/experimental/env/table30/04_hang_toothbrush_cup_data_qpos.py
```

### 2) IK example (Mink)

```bash
python examples/ik/mink_franka.py
```

---

## Benchmarks

Notebooks are located at:

- `gs_playground/experimental/env/batch_render/`

Included:

- `motrix_vs_mujoco.ipynb`: MotrixSim (3DGS) vs MuJoCo renderer FPS
- `mjx_batch.ipynb`: MJX (JAX) large-scale parallel physics throughput

---

## FAQ

- **Install / download issues**: use `UV_INDEX_URL` mirror.
- **Gym registry errors**: run `test_registry.py` first.
- **MJX performance**: verify JAX/CUDA compatibility if using GPU.

---

## License

Specify your license here (MIT / Apache-2.0 / Proprietary).

---

# Project Notes (Features / Architecture / Directory Structure)

## Features

- **Double-backend design (Rendering / Physics decoupled)**
  - Rendering: `motrixsim` (high-quality 3D Gaussian Splatting rendering)
  - Physics: MuJoCo (high-fidelity) / MJX (JAX-accelerated parallel simulation)
- **End-to-end workflow**
  - Expert data generation (imitation learning / testing)
  - Large-scale RL training (MJX acceleration)
  - Sim-to-Real adaptation (robot arm hardware wrappers)
- **Task coverage**
  - Locomotion: Unitree Go1 / Go2 / G1
  - Manipulation: Airbot Play / Franka Panda / UR5e
  - Desktop task suite: Table30 (stacking, grasping, buttons, etc.)

---

## Architecture

GS Playground splits the system into two primary pipelines:

- **Rendering Backend**
  - `motrixsim`: high-quality visual rendering (3DGS)
- **Physics Backend**
  - `MuJoCo`: single-machine, high-fidelity physics
  - `MJX`: JAX-based parallel physics (for large-scale training)

This enables:

- higher throughput with MJX during training,
- higher visual quality with motrixsim during evaluation/visualization,
- backend switching within a consistent task/interface layer.

---

## Directory Structure

```text
gs_playground/
├── experimental/env/               # Experimental envs and test scripts
│   ├── batch_render/               # Rendering benchmarks (MuJoCo/MJX vs Motrix)
│   └── table30/                    # Table30 manipulation suite (data + tests)
│       ├── *_data.py               # Expert data generation scripts
│       ├── test_action.py          # Action execution sanity checks
│       └── test_registry.py        # Gym registry checks
├── models/                         # Assets (URDF/MJCF/3DGS)
│   ├── robots/
│   │   ├── manipulation/           # Arms (Airbot, Franka, UR5e)
│   │   └── locomotion/             # Legged robots (Unitree G1, Go1, Go2)
│   └── tasks/table30/              # Table30 assets (3DGS models: buttons, blocks, etc.)
└── src/                            # Core source
    ├── env/                        # Backend interface layer
    │   ├── motrix_env/             # Motrix backend (visual rendering)
    │   ├── mujoco_env/             # MuJoCo/MJX backend (physics + parallelism)
    │   └── __init__.py             # Gym env registry entry
    ├── locomotion/                 # Legged RL (based on rsl_rl)
    │   ├── legged_robots/          # Robot configs (reward, config)
    │   └── scripts/                # Train (train.py) and play (play.py)
    └── manipulation/               # Manipulation core logic
        ├── robots/                 # Robot control wrappers
        │   ├── airbot_play/        # Airbot hardware adapter
        │   ├── franka_.../         # Franka Panda hardware adapter
        │   └── universal_.../      # UR5e hardware adapter
        ├── tasks/                  # Task logic
        │   ├── table30/            # Table30 task rules (stacking, grasp success, etc.)
        │   └── task_env.py         # Task env base (Gym wrapper)
        └── base_robot.py           # Base robot controller interface
```