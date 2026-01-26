[![中文](https://img.shields.io/badge/lang-%E4%B8%AD%E6%96%87-brightgreen)](README_zh.md) [![English](https://img.shields.io/badge/lang-English-blue)](README.md)

# GS Playground

GS Playground 是一个高性能机器人仿真与学习平台，专注于 **Locomotion（足式）** 与 **Manipulation（机械臂）** 任务。  
它支持从 **专家数据生成**、**大规模强化学习训练** 到 **Sim-to-Real** 的完整工作流。

---

## 快速链接

- [环境安装](#环境安装)
- [快速开始（5 分钟跑通）](#快速开始5-分钟跑通)
- [Locomotion：足式训练与回放](#locomotion足式训练与回放)
- [Manipulation：Table30 数据生成与验证](#manipulationtable30-数据生成与验证)
- [性能基准（Benchmarks）](#性能基准benchmarks)
- [常见问题（FAQ）](#常见问题faq)
- [项目说明（Features / Architecture / Directory Structure）](#项目说明features--architecture--directory-structure)

---

## 环境安装

推荐使用 **Python 3.10+**。本项目使用 **uv** 进行极速依赖管理。

### 1) 同步依赖（Sync）

为避免国内网络环境下载超时，建议使用镜像源：

```bash
# 设置清华源并同步依赖（自动创建 .venv）
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --all-extras --reinstall-package motrixsim
```

### 2) 激活环境（Activate）

```bash
# Linux / macOS
source .venv/bin/activate
```

### 3) 安装 rsl_rl（RL Core）

本项目依赖特定版本的 `rsl_rl`：

```bash
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl
git checkout v1.0.2
uv pip install -e .
cd ..
```

---

## 快速开始（5 分钟跑通）

按以下顺序执行，快速验证安装与基础功能是否正常。

### A) 验证 Manipulation 环境与 Action

```bash
python gs_playground/experimental/env/table30/test_action.py
```

（可选）验证 Gym 环境注册：

```bash
python gs_playground/experimental/env/table30/test_registry.py
```

### B) 启动 Locomotion 训练（最小路径）

```bash
python gs_playground/src/locomotion/scripts/train.py
```

训练日志默认输出到 `logs/`。

---

## Locomotion：足式训练与回放

基于 `rsl_rl (PPO)` 的训练流程，支持 Unitree Go1 / Go2 / G1 等模型。

### 1) 训练（Train）

```bash
python gs_playground/src/locomotion/scripts/train.py
```

你可以通过修改以下目录的配置来调整奖励函数与超参：

- `gs_playground/src/locomotion/legged_robots/`

### 2) 回放（Play）

```bash
python gs_playground/src/locomotion/scripts/play.py \
  --resume_path "logs/your_experiment_name/model_latest.pt" \
  --num_envs 10
```

参数说明：
- `--resume_path`：训练产物路径（示例：`model_latest.pt`）
- `--num_envs`：并行环境数量，用于观察策略鲁棒性

---

## Manipulation：Table30 数据生成与验证

Table30 提供桌面操作任务集（堆叠、抓取、按钮等），支持 Airbot / Franka / UR5e 等机械臂。

### 1) 生成专家数据（Data Generation）

```bash
# 生成“堆叠颜色积木”演示数据
python gs_playground/experimental/env/table30/02_stack_color_blocks_data.py

# 生成“悬挂牙刷杯”的关节空间数据
python gs_playground/experimental/env/table30/04_hang_toothbrush_cup_data_qpos.py
```

### 2) IK 示例（Mink）

```bash
python examples/ik/mink_franka.py
```

---

## 性能基准（Benchmarks）

性能测试 Notebook 位于：

- `gs_playground/experimental/env/batch_render/`

包含：
- `motrix_vs_mujoco.ipynb`：对比 MotrixSim（3DGS）与 MuJoCo 原生渲染帧率
- `mjx_batch.ipynb`：测试 MJX（JAX）的大规模并行物理仿真吞吐量

---

## 常见问题（FAQ）

### 1) 依赖安装失败 / 下载超时？
建议使用 `UV_INDEX_URL` 镜像源（见 [环境安装](#环境安装)）。

### 2) Gym 环境注册失败？
优先运行：

```bash
python gs_playground/experimental/env/table30/test_registry.py
```

### 3) MJX/JAX 性能不符合预期？
请确认 JAX 与本机 CUDA/驱动环境匹配（如果使用 GPU），并检查是否正确启用了加速后端。

---

## License

请在此处填写项目 License（例如 MIT / Apache-2.0 / Proprietary）。

---

# 项目说明（Features / Architecture / Directory Structure）

## Features

- **双后端架构（Rendering / Physics 解耦）**
  - Rendering：`motrixsim`（基于 3D Gaussian Splatting 的高质量渲染）
  - Physics：MuJoCo（高保真物理）/ MJX（JAX 加速并行仿真）
- **端到端训练与数据工作流**
  - 专家数据生成（用于模仿学习 / 测试）
  - 大规模并行 RL 训练（MJX 加速）
  - Sim-to-Real 适配（机械臂硬件封装）
- **任务覆盖**
  - 足式：Unitree Go1 / Go2 / G1
  - 机械臂：Airbot Play / Franka Panda / UR5e
  - 桌面操作任务集：Table30（堆叠、抓取、按钮等）

---

## Architecture

GS Playground 将仿真系统拆为两条主链路：

- **Rendering Backend**
  - `motrixsim`：用于高质量视觉渲染（3DGS）
- **Physics Backend**
  - `MuJoCo`：单机高保真物理仿真
  - `MJX`：基于 JAX 的并行物理仿真（适合大规模训练）

这意味着你可以：
- 用 MJX 在训练阶段获得更高吞吐量；
- 用 motrixsim 在验证/可视化阶段获得更高视觉质量；
- 在同一套任务与接口层中切换后端。

---

## Directory Structure

```text
gs_playground/
├── experimental/env/               # 实验性环境与测试脚本
│   ├── batch_render/               # 渲染性能基准 (MuJoCo/MJX vs Motrix)
│   └── table30/                    # Table30 桌面操作任务集 (数据生成与测试)
│       ├── *_data.py               # 专家数据生成脚本 (用于模仿学习/测试)
│       ├── test_action.py          # 基础 Action 验证脚本
│       └── test_registry.py        # 环境注册测试脚本
├── models/                         # 机器人与场景资产 (URDF/MJCF/3DGS)
│   ├── robots/
│   │   ├── manipulation/           # 机械臂定义 (Airbot Play, Franka Panda, UR5e)
│   │   └── locomotion/             # 足式机器人定义 (Unitree G1, Go1, Go2)
│   └── tasks/table30/              # 任务场景资产 (3DGS 模型文件，如按钮、积木等)
└── src/                            # 核心源码
    ├── env/                        # 仿真后端接口层
    │   ├── motrix_env/             # Motrix 后端实现 (高质量视觉渲染)
    │   ├── mujoco_env/             # MuJoCo/MJX 后端实现 (并行物理仿真)
    │   └── __init__.py             # Gym 环境注册入口 (Registry)
    ├── locomotion/                 # 足式机器人 RL (基于 rsl_rl)
    │   ├── legged_robots/          # 机器人配置 (Reward, Config)
    │   └── scripts/                # 训练 (train.py) 与推理 (play.py)
    └── manipulation/               # 机械臂操作核心逻辑
        ├── robots/                 # 机器人控制封装
        │   ├── airbot_play/        # Airbot 硬件适配
        │   ├── franka_.../         # Franka Panda 硬件适配
        │   └── universal_.../      # UR5e 硬件适配
        ├── tasks/                  # 任务逻辑实现
        │   ├── table30/            # Table30 具体任务规则 (如堆叠、抓取判定)
        │   └── task_env.py         # 任务环境基类 (Gym Wrapper)
        └── base_robot.py           # 机器人控制基类 (定义通用接口)
```
