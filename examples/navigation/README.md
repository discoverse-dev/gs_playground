# Navigation Demo

基于 3D Gaussian Splatting 的机器人导航演示。

## 快速开始

```bash
python examples/navigation/robot_locomotion.py \
  --robot go2 \
  --scene examples/navigation/nav_scene_1/mjcf/scene.xml \
  --gs_ply examples/navigation/nav_scene_1/3dgs/point_cloud.ply
```

## 参数说明

- `--robot`: 机器人类型 (go1/go2/g1)
- `--scene`: 场景 MJCF 文件路径
- `--gs_ply`: 3D Gaussian Splatting PLY 文件路径
- `--no-sync`: 禁用实时时钟同步（最快速度运行）
- `--save_data`: 启用数据收集
- `--save_dir`: 数据保存目录（默认: ./data/navigation）
- `--prompt`: 任务描述（默认: "Navigate in the scene"）

## 控制方式

- W/↑: 前进
- S/↓: 后退
- ←/→: 左右平移
- A/D: 旋转
- R: 保存当前 episode（仅在 --save_data 时可用）
- ESC: 退出

## 显示说明

- 左上角: 机器人头部相机视角（GS 渲染）
- 左下角: 系统相机视角（GS 渲染）
- 主窗口: MuJoCo 物理仿真视图

## 数据收集

启用数据收集后，会保存以下内容：

```bash
python examples/navigation/robot_locomotion.py \
  --robot go2 \
  --scene examples/navigation/nav_scene_1/mjcf/scene.xml \
  --gs_ply examples/navigation/nav_scene_1/3dgs/point_cloud.ply \
  --save_data \
  --save_dir ./data/nav_demo \
  --prompt "Navigate to the target location"
```

### 数据格式

- **JSONL 文件**: 每个 episode 一个 `.jsonl` 文件，每行包含：
  - `prompt`: 任务描述
  - `base_pose`: 机器人基座位姿 [x, y, z, roll, pitch, yaw]
  - `ctrl`: 控制指令 [vx, vy, omega]
  - `images_1`: 头部相机视频帧索引
  - `images_2`: 系统相机视频帧索引

- **视频文件**: MP4 格式，480x360 @ 30fps，存储在 `videos/` 子目录
