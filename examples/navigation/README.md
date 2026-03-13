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

## 控制方式

- W/↑: 前进
- S/↓: 后退
- ←/→: 左右平移
- A/D: 旋转
- ESC: 退出

## 显示说明

- 左上角: 机器人头部相机视角（GS 渲染）
- 左下角: 系统相机视角（GS 渲染）
- 主窗口: MuJoCo 物理仿真视图
