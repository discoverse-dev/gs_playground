
# draw_code

## 范围

当前推荐链路已经统一到：

- replay 配置：
  [replay_task_config.py](/home/xyys2003/ws/gsp/gs_playground/examples/draw/draw_code/replay_task_config.py)
- replay 抓帧：
  [replay.py](/home/xyys2003/ws/gsp/gs_playground/examples/draw/draw_code/replay.py)
- overlay 合成：
  [make_video_from_steps.py](/home/xyys2003/ws/gsp/gs_playground/examples/draw/draw_code/make_video_from_steps.py)

目前统一支持两个任务：

- `04`: `hang_toothbrush_cup`
- `13`: `arrange_flowers`

## 依赖

- Python `>=3.10`
- NVIDIA GPU
- CUDA 可用
- `ffmpeg` / `ffprobe`
- `gs_playground` 可导入

## 当前约定

- 数据采集保持低位场景。
- replay 使用高位显示逻辑。
- replay 默认背景使用：
  [background_085.ply](/home/xyys2003/ws/gsp/gs_playground/gs_playground/models/robots/manipulation/franka_emika_panda_robotiq/3dgs/background_085.ply)
- 高位 replay 运行时只走 `motrixsim.msd` 的内存场景覆盖，不再生成临时 XML。
- 运行时覆盖统一会处理：
  - 机器人 `link0 -> 0,0,0.85`
  - 桌子、相机、灯、任务场景 body/geom 的高位抬升
  - 从
    [test.xml](/home/xyys2003/ws/gsp/gs_playground/gs_playground/models/robots/manipulation/franka_emika_panda_robotiq/xmls/test.xml)
    注入 `pedestal/base`
  - 从 `test.xml` 注入四条桌腿
- `pedestal/base` 保持原高度，不额外抬高。

任务差异通过
[replay_task_config.py](/home/xyys2003/ws/gsp/gs_playground/examples/draw/draw_code/replay_task_config.py)
管理，不再分别维护两份 replay 主逻辑。

## 任务差异

### Task 04

- 数据采集脚本：
  [04_hang_toothbrush_cup_date.py](/home/xyys2003/ws/gsp/gs_playground/gs_playground/experimental/env/table30/04_hang_toothbrush_cup_date.py)
- 任务环境：
  [_04_hang_toothbrush_cup.py](/home/xyys2003/ws/gsp/gs_playground/gs_playground/src/manipulation/tasks/table30/_04_hang_toothbrush_cup.py)
- 采集保持低位。
- replay 使用高位显示。
- `toothbrush_cup` 是 replay 初始化物体；`rack` 跟随场景整体抬升，不单独恢复 pose。

### Task 13

- 数据采集脚本：
  [13_arrange_flowers_data_refactored.py](/home/xyys2003/ws/gsp/gs_playground/gs_playground/experimental/env/table30/13_arrange_flowers_data_refactored.py)
- 任务环境：
  [_13_arrange_flowers.py](/home/xyys2003/ws/gsp/gs_playground/gs_playground/src/manipulation/tasks/table30/_13_arrange_flowers.py)
- 采集保持低位。
- replay 使用
  [table30_13_arrange_flower.xml](/home/xyys2003/ws/gsp/gs_playground/gs_playground/models/robots/manipulation/franka_emika_panda_robotiq/xmls/table30_13_arrange_flower.xml)
  作为低位基底，再在运行时临时抬高。
- 不再依赖 `test.xml` 作为 13 replay 的主模型来源。

## 数据生成

调用入口就是各任务自己的采集脚本。最小命令分别如下。

### 04

```bash
python /home/xyys2003/ws/gsp/gs_playground/gs_playground/experimental/env/table30/04_hang_toothbrush_cup_date.py \
  --save_dir /home/xyys2003/ws/gsp/gs_playground/table30_04_fulltraj \
  --data_size 1 \
  --num_envs 1 \
  --max_ctrl_steps 1200
```

输出包含：

- `episode_00000.jsonl`
- `videos/*.mp4`
- `replay/ep_00000.npz`

常用可选参数：

- `--max_ctrl_steps`
- `--seed`
- `--no_video`

### 13

```bash
python /home/xyys2003/ws/gsp/gs_playground/gs_playground/experimental/env/table30/13_arrange_flowers_data_refactored.py \
  --save_dir /home/xyys2003/ws/gsp/gs_playground/table30_13_fulltraj \
  --data_size 1 \
  --num_envs 1
```

输出包含：

- `episode_00000.jsonl`
- `videos/*.mp4`
- `replay/ep_00000.npz`

常用可选参数：

- `--max_ctrl_steps`
- `--seed`
- `--no_video`

## Unified Replay

统一脚本：

- [replay.py](/home/xyys2003/ws/gsp/gs_playground/examples/draw/draw_code/replay.py)

### 如何调用

单条 episode 回放：

```bash
python /home/xyys2003/ws/gsp/gs_playground/examples/draw/draw_code/replay.py \
  --task <04|13> \
  --replay_npz /abs/path/to/ep_xxxxx.npz \
  --batch_size 1 \
  --num_steps 200 \
  --auto_start
```

批量回放一个目录里的多个 episode：

```bash
python /home/xyys2003/ws/gsp/gs_playground/examples/draw/draw_code/replay.py \
  --task <04|13> \
  --replay_dir /abs/path/to/replay_dir \
  --batch_size 3 \
  --num_steps 200 \
  --auto_start
```

说明：

- `--replay_npz` 用于单个 episode，脚本会复制到整个 batch。
- `--replay_dir` 用于一个目录内的多个 `.npz`。
- `--batch_size` 直接指定 batch 大小。
- `--replay_dir` 不传 `--batch_size` 时，默认使用目录里全部 `.npz`。
- 布局固定为 grid；`--grid_cols` 可选，不传时默认取 `ceil(sqrt(batch_size))`。
- 不传 `--num_steps` 时，默认回放到当前 episode 实际动作长度。
- `--auto_start` 表示启动后自动开始，不用手动按 `v`。

关键参数：

- `--task 04|13`
- `--replay_npz` 或 `--replay_dir`
- `--batch_size`
- `--grid_cols`
- `--num_steps`
- `--auto_start`

### 04 示例

```bash
python /home/xyys2003/ws/gsp/gs_playground/examples/draw/draw_code/replay.py \
  --task 04 \
  --replay_npz /home/xyys2003/ws/gsp/gs_playground/table30_04_pipeline_recheck/replay/ep_00000.npz \
  --batch_size 1 \
  --num_steps 200 \
  --auto_start
```

### 13 示例

```bash
python /home/xyys2003/ws/gsp/gs_playground/examples/draw/draw_code/replay.py \
  --task 13 \
  --replay_npz /home/xyys2003/ws/gsp/gs_playground/examples/draw/draw_code/data/table30_arrange_flowers_refactored_continue/replay/ep_000003.npz \
  --batch_size 1 \
  --num_steps 200 \
  --auto_start
```

## 当前验证结果

代码层面已验证：

- `replay_task_config.py` 可编译
- `replay.py` 可编译
- `make_video_from_steps.py` 可编译

说明：

- 04 这次重跑时，视觉上对齐已确认正常。
- 13 之前的 unified smoke 和 200-step replay 已验证通过。

## 排查

### 1. 沙盒里跑 replay 直接失败

如果报：

- `assert torch.cuda.is_available()`

这不是脚本逻辑错误，而是 CUDA 在当前运行环境不可见。需要在可见 GPU 的环境里运行。

### 2. 04 看起来物体对了，但步数没到预期

先检查 replay pack 本身长度：

```bash
python - <<'PY'
import numpy as np
p = np.load('/path/to/replay/ep_00000.npz')
print(p['actions'].shape)
PY
```

如果 pack 长度本身足够，但 replay 提前停了，再检查是否有运行时交互逻辑、窗口关闭或 episode 提前结束。

### 3. 背景不对

当前统一 replay 默认应加载：

- [background_085.ply](/home/xyys2003/ws/gsp/gs_playground/gs_playground/models/robots/manipulation/franka_emika_panda_robotiq/3dgs/background_085.ply)

如果仍是低位背景，通常说明没有走统一脚本，或者被命令行参数手动覆盖了。
