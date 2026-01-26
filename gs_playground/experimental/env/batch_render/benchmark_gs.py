import argparse
import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from pathlib import Path
import mediapy

# 引入原有依赖
from motrixsim import SceneData, load_model, step
from gaussian_renderer import GSRendererMotrixSim, BatchSplatConfig, MtxBatchSplatRenderer
from gs_playground import ROOT_PATH

def get_args():
    parser = argparse.ArgumentParser(description="Benchmark 3DGS Rendering in MotrixSim")
    
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--minibatch", type=int, default=8, help="Minibatch size for splatter")
    parser.add_argument("--height", type=int, default=480, help="Render height")
    parser.add_argument("--width", type=int, default=480, help="Render width")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("--save_img", action="store_true", default=True, help="Save the output image for verification")
    
    return parser.parse_args()

def smart_tile(images, max_display=16):
    """
    动态拼图函数：不强制要求完全平方数，适应任意 num_envs。
    Args:
        images: (N, H, W, C) numpy array
        max_display: 最大显示的图片数量，防止图片过大
    """
    N = images.shape[0]
    if N > max_display:
        images = images[:max_display]
        N = max_display
    
    # 动态计算网格行列
    cols = int(math.ceil(math.sqrt(N)))
    rows = int(math.ceil(N / cols))
    
    H, W, C = images.shape[1:]
    
    # 如果图片数量不足填满网格，用黑色填充
    pad_num = rows * cols - N
    if pad_num > 0:
        padding = np.zeros((pad_num, H, W, C), dtype=images.dtype)
        images = np.concatenate([images, padding], axis=0)
        
    # 此时 images shape 为 (rows*cols, H, W, C)
    # Reshape logic: (rows, cols, H, W, C) -> (rows, H, cols, W, C) -> (rows*H, cols*W, C)
    grid = images.reshape(rows, cols, H, W, C)
    grid = grid.transpose(0, 2, 1, 3, 4)
    grid = grid.reshape(rows * H, cols * W, C)
    
    return grid

def main():
    args = get_args()
    
    print(f"--- Benchmark Configuration ---")
    print(f"Num Envs: {args.num_envs}")
    print(f"Resolution: {args.height}x{args.width}")
    print(f"Background: Enabled") # 既然你说了必须要有，这里就固定开启
    print(f"-------------------------------")

    # === 1. 路径与资源设置 ===
    mjcf_path = ROOT_PATH / "models" / "robots" / "manipulation" / "franka_emika_panda_robotiq" / "xmls" / "table30_00_simple_room.xml"
    _ASSETS_FRANKA_DIR = ROOT_PATH / "models" / "robots" / "manipulation" / "franka_emika_panda_robotiq"

    gaussians = {
        "link1" : (_ASSETS_FRANKA_DIR / "3dgs" / "franka" / "link1.ply").as_posix(),
        "link2" : (_ASSETS_FRANKA_DIR / "3dgs" / "franka" / "link2.ply").as_posix(),
        "link3" : (_ASSETS_FRANKA_DIR / "3dgs" / "franka" / "link3.ply").as_posix(),
        "link4" : (_ASSETS_FRANKA_DIR / "3dgs" / "franka" / "link4.ply").as_posix(),
        "link5" : (_ASSETS_FRANKA_DIR / "3dgs" / "franka" / "link5.ply").as_posix(),
        "link6" : (_ASSETS_FRANKA_DIR / "3dgs" / "franka" / "link6.ply").as_posix(),
        "link7" : (_ASSETS_FRANKA_DIR / "3dgs" / "franka" / "link7.ply").as_posix(),
        "robotiq_base"      : (_ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "robotiq_base.ply").as_posix(),
        "left_driver"       : (_ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "left_driver.ply").as_posix(),
        "left_coupler"      : (_ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "left_coupler.ply").as_posix(),
        "left_spring_link"  : (_ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "left_spring_link.ply").as_posix(),
        "left_follower"     : (_ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "left_follower.ply").as_posix(),
        "right_driver"      : (_ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "right_driver.ply").as_posix(),
        "right_coupler"     : (_ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "right_coupler.ply").as_posix(),
        "right_spring_link" : (_ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "right_spring_link.ply").as_posix(),
        "right_follower"    : (_ASSETS_FRANKA_DIR / "3dgs" / "robotiq" / "right_follower.ply").as_posix(),
    }
    background_path = (_ASSETS_FRANKA_DIR / "3dgs" / "table.ply").as_posix()

    # === 2. 初始化 Simulation ===
    mx_model = load_model(mjcf_path.as_posix())
    mx_data = SceneData(mx_model, batch=(args.num_envs, ))

    mx_model.keyframes[0].apply(mx_data)
    step(mx_model, mx_data)

    # === 3. 初始化 Renderer (前景) ===
    cfg = BatchSplatConfig(
        body_gaussians=gaussians, 
        background_ply=None, 
        minibatch=args.minibatch
    )
    renderer = MtxBatchSplatRenderer(cfg, mx_model)

    # === 4. 准备数据 ===
    link_poses = mx_model.get_link_poses(mx_data)
    body_pos = link_poses[...,:3]
    body_quat = link_poses[...,3:]
    
    cam_pos_lst = []
    cam_xmat_lst = []
    fovy_lst = []
    for cid in range(len(mx_model.cameras)):
        cam = mx_model.cameras[cid]
        cam_pose = cam.get_pose(mx_data)
        cam_pos_lst.append(cam_pose[...,:3])
        cam_xmat_lst.append(Rotation.from_quat(cam_pose[... ,3:7]).as_matrix().reshape(args.num_envs, 9))
        fovy_lst.append(mx_model.cameras[cid].fovy)
    
    cam_pos = np.array(cam_pos_lst).transpose(1, 0, 2)
    cam_xmat = np.array(cam_xmat_lst).transpose(1, 0, 2)
    fovy = np.array(fovy_lst)
    fovy = np.tile(fovy, (args.num_envs, 1))

    # 更新前景高斯状态
    gsb = renderer.batch_update_gaussians(body_pos, body_quat)

    # === 5. 背景处理 (固定执行) ===
    print("Pre-rendering background...")
    bg_renderer = MtxBatchSplatRenderer(BatchSplatConfig(body_gaussians=dict(), background_ply=background_path, minibatch=args.minibatch), mx_model)
    bg_gsb = bg_renderer.batch_update_gaussians(body_pos, body_quat)
    # 渲染出背景图，后续反复使用
    bg_imgs, _ = bg_renderer.batch_env_render(bg_gsb, cam_pos, cam_xmat, args.height, args.width, fovy)

    # === 6. Benchmarking ===
    print("Starting warmup...")
    # Warmup
    for _ in range(3):
        renderer.batch_env_render(gsb, cam_pos, cam_xmat, args.height, args.width, fovy, bg_imgs)

    print(f"Starting benchmark ({args.runs} runs)...")
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    elapsed_ms = []

    # 用于保存最后一次渲染结果以便验证
    final_rgb = None

    for _ in range(args.runs):
        start_event.record()
        
        # 必须传入 bg_imgs 以进行合成
        rgb, depth = renderer.batch_env_render(gsb, cam_pos, cam_xmat, args.height, args.width, fovy, bg_imgs)
        
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms.append(start_event.elapsed_time(end_event))
        final_rgb = rgb

    avg_ms = float(np.mean(elapsed_ms))
    print(f"Average render time over {args.runs} runs: {avg_ms:.3f} ms")
    
    total_images_per_step = args.num_envs * len(mx_model.cameras)
    ips = total_images_per_step / (avg_ms / 1000.0)
    print(f"Throughput: {ips:.1f} images/s")

    # === 7. 保存结果 (使用修复后的 smart_tile) ===
    if args.save_img and final_rgb is not None:
        print("Saving visualization to 'render_output.jpg'...")
        # 取每个环境的第一个相机视角来拼图: final_rgb shape [num_envs*num_cams, H, W, 3]
        # 假设我们只想看每个环境的第0个相机
        try:
            # 提取数据: (N_env, N_cam, H, W, C) -> 取 N_cam=0 -> (N_env, H, W, C)
            rgb_numpy = final_rgb.cpu().numpy()
            n_cams = len(mx_model.cameras)
            
            # 这里的 reshape 取决于 renderer 返回的 flat 形式
            # 假设 batch_env_render 返回的是 (num_envs * num_cams, H, W, 3)
            # 我们先 reshape 回去分离 env 和 camera
            rgb_reshaped = rgb_numpy.reshape(args.num_envs, n_cams, args.height, args.width, 3)
            
            # 只展示每个环境的第一个相机视角
            imgs_to_show = rgb_reshaped[:, 0, ...] 
            
            tiled_img = smart_tile(imgs_to_show, max_display=16)
            mediapy.write_image("render_output.jpg", tiled_img)
            print("Image saved successfully.")
        except Exception as e:
            print(f"Failed to save image: {e}")

if __name__ == "__main__":
    main()