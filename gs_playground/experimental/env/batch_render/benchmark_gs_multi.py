import argparse
import time
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path

# 引入原有依赖
from motrixsim import SceneData, load_model, step
from gaussian_renderer import BatchSplatConfig, MtxBatchSplatRenderer
from gs_playground import ROOT_PATH

# === 全局资源加载 (只加载一次以节省时间) ===
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
background_path = (_ASSETS_FRANKA_DIR / "3dgs" / "simple_room.ply").as_posix()

# 加载模型结构
mx_model = load_model(mjcf_path.as_posix())

def run_benchmark_case(num_envs, height, width, minibatch, runs=10):
    """
    运行单个测试用例并返回指标
    """
    try:
        # 清理显存统计
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 1. 初始化 SceneData
        mx_data = SceneData(mx_model, batch=(num_envs, ))
        mx_model.keyframes[0].apply(mx_data)
        step(mx_model, mx_data)

        # 2. 初始化 Renderer
        cfg = BatchSplatConfig(
            body_gaussians=gaussians, 
            background_ply=None, 
            minibatch=minibatch
        )
        renderer = MtxBatchSplatRenderer(cfg, mx_model)

        # 3. 准备数据
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
            cam_xmat_lst.append(Rotation.from_quat(cam_pose[... ,3:7]).as_matrix().reshape(num_envs, 9))
            fovy_lst.append(mx_model.cameras[cid].fovy)
        
        cam_pos = np.array(cam_pos_lst).transpose(1, 0, 2)
        cam_xmat = np.array(cam_xmat_lst).transpose(1, 0, 2)
        fovy = np.array(fovy_lst)
        fovy = np.tile(fovy, (num_envs, 1))

        # 更新高斯
        gsb = renderer.batch_update_gaussians(body_pos, body_quat)

        # 准备背景 (只做一次)
        bg_renderer = MtxBatchSplatRenderer(BatchSplatConfig(body_gaussians=dict(), background_ply=background_path, minibatch=minibatch), mx_model)
        bg_gsb = bg_renderer.batch_update_gaussians(body_pos, body_quat)
        bg_imgs, _ = bg_renderer.batch_env_render(bg_gsb, cam_pos, cam_xmat, height, width, fovy)

        # Warmup
        for _ in range(3):
            renderer.batch_env_render(gsb, cam_pos, cam_xmat, height, width, fovy, bg_imgs)
        
        # Benchmark
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        elapsed_ms = []

        for _ in range(runs):
            start_event.record()
            renderer.batch_env_render(gsb, cam_pos, cam_xmat, height, width, fovy, bg_imgs)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms.append(start_event.elapsed_time(end_event))

        avg_ms = float(np.mean(elapsed_ms))
        
        # 计算指标
        total_cameras = num_envs * len(mx_model.cameras)
        images_per_sec = total_cameras / (avg_ms / 1000.0)
        fps = 1000.0 / avg_ms # Batch FPS
        
        # 显存占用 (GB)
        vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        
        return images_per_sec, fps, vram_gb

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return None, None, "OOM"
        else:
            raise e
    except Exception as e:
        print(f"Error: {e}")
        return None, None, "Error"

def main():
    # 定义测试配置
    # 格式: (Height, Width, Label)
    resolutions = [
        (128, 128, "128x128"),
        (256, 256, "256x256"),
        (480, 480, "480x480"),
        (720, 1280, "1280x720") 
    ]
    
    # 基础环境数列表
    base_envs = [2, 4, 8, 16, 32, 64, 128, 256]
    
    print("Starting Benchmark Suite...")
    
    for H, W, label in resolutions:
        print(f"\n=== {label} ===")
        print("| 相机数 | images/s | FPS | 显存占用(GB) | GPU利用率(%) | 备注 |")
        print("|---:|---:|---:|---:|---:|---|")
        
        configs = []
        # 1. 构建基础配置
        for e in base_envs:
            minibatch = e
            # [修正] 1280x720下256环境显存溢出，强制minibatch=128
            if label == "1280x720" and e == 256:
                minibatch = 128
            configs.append((e, minibatch))
        
        # 2. 添加额外的大规模测试配置
        if label == "128x128" :
            configs.append((512, 128))
            configs.append((1024, 64))
        elif label == "256x256":
            configs.append((512, 128))
            configs.append((1024, 64))
        elif label == "480x480":
            configs.append((512, 128))
            configs.append((1024, 32))
        elif label == "1280x720":
            configs.append((512, 64))
            configs.append((1024, 1))

        for num_envs, minibatch in configs:
            img_s, fps, vram = run_benchmark_case(num_envs, H, W, minibatch)
            
            # 格式化输出
            env_str = f"{num_envs}"
            if minibatch != num_envs:
                env_str = f"{num_envs}({minibatch})"
            
            if img_s is None:
                # OOM 或 错误
                res_str = f"| {env_str:>6} | {'-':>8} | {'-':>6} | {vram:>10} | {'-':>10} | Error |"
            else:
                res_str = f"| {env_str:>6} | {img_s:>8.1f} | {fps:>6.1f} | {vram:>10.2f} | {'-':>10} | |"
            
            print(res_str)

if __name__ == "__main__":
    main()