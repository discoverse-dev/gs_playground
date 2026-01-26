import cv2
import numpy as np
import os
import time

# =========================================================
# 1. 路径设置
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 假设你的视频路径如下（请修改为你实际的视频路径）
video_path2 = "/home/xyys2003/ws/gs_playground/gs_playground/gs_playground/experimental/env/table30/data/table30_hang_toothbrush_cup_collect_yaw_stack_style_debug/videos/episode_00000.mp4"
video_path1 = "/home/xyys2003/ws/gs_playground/gs_playground/gs_playground/experimental/env/table30/data/real.mp4"

# 输出视频保存路径
out_dir = os.path.join(BASE_DIR, "compare_results")
os.makedirs(out_dir, exist_ok=True)
out_video_path = os.path.join(out_dir, "comparison_output.mp4")

# =========================================================
# 2. 初始化视频读取
# =========================================================
cap1 = cv2.VideoCapture(video_path1)
cap2 = cv2.VideoCapture(video_path2)

if not cap1.isOpened() or not cap2.isOpened():
    print("❌ Error: Could not open one of the video files.")
    exit()

# 获取视频 1 的基本信息（作为输出基准）
fps = cap1.get(cv2.CAP_PROP_FPS)
w = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"ℹ Video info: {w}x{h} @ {fps}fps")
print(f"ℹ Processing approximately {total_frames} frames...")

# =========================================================
# 3. 初始化视频写入 (VideoWriter)
# =========================================================
# 输出宽度为 2倍宽度 (因为我们要左右拼接: 左边原图 | 右边差异图)
output_size = (w * 2, h)

# 编码器设置 (mp4v 兼容性较好，Linux环境下也可以尝试 'XVID')
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out_writer = cv2.VideoWriter(out_video_path, fourcc, fps, output_size)

# =========================================================
# 4. 循环处理每一帧
# =========================================================
frame_idx = 0
start_time = time.time()

while True:
    # 1. 读取两帧
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # 如果任一视频结束，则停止循环
    if not ret1 or not ret2:
        print(f"\n⏹ End of stream reached at frame {frame_idx}.")
        break

    # 2. 尺寸对齐 (以 frame1 为基准)
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

    # 3. 图像处理 (灰度 -> 差分 -> 二值化 -> 去噪)
    g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(g1, g2)
    
    # 阈值 (可调: 25 是经验值，越小越敏感)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # 形态学去噪 (去除孤立噪点)
    kernel = np.ones((3, 3), np.uint8) # 视频连续帧可以用稍微小一点的核，或者(5,5)
    diff_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 4. 绘制结果
    # 复制一份原图做叠加
    overlay = frame1.copy()
    # 将差异区域涂成红色 (BGR: 0, 0, 255)
    overlay[diff_clean > 0] = (0, 0, 255)
    
    # 混合原图和红色图层 (addWeighted)
    result_view = cv2.addWeighted(frame1, 0.7, overlay, 0.3, 0)

    # 可选：在结果图上写帧号
    cv2.putText(result_view, f"Frame: {frame_idx}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 5. 拼接显示 (左: 原图, 右: 差异图)
    final_frame = np.hstack((frame1, result_view))

    # 6. 写入视频文件
    out_writer.write(final_frame)

    # 打印进度 (每50帧打印一次)
    if frame_idx % 50 == 0:
        print(f"Processing frame {frame_idx}/{total_frames}...", end='\r')

    frame_idx += 1

# =========================================================
# 5. 资源释放
# =========================================================
cap1.release()
cap2.release()
out_writer.release()

end_time = time.time()
duration = end_time - start_time

print("\n" + "="*40)
print(f"✅ Done! Processed {frame_idx} frames in {duration:.2f} seconds.")
print(f"📂 Output saved to: {out_video_path}")
print("="*40)