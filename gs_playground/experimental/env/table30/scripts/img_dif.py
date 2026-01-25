import cv2
import numpy as np
import os

# =========================================================
# 1. 路径设置（不依赖运行目录）
# =========================================================

# 当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# RoboArena 根目录（按你的目录结构）
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# 输入图像路径
path1 = os.path.join(ROOT_DIR, "/home/xyys2003/ws/gs_playground/gs_playground/gs_playground/experimental/env/table30/data/real.png")
path2 = os.path.join(ROOT_DIR, "/home/xyys2003/ws/gs_playground/gs_playground/gs_playground/experimental/env/table30/data/sim.png")

# 输出目录
out_dir = os.path.join(ROOT_DIR, "/home/xyys2003/ws/gs_playground/gs_playground/gs_playground/experimental/env/table30/data/dif_images")
os.makedirs(out_dir, exist_ok=True)

# =========================================================
# 2. 安全读取图像
# =========================================================

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"❌ Cannot read image: {path}")
    return img

img1 = load_image(path1)
img2 = load_image(path2)

print("✔ Image1 shape:", img1.shape)
print("✔ Image2 shape:", img2.shape)

# =========================================================
# 3. 保证尺寸一致
# =========================================================

if img1.shape != img2.shape:
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    print("⚠ Resized img2 to match img1")

# =========================================================
# 4. 灰度转换
# =========================================================

g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# =========================================================
# 5. 灰度帧差
# =========================================================

diff_gray = cv2.absdiff(g2, g1)

# =========================================================
# 6. 二值化
# =========================================================

_, diff_thresh = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)

# =========================================================
# 7. 去噪（开运算）
# =========================================================

kernel = np.ones((5, 5), np.uint8)
diff_clean = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel)

# =========================================================
# 8. 叠加显示差异（红色）
# =========================================================

overlay = img1.copy()
overlay[diff_clean > 0] = (0, 0, 255)  # 红色高亮变化区域
vis = cv2.addWeighted(img1, 0.7, overlay, 0.3, 0)

# =========================================================
# 9. 保存结果
# =========================================================

cv2.imwrite(os.path.join(out_dir, "diff_gray.png"), diff_gray)
cv2.imwrite(os.path.join(out_dir, "diff_thresh.png"), diff_thresh)
cv2.imwrite(os.path.join(out_dir, "diff_clean.png"), diff_clean)
cv2.imwrite(os.path.join(out_dir, "diff_overlay.png"), vis)

print("✅ All results saved to:", out_dir)
