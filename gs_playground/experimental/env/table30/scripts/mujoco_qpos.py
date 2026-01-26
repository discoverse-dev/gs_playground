import numpy as np
import mujoco

# ===== 1) 载入你的“完整场景xml”（包含机器人+你挂相机的那个xml） =====
MODEL_XML_PATH = "/home/xyys2003/ws/gs_playground/gs_playground/gs_playground/models/robots/manipulation/franka_emika_panda_robotiq/xmls/table30_13_arrange_flower.xml"   # 改成你的文件路径
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)

# ===== 2) 让模型处于你认为这组 base 相机位姿成立的参考姿态（推荐用 keyframe home）=====
kf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
data.qpos[:] = model.key_qpos[kf]
mujoco.mj_forward(model, data)

# ===== 3) 读取 link7 在 base/world 下的位姿 =====
bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
p_bl7 = data.xpos[bid].copy()                     # base->link7 平移
R_bl7 = data.xmat[bid].reshape(3, 3).copy()       # base->link7 旋转（列向量含义见下）

# ===== 4) 你的 base 下相机位姿（来自你认为“合理”的那组）=====
p_bc = np.array([0.45622845, 0.00849983, 0.44796835], dtype=float)

x = np.array([-0.0305, -0.9993, -0.0228], dtype=float)   # camera +X in base
y = np.array([ 0.8935, -0.0171, -0.4487], dtype=float)   # camera +Y in base

# 保守起见正交化一下，避免数值不正交导致漂
x = x / np.linalg.norm(x)
y = y - x * np.dot(x, y)
y = y / np.linalg.norm(y)
z = np.cross(x, y)
z = z / np.linalg.norm(z)

R_bc = np.column_stack([x, y, z])  # cam->base 旋转：列分别是 cam轴在base下的表达

# ===== 5) 计算 link7->cam（把 base 下的相机位姿换到 link7 局部）=====
# p_bc = p_bl7 + R_bl7 @ p_l7c  => p_l7c = R_bl7.T @ (p_bc - p_bl7)
p_l7c = R_bl7.T @ (p_bc - p_bl7)

# R_bc = R_bl7 @ R_l7c  => R_l7c = R_bl7.T @ R_bc
R_l7c = R_bl7.T @ R_bc

x_l = R_l7c[:, 0]
y_l = R_l7c[:, 1]

print("Paste into <camera> under link7:")
print(f'pos="{p_l7c[0]:.8f} {p_l7c[1]:.8f} {p_l7c[2]:.8f}"')
print(f'xyaxes="{x_l[0]:.8f} {x_l[1]:.8f} {x_l[2]:.8f}   {y_l[0]:.8f} {y_l[1]:.8f} {y_l[2]:.8f}"')
