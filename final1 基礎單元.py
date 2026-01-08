import numpy as np

from compas.geometry import Point, Vector, Line
from compas.datastructures import Mesh
from compas_viewer.viewer import Viewer


# ----------------------------
# 0) 基本工具
# ----------------------------
sqrt3 = np.sqrt(3)

def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector")
    return v / n

def rodrigues(p, c, axis, theta):
    """
    Rotate point p around axis passing through c by theta (radians).
    axis can be any non-zero vector (will be normalized).
    """
    p = np.asarray(p, float)
    c = np.asarray(c, float)
    k = unit(axis)

    v = p - c
    v_rot = (v * np.cos(theta)
             + np.cross(k, v) * np.sin(theta)
             + k * (np.dot(k, v)) * (1 - np.cos(theta)))
    return c + v_rot

def as_point(a):
    return Point(float(a[0]), float(a[1]), float(a[2]))


# ----------------------------
# 1) 你的基礎幾何（照你總整）
# ----------------------------
P1  = np.array([0.0, 0.0, 1.0])
P2  = np.array([sqrt3, 0.0, 1.0])
P3  = np.array([sqrt3 + 1.0, -sqrt3, 1.0])
P4  = np.array([sqrt3, 0.0, 1.0])
P5  = np.array([sqrt3 + 1.0,  sqrt3, 1.0])
P6  = np.array([sqrt3, 0.0, 1.0])

P7  = np.array([0.0, 0.0, 0.0])
P8  = np.array([sqrt3, 0.0, 0.0])
P9  = np.array([sqrt3 + 1.0, -sqrt3, 0.0])
P10 = np.array([sqrt3 + 1.0,  sqrt3, 0.0])


# ----------------------------
# 2) 角度
# ----------------------------
# 你說「往下 (z 變小)」：用 +theta1 比較一致
theta1 = 0.15552   # rad
theta3 = -0.96043    # rad


# ----------------------------
# 3) 平面 S1~S3 的法向量（繞此軸=在平面內旋轉）
# ----------------------------
n1 = np.array([0.0, -1.0, 0.0])                 # S1: Plate7 平面 y=0
n2 = unit(np.array([sqrt3, 1.0, 0.0]))         # S2: Plate8 平面
n3 = unit(np.array([-sqrt3, 1.0, 0.0]))        # S3: Plate9 平面


# ----------------------------
# 4) 依你的規則算 prime 點
# ----------------------------
# Plate7' (S1, center P8, theta1): affects P1, P7
P1p = rodrigues(P1, P8, n1, theta1)
P7p = rodrigues(P7, P8, n1, theta1)

# Plate8' (S2, center P8, theta1): affects P3, P9
P3p = rodrigues(P3, P8, n2, theta1)
P9p = rodrigues(P9, P8, n2, theta1)

# Plate9' (S3, center P8, theta1): affects P5, P10
P5p  = rodrigues(P5,  P8, n3, theta1)
P10p = rodrigues(P10, P8, n3, theta1)

# P2', P4', P6' with theta3 on specific planes
P2p = rodrigues(P2, P8, n3, theta3)  # P2 in S3
P4p = rodrigues(P4, P8, n1, theta3)  # P4 in S1
P6p = rodrigues(P6, P8, n2, theta3)  # P6 in S2


# 統一收集點（給 Viewer 用）
points = {
    "P1'": P1p,
    "P2'": P2p,
    "P3'": P3p,
    "P4'": P4p,
    "P5'": P5p,
    "P6'": P6p,
    "P7'": P7p,
    "P8":  P8,
    "P9'": P9p,
    "P10'": P10p,
}


# ----------------------------
# 5) 板片 Plate1'~Plate9'（用 Mesh 顯示面）
# ----------------------------
plates = {
    "Plate1'": ["P1'", "P2'", "P8"],
    "Plate2'": ["P1'", "P6'", "P8"],
    "Plate3'": ["P3'", "P2'", "P8"],
    "Plate4'": ["P3'", "P4'", "P8"],
    "Plate5'": ["P5'", "P4'", "P8"],
    "Plate6'": ["P5'", "P6'", "P8"],
    "Plate7'": ["P1'", "P7'", "P8"],
    "Plate8'": ["P3'", "P9'", "P8"],
    "Plate9'": ["P5'", "P10'", "P8"],
}

# 做一個總 mesh（把每塊三角形都加進去）
mesh = Mesh()

# 先加所有點到 mesh（固定 key，讓 face 直接引用）
vkeys = {}
for name, xyz in points.items():
    vkeys[name] = mesh.add_vertex(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))

# 加面
for pl, verts in plates.items():
    mesh.add_face([vkeys[v] for v in verts])


# ----------------------------
# 6) 摺線 L1~L6（連到 P8）
# ----------------------------
fold_lines = [
    ("P1'", "P8"),
    ("P2'", "P8"),
    ("P3'", "P8"),
    ("P4'", "P8"),
    ("P5'", "P8"),
    ("P6'", "P8"),
]


# ----------------------------
# 7) 用 COMPAS Viewer 顯示
# ----------------------------
viewer = Viewer()
viewer.scene.add(mesh, name="plates", show_faces=True, show_edges=True, opacity=0.55)

# 點 + 標籤
for name, xyz in points.items():
    viewer.scene.add(as_point(xyz), name=name, size=10)
    viewer.scene.add(as_point(xyz), name=f"label_{name}", text=name)

# 折線
for a, b in fold_lines:
    pa = points[a]
    pb = points[b]
    line = Line(as_point(pa), as_point(pb))
    viewer.scene.add(line, name=f"{a}-{b}")

viewer.show()
