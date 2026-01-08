import numpy as np

from compas.geometry import Point, Line
from compas.datastructures import Mesh
from compas_viewer.viewer import Viewer


sqrt3 = np.sqrt(3)

def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector")
    return v / n

def rodrigues(p, c, axis, theta):
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

def clamp(a, lo, hi):
    return max(lo, min(hi, a))

def Q_arc(theta_arc):
    # C=(sqrt3,0,0), radius=1, azimuth=240°, polar angle theta_arc from +z
    az = np.deg2rad(240.0)
    s = np.sin(theta_arc)
    return np.array([sqrt3 + s*np.cos(az),
                     0.0   + s*np.sin(az),
                     np.cos(theta_arc)], float)

# Plane/plate axes (your definition)
n1 = np.array([0.0, -1.0, 0.0])          # S1
n2 = unit(np.array([sqrt3, 1.0, 0.0]))   # S2
n3 = unit(np.array([-sqrt3, 1.0, 0.0]))  # S3

def base_points(h):
    # top layer z=h
    P1  = np.array([0.0, 0.0, h])
    P2  = np.array([sqrt3, 0.0, h])
    P3  = np.array([sqrt3 + 1.0, -sqrt3, h])
    P4  = np.array([sqrt3, 0.0, h])
    P5  = np.array([sqrt3 + 1.0,  sqrt3, h])
    P6  = np.array([sqrt3, 0.0, h])

    # bottom layer z=0
    P7  = np.array([0.0, 0.0, 0.0])
    P8  = np.array([sqrt3, 0.0, 0.0])
    P9  = np.array([sqrt3 + 1.0, -sqrt3, 0.0])
    P10 = np.array([sqrt3 + 1.0,  sqrt3, 0.0])
    return P1,P2,P3,P4,P5,P6,P7,P8,P9,P10

def compute_primes(h, theta1, theta_fold):
    P1,P2,P3,P4,P5,P6,P7,P8,P9,P10 = base_points(h)

    # Plate7' (S1, center P8, theta1): affects P1, P7
    P1p = rodrigues(P1, P8, n1, theta1)
    P7p = rodrigues(P7, P8, n1, theta1)

    # Plate8' (S2, center P8, theta1): affects P3, P9
    P3p = rodrigues(P3, P8, n2, theta1)
    P9p = rodrigues(P9, P8, n2, theta1)

    # Plate9' (S3, center P8, theta1): affects P5, P10
    P5p  = rodrigues(P5,  P8, n3, theta1)
    P10p = rodrigues(P10, P8, n3, theta1)

    # P2', P4', P6' with theta_fold (THIS is the one you said must rotate out)
    P2p = rodrigues(P2, P8, n3, theta_fold)
    P4p = rodrigues(P4, P8, n1, theta_fold)
    P6p = rodrigues(P6, P8, n2, theta_fold)

    return {
        "P1'": P1p, "P2'": P2p, "P3'": P3p, "P4'": P4p, "P5'": P5p, "P6'": P6p,
        "P7'": P7p, "P8": P8, "P9'": P9p, "P10'": P10p,
        "P1": P1, "P2": P2, "P3": P3, "P4": P4, "P5": P5, "P6": P6,
        "P7": P7, "P9": P9, "P10": P10
    }

def midpoint_M(pts):
    # If your "M" is different, change THIS ONE LINE.
    return 0.5*(pts["P1'"] + pts["P2'"])

def solve_best(
    h_range=(0.2, 3.0),
    theta1_range=(np.deg2rad(1.0), np.pi/2),     # enforce theta1 != 0
    theta_fold_range=(0.0, np.pi/2),
    theta_arc_range=(-np.pi/2, 0.0),              # your "theta3 definition" <= 90deg
    steps=(81, 121, 121, 121),                   # (h, theta1, theta_fold, theta_arc)
    refine_rounds=2,
    verbose=True
):
    Hn, T1n, TFn, TAn = steps

    def eval_grid(h_lo,h_hi, t1_lo,t1_hi, tf_lo,tf_hi, ta_lo,ta_hi, Hn,T1n,TFn,TAn):
        hs  = np.linspace(h_lo, h_hi, Hn)
        t1s = np.linspace(t1_lo, t1_hi, T1n)
        tfs = np.linspace(tf_lo, tf_hi, TFn)
        tas = np.linspace(ta_lo, ta_hi, TAn)

        best = None
        for h in hs:
            for t1 in t1s:
                for tf in tfs:
                    pts = compute_primes(h, t1, tf)
                    M = midpoint_M(pts)
                    for ta in tas:
                        Q = Q_arc(ta)
                        err = np.linalg.norm(M - Q)
                        if (best is None) or (err < best["err"]):
                            best = {"err": err, "h": h, "theta1": t1, "theta_fold": tf, "theta_arc": ta,
                                    "M": M, "Q": Q, "pts": pts}
        return best

    best = eval_grid(h_range[0],h_range[1],
                     theta1_range[0],theta1_range[1],
                     theta_fold_range[0],theta_fold_range[1],
                     theta_arc_range[0],theta_arc_range[1],
                     Hn,T1n,TFn,TAn)

    if verbose:
        print("coarse best err =", best["err"])
        print("h =", best["h"])
        print("theta1 =", best["theta1"], "rad")
        print("theta_fold =", best["theta_fold"], "rad")
        print("theta_arc =", best["theta_arc"], "rad")
        print("M =", best["M"])
        print("Q =", best["Q"])

    for r in range(refine_rounds):
        shrink = 0.25**(r+1)

        def win(center, lo, hi):
            span = (hi - lo) * shrink
            return (clamp(center - span/2, lo, hi), clamp(center + span/2, lo, hi))

        h_lo, h_hi   = win(best["h"], h_range[0], h_range[1])
        t1_lo, t1_hi = win(best["theta1"], theta1_range[0], theta1_range[1])
        tf_lo, tf_hi = win(best["theta_fold"], theta_fold_range[0], theta_fold_range[1])
        ta_lo, ta_hi = win(best["theta_arc"], theta_arc_range[0], theta_arc_range[1])

        best = eval_grid(h_lo,h_hi, t1_lo,t1_hi, tf_lo,tf_hi, ta_lo,ta_hi,
                         Hn,T1n,TFn,TAn)

        if verbose:
            print(f"refine round {r+1} best err =", best["err"])
            print("h =", best["h"])
            print("theta1 =", best["theta1"], "rad")
            print("theta_fold =", best["theta_fold"], "rad")
            print("theta_arc =", best["theta_arc"], "rad")
            print("M =", best["M"])
            print("Q =", best["Q"])

    return best


best = solve_best(
    h_range=(0.2, 3.0),
    theta1_range=(np.deg2rad(5), np.pi/2),
    theta_fold_range=(0.0, np.pi/2),
    theta_arc_range=(-np.pi/2, 0.0),
    steps=(11, 11, 11, 11),
    refine_rounds=1,
    verbose=True
)

h_best = best["h"]
theta1_best = best["theta1"]
theta_fold_best = best["theta_fold"]
theta_arc_best = best["theta_arc"]

print("\nFINAL (copy-paste friendly):")
print("h =", h_best)
print("theta1 =", theta1_best, "rad")
print("theta_fold =", theta_fold_best, "rad")
print("theta_arc =", theta_arc_best, "rad")
print("min_error =", best["err"])

pts = compute_primes(h_best, theta1_best, theta_fold_best)
M = midpoint_M(pts)
Q = Q_arc(theta_arc_best)

pts["M"] = M
pts["Q_arc"] = Q

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

mesh = Mesh()
vkeys = {}

# Add only needed vertices to mesh
needed = set([v for vs in plates.values() for v in vs] + ["M", "Q_arc"])
for name in needed:
    xyz = pts[name]
    vkeys[name] = mesh.add_vertex(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))

for pl, verts in plates.items():
    mesh.add_face([vkeys[v] for v in verts])

fold_lines = [
    ("P1'", "P8"),
    ("P2'", "P8"),
    ("P3'", "P8"),
    ("P4'", "P8"),
    ("P5'", "P8"),
    ("P6'", "P8"),
]

extra_lines = [
    ("M", "Q_arc"),
    ("P1'", "P2'"),
]

viewer = Viewer()
viewer.scene.add(mesh, name="plates", show_faces=True, show_edges=True, opacity=0.55)

for name, xyz in pts.items():
    if name in ["P1'", "P2'", "P3'", "P4'", "P5'", "P6'", "P7'", "P8", "P9'", "P10'", "M", "Q_arc"]:
        viewer.scene.add(as_point(xyz), name=name, size=10)
        viewer.scene.add(as_point(xyz), name=f"label_{name}", text=name)

for a, b in fold_lines:
    pa = pts[a]
    pb = pts[b]
    viewer.scene.add(Line(as_point(pa), as_point(pb)), name=f"{a}-{b}")

for a, b in extra_lines:
    pa = pts[a]
    pb = pts[b]
    viewer.scene.add(Line(as_point(pa), as_point(pb)), name=f"{a}-{b}")

viewer.show()
