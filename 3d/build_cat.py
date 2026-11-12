"""Sculpts the pouncing orange tabby and its yarn ball and writes 3d/cat.glb.

    pip install numpy scipy scikit-image trimesh xatlas pillow
    python3 3d/build_cat.py            # -> 3d/cat.glb

The cat is one smooth surface (blended round shapes -> marching cubes), unwrapped
and painted from a procedural colour function: tabby strokes, yellow suns/stars/
spirals, eyes, nose, ears. The head star, yarn ball and loose thread are separate
nodes so a viewer can move them.

Axes: +X is the way the cat faces, +Y is up, +Z is its left flank. 1 unit ~ 10 cm.
"""
import sys, os
import numpy as np
import trimesh, xatlas
from skimage import measure
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.interpolate import CubicSpline

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "cat.glb")
PREVIEW = sys.argv[2] if len(sys.argv) > 2 else None   # optional folder for texture previews
STEP = 0.026        # marching-cubes cell
TEX = 2048

def rgb(h): return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float) / 255
BASE, MID, DARK = rgb("#F9BE70"), rgb("#EDA354"), rgb("#DD8A3E")
BELLY = rgb("#FBCB88")
PINK, PINK2 = rgb("#F6A08D"), rgb("#FBC3B0")
Y1, Y2 = rgb("#F8D448"), rgb("#FCEB9A")
LINE = rgb("#D4803A")
EYE_Y, EYE_B, EYE_LID, EYE_HI = rgb("#F8C51C"), rgb("#A35E22"), rgb("#93531F"), rgb("#FFECB8")
NOSE = rgb("#F0786A")
YARN, YARN_LINE = rgb("#EFA658"), rgb("#CC7632")

# ---------------------------------------------------------------- helpers
def norm(v): return v / np.linalg.norm(v)
def smooth(a, b, x): t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)
def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)

def _hash(i, j, k):
    n = (i * 374761393 + j * 668265263 + k * 1274126177) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1103515245) & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFF) / 65535.0

def vnoise(P):
    f = np.floor(P); i = f.astype(np.int64); t = P - f; t = t * t * (3 - 2 * t)
    x, y, z = i[:, 0], i[:, 1], i[:, 2]; tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]
    def L(a, b, w): return a + (b - a) * w
    c = lambda dx, dy, dz: _hash(x + dx, y + dy, z + dz)
    return L(L(L(c(0, 0, 0), c(1, 0, 0), tx), L(c(0, 1, 0), c(1, 1, 0), tx), ty),
             L(L(c(0, 0, 1), c(1, 0, 1), tx), L(c(0, 1, 1), c(1, 1, 1), tx), ty), tz)

def fbm(P): return (vnoise(P) + 0.5 * vnoise(P * 2.03 + 11.7)) / 1.5

def resample(pts, radii, n=6):
    pts, radii = np.array(pts, float), np.array(radii, float)
    if len(pts) < 3: return pts, radii
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    out = CubicSpline(s, np.c_[pts, radii])(np.linspace(0, s[-1], (len(pts) - 1) * n + 1))
    return out[:, :3], out[:, 3]

def chain(P, pts, rad):
    """Distance to a tube of round cones, plus arc length along it."""
    best = np.full(len(P), 1e9); tb = np.zeros(len(P)); cum = 0.0
    for i in range(len(pts) - 1):
        a = pts[i]; ba = pts[i + 1] - a; L2 = ba @ ba
        pa = P - a; h = np.clip(pa @ ba / L2, 0, 1)
        d = np.linalg.norm(pa - h[:, None] * ba, axis=1) - (rad[i] + (rad[i + 1] - rad[i]) * h)
        m = d < best; best[m] = d[m]; L = np.sqrt(L2); tb[m] = cum + h[m] * L; cum += L
    return best, tb

def ellipsoid(Q, c, r):
    q = (Q - np.array(c)) / np.array(r); k0 = np.linalg.norm(q, axis=1)
    k1 = np.linalg.norm(q / np.array(r), axis=1) + 1e-9
    return k0 * (k0 - 1) / k1

# ---------------------------------------------------------------- the pose
TORSO = resample([(-0.82, 1.30, 0), (-0.25, 1.12, 0), (0.36, 0.84, 0), (0.78, 0.74, 0.03)], [0.50, 0.50, 0.44, 0.34])
TAIL = resample([(-1.12, 1.50, 0), (-1.44, 1.98, 0.03), (-1.54, 2.52, 0.07), (-1.38, 3.02, 0.11),
                 (-1.00, 3.26, 0.14), (-0.66, 3.10, 0.15), (-0.60, 2.84, 0.15)],
                [0.17, 0.155, 0.15, 0.15, 0.15, 0.155, 0.15])
HIND_L = resample([(-0.68, 0.98, 0.34), (-0.50, 0.50, 0.42), (-0.44, 0.16, 0.42), (-0.24, 0.125, 0.43)], [0.33, 0.185, 0.14, 0.15])
HIND_R = resample([(-0.92, 1.00, -0.34), (-0.98, 0.50, -0.42), (-0.90, 0.16, -0.42), (-0.70, 0.125, -0.43)], [0.33, 0.185, 0.14, 0.15])
FORE_L = resample([(0.42, 0.70, 0.30), (0.50, 0.36, 0.38), (0.60, 0.15, 0.40), (0.80, 0.125, 0.41)], [0.25, 0.16, 0.135, 0.15])   # tucked
FORE_R = resample([(0.55, 0.62, -0.30), (1.15, 0.24, -0.50), (1.85, 0.135, -0.36), (2.04, 0.13, -0.30)], [0.23, 0.155, 0.135, 0.15])  # reaching
LIMBS = [TAIL, HIND_L, HIND_R, FORE_L, FORE_R]

HEAD_C = np.array([1.20, 0.93, 0.08]); HEAD_S = 1.27
_fx = norm(np.array([np.cos(np.radians(30)), 0.02, np.sin(np.radians(30))]))
_fz = norm(np.cross(_fx, [0, 1, 0])); _fy = np.cross(_fz, _fx)
HEAD_R = np.stack([_fx, _fy, _fz], axis=1)          # columns: face-forward, up, left
EAR_BASE, EAR_TIP = np.array([-0.03, 0.22, 0.31]), np.array([-0.07, 0.66, 0.45])

def head_local(P): return (P - HEAD_C) @ HEAD_R / HEAD_S

def head_sdf(H):
    d = ellipsoid(H, (0, 0, 0), (0.45, 0.40, 0.50))
    for s in (1, -1):
        d = smin(d, ellipsoid(H, (0.08, -0.13, 0.25 * s), (0.33, 0.27, 0.30)), 0.08)
    d = smin(d, ellipsoid(H, (0.33, -0.17, 0), (0.17, 0.12, 0.20)), 0.06)
    Q = H.copy(); Q[:, 0] = (Q[:, 0] + 0.045) * 1.9 - 0.045     # ears are flat front-to-back
    for s in (1, -1):
        e, _ = chain(Q, np.array([EAR_BASE * [1, 1, s], EAR_TIP * [1, 1, s]]), [0.21, 0.062])
        d = smin(d, e, 0.05)
    return d

def parts(P):
    """[(distance, arc length)] for torso, head, tail and the four legs."""
    out = [chain(P, *TORSO), (head_sdf(head_local(P)) * HEAD_S, None)]
    out += [chain(P, *c) for c in LIMBS]
    return out

def cat_sdf(P):
    ps = parts(P); torso = ps[0][0]
    d = smin(torso, ps[1][0], 0.10)
    for dl, _ in ps[2:]:
        d = np.minimum(d, smin(torso, dl, 0.09))
    return d

def chunked(fn, P, n=400000):
    return np.concatenate([fn(P[i:i + n]) for i in range(0, len(P), n)])

def mesh_sdf(fn, lo, hi, step):
    ax = [np.arange(lo[i], hi[i] + step, step) for i in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    vol = chunked(fn, G).reshape([len(a) for a in ax])
    V, F, _, _ = measure.marching_cubes(vol, 0.0, spacing=(step,) * 3)
    V += np.array(lo)
    e = 1e-3; N = np.zeros_like(V)
    for i in range(3):
        o = np.zeros(3); o[i] = e
        N[:, i] = chunked(fn, V + o) - chunked(fn, V - o)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    F = F[:, ::-1]                                    # outward winding
    fn_ = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    if (fn_ * N[F[:, 0]]).sum() < 0: F = F[:, ::-1]
    return V, F, N

# ---------------------------------------------------------------- painting
def decal(P, N, A, D, up=(0, 1, 0)):
    """Project points onto a plane at A facing -D. Returns u, v and a 'this side' mask."""
    A, D = np.array(A, float), norm(np.array(D, float))
    U = norm(np.cross(up, D)) if abs(np.dot(up, D)) < 0.95 else norm(np.cross((1, 0, 0), D))
    Vv = np.cross(D, U); R = P - A
    return R @ U, R @ Vv, (np.abs(R @ D) < 0.3) & (N @ -D > 0.2)

def star4(u, v, R, rot=0.0, p=0.55):
    c, s = np.cos(rot), np.sin(rot); a, b = u * c + v * s, -u * s + v * c
    return np.abs(a) ** p + np.abs(b) ** p < R ** p

def spiral(u, v, R=0.085, pitch=0.037):
    r = np.hypot(u, v); th = np.arctan2(v, u)
    f = (r / pitch - th / (2 * np.pi)) % 1.0
    return (r < R) & (r > 0.006) & (np.abs(f - 0.5) < 0.2)

def sun(u, v):
    r = np.hypot(u, v); th = np.arctan2(v, u)
    k = (th / (2 * np.pi) * 8) % 1.0 - 0.5                       # 8 rays
    ray = (r > 0.15) & (r < 0.30) & (np.abs(k) * (2 * np.pi / 8) * r < 0.042 * (0.30 - r) / 0.15)
    strong = (r < 0.05) | ((r > 0.085) & (r < 0.122)) | ray
    pale = (r >= 0.05) & (r <= 0.085)
    return strong, pale

def bullseye(u, v):
    r = np.hypot(u, v)
    return (r < 0.04) | ((r > 0.075) & (r < 0.11)), ((r >= 0.04) & (r <= 0.075)) | ((r >= 0.11) & (r < 0.14))

def tail_point(f):
    pts, rad = TAIL; i = int(f * (len(pts) - 1)); return pts[i], rad[i]

def paint(P, N):
    ps = parts(P)
    D = np.stack([p[0] for p in ps], 1)
    W = np.exp(-(D - D.min(1, keepdims=True)) / 0.045)
    warp = (fbm(P * 2.3) - 0.5) * 5.0
    H = head_local(P)

    # stripe field per part: bands across each bone, brushed apart by noise
    f = np.zeros_like(D)
    f[:, 0] = np.sin(2 * np.pi * (ps[0][1] - 0.38 * P[:, 1]) / 0.17 + warp)
    lam = [0.17, 0.14, 0.14, 0.14, 0.14]
    for i in range(5):
        f[:, 2 + i] = np.sin(2 * np.pi * ps[2 + i][1] / lam[i] + 0.6 * warp + i * 1.7)
    hx, hy, hz = H[:, 0], H[:, 1], H[:, 2]
    brow = np.sin(2 * np.pi * hz / 0.125 + 0.35 * warp)                       # forehead lines run front to back
    brow_m = smooth(0.30, 0.22, np.abs(hz)) * np.where(hx > 0.12, smooth(0.13, 0.19, hy), smooth(-0.12, 0.0, hy))
    cheek = np.sin(2 * np.pi * (hy + 0.3 * hx) / 0.105 + 0.3 * warp)
    cheek_m = smooth(0.36, 0.42, np.abs(hz)) * smooth(-0.26, -0.18, hy) * smooth(0.12, 0.04, hy) * smooth(0.30, 0.2, hx) * smooth(-0.2, -0.1, hx)
    back = np.sin(2 * np.pi * hz / 0.15 + warp) * smooth(0.0, -0.12, hx) * smooth(-0.15, 0.0, hy)
    f[:, 1] = np.maximum.reduce([brow * brow_m, cheek * cheek_m, back]) - 0.05
    F = (W * f).sum(1) / W.sum(1)
    F -= 2.2 * np.maximum(0, 0.46 - fbm(P * 4.6 + 17.0))                      # break the bands into dashes
    F -= 1.5 * smooth(-0.30, -0.80, N[:, 1])                                  # clean belly
    col = BASE + (BELLY - BASE) * smooth(-0.2, -0.8, N[:, 1])[:, None]
    col = col + (MID - col) * smooth(0.10, 0.15, F)[:, None]
    col = col + (DARK - col) * smooth(0.46, 0.51, F)[:, None]

    def put(mask, c): col[mask] = c

    # ears: pink inside, on the face side only
    for s in (1, -1):
        b, t = EAR_BASE * [1, 1, s], EAR_TIP * [1, 1, s]
        ax = norm((t - b)[1:]); rel = np.stack([hy - b[1], hz - b[2]], 1)
        along = rel @ ax; across = np.abs(rel @ np.array([-ax[1], ax[0]]))
        Lr = np.linalg.norm((t - b)[1:])
        inside = (along > 0.10) & (across < 0.135 * (1 - along / (Lr - 0.02))) & (hx > -0.06) & (hy > 0.2)
        put(inside, PINK)
        streak = inside & (np.abs((across - 0.02) % 0.07 - 0.035) < 0.011) & (along < Lr * 0.62) & (along > 0.14)
        put(streak, PINK2)

    # yellow doodles
    def doodle(A, Dv, shape, up=(0, 1, 0)):
        u, v, m = decal(P, N, A, Dv, up); r = shape(u, v)
        if isinstance(r, tuple): put(m & r[0], Y1); put(m & r[1], Y2)
        else: put(m & r, Y1)
    doodle((0.10, 1.27, 0.33), (0, -0.7, -0.7), sun)
    doodle((-0.52, 1.30, 0.47), (0, -0.3, -0.95), lambda u, v: star4(u, v, 0.21, 0.35))
    doodle((-0.14, 0.93, 0.47), (0, 0.25, -0.95), lambda u, v: (np.zeros(len(u), bool), star4(u, v, 0.115, 0.2)))
    for a in [(0.30, 0.80, 0.43), (0.40, 0.96, 0.38), (0.20, 0.66, 0.42)]:
        doodle(a, (0, 0, -1), lambda u, v: np.hypot(u, v) < 0.03)
    doodle((-0.30, 1.28, -0.46), (0, -0.3, 0.95), lambda u, v: star4(u, v, 0.20, -0.3))
    doodle((0.20, 1.02, -0.44), (0, -0.1, 1), lambda u, v: (np.zeros(len(u), bool), star4(u, v, 0.11, 0.5)))
    doodle((-0.78, 1.52, -0.36), (0, -0.7, 0.7), sun)
    for a in [(-0.02, 0.80, -0.46), (0.08, 0.66, -0.42), (-0.14, 0.68, -0.44)]:
        doodle(a, (0, 0, 1), lambda u, v: np.hypot(u, v) < 0.03)
    for fr in (0.27, 0.50):
        c, r = tail_point(fr)
        for s in (1, -1):
            doodle(c + [-0.03, 0, s * r], (0, 0, -s), bullseye)
    doodle((-0.46, 0.34, 0.60), (0, 0, -1), spiral)          # near hind leg
    doodle((0.56, 0.33, 0.56), (0, 0, -1), spiral)           # tucked front leg
    doodle((-0.95, 0.34, -0.60), (0, 0, 1), spiral)          # far hind leg
    doodle((1.66, 0.33, -0.40), (0, -1, 0.15), spiral, up=(1, 0, 0))   # reaching paw

    # toes
    for ch in (HIND_L, HIND_R, FORE_L, FORE_R):
        c = ch[0][-1]; fw = norm((ch[0][-1] - ch[0][-4]) * [1, 0, 1]); sd = np.cross([0, 1, 0], fw)
        R = P - c; a = R @ fw; b = R @ sd
        near = (np.linalg.norm(R, axis=1) < 0.22) & (a > 0.045) & (R[:, 1] > -0.07) & (R[:, 1] < 0.12)
        put(near & (np.abs(np.abs(b) - 0.052) < 0.011), LINE)

    # face
    face = hx > 0.12
    a = np.abs(hz) - 0.215; b = hy + 0.07
    ell = (a / 0.118) ** 2 + (b / 0.108) ** 2
    lid = 0.07 + 0.04 * a
    eye = face & (ell < 1) & (b < lid)
    put(face & (ell < 1.26) & (ell >= 1) & (b < lid), LINE)
    put(eye, EYE_Y)
    put(eye & ((a / 0.100) ** 2 + ((b - 0.052) / 0.082) ** 2 < 1), EYE_B)
    put(eye & (((a + 0.03) / 0.014) ** 2 + ((b + 0.05) / 0.023) ** 2 < 1), EYE_HI)
    put(face & (ell < 1.12) & (np.abs(b - lid) < 0.011), EYE_LID)
    nb = hy + 0.205
    put(face & (nb > -0.028) & (nb < 0.014) & (np.abs(hz) < 0.036 * (nb + 0.028) / 0.042), NOSE)
    return col

# ---------------------------------------------------------------- build the cat
print("sculpting cat…")
V, F, N = mesh_sdf(cat_sdf, (-1.95, -0.12, -0.80), (2.35, 3.60, 0.80), STEP)
print("  verts", len(V), "faces", len(F))

print("unwrapping…")
atlas = xatlas.Atlas(); atlas.add_mesh(V.astype(np.float32), F.astype(np.uint32))
po = xatlas.PackOptions(); po.padding = 5; po.resolution = TEX; po.bruteForce = False
atlas.generate(pack_options=po)
vmap, F2, UV = atlas[0]
V2, N2 = V[vmap], N[vmap]; F2 = F2.astype(np.int64)

print("baking texture…")
px = np.stack([UV[:, 0] * TEX, (1 - UV[:, 1]) * TEX], 1)
ids = Image.new("I", (TEX, TEX), 0); dr = ImageDraw.Draw(ids)
for i, t in enumerate(F2):
    xy = [tuple(px[j]) for j in t]; dr.polygon(xy, fill=i + 1, outline=i + 1)
ids = np.array(ids); ys, xs = np.nonzero(ids); fid = ids[ys, xs] - 1
t = F2[fid]; a, b, c = px[t[:, 0]], px[t[:, 1]], px[t[:, 2]]
q = np.stack([xs + 0.5, ys + 0.5], 1)
v0, v1, v2 = b - a, c - a, q - a
den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]; den[np.abs(den) < 1e-12] = 1e-12
w1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
w2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
w = np.clip(np.stack([1 - w1 - w2, w1, w2], 1), -0.5, 1.5)[:, :, None]
TP = (V2[t] * w).sum(1); TN = (N2[t] * w).sum(1); TN /= np.linalg.norm(TN, axis=1, keepdims=True) + 1e-12
cols = np.concatenate([paint(TP[i:i + 300000], TN[i:i + 300000]) for i in range(0, len(TP), 300000)])
img = np.zeros((TEX, TEX, 3)); img[ys, xs] = cols
filled = ids > 0
iy, ix = ndimage.distance_transform_edt(~filled, return_distances=False, return_indices=True)
img = img[iy, ix]
cat_tex = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))

def textured(Vt, Ft, Nt, UVt, image, name):
    mat = trimesh.visual.material.PBRMaterial(name=name, baseColorTexture=image, metallicFactor=0.0, roughnessFactor=1.0)
    m = trimesh.Trimesh(Vt, Ft, vertex_normals=Nt, process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=UVt, material=mat)
    return m

def flat(Vt, Ft, Nt, colour, name):
    mat = trimesh.visual.material.PBRMaterial(name=name, baseColorFactor=[*colour, 1.0], metallicFactor=0.0, roughnessFactor=1.0)
    m = trimesh.Trimesh(Vt, Ft, vertex_normals=Nt, process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(Vt), 2)), material=mat)
    return m

scene = trimesh.Scene()
scene.add_geometry(textured(V2, F2, N2, UV, cat_tex, "cat"), node_name="Cat", geom_name="Cat")

# ---------------------------------------------------------------- the star on the head
def star5_2d(x, y, r, rf=0.52):
    """Signed distance to a 5-point star (after Inigo Quilez)."""
    k1 = np.array([0.809016994375, -0.587785252292]); k2 = np.array([-k1[0], k1[1]])
    p = np.stack([np.abs(x), y], 1)
    p -= 2 * np.maximum(p @ k1, 0)[:, None] * k1
    p -= 2 * np.maximum(p @ k2, 0)[:, None] * k2
    p[:, 0] = np.abs(p[:, 0]); p[:, 1] -= r
    ba = rf * np.array([-k1[1], k1[0]]) - np.array([0, 1])
    h = np.clip(p @ ba / (ba @ ba), 0, r)
    return np.linalg.norm(p - h[:, None] * ba, axis=1) * np.sign(p[:, 1] * ba[0] - p[:, 0] * ba[1])

def star_mesh(r, thick, puff):
    def f(Q):
        d2 = star5_2d(Q[:, 0], Q[:, 1], r) - 0.03
        h = thick + puff * smooth(0.0, -0.12, d2)                # pillowy middle
        w = np.stack([np.maximum(d2, 0), np.maximum(np.abs(Q[:, 2]) - h, 0)], 1)
        return np.minimum(np.maximum(d2, np.abs(Q[:, 2]) - h), 0) + np.linalg.norm(w, axis=1) - 0.03
    return mesh_sdf(f, (-0.36, -0.36, -0.16), (0.36, 0.36, 0.16), 0.016)

def place(Vs, Ns, centre_h, normal_h, roll):
    n = norm(np.array(normal_h, float)); x = norm(np.cross([0, 1, 0], n)); y = np.cross(n, x)
    c, s = np.cos(roll), np.sin(roll)
    M = np.stack([x * c + y * s, -x * s + y * c, n], 1)       # star-local -> head-local
    M = HEAD_R @ M
    return Vs @ M.T + HEAD_C + HEAD_R @ np.array(centre_h) * HEAD_S, Ns @ M.T

print("star…")
Vs, Fs, Ns = star_mesh(0.215, 0.03, 0.02)
Vp, Np_ = place(Vs, Ns, (0.17, 0.44, 0.0), (0.72, 0.69, 0.0), 0.10)
scene.add_geometry(flat(Vp, Fs, Np_, Y2, "star"), node_name="Star", geom_name="Star")
Vs, Fs, Ns = star_mesh(0.225, 0.02, 0.0)
Vp, Np_ = place(Vs, Ns, (0.125, 0.425, -0.03), (0.72, 0.69, 0.0), -0.2)
scene.add_geometry(flat(Vp, Fs, Np_, Y1, "star-shadow"), node_name="StarShadow", geom_name="StarShadow")

# ---------------------------------------------------------------- yarn ball + thread
print("yarn…")
YARN_R = 0.31; YARN_C = np.array([2.62, YARN_R + 0.005, -0.06])
rng = np.random.default_rng(7)
AXES = np.array([norm(v) for v in rng.normal(size=(6, 3))])

def yarn_pattern(Dr):
    """Wound bands: each point belongs to the great-circle band it lies closest to."""
    lat = np.arcsin(np.clip(Dr @ AXES.T, -1, 1)); al = np.abs(lat) + np.arange(len(AXES)) * 0.035
    o = np.argsort(al, 1); i0 = o[:, 0]; r = np.arange(len(Dr))
    l0 = lat[r, i0]; gap = al[r, o[:, 1]] - al[r, i0]
    wave = np.cos(l0 * 2 * np.pi / 0.125)
    return wave, gap

def yarn_height(Dr):
    wave, gap = yarn_pattern(Dr)
    return 0.012 * (0.5 + 0.5 * wave) * smooth(0.0, 0.05, gap) - 0.010 * smooth(0.035, 0.0, gap)

nu, nv = 128, 64
uu, vv = np.meshgrid(np.linspace(0, 1, nu + 1), np.linspace(0, 1, nv + 1))
def sph(u, v):
    th = u * 2 * np.pi; ph = v * np.pi
    return np.stack([np.sin(ph) * np.cos(th), np.cos(ph), np.sin(ph) * np.sin(th)], -1)
Dn = sph(uu, vv).reshape(-1, 3)
Vy = Dn * (YARN_R + yarn_height(Dn))[:, None]
idx = np.arange((nu + 1) * (nv + 1)).reshape(nv + 1, nu + 1)
qa, qb, qc, qd = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
Fy = np.concatenate([np.stack([qa, qb, qc], 1), np.stack([qa, qc, qd], 1)])
UVy = np.stack([uu.ravel(), 1 - vv.ravel()], 1)
tw, th_ = 1536, 768
gu, gv = np.meshgrid((np.arange(tw) + 0.5) / tw, (np.arange(th_) + 0.5) / th_)
wave, gap = yarn_pattern(sph(gu, gv).reshape(-1, 3))
line = np.maximum(smooth(-0.72, -0.80, wave), smooth(0.022, 0.012, gap))
ycol = YARN + (YARN_LINE - YARN) * line[:, None]
yarn_tex = Image.fromarray((ycol.reshape(th_, tw, 3) * 255 + 0.5).astype(np.uint8))
ball = textured(Vy, Fy, Dn, UVy, yarn_tex, "yarn")
scene.add_geometry(ball, node_name="YarnBall", geom_name="YarnBall", transform=trimesh.transformations.translation_matrix(YARN_C))

def tube(points, r=0.021, sides=10, n=14):
    pts, _ = resample(points, np.full(len(points), r), n)
    tg = np.gradient(pts, axis=0); tg /= np.linalg.norm(tg, axis=1, keepdims=True)
    side = np.cross(tg, [0, 1, 0]); side /= np.linalg.norm(side, axis=1, keepdims=True); upv = np.cross(side, tg)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = np.cos(ang)[None, :, None] * side[:, None, :] + np.sin(ang)[None, :, None] * upv[:, None, :]
    Vt = (pts[:, None, :] + r * ring).reshape(-1, 3); Nt = ring.reshape(-1, 3)
    k = len(pts); i, j = np.meshgrid(np.arange(k - 1), np.arange(sides), indexing="ij")
    a = (i * sides + j).ravel(); b = (i * sides + (j + 1) % sides).ravel(); c = a + sides; d = b + sides
    Ft = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])
    caps = np.array([pts[0], pts[-1]]); o = len(Vt)
    Vt = np.concatenate([Vt, caps]); Nt = np.concatenate([Nt, [-tg[0], tg[-1]]])
    j = np.arange(sides); jn = (j + 1) % sides
    Ft = np.concatenate([Ft, np.stack([np.full(sides, o), j, jn], 1), np.stack([np.full(sides, o + 1), (k - 1) * sides + jn, (k - 1) * sides + j], 1)])
    return Vt, Ft, Nt

g = 0.022
strands = [
    [(2.50, 0.05, 0.16), (2.30, g, 0.42), (2.02, g, 0.40), (1.80, g, 0.62), (1.52, g, 0.66), (1.36, g, 0.56)],
    [(2.84, 0.05, -0.22), (3.02, g, -0.30), (3.20, g, -0.22), (3.36, g, -0.34), (3.46, g, -0.30)],
]
tv, tf, tn = [], [], []; off = 0
for s in strands:
    a, b, c = tube(s); tv.append(a); tf.append(b + off); tn.append(c); off += len(a)
scene.add_geometry(flat(np.concatenate(tv), np.concatenate(tf), np.concatenate(tn), YARN_LINE, "thread"), node_name="YarnThread", geom_name="YarnThread")

scene.export(OUT)
print("wrote", OUT, round(os.path.getsize(OUT) / 1e6, 2), "MB")
if PREVIEW:
    cat_tex.save(os.path.join(PREVIEW, "cat_tex.png")); yarn_tex.save(os.path.join(PREVIEW, "yarn_tex.png"))
