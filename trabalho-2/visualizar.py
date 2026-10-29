import json
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pycolmap
import trimesh

VIDEO_SIZE, VIDEO_FPS, VIDEO_SECONDS = (1280, 720), 30, 40
RING_COLORS = {"h1": (230, 80, 60), "h2": (60, 160, 230), "h3": (80, 200, 90), "topo": (240, 190, 40)}


class Renderer:
    def __init__(self, path):
        mesh = o3d.io.read_triangle_mesh(str(path), True)
        self.uv = np.asarray(mesh.triangle_uvs).reshape(-1, 3, 2)
        self.material = np.asarray(mesh.triangle_material_ids)
        self.textures = [np.asarray(t)[..., :3] for t in mesh.textures]
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def __call__(self, K, cam_from_world, width, height, background=235):
        rays = self.scene.create_rays_pinhole(o3d.core.Tensor(K), o3d.core.Tensor(cam_from_world), width, height)
        hit = self.scene.cast_rays(rays)
        ids = hit["primitive_ids"].numpy().astype(np.int64)
        valid = ids != self.scene.INVALID_ID
        ids, b = ids[valid], hit["primitive_uvs"].numpy()[valid]
        uv = (1 - b.sum(1, keepdims=True)) * self.uv[ids, 0] + b[:, :1] * self.uv[ids, 1] + b[:, 1:] * self.uv[ids, 2]
        colors = np.zeros((len(ids), 3), np.float32)
        for m, tex in enumerate(self.textures):
            sel = self.material[ids] == m
            colors[sel] = bilinear(tex, uv[sel, 0] * tex.shape[1] - 0.5, uv[sel, 1] * tex.shape[0] - 0.5)
        image = np.full((height, width, 3), background, np.uint8)
        image[valid] = np.clip(colors, 0, 255).astype(np.uint8)
        return image, valid


def bilinear(tex, x, y):
    h, w = tex.shape[:2]
    x, y = np.clip(x, 0, w - 1.001), np.clip(y, 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    at = lambda yy, xx: tex[yy, xx].astype(np.float32)
    return (at(y0, x0) * (1 - fx) + at(y0, x0 + 1) * fx) * (1 - fy) + (at(y0 + 1, x0) * (1 - fx) + at(y0 + 1, x0 + 1) * fx) * fy


def pose4(image):
    return np.r_[image.cam_from_world().matrix(), [[0, 0, 0, 1]]]


def look_at(eye, target):
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(z, [0, 0, 1])
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z])
    return np.r_[np.c_[R, -R @ eye], [[0, 0, 0, 1]]]


def ring(name):
    return name.split("/")[0] if "/" in name else "h2"


def plot_cameras(rec, path):
    pts = np.array([p.xyz for p in rec.points3D.values()])
    colors = np.array([p.color for p in rec.points3D.values()]) / 255
    idx = np.random.default_rng(0).permutation(len(pts))[:20000]
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*pts[idx].T, c=colors[idx], s=0.5)
    for im in rec.images.values():
        c, d = im.projection_center(), im.viewing_direction()
        color = np.array(RING_COLORS.get(ring(im.name), (120, 120, 120))) / 255
        ax.scatter(*c, color=color, s=12)
        ax.plot(*np.c_[c, c + 40 * d], color=color, lw=0.8)
    centers = np.array([im.projection_center() for im in rec.images.values()])
    allp = np.r_[pts[idx], centers]
    ax.set_box_aspect(np.ptp(allp, 0))
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def frustum(image, depth=15.0):
    cam, T = image.camera, pose4(image)
    R, t = T[:3, :3], T[:3, 3]
    w, h = cam.width, cam.height
    corners = cam.cam_from_img(np.array([[0, 0], [w, 0], [w, h], [0, h]], float))
    corners = np.c_[corners, np.ones(4)] * depth
    verts = np.r_[[image.projection_center()], (corners - t) @ R]
    faces = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 3, 2], [1, 4, 3]]
    color = list(RING_COLORS.get(ring(image.name), (120, 120, 120))) + [255]
    return trimesh.Trimesh(verts, faces, vertex_colors=np.tile(color, (5, 1)), process=False)


def export_glb(rec, textured, path):
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.load(textured, process=False, force="mesh"), node_name="modelo")
    for im in rec.images.values():
        scene.add_geometry(frustum(im), node_name=im.name.replace("/", "_"))
    pts = rec.points3D.values()
    scene.add_geometry(trimesh.PointCloud([p.xyz for p in pts], colors=[list(p.color) + [255] for p in pts]), node_name="esparsa")
    scene.export(path)


def photo_vs_render(render, mvs_rec, mvs_dir, figures, scale=0.5):
    names = sorted(im.name for im in mvs_rec.images.values())
    chosen = [next(n for n in names if ring(n) == r) for r in ("h1", "h2", "h3") if any(ring(n) == r for n in names)] or names[:3]
    stats = {}
    for name in chosen:
        im = mvs_rec.find_image_with_name(name)
        K = im.camera.calibration_matrix() * [[scale], [scale], [1]]
        w, h = int(im.camera.width * scale), int(im.camera.height * scale)
        photo = cv2.resize(cv2.imread(str(mvs_dir / "images" / name), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION), (w, h), interpolation=cv2.INTER_AREA)
        rgb, valid = render(K, pose4(im), w, h)
        rendered = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        diff = np.abs(photo.astype(np.float32) - rendered).mean(2)
        mse = (diff[valid] ** 2).mean()
        stats[name] = {"cobertura_pct": 100 * valid.mean(), "erro_abs_medio": float(diff[valid].mean()),
                       "psnr_db": float(10 * np.log10(255 ** 2 / mse))}
        heat = cv2.applyColorMap(np.clip(diff * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        heat[~valid] = 0
        cv2.imwrite(str(figures / f"foto_vs_render_{Path(name).stem}.jpg"), np.hstack([photo, rendered, heat]))
    return stats


def splat(points, colors, K, T, w, h, background=235):
    X = points @ T[:3, :3].T + T[:3, 3]
    front = X[:, 2] > 1
    uv = (X[front] @ K.T)[:, :2] / X[front, 2:]
    u, v = uv[:, 0].astype(int), uv[:, 1].astype(int)
    inside = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
    order = np.argsort(-X[front, 2][inside])  # far to near, so nearer points are written last
    u, v, c = u[inside][order], v[inside][order], colors[front][inside][order]
    image = np.full((h, w, 3), background, np.uint8)
    for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
        image[v + dv, u + du] = c
    return image


def draw_cameras(image, rec, K, T):
    for im in rec.images.values():
        f = frustum(im, 25.0)
        X = f.vertices @ T[:3, :3].T + T[:3, 3]
        if (X[:, 2] < 1).any():
            continue
        uv = ((X @ K.T)[:, :2] / X[:, 2:]).astype(int)
        color = tuple(int(c) for c in RING_COLORS.get(ring(im.name), (120, 120, 120))[::-1])
        for i, j in ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)):
            cv2.line(image, tuple(uv[i]), tuple(uv[j]), color, 1, cv2.LINE_AA)
    return image


def navigation_video(render, rec, dense, center, radius, path):
    # Three acts, one orbit each: sparse cloud with the cameras, dense cloud, textured mesh.
    w, h = VIDEO_SIZE
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1.0]])
    sparse = np.array([p.xyz for p in rec.points3D.values()]), np.array([p.color for p in rec.points3D.values()])
    reach = np.percentile([np.linalg.norm(im.projection_center() - center) for im in rec.images.values()], 90)
    lift = 0.4 * np.median([im.projection_center()[2] for im in rec.images.values()])  # aim between object and rings
    acts = [("nuvem esparsa e cameras", 2.4 * reach, lift), ("nuvem densa", 2.6 * radius, 0.0), ("malha texturizada", 2.6 * radius, 0.0)]
    n = VIDEO_FPS * VIDEO_SECONDS
    writer = imageio_ffmpeg.write_frames(str(path), (w, h), fps=VIDEO_FPS, codec="libx264", quality=5, macro_block_size=8)
    writer.send(None)
    for k in range(n):
        t = 3 * k / n
        act = min(int(t), 2)
        blend = np.clip((t - act) * 3, 0, 1) if act else 1.0  # ease the zoom over the first third of each act
        ease = 3 * blend ** 2 - 2 * blend ** 3
        dist, z = (np.array(acts[act - 1][1:]) * (1 - ease) + np.array(acts[act][1:]) * ease) if act else acts[0][1:]
        target = center + [0, 0, z]
        az = 2 * np.pi * t
        el = np.deg2rad(38 + 18 * np.sin(2 * np.pi * t / 3))
        eye = target + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        T = look_at(eye, target)
        if act == 0:
            frame = cv2.cvtColor(draw_cameras(cv2.cvtColor(splat(*sparse, K, T, w, h), cv2.COLOR_RGB2BGR), rec, K, T), cv2.COLOR_BGR2RGB)
        elif act == 1:
            frame = splat(*dense, K, T, w, h)
        else:
            frame, _ = render(K, T, w, h)
        frame = np.ascontiguousarray(frame)
        cv2.putText(frame, acts[act][0], (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (40, 40, 40), 2, cv2.LINE_AA)
        writer.send(frame)
    writer.close()


def portraits(render, center, radius, figures, size=(2400, 1600)):
    w, h = size
    K = np.array([[1.1 * w, 0, w / 2], [0, 1.1 * w, h / 2], [0, 0, 1.0]])

    def shot(az, el=30, dist=2.2):
        el, az = np.radians(el), np.radians(az)
        eye = center + dist * radius * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        return cv2.cvtColor(render(K, look_at(eye, center), w, h)[0], cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(figures / "modelo_3d.jpg"), shot(-60), [cv2.IMWRITE_JPEG_QUALITY, 92])
    views = [cv2.resize(shot(az, dist=2.6), (w // 2, h // 2), interpolation=cv2.INTER_AREA) for az in (0, 90, 180, 270)]
    cv2.imwrite(str(figures / "modelo_3d_4vistas.jpg"), np.vstack([np.hstack(views[:2]), np.hstack(views[2:])]), [cv2.IMWRITE_JPEG_QUALITY, 92])


def dense_points(path, center, radius, limit=2_000_000):
    cloud = o3d.io.read_point_cloud(str(path))
    pts, colors = np.asarray(cloud.points), (np.asarray(cloud.colors) * 255).astype(np.uint8)
    near = np.linalg.norm(pts - center, axis=1) < 1.5 * radius
    idx = np.flatnonzero(near)
    idx = np.random.default_rng(0).permutation(idx)[:limit]
    return pts[idx], colors[idx]


def main(data, out):
    figures = out / "figuras"
    figures.mkdir(exist_ok=True)
    rec = pycolmap.Reconstruction(out / "roi")
    plot_cameras(rec, figures / "cameras.png")
    textured = out / "mvs/scene_texture.obj"
    export_glb(rec, textured, out / "cena.glb")
    render = Renderer(textured)
    stats = photo_vs_render(render, pycolmap.Reconstruction(out / "mvs/sparse"), out / "mvs", figures)
    (out / "foto_vs_render.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    obj = np.asarray(o3d.io.read_triangle_mesh(str(out / "objeto.ply")).vertices)
    center = np.r_[(obj[:, :2].min(0) + obj[:, :2].max(0)) / 2, obj[:, 2].max() / 2]
    radius = np.linalg.norm(np.ptp(obj, 0))
    portraits(render, center, radius, figures)
    navigation_video(render, rec, dense_points(out / "mvs/scene_dense.ply", center, radius), center, radius, out / "navegacao.mp4")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
