import base64
import io
import json
import struct
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap
import trimesh
from PIL import Image

from visualizar import frustum

MARGIN_MM, MAX_FACES, TEXTURE_PX = 40, 1_000_000, 8192
VIEWER = """<!doctype html>
<html lang="pt-br"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script type="module" src="https://cdn.jsdelivr.net/npm/@google/model-viewer@3.5.0/dist/model-viewer.min.js"></script>
<style>html,body{{margin:0;height:100%;background:#eee;font-family:sans-serif}}model-viewer{{width:100%;height:100%}}
p{{position:fixed;left:16px;bottom:8px;margin:0;color:#444}}</style></head>
<body><model-viewer src="data:model/gltf-binary;base64,{glb}" camera-controls touch-action="pan-y"
camera-target="0m 0m 0m" camera-orbit="-60deg 60deg {dist}m" min-camera-orbit="auto auto {near}m"
max-camera-orbit="auto auto {far}m" field-of-view="30deg"></model-viewer>
<p>{title}: malha texturizada, câmeras (cores por altura) e nuvem esparsa, em mm. Arraste para girar, role para aproximar.</p>
</body></html>
"""


def crop_mesh(out):
    obj = np.asarray(o3d.io.read_triangle_mesh(str(out / "objeto.ply")).vertices)
    lo, hi = obj.min(0) - MARGIN_MM, obj.max(0) + MARGIN_MM
    center, size = (obj.min(0) + obj.max(0)) / 2, np.linalg.norm(np.ptp(obj, 0))
    lo[2] = -5
    mesh = o3d.io.read_triangle_mesh(str(out / "mvs/scene_refine.ply"))
    V = np.asarray(mesh.vertices)
    mesh.remove_vertices_by_mask(~np.all((V > lo) & (V < hi), 1))
    # The cut leaves non-manifold edges and stray pieces that make TextureMesh abort.
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    labels, counts, _ = mesh.cluster_connected_triangles()
    mesh.remove_triangles_by_mask(np.asarray(counts)[np.asarray(labels)] < 100)
    mesh.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(str(out / "mvs/web_mesh.ply"), mesh)
    return len(mesh.triangles), center, size


def jpeg_textures(mesh):
    # Keep the PNG atlas out of the GLB: as JPEG the texture is several times smaller.
    for geom in mesh.geometry.values() if isinstance(mesh, trimesh.Scene) else [mesh]:
        material = geom.visual.material
        image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
        buf = io.BytesIO()
        image.convert("RGB").save(buf, "JPEG", quality=95)
        geom.visual.material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.open(buf), metallicFactor=0, roughnessFactor=1)
    return mesh


def unlit(glb):
    # The texture already holds the real lighting of the photos; without KHR_materials_unlit the viewer shades it again.
    length = struct.unpack_from("<I", glb, 12)[0]
    doc = json.loads(glb[20:20 + length])
    for material in doc.get("materials", []):
        material.setdefault("extensions", {})["KHR_materials_unlit"] = {}
    doc["extensionsUsed"] = sorted(set(doc.get("extensionsUsed", [])) | {"KHR_materials_unlit"})
    chunk = json.dumps(doc, separators=(",", ":")).encode()
    chunk += b" " * (-len(chunk) % 4)
    rest = glb[20 + length:]
    return struct.pack("<III", 0x46546C67, 2, 20 + len(chunk) + len(rest)) + struct.pack("<II", len(chunk), 0x4E4F534A) + chunk + rest


def viewer(data, out, glb, size):
    (out / "modelo.html").write_text(VIEWER.format(title=f"Reconstrução 3D: {data.name}", glb=base64.b64encode(glb).decode(),
                                                   dist=2.5 * size, near=0.5 * size, far=12 * size))


def main(data, out):
    mvs = Path(os.environ.get("OPENMVS_BIN", Path.home() / ".local/openmvs/bin/OpenMVS"))
    faces, center, size = crop_mesh(out)
    subprocess.run([mvs / "TextureMesh", "-w", out / "mvs", "-i", "scene_dense.mvs", "-m", "web_mesh.ply", "-o", "web.mvs",
                    "--decimate", str(min(1.0, MAX_FACES / faces)), "--max-texture-size", str(TEXTURE_PX),
                    "--export-type", "obj"], check=True, stdout=subprocess.DEVNULL)
    scene = trimesh.Scene()
    scene.add_geometry(jpeg_textures(trimesh.load(out / "mvs/web.obj", process=False)), node_name="modelo")
    rec = pycolmap.Reconstruction(out / "roi")
    for im in rec.images.values():
        scene.add_geometry(frustum(im), node_name=im.name.replace("/", "_"))
    points = list(rec.points3D.values())
    scene.add_geometry(trimesh.PointCloud([p.xyz for p in points], colors=[list(p.color) + [255] for p in points]), node_name="esparsa")
    # The viewer orbits the origin: put the object there (not the middle of the camera rings), then go y-up as glTF expects.
    scene.apply_transform(trimesh.transformations.translation_matrix(-center))
    scene.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    glb = unlit(scene.export(file_type="glb"))
    (out / "modelo.glb").write_bytes(glb)
    viewer(data, out, glb, size)
    print(f"modelo.glb: {len(glb) / 2**20:.1f} MB, {faces} faces antes da redução")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
