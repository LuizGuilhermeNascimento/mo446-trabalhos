"""Pipeline SfM/MVS do COLMAP e escala métrica baseada no papel quadriculado."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


def resolve_colmap(executable: str | Path | None = None) -> Path:
    """Resolve COLMAP CLI or the Windows COLMAP.bat launcher."""
    if executable:
        path = Path(executable).expanduser()
        if path.is_file():
            return path.resolve()
        found = shutil.which(str(executable))
        if found:
            return Path(found).resolve()
        raise FileNotFoundError(f"COLMAP executable not found: {executable}")
    found = shutil.which("colmap") or shutil.which("COLMAP.bat")
    if found:
        return Path(found).resolve()
    raise FileNotFoundError(
        "COLMAP is not installed or not on PATH. Set COLMAP_EXECUTABLE to colmap.exe "
        "or the Windows COLMAP.bat launcher."
    )


def _command_prefix(executable: Path) -> list[str]:
    if executable.suffix.lower() == ".bat":
        return ["cmd.exe", "/c", "call", str(executable)]
    return [str(executable)]


def _run_colmap(executable: Path, command: str, args: Sequence[str], log_path: Path) -> None:
    full_command = _command_prefix(executable) + [command, *map(str, args)]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write("\n$ " + subprocess.list2cmdline(full_command) + "\n")
        log.flush()
        result = subprocess.run(full_command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"COLMAP command '{command}' failed with exit code {result.returncode}. "
            f"See log: {log_path}"
        )


def _gpu_option_args(
    executable: Path,
    command: str,
    option_prefixes: Sequence[str],
    gpu_index: str,
) -> list[str]:
    """Choose GPU option names supported by this COLMAP CLI version."""
    help_result = subprocess.run(
        _command_prefix(executable) + [command, "-h"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    help_text = help_result.stdout or ""
    for prefix in option_prefixes:
        use_gpu_option = f"--{prefix}.use_gpu"
        gpu_index_option = f"--{prefix}.gpu_index"
        if use_gpu_option in help_text and gpu_index_option in help_text:
            return [use_gpu_option, "1", gpu_index_option, gpu_index]
    raise RuntimeError(
        f"Could not find supported GPU options for COLMAP command '{command}'. "
        f"Check whether this COLMAP build supports GPU processing."
    )


def _parse_images_text(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    parsed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        image_name = " ".join(parts[9:])
        observations = np.empty((0, 3), dtype=np.float64)
        if index < len(lines):
            values = lines[index].strip().split()
            index += 1
            if values:
                raw = np.asarray(values, dtype=np.float64)
                if raw.size % 3:
                    raise ValueError(f"Invalid point observations in {path} for {image_name}")
                observations = raw.reshape(-1, 3)
        parsed[image_name] = (np.asarray(parts[:1], dtype=np.int64), observations)
    return parsed


def _read_points3d_text(path: Path) -> dict[int, np.ndarray]:
    points: dict[int, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 8:
            points[int(fields[0])] = np.asarray(fields[1:4], dtype=np.float64)
    return points


def estimate_metric_scale_from_grid(
    text_model: str | Path,
    image_name: str,
    grid_image_points: np.ndarray,
    *,
    grid_size: tuple[int, int],
    square_size_m: float = 0.005,
    max_reprojection_distance_px: float = 5.0,
    min_grid_edges: int = 4,
) -> dict[str, Any]:
    """Estima a escala do COLMAP usando pontos 3D nas interseções da grade."""
    model = Path(text_model)
    image_rows = _parse_images_text(model / "images.txt")
    point3d = _read_points3d_text(model / "points3D.txt")
    if image_name not in image_rows:
        raise ValueError(f"Calibration view {image_name!r} was not registered by COLMAP.")
    _, observations = image_rows[image_name]
    if not len(observations):
        raise ValueError(f"No registered feature observations in {image_name}.")

    detected_xy = observations[:, :2]
    point_ids = observations[:, 2].astype(np.int64)
    grid_xy = np.asarray(grid_image_points, dtype=np.float64).reshape(-1, 2)
    grid_to_3d: list[np.ndarray | None] = []
    nearest_errors: list[float | None] = []
    for target in grid_xy:
        distances = np.linalg.norm(detected_xy - target, axis=1)
        order = np.argsort(distances)
        match = next((int(i) for i in order if point_ids[i] > 0 and int(point_ids[i]) in point3d), None)
        if match is None or distances[match] > max_reprojection_distance_px:
            grid_to_3d.append(None)
            nearest_errors.append(None)
        else:
            grid_to_3d.append(point3d[int(point_ids[match])])
            nearest_errors.append(float(distances[match]))

    columns, rows = grid_size
    edge_distances = []
    used_pairs = []
    for row in range(rows):
        for col in range(columns):
            idx = row * columns + col
            for neighbor in ([idx + 1] if col + 1 < columns else []) + ([idx + columns] if row + 1 < rows else []):
                a, b = grid_to_3d[idx], grid_to_3d[neighbor]
                if a is None or b is None or np.array_equal(a, b):
                    continue
                distance = float(np.linalg.norm(a - b))
                if distance > 1e-9:
                    edge_distances.append(distance)
                    used_pairs.append((idx, neighbor))
    if len(edge_distances) < min_grid_edges:
        raise RuntimeError(
            f"Only {len(edge_distances)} adjacent 5 mm grid edges map to COLMAP 3D points; "
            f"need at least {min_grid_edges}. The grid may lack stable SIFT tracks in {image_name}."
        )

    edge_distances = np.asarray(edge_distances)
    candidate_scales = square_size_m / edge_distances
    scale = float(np.median(candidate_scales))
    relative_mad = float(np.median(np.abs(candidate_scales - scale)) / max(scale, 1e-12))
    if relative_mad > 0.25:
        raise RuntimeError(
            f"Grid scale estimates are inconsistent (relative MAD={relative_mad:.1%}); "
            "inspect the paper flatness and grid-point overlays before applying a metric scale."
        )
    return {
        "scale_factor": scale,
        "grid_edge_count": int(len(edge_distances)),
        "relative_mad": relative_mad,
        "median_reconstructed_grid_edge": float(np.median(edge_distances)),
        "median_grid_reprojection_distance_px": float(np.nanmedian(
            [value for value in nearest_errors if value is not None]
        )) if any(value is not None for value in nearest_errors) else None,
        "grid_to_3d": grid_to_3d,
        "nearest_errors_px": nearest_errors,
        "used_pairs": used_pairs,
    }


def _write_scaled_textured_obj(mesh, texture_path: Path, output_obj: Path, scale: float) -> None:
    """Grava a geometria escalada e preserva as coordenadas UV e a textura."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    uv = np.asarray(mesh.triangle_uvs, dtype=np.float64)
    if len(uv) != len(triangles) * 3:
        raise RuntimeError("Textured COLMAP mesh has no per-corner UV coordinates.")
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    texture_copy = output_obj.parent / texture_path.name
    if texture_path.resolve() != texture_copy.resolve():
        shutil.copy2(texture_path, texture_copy)
    mtl = output_obj.with_suffix(".mtl")
    mtl.write_text(
        f"newmtl colmap_texture\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\nd 1.0\nillum 2\nmap_Kd {texture_copy.name}\n",
        encoding="utf-8",
    )
    with output_obj.open("w", encoding="utf-8") as obj:
        obj.write(f"mtllib {mtl.name}\nusemtl colmap_texture\n")
        for x, y, z in vertices:
            obj.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for u, v in uv:
            obj.write(f"vt {u:.9g} {1.0-v:.9g}\n")
        for tri_idx, tri in enumerate(triangles):
            corners = []
            for local in range(3):
                vi = int(tri[local]) + 1
                ti = tri_idx * 3 + local + 1
                corners.append(f"{vi}/{ti}")
            obj.write("f " + " ".join(corners) + "\n")


def _render_mesh(mesh, texture_path: Path | None, output_path: Path) -> None:
    """Try an Open3D textured offscreen render, with a shaded-mesh fallback."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    rendered = False
    try:
        import open3d as o3d
        renderer = o3d.visualization.rendering.OffscreenRenderer(960, 720)
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"
        if texture_path and texture_path.is_file():
            material.albedo_img = o3d.io.read_image(str(texture_path))
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        renderer.scene.add_geometry("reconstruction", tensor_mesh, material)
        bounds = mesh.get_axis_aligned_bounding_box()
        center = bounds.get_center()
        extent = np.linalg.norm(bounds.get_extent())
        eye = center + np.asarray([1.5, -2.0, 1.4]) * max(extent, 1e-3)
        renderer.setup_camera(55.0, center, eye, np.asarray([0.0, 0.0, 1.0]))
        image = renderer.render_to_image()
        o3d.io.write_image(str(output_path), image)
        renderer.release()
        rendered = output_path.is_file()
    except Exception:
        rendered = False
    if rendered:
        return

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(triangles) > 12000:
        chosen = np.linspace(0, len(triangles) - 1, 12000).astype(int)
        triangles = triangles[chosen]
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    facecolors = None
    colors = np.asarray(mesh.vertex_colors)
    if len(colors) == len(vertices):
        facecolors = colors[triangles].mean(axis=1)
    collection = Poly3DCollection(vertices[triangles], facecolors=facecolors, linewidths=0)
    if facecolors is None:
        collection.set_facecolor((0.72, 0.74, 0.78, 1.0))
    ax.add_collection3d(collection)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    center = (low + high) / 2
    radius = max(float((high - low).max()) / 2, 1e-3)
    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)
    ax.view_init(elev=24, azim=42)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_etapa4_colmap(
    image_paths: Sequence[str | Path],
    *,
    output_dir: str | Path,
    colmap_executable: str | Path | None = None,
    calibration: dict[str, Any] | None = None,
    scale_view_index: int = 0,
    grid_size: tuple[int, int] = (5, 5),
    square_size_m: float = 0.005,
    max_image_size: int = 1600,
    sequential_overlap: int = 10,
    gpu_index: str = "-1",
) -> dict[str, Any]:
    """Executa SfM/MVS e grava os produtos da reconstrução em escala métrica."""
    executable = resolve_colmap(colmap_executable)
    import open3d as o3d
    paths = [Path(path).resolve() for path in image_paths]
    frame_tag = f"{len(paths)}quadros"
    if len(paths) < 5:
        raise ValueError("COLMAP reconstruction needs at least five input frames.")
    if calibration is None or "image_points" not in calibration:
        raise ValueError("A per-capture grid calibration, including image_points, is required for scale.")
    if not 0 <= scale_view_index < len(calibration["image_paths"]):
        raise ValueError("scale_view_index does not select one of the calibrated grid views.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / f"imagens_sfm_{frame_tag}"
    image_dir.mkdir(parents=True, exist_ok=True)
    for source in paths:
        target = image_dir / source.name
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            shutil.copy2(source, target)

    db_path = output / f"banco_dados_{frame_tag}.db"
    sparse_dir = output / f"modelo_esparso_{frame_tag}"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    dense_dir = output / f"nuvem_densa_{frame_tag}"
    log_path = output / f"registro_colmap_{frame_tag}.log"
    camera_matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(calibration["dist_coeffs"], dtype=np.float64).ravel()[:4]
    distortion = np.pad(distortion, (0, max(0, 4 - len(distortion))))
    camera_params = ",".join(map(str, [
        camera_matrix[0, 0], camera_matrix[1, 1],
        camera_matrix[0, 2] + 0.5, camera_matrix[1, 2] + 0.5,
        *distortion.tolist(),
    ]))
    feature_gpu_args = _gpu_option_args(
        executable, "feature_extractor",
        ("FeatureExtraction", "SiftExtraction"), gpu_index,
    )
    matching_gpu_args = _gpu_option_args(
        executable, "sequential_matcher",
        ("FeatureMatching", "SiftMatching"), gpu_index,
    )
    feature_args = [
        "--database_path", db_path, "--image_path", image_dir,
        "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.camera_params", camera_params,
        *feature_gpu_args,
    ]
    try:
        _run_colmap(executable, "feature_extractor", feature_args, log_path)
    except RuntimeError as error:
        # COLMAP on Windows can terminate inside its GPU SIFT code without
        # writing a diagnostic (STATUS_STACK_BUFFER_OVERRUN / 0xC0000409).
        # Retry on CPU in a fresh database so an interrupted GPU run cannot
        # leave partially written feature tables behind.
        if "3221226505" not in str(error) and "0xC0000409" not in str(error):
            raise
        cpu_db_path = output / f"banco_dados_cpu_{frame_tag}.db"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write(
                "\nCOLMAP encerrou inesperadamente durante a extração SIFT na GPU "
                "(código 0xC0000409); repetindo a extração na CPU.\n"
            )
        cpu_feature_args = list(feature_args)
        cpu_feature_args[cpu_feature_args.index("--database_path") + 1] = cpu_db_path
        use_gpu_index = next(
            (i for i, value in enumerate(cpu_feature_args) if str(value).endswith(".use_gpu")),
            None,
        )
        if use_gpu_index is None:
            raise
        cpu_feature_args[use_gpu_index + 1] = "0"
        _run_colmap(executable, "feature_extractor", cpu_feature_args, log_path)
        db_path = cpu_db_path
    _run_colmap(executable, "sequential_matcher", [
        "--database_path", db_path, "--SequentialMatching.overlap", str(sequential_overlap),
        *matching_gpu_args,
    ], log_path)
    _run_colmap(executable, "mapper", [
        "--database_path", db_path, "--image_path", image_dir, "--output_path", sparse_dir,
    ], log_path)

    sparse_models = sorted(
        [p for p in sparse_dir.iterdir() if p.is_dir() and (p / "images.bin").is_file()],
        key=lambda p: p.name,
    )
    if not sparse_models:
        raise RuntimeError(f"COLMAP did not produce a sparse model. Inspect {log_path}")
    sparse_text_models = output / f"modelos_texto_esparsos_{frame_tag}"
    sparse_text_models.mkdir(parents=True, exist_ok=True)
    converted_models = []
    for candidate in sparse_models:
        candidate_text = sparse_text_models / candidate.name
        _run_colmap(executable, "model_converter", [
            "--input_path", candidate, "--output_path", candidate_text, "--output_type", "TXT",
        ], log_path)
        converted_models.append((len(_parse_images_text(candidate_text / "images.txt")), candidate, candidate_text))
    _, sparse_model, sparse_text = max(converted_models, key=lambda row: row[0])
    _run_colmap(executable, "model_converter", [
        "--input_path", sparse_model, "--output_path", output / f"nuvem_esparsa_{frame_tag}.ply", "--output_type", "PLY",
    ], log_path)
    _run_colmap(executable, "image_undistorter", [
        "--image_path", image_dir, "--input_path", sparse_model, "--output_path", dense_dir,
        "--output_type", "COLMAP", "--max_image_size", str(max_image_size),
    ], log_path)
    _run_colmap(executable, "patch_match_stereo", [
        "--workspace_path", dense_dir, "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
        "--PatchMatchStereo.max_image_size", str(max_image_size),
        "--PatchMatchStereo.gpu_index", gpu_index,
    ], log_path)
    _run_colmap(executable, "stereo_fusion", [
        "--workspace_path", dense_dir, "--workspace_format", "COLMAP",
        "--input_type", "geometric", "--output_path", dense_dir / "fused.ply",
        "--StereoFusion.max_image_size", str(max_image_size),
    ], log_path)
    _run_colmap(executable, "patch_match_stereo", [
        "--workspace_path", dense_dir, "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "false",
        "--PatchMatchStereo.max_image_size", str(max_image_size),
        "--PatchMatchStereo.gpu_index", gpu_index,
    ], log_path)
    _run_colmap(executable, "stereo_fusion", [
        "--workspace_path", dense_dir, "--workspace_format", "COLMAP",
        "--input_type", "photometric", "--output_path", dense_dir / "fused_photometric.ply",
        "--StereoFusion.max_image_size", str(max_image_size),
    ], log_path)
    _run_colmap(executable, "poisson_mesher", [
        "--input_path", dense_dir / "fused.ply", "--output_path", dense_dir / "meshed-poisson.ply",
    ], log_path)
    textured_dir = dense_dir / f"malha_texturizada_{frame_tag}"
    _run_colmap(executable, "mesh_texturer", [
        "--workspace_path", dense_dir, "--input_path", dense_dir / "meshed-poisson.ply",
        "--output_path", textured_dir,
    ], log_path)

    image_name = Path(calibration["image_paths"][scale_view_index]).name
    grid_points = np.asarray(calibration["image_points"][scale_view_index])
    try:
        scale_info = estimate_metric_scale_from_grid(
            sparse_text, image_name, grid_points, grid_size=grid_size, square_size_m=square_size_m
        )
    except Exception as exc:
        (output / f"falha_escala_metrica_{frame_tag}.txt").write_text(
            f"SfM/MVS products were saved without metric scaling.\n{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        raise
    scale = scale_info["scale_factor"]
    scale_info_path = output / f"escala_metrica_{frame_tag}.json"
    scale_info_path.write_text(json.dumps({k: v for k, v in scale_info.items() if k not in ("grid_to_3d", "nearest_errors_px", "used_pairs")}, indent=2), encoding="utf-8")

    sparse_cloud = o3d.io.read_point_cloud(str(output / f"nuvem_esparsa_{frame_tag}.ply"))
    sparse_cloud.scale(scale, center=(0, 0, 0))
    o3d.io.write_point_cloud(str(output / f"nuvem_esparsa_metrica_{frame_tag}.ply"), sparse_cloud)
    fused = o3d.io.read_point_cloud(str(dense_dir / "fused.ply"))
    fused.scale(scale, center=(0, 0, 0))
    o3d.io.write_point_cloud(str(dense_dir / f"nuvem_densa_geometrica_metrica_{frame_tag}.ply"), fused)
    fused_photometric = o3d.io.read_point_cloud(str(dense_dir / "fused_photometric.ply"))
    fused_photometric.scale(scale, center=(0, 0, 0))
    o3d.io.write_point_cloud(str(dense_dir / f"nuvem_densa_fotometrica_metrica_{frame_tag}.ply"), fused_photometric)
    mesh = o3d.io.read_triangle_mesh(str(dense_dir / "meshed-poisson.ply"), enable_post_processing=True)
    if mesh.is_empty():
        raise RuntimeError("Could not load the COLMAP Poisson mesh for metric scaling.")
    mesh.scale(scale, center=(0, 0, 0))
    o3d.io.write_triangle_mesh(str(dense_dir / f"malha_poisson_metrica_{frame_tag}.ply"), mesh)
    textured_mesh_path = textured_dir / "mesh.ply"
    texture_path = textured_dir / "texture.png"
    textured_mesh = o3d.io.read_triangle_mesh(str(textured_mesh_path), enable_post_processing=True)
    if textured_mesh.is_empty():
        raise RuntimeError(f"Could not read COLMAP textured mesh: {textured_mesh_path}")
    metric_obj = textured_dir / f"malha_texturizada_metrica_{frame_tag}.obj"
    _write_scaled_textured_obj(textured_mesh, texture_path, metric_obj, scale)
    preview_path = output / f"render_malha_{frame_tag}.png"
    _render_mesh(textured_mesh, texture_path, preview_path)

    source_path = Path(calibration["image_paths"][scale_view_index])
    source_image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    render_image = cv2.imread(str(preview_path), cv2.IMREAD_COLOR)
    if source_image is not None and render_image is not None:
        height = 640
        source_image = cv2.resize(source_image, (round(source_image.shape[1] * height / source_image.shape[0]), height))
        render_image = cv2.resize(render_image, (round(render_image.shape[1] * height / render_image.shape[0]), height))
        gap = np.full((height, 32, 3), 245, dtype=np.uint8)
        side_by_side = np.hstack([source_image, gap, render_image])
        cv2.putText(side_by_side, "Foto original", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2)
        cv2.putText(side_by_side, f"Malha COLMAP, escala x{scale:.6g}", (source_image.shape[1] + 45, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
        cv2.imwrite(str(output / f"foto_e_render_lado_a_lado_{frame_tag}.png"), side_by_side)

    report = {
        "frame_tag": frame_tag,
        "registered_images": len(_parse_images_text(sparse_text / "images.txt")),
        "sparse_points": len(_read_points3d_text(sparse_text / "points3D.txt")),
        "dense_points": len(fused.points),
        "dense_points_photometric": len(fused_photometric.points),
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
        "metric_scale_factor": scale,
        "grid_edges_used": scale_info["grid_edge_count"],
        "grid_scale_relative_mad": scale_info["relative_mad"],
        "sparse_model": str(sparse_model),
        "sparse_text_model": str(sparse_text),
        "sparse_metric_cloud": str(output / f"nuvem_esparsa_metrica_{frame_tag}.ply"),
        "dense_workspace": str(dense_dir),
        "dense_geometric_cloud": str(dense_dir / f"nuvem_densa_geometrica_metrica_{frame_tag}.ply"),
        "dense_photometric_cloud": str(dense_dir / f"nuvem_densa_fotometrica_metrica_{frame_tag}.ply"),
        "mesh_metric": str(dense_dir / f"malha_poisson_metrica_{frame_tag}.ply"),
        "textured_metric_mesh": str(metric_obj),
        "render": str(preview_path),
        "side_by_side": str(output / f"foto_e_render_lado_a_lado_{frame_tag}.png"),
        "source_image": str(source_path),
    }
    (output / f"relatorio_etapa4_{frame_tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
