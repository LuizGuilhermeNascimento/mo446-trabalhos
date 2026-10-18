"""Camera calibration from manually selected rectangular patches of graph paper.

The four clicks in each image must mark the corners of the same sized grid
patch (same number of 5 mm intervals). The paper must be locally flat.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


def _select_patch_corners(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read calibration image: {image_path}")
    height, width = image.shape[:2]
    scale = min(1200 / width, 760 / height, 1.0)
    display_width, display_height = round(width * scale), round(height * scale)
    scale_x, scale_y = display_width / width, display_height / height
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb).resize((display_width, display_height), Image.Resampling.LANCZOS)

    window = tk.Tk()
    window.title(f"Calibração do grid — {image_path.name}")
    window.resizable(False, False)
    instruction = tk.Label(
        window,
        text="Marque 4 cantos do mesmo retângulo: superior esquerdo, superior direito, inferior direito, inferior esquerdo.",
        wraplength=display_width,
    )
    instruction.pack(padx=8, pady=6)
    canvas = tk.Canvas(window, width=display_width, height=display_height, cursor="crosshair")
    photo = ImageTk.PhotoImage(pil_image)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.pack(padx=8, pady=4)
    selected: list[tuple[float, float]] = []
    marker_ids: list[int] = []
    status = tk.Label(window, text="0/4 pontos selecionados")
    status.pack(pady=4)
    confirm = tk.Button(window, text="Confirmar pontos", state=tk.DISABLED)

    def redraw_markers() -> None:
        for marker_id in marker_ids:
            canvas.delete(marker_id)
        marker_ids.clear()
        for number, (x, y) in enumerate(selected, start=1):
            display_x, display_y = x * scale_x, y * scale_y
            marker_ids.append(canvas.create_line(display_x-6, display_y, display_x+6, display_y,
                                                  fill="#ff00ff", width=1))
            marker_ids.append(canvas.create_line(display_x, display_y-6, display_x, display_y+6,
                                                  fill="#ff00ff", width=1))
            marker_ids.append(canvas.create_text(display_x+9, display_y-9, text=str(number),
                                                  fill="yellow", font=("Arial", 9, "bold")))
        status.config(text=f"{len(selected)}/4 pontos selecionados")
        confirm.config(state=tk.NORMAL if len(selected) == 4 else tk.DISABLED)

    def on_click(event) -> None:
        if len(selected) < 4:
            selected.append((event.x / scale_x, event.y / scale_y))
            redraw_markers()

    def undo() -> None:
        if selected:
            selected.pop()
            redraw_markers()

    def clear() -> None:
        selected.clear()
        redraw_markers()

    def finish() -> None:
        window.destroy()

    canvas.bind("<Button-1>", on_click)
    controls = tk.Frame(window)
    controls.pack(pady=6)
    tk.Button(controls, text="Desfazer último", command=undo).pack(side=tk.LEFT, padx=4)
    tk.Button(controls, text="Limpar", command=clear).pack(side=tk.LEFT, padx=4)
    confirm.config(command=finish)
    confirm.pack(pady=4)
    window.mainloop()
    if len(selected) != 4:
        raise RuntimeError(f"Seleção cancelada para {image_path.name}; eram necessários quatro pontos.")
    return np.asarray(selected, dtype=np.float32)


def _grid_points_from_patch(
    image_path: Path,
    corners_px: np.ndarray,
    grid_size: tuple[int, int],
) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read calibration image: {image_path}")
    columns, rows = grid_size
    ideal_corners = np.asarray(
        [[0, 0], [columns - 1, 0], [columns - 1, rows - 1], [0, rows - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(ideal_corners, corners_px)
    lattice = np.asarray(
        [(x, y) for y in range(rows) for x in range(columns)], dtype=np.float32
    ).reshape(-1, 1, 2)
    predicted = cv2.perspectiveTransform(lattice, homography)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    refined = cv2.cornerSubPix(
        gray,
        predicted.astype(np.float32),
        winSize=(5, 5),
        zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01),
    )
    return refined.reshape(-1, 2)


def calibrar_camera_papel_quadriculado(
    image_paths: list[str | Path],
    *,
    grid_size: tuple[int, int] = (5, 5),
    square_size_m: float = 0.005,
) -> dict[str, Any]:
    """Calibrate intrinsics from several views of a locally flat graph-paper patch.

    ``grid_size`` counts grid intersections, including the four clicked corners.
    In every image, click the corners of a patch spanning exactly
    ``(columns - 1) x (rows - 1)`` squares.
    """
    paths = [Path(path) for path in image_paths]
    columns, rows = grid_size
    if len(paths) < 5:
        raise ValueError("Use at least five differently tilted calibration views.")
    if columns < 3 or rows < 3 or square_size_m <= 0:
        raise ValueError("grid_size must be >= (3, 3) and square_size_m must be positive.")

    first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(f"Could not read calibration image: {paths[0]}")
    image_size = (first.shape[1], first.shape[0])
    object_grid = np.asarray(
        [(x * square_size_m, y * square_size_m, 0.0)
         for y in range(rows) for x in range(columns)],
        dtype=np.float32,
    )
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    overlays: list[np.ndarray] = []

    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read calibration image: {path}")
        if (image.shape[1], image.shape[0]) != image_size:
            raise ValueError("Calibration images must all have the same pixel dimensions.")
        corners = _select_patch_corners(path)
        points = _grid_points_from_patch(path, corners, grid_size)
        object_points.append(object_grid.copy())
        image_points.append(points.reshape(-1, 1, 2))

        overlay = image.copy()
        for point in points:
            cv2.circle(overlay, tuple(np.round(point).astype(int)), 5, (0, 0, 255), -1)
        cv2.polylines(overlay, [np.round(corners).astype(np.int32)], True, (255, 80, 0), 2)
        overlays.append(overlay)

    rms, camera_matrix, dist_coeffs, rotations, translations = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    per_view_rms = []
    for obj, img, rotation, translation in zip(
        object_points, image_points, rotations, translations
    ):
        projected, _ = cv2.projectPoints(obj, rotation, translation, camera_matrix, dist_coeffs)
        error = cv2.norm(img, projected, cv2.NORM_L2) / np.sqrt(len(obj))
        per_view_rms.append(float(error))

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "rms_px": float(rms),
        "per_view_rms_px": per_view_rms,
        "image_size": image_size,
        "grid_size": grid_size,
        "square_size_m": square_size_m,
        "image_paths": paths,
        "image_points": [points.reshape(-1, 2) for points in image_points],
        "overlays": overlays,
    }
