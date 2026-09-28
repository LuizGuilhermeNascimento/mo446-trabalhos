from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # figures are only saved to disk, never shown
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SEED = 0

# Categorical slots are assigned to entities in this fixed order, never cycled.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
# Font sizes of the (wide) figures that show images: larger than the chart defaults to stay readable.
IMAGE_SUPTITLE, IMAGE_TITLE, IMAGE_TEXT = 18, 15, 13

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "text.color": INK,
    "font.size": 10,
})


def hex_to_bgr(color: str) -> tuple[int, int, int]:
    """'#rrggbb' -> (b, g, r), for drawing with OpenCV in the same colors as the charts."""
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return b, g, r


@dataclass
class ImageSet:
    """Images of one panorama group, in the (meaningless) alphabetical order of their shuffled names."""

    group: str
    names: list[str]
    images: list[np.ndarray]  # BGR, largest side <= max_side

    def __len__(self) -> int:
        return len(self.images)


def set_seed(seed: int = SEED) -> None:
    """Makes RANSAC (OpenCV RNG) and FLANN/numpy sampling reproducible."""
    cv2.setRNGSeed(seed)
    np.random.seed(seed)


def list_image_files(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def load_image_set(folder: str | Path, max_side: int = 1024, shuffle_seed: int | None = SEED) -> ImageSet:
    """Loads every image of a group folder, resized so the largest side is at most `max_side`.

    Requirement 4.1: the list of files is shuffled (`shuffle_seed`; None keeps the alphabetical order),
    so the pipeline receives the images out of order. Names are only used as labels, and only pixel data
    is read: cv2.imread applies the EXIF orientation tag, but no timestamp or other metadata is ever used.
    """
    folder = Path(folder)
    files = list_image_files(folder)
    if not files:
        raise FileNotFoundError(f"Nenhuma imagem (.jpg, .jpeg, .png) encontrada em {folder}")
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(files)

    images = []
    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Não foi possível ler a imagem {path}")
        images.append(resize_max_side(image, max_side))

    log("Entrada", f"{len(images)} imagens carregadas de {folder} (lado maior <= {max_side} px)")
    log("Entrada", f"ordem de entrada{' (embaralhada)' if shuffle_seed is not None else ''}: "
                   f"{', '.join(p.stem for p in files)}")
    return ImageSet(group=folder.name, names=[p.stem for p in files], images=images)


def stage_output_dir(stage: str, group: str) -> Path:
    """Returns (and creates) `output/<stage>/<group>/`."""
    out = OUTPUT_DIR / stage / group
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", pil_kwargs={"quality": 92} if path.suffix == ".jpg" else None)
    plt.close(fig)


def format_value(value) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{value:.2f}"
    return str(value)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: format_value(v) for k, v in row.items()} for row in rows)


def print_table(rows: list[dict], title: str | None = None) -> None:
    """Prints rows (dicts with the same keys) as an aligned table in the terminal."""
    headers = list(rows[0])
    cells = [[format_value(row[h]) for h in headers] for row in rows]
    widths = [max(len(h), *(len(c[i]) for c in cells)) for i, h in enumerate(headers)]
    if title:
        print(f"\n{title}")
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for c in cells:
        print("  ".join(v.ljust(w) for v, w in zip(c, widths)))
    print(flush=True)


def pair_name(names: list[str], i: int, j: int) -> str:
    """File-name tag of a pair of images, independent of the (shuffled) input order."""
    return "_".join(sorted((names[i], names[j])))


def log(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", flush=True)
