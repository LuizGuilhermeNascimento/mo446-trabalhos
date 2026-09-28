from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle

from common import (IMAGE_SUPTITLE, IMAGE_TITLE, INK_SECONDARY, SERIES_COLORS, ImageSet, log, print_table,
                    save_figure, write_csv)

STAGE = Path(__file__).resolve().parent.name
METHODS = ("SIFT", "ORB", "AKAZE")
DETECTOR_COLORS = dict(zip(METHODS, SERIES_COLORS))  # bar colors of each detector in the charts
KEYPOINT_COLOR = "yellow"  # color of the keypoint marks drawn over the images (any matplotlib color)
MAX_DRAW = 300  # strongest keypoints drawn with scale circle and orientation segment
GRID = 8  # grid size of the spatial distribution metrics


@dataclass
class Features:
    """Keypoints and descriptors of one image."""

    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray  # one row per keypoint
    method: str
    time_ms: float  # detection + description time

    @property
    def is_binary(self) -> bool:
        """ORB/AKAZE descriptors are bit strings (Hamming distance); SIFT is float (L2)."""
        return self.descriptors.dtype == np.uint8

    @property
    def points(self) -> np.ndarray:
        return np.float32([kp.pt for kp in self.keypoints]).reshape(-1, 2)

    def __len__(self) -> int:
        return len(self.keypoints)


class FeatureDetector:
    """Detects keypoints and computes descriptors with SIFT, ORB or AKAZE.

    At most `max_features` keypoints are kept, the strongest by response, with the
    same rule for every method so that the comparison between them is fair.
    """

    def __init__(self, method: str = "SIFT", max_features: int = 3000) -> None:
        self.method = method.upper()
        if self.method not in METHODS:
            raise ValueError(f"Detector desconhecido: {method}. Opções: {', '.join(METHODS)}")
        self.max_features = max_features
        if self.method == "SIFT":
            self._backend = cv2.SIFT_create(nfeatures=max_features)
        elif self.method == "ORB":
            self._backend = cv2.ORB_create(nfeatures=max_features)
        else:
            self._backend = cv2.AKAZE_create()  # has no count parameter: trimmed in detect()

    def detect(self, image: np.ndarray) -> Features:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        start = time.perf_counter()
        keypoints, descriptors = self._backend.detectAndCompute(gray, None)
        time_ms = (time.perf_counter() - start) * 1000

        if descriptors is None:  # textureless image: no keypoints at all
            dtype = np.float32 if self.method == "SIFT" else np.uint8
            return Features([], np.empty((0, self._backend.descriptorSize()), dtype), self.method, time_ms)
        if len(keypoints) > self.max_features:
            strongest = np.argsort([-kp.response for kp in keypoints])[: self.max_features]
            keypoints, descriptors = [keypoints[i] for i in strongest], descriptors[strongest]
        return Features(list(keypoints), descriptors, self.method, time_ms)

    def detect_all(self, images: list[np.ndarray]) -> list[Features]:
        self.detect(images[0])  # warm-up: the first call pays OpenCV's lazy initialization, not timed
        features = [self.detect(image) for image in images]
        mean_count = np.mean([len(f) for f in features])
        mean_time = np.mean([f.time_ms for f in features])
        log("Passo 1", f"{self.method}: {mean_count:.0f} keypoints por imagem, {mean_time:.0f} ms por imagem (média)")
        return features


def spatial_distribution(features: Features, shape: tuple) -> tuple[float, float]:
    """Coverage (% of the GRID x GRID cells holding a keypoint) and uniformity
    (normalized entropy of the keypoint count per cell: 1 = perfectly even)."""
    h, w = shape[:2]
    pts = features.points
    if len(pts) == 0:
        return 0.0, 0.0
    rows = np.minimum(pts[:, 1] * GRID // h, GRID - 1)
    cols = np.minimum(pts[:, 0] * GRID // w, GRID - 1)
    counts = np.bincount((rows * GRID + cols).astype(int), minlength=GRID * GRID)
    p = counts[counts > 0] / counts.sum()
    return 100 * np.count_nonzero(counts) / GRID ** 2, float(-(p * np.log(p)).sum() / np.log(GRID ** 2))


def describe_descriptor(features: Features) -> str:
    size = features.descriptors.shape[1]
    return f"{size * 8} bits (Hamming)" if features.is_binary else f"{size} floats (L2)"


def detection_rows(image_set: ImageSet, features: list[Features]) -> list[dict]:
    """One row of statistics per image."""
    rows = []
    for name, image, feats in zip(image_set.names, image_set.images, features):
        coverage, uniformity = spatial_distribution(feats, image.shape)
        rows.append({"detector": feats.method, "imagem": name, "keypoints": len(feats), "tempo_ms": feats.time_ms,
                     "cobertura_%": coverage, "uniformidade": uniformity,
                     "diametro_medio_px": float(np.mean([kp.size for kp in feats.keypoints])) if len(feats) else 0.0})
    return rows


STAT_KEYS = ("keypoints", "tempo_ms", "cobertura_%", "uniformidade")


def compare_detectors(image_set: ImageSet, out_dir: Path, max_features: int = 3000,
                      methods: tuple[str, ...] = METHODS) -> dict[str, list[Features]]:
    """Runs every detector on every image; prints and saves statistics, draws the comparison figures.
    Returns the features of every detector (reused by the next steps)."""
    features = {m: FeatureDetector(m, max_features).detect_all(image_set.images) for m in methods}
    per_image = {m: detection_rows(image_set, feats) for m, feats in features.items()}

    summary = []
    for method, rows in per_image.items():
        row = {"detector": method}
        for key in STAT_KEYS:
            values = [r[key] for r in rows]
            row[f"{key}_media"], row[f"{key}_desvio"] = float(np.mean(values)), float(np.std(values))
        row["diametro_medio_px"] = float(np.mean([r["diametro_medio_px"] for r in rows]))
        row["descritor"] = describe_descriptor(features[method][0])
        summary.append(row)

    print_table([{"detector": r["detector"],
                  "keypoints": f"{r['keypoints_media']:.0f} ± {r['keypoints_desvio']:.0f}",
                  "tempo (ms)": f"{r['tempo_ms_media']:.1f} ± {r['tempo_ms_desvio']:.1f}",
                  "cobertura (%)": f"{r['cobertura_%_media']:.0f} ± {r['cobertura_%_desvio']:.0f}",
                  "uniformidade": f"{r['uniformidade_media']:.2f} ± {r['uniformidade_desvio']:.2f}",
                  "diâmetro médio (px)": f"{r['diametro_medio_px']:.1f}",
                  "descritor": r["descritor"]} for r in summary],
                f"Passo 1: comparação entre detectores (média ± desvio padrão sobre {len(image_set)} imagens, "
                f"limite de {max_features} keypoints)")

    write_csv(summary, out_dir / "comparacao_detectores.csv")
    write_csv([r for rows in per_image.values() for r in rows], out_dir / "comparacao_detectores_por_imagem.csv")
    plot_metrics(summary, len(image_set), out_dir / "comparacao_detectores.png")
    for i, name in enumerate(image_set.names):
        plot_detectors_on_image(name, image_set.images[i], [features[m][i] for m in methods],
                                [per_image[m][i] for m in methods], out_dir / f"comparacao_detectores_{name}.jpg")
    log("Passo 1", f"estudo comparativo salvo em {out_dir}")
    return features


def draw_keypoints(ax: plt.Axes, image: np.ndarray, features: Features) -> None:
    """Every keypoint as a dot; the MAX_DRAW strongest also get a circle (radius = scale) and a
    segment (orientation), as in Figure 2 of the assignment. A thin dark halo keeps the marks
    visible over bright regions."""
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pts = features.points
    ax.scatter(pts[:, 0], pts[:, 1], s=3, c=KEYPOINT_COLOR, edgecolors="black", linewidths=0.2)

    strongest = sorted(features.keypoints, key=lambda k: -k.response)[:MAX_DRAW]
    radii = [max(kp.size / 2, 3.0) for kp in strongest]
    segments = [[kp.pt, (kp.pt[0] + r * np.cos(np.deg2rad(kp.angle)), kp.pt[1] + r * np.sin(np.deg2rad(kp.angle)))]
                for kp, r in zip(strongest, radii) if kp.angle >= 0]
    for color, width in (("black", 2.0), (KEYPOINT_COLOR, 1.0)):  # halo first, then the colored mark
        circles = [Circle(kp.pt, r) for kp, r in zip(strongest, radii)]
        ax.add_collection(PatchCollection(circles, facecolor="none", edgecolor=color, linewidth=width))
        ax.add_collection(LineCollection(segments, colors=color, linewidths=width))
    ax.axis("off")


def plot_detectors_on_image(name: str, image: np.ndarray, features: list[Features], rows: list[dict],
                            path: Path) -> None:
    h, w = image.shape[:2]
    panel_w = 7.0
    fig, axes = plt.subplots(1, len(features), figsize=(panel_w * len(features), panel_w * h / w + 1.3))
    for ax, feats, row in zip(np.atleast_1d(axes), features, rows):
        draw_keypoints(ax, image, feats)
        ax.set_title(f"{feats.method}: {len(feats)} keypoints · {feats.time_ms:.0f} ms\n"
                     f"cobertura {row['cobertura_%']:.0f}% · uniformidade {row['uniformidade']:.2f} · "
                     f"diâmetro médio {row['diametro_medio_px']:.1f} px", fontsize=IMAGE_TITLE)
    fig.suptitle(f"Imagem {name}: todos os keypoints (pontos) e os {MAX_DRAW} mais fortes "
                 f"(círculo = escala, segmento = orientação)", fontsize=IMAGE_SUPTITLE)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))  # room for the suptitle
    save_figure(fig, path)


def plot_metrics(summary: list[dict], n_images: int, path: Path) -> None:
    """Small multiples, one metric per panel (different scales never share an axis)."""
    panels = [("keypoints", "Keypoints por imagem", "{:.0f}"),
              ("tempo_ms", "Tempo de detecção + descrição (ms)", "{:.1f}"),
              ("cobertura_%", f"Cobertura da grade {GRID}×{GRID} (%)", "{:.0f}"),
              ("uniformidade", "Uniformidade (entropia normalizada)", "{:.2f}")]
    names = [row["detector"] for row in summary]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.8))
    for ax, (key, title, fmt) in zip(axes, panels):
        means = [row[f"{key}_media"] for row in summary]
        stds = [row[f"{key}_desvio"] for row in summary]
        ax.bar(names, means, yerr=stds, color=[DETECTOR_COLORS[n] for n in names], width=0.6, capsize=4,
               error_kw={"elinewidth": 1, "ecolor": INK_SECONDARY})
        top = max(m + s for m, s in zip(means, stds))
        for x, (m, s) in enumerate(zip(means, stds)):
            ax.text(x, m + s + 0.02 * top, fmt.format(m), ha="center", va="bottom", fontsize=10, color=INK_SECONDARY)
        ax.set_ylim(0, top * 1.15)
        ax.set_title(title, fontsize=11)
    fig.suptitle(f"Comparação entre detectores: média ± desvio padrão sobre {n_images} imagens", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))  # room for the suptitle
    save_figure(fig, path)
