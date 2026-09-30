from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from common import (IMAGE_SUPTITLE, IMAGE_TEXT, IMAGE_TITLE, INK_SECONDARY, ImageSet, log, pair_name,
                    plot_metric_bars, print_table, save_figure, write_csv)
from step1_key_points_detection.detector import DETECTOR_COLORS, METHODS as DETECTORS, Features

STAGE = Path(__file__).resolve().parent.name
RATIO = 0.8  # Lowe's ratio threshold, as proposed by Lowe (2004) and widely used
RATIOS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)  # thresholds of the ratio test curve
RANSAC_THRESHOLD = 3.0  # px; OpenCV's default ransacReprojThreshold. Only used here to *evaluate* matches

CANDIDATE_COLOR = "yellow"  # matches before the ratio test
ACCEPTED_COLOR = "lime"  # matches accepted by the ratio test (solid lines)
REJECTED_COLOR = "red"  # matches discarded by the ratio test (dashed lines)
MAX_LINES = 150  # accepted matches drawn per detector in the detector comparison figure
BEFORE_AFTER_LINES = 80  # candidates drawn in the before / after figures (the same ones in both panels)


@dataclass
class MatchResult:
    """Matches from image A (query) to image B (train)."""

    knn: list[tuple[cv2.DMatch, ...]]  # the 2 nearest neighbors in B of each descriptor of A
    good: list[cv2.DMatch]  # accepted by the ratio test
    rejected: list[cv2.DMatch]  # nearest neighbor discarded by the ratio test
    ratio: float
    time_ms: float  # knnMatch time

    @property
    def candidates(self) -> list[cv2.DMatch]:
        """Nearest neighbor of every descriptor: the matches *before* the ratio test."""
        return [pair[0] for pair in self.knn if pair]

    def swapped(self) -> MatchResult:
        """Same matches seen from image B to image A (used to draw a pair in panorama order)."""
        flip = lambda ms: [cv2.DMatch(m.trainIdx, m.queryIdx, m.distance) for m in ms]  # noqa: E731
        return MatchResult([tuple(flip(pair)) for pair in self.knn], flip(self.good), flip(self.rejected),
                           self.ratio, self.time_ms)


def ratio_test(knn: list[tuple[cv2.DMatch, ...]], ratio: float) -> tuple[list, list]:
    """Lowe: keep m only if d(m) < ratio * d(second neighbor). Without a second neighbor it is rejected."""
    good, rejected = [], []
    for pair in knn:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append(pair[0])
        elif pair:
            rejected.append(pair[0])
    return good, rejected


class FeatureMatcher:
    """Brute Force matcher: L2 norm for float descriptors (SIFT), Hamming for binary ones (ORB, AKAZE)."""

    def __init__(self, ratio: float = RATIO):
        self.ratio = ratio

    def match(self, feat_a: Features, feat_b: Features) -> MatchResult:
        if len(feat_a) < 2 or len(feat_b) < 2:
            return MatchResult([], [], [], self.ratio, 0.0)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING if feat_a.is_binary else cv2.NORM_L2)
        start = time.perf_counter()
        knn = matcher.knnMatch(feat_a.descriptors, feat_b.descriptors, k=2)
        time_ms = (time.perf_counter() - start) * 1000
        good, rejected = ratio_test(knn, self.ratio)
        return MatchResult(list(knn), good, rejected, self.ratio, time_ms)


def geometric_inliers(feat_a: Features, feat_b: Features, matches: list[cv2.DMatch]) -> np.ndarray:
    """Boolean mask of the matches consistent with a RANSAC homography (evaluation only)."""
    if len(matches) < 4:
        return np.zeros(len(matches), bool)
    pts_a = feat_a.points[[m.queryIdx for m in matches]]
    pts_b = feat_b.points[[m.trainIdx for m in matches]]
    _, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, RANSAC_THRESHOLD)
    return np.zeros(len(matches), bool) if mask is None else mask.ravel().astype(bool)


def pair_stats(names: list[str], i: int, j: int, feat_a: Features, feat_b: Features, result: MatchResult) -> dict:
    inliers = int(geometric_inliers(feat_a, feat_b, result.good).sum())
    accepted = len(result.good)
    return {"detector": feat_a.method, "par": f"{names[i]} x {names[j]}",
            "candidatos": len(result.candidates), "aceitos": accepted,
            "taxa_aceitacao_%": 100 * accepted / max(len(result.candidates), 1),
            "inliers": inliers, "precisao_%": 100 * inliers / max(accepted, 1), "tempo_ms": result.time_ms}


def summarize(detector: str, rows: dict, overlapping: list, separate: list) -> dict:
    """Mean and standard deviation over the overlapping pairs; false matches over the separate pairs."""
    summary = {"detector": detector}
    for key in ("tempo_ms", "candidatos", "aceitos", "taxa_aceitacao_%", "inliers", "precisao_%"):
        values = [rows[p][key] for p in overlapping] or [0.0]
        summary[f"{key}_media"], summary[f"{key}_desvio"] = float(np.mean(values)), float(np.std(values))
    false = [rows[p]["aceitos"] for p in separate] or [0.0]
    summary["falsos_aceitos_media"], summary["falsos_aceitos_desvio"] = float(np.mean(false)), float(np.std(false))
    return summary


def ratio_curve(features: dict, results: dict, overlapping: list) -> list[dict]:
    """Re-filters the same kNN matches with several thresholds (requirement 3.2: choosing the threshold)."""
    rows = []
    for d in DETECTORS:
        for r in RATIOS:
            accepted, precision = [], []
            for i, j in overlapping:
                good, _ = ratio_test(results[d][i, j].knn, r)
                accepted.append(len(good))
                precision.append(100 * geometric_inliers(features[d][i], features[d][j], good).sum() / max(len(good), 1))
            rows.append({"detector": d, "ratio": r, "aceitos": float(np.mean(accepted)) if accepted else 0.0,
                         "precisao_%": float(np.mean(precision)) if precision else 0.0})
    return rows


def compare_matching(image_set: ImageSet, out_dir: Path, features: dict[str, list[Features]],
                     overlapping: list[tuple[int, int]], order: list[int], ratio: float = RATIO,
                     detector: str = "SIFT", comparison_pair: tuple[str, str] | None = None) -> list[dict]:
    """Matches every pair of images with Brute Force for every detector and compares them.

    `features` holds the keypoints of every detector (step 1); `overlapping` (pairs that overlap) and
    `order` (inferred sequence) come from step 3. Pairs are drawn in panorama order (left image first).
    `detector` is the one drawn in the per-pair figures; `comparison_pair` (two image names) is the pair
    seen by every detector (default or when absent from the group: the pair with the median inliers).
    """
    names, group = image_set.names, image_set.group
    pairs = list(itertools.combinations(range(len(image_set)), 2))
    matcher = FeatureMatcher(ratio)

    results, rows = {}, {}
    for d in DETECTORS:
        matcher.match(features[d][0], features[d][1])  # warm-up, not timed
        results[d] = {(i, j): matcher.match(features[d][i], features[d][j]) for i, j in pairs}
        rows[d] = {(i, j): pair_stats(names, i, j, features[d][i], features[d][j], r) for (i, j), r in results[d].items()}
        log("Passo 2", f"{d} + Brute Force: {len(pairs)} pares emparelhados")
    separate = [p for p in pairs if p not in overlapping]
    log("Passo 2", f"{len(overlapping)} pares com sobreposição e {len(separate)} sem (grafo do passo 3)")

    summary = [summarize(d, rows[d], overlapping, separate) for d in DETECTORS]
    pm = lambda r, key, f: f"{r[key + '_media']:{f}} ± {r[key + '_desvio']:{f}}"  # noqa: E731
    print_table([{"detector": r["detector"], "tempo (ms)": pm(r, "tempo_ms", ".1f"), "aceitos": pm(r, "aceitos", ".0f"),
                  "aceitação (%)": pm(r, "taxa_aceitacao_%", ".1f"), "inliers": pm(r, "inliers", ".0f"),
                  "precisão (%)": pm(r, "precisao_%", ".1f"), "falsos aceitos": pm(r, "falsos_aceitos", ".1f")}
                 for r in summary],
                f"Passo 2: emparelhamento com Brute Force, ratio test {ratio} (média ± desvio sobre {len(overlapping)} "
                f"pares com sobreposição; falsos aceitos sobre {len(separate)} pares sem sobreposição)")
    curve = ratio_curve(features, results, overlapping)
    print_table([r for r in curve if r["detector"] == detector],
                f"Passo 2: curva do ratio test ({detector}, média sobre os pares com sobreposição)")

    write_csv(summary, out_dir / "comparacao_emparelhamento.csv")
    write_csv([r for d in DETECTORS for r in rows[d].values()], out_dir / "comparacao_emparelhamento_por_par.csv")
    write_csv(curve, out_dir / "curva_ratio_test.csv")
    plot_metrics(summary, group, out_dir / "comparacao_emparelhamento.png")
    plot_ratio_curve(curve, ratio, group, out_dir / "curva_ratio_test.png")

    position = {k: n for n, k in enumerate(order)}  # images outside the sequence (intruders) go last

    def drawn(d: str, a: int, b: int) -> tuple[MatchResult, dict]:
        """Result and statistics of the pair oriented from image a (left) to image b (right)."""
        key = (min(a, b), max(a, b))
        result, stats = results[d][key], dict(rows[d][key], par=f"{names[a]} x {names[b]}")
        return (result if a < b else result.swapped()), stats

    def in_order(pair: tuple[int, int]) -> tuple[int, int]:
        return tuple(sorted(pair, key=lambda k: position.get(k, len(order))))

    feats = features[detector]
    for a, b in zip(order, order[1:]):  # requirement 3.3: before / after the ratio test along the sequence
        plot_before_after(image_set, a, b, feats[a], feats[b], *drawn(detector, a, b),
                          out_dir / f"matches_{pair_name(names, a, b)}.jpg")
    if overlapping:  # one pair seen by every detector: the chosen one or the median number of inliers
        if comparison_pair and set(comparison_pair) <= set(names):
            a, b = in_order(tuple(names.index(n) for n in comparison_pair))
        else:
            a, b = in_order(sorted(overlapping, key=lambda p: rows[detector][p]["inliers"])[len(overlapping) // 2])
        plot_detectors(image_set, a, b, features, {d: drawn(d, a, b) for d in DETECTORS}, ratio,
                       out_dir / f"comparacao_emparelhamento_{pair_name(names, a, b)}.jpg")
    if separate:  # requirement 3.4: false matches of the non-overlapping pair with most accepted matches
        a, b = in_order(max(separate, key=lambda p: rows[detector][p]["aceitos"]))
        plot_before_after(image_set, a, b, feats[a], feats[b], *drawn(detector, a, b),
                          out_dir / f"matches_sem_sobreposicao_{pair_name(names, a, b)}.jpg", overlapping_pair=False)
    log("Passo 2", f"estudo comparativo salvo em {out_dir}")
    return summary


def pair_canvas(img_a: np.ndarray, img_b: np.ndarray, gap: int = 20) -> tuple[np.ndarray, int]:
    """Both images side by side (RGB) and the x offset of image B."""
    h = max(img_a.shape[0], img_b.shape[0])
    canvas = np.full((h, img_a.shape[1] + gap + img_b.shape[1], 3), 255, np.uint8)
    canvas[: img_a.shape[0], : img_a.shape[1]] = img_a
    canvas[: img_b.shape[0], img_a.shape[1] + gap:] = img_b
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), img_a.shape[1] + gap


def sample(matches: list, n: int = MAX_LINES) -> list:
    if len(matches) <= n:
        return matches
    return [matches[k] for k in np.random.default_rng(0).choice(len(matches), n, replace=False)]


def draw_lines(ax: plt.Axes, feat_a: Features, feat_b: Features, matches: list, offset: int, color: str,
               style: str = "-") -> None:
    if not matches:
        return
    pa = feat_a.points[[m.queryIdx for m in matches]]
    pb = feat_b.points[[m.trainIdx for m in matches]] + [offset, 0]
    ax.add_collection(LineCollection(np.stack([pa, pb], axis=1), colors=color, linestyles=style, linewidths=0.9),
                      autolim=False)
    ax.scatter(np.r_[pa[:, 0], pb[:, 0]], np.r_[pa[:, 1], pb[:, 1]], s=6, c=color, edgecolors="black",
               linewidths=0.2, zorder=3)


def show_canvas(ax: plt.Axes, canvas: np.ndarray) -> None:
    ax.imshow(canvas)
    ax.set_xlim(-0.5, canvas.shape[1] - 0.5)  # lines never enlarge the panel
    ax.set_ylim(canvas.shape[0] - 0.5, -0.5)
    ax.axis("off")


def pair_title(stats: dict) -> str:
    return f"{stats['aceitos']} aceitos ({stats['taxa_aceitacao_%']:.1f}%), {stats['inliers']} inliers " \
           f"(precisão {stats['precisao_%']:.1f}%)"


def plot_before_after(image_set: ImageSet, i: int, j: int, feat_a: Features, feat_b: Features,
                      result: MatchResult, stats: dict, path: Path, overlapping_pair: bool = True) -> None:
    """Requirement 3.3, as in Figure 3 of the assignment. The same BEFORE_AFTER_LINES candidates (random sample)
    are drawn in both panels: yellow before the ratio test; after it, each one is green (accepted, solid)
    or red (discarded, dashed)."""
    canvas, offset = pair_canvas(image_set.images[i], image_set.images[j])
    h, w = canvas.shape[:2]
    fig, axes = plt.subplots(2, 1, figsize=(14, 2 * 14 * h / w + 1.4), layout="constrained")
    show_canvas(axes[0], canvas)
    drawn = sample(result.candidates, BEFORE_AFTER_LINES)
    accepted_ids = {(m.queryIdx, m.trainIdx) for m in result.good}
    accepted = [m for m in drawn if (m.queryIdx, m.trainIdx) in accepted_ids]
    discarded = [m for m in drawn if (m.queryIdx, m.trainIdx) not in accepted_ids]
    draw_lines(axes[0], feat_a, feat_b, drawn, offset, CANDIDATE_COLOR)
    axes[0].set_title(f"Antes do ratio test: {len(result.candidates)} candidatos", fontsize=IMAGE_TITLE)
    show_canvas(axes[1], canvas)
    draw_lines(axes[1], feat_a, feat_b, discarded, offset, REJECTED_COLOR, "--")
    draw_lines(axes[1], feat_a, feat_b, accepted, offset, ACCEPTED_COLOR)
    axes[1].set_title(f"Depois do ratio test (limiar {result.ratio}): {pair_title(stats)}", fontsize=IMAGE_TITLE)
    axes[1].legend(handles=[Line2D([], [], color=ACCEPTED_COLOR, lw=2, label="aceito"),
                            Line2D([], [], color=REJECTED_COLOR, lw=2, ls="--", label="descartado")],
                   loc="upper center", bbox_to_anchor=(0.5, 0), ncols=2, fontsize=IMAGE_TEXT, frameon=False)
    kind = "Correspondências" if overlapping_pair else "Correspondências Falsas (Par sem Sobreposição)"
    fig.suptitle(f"Imagens {stats['par']}: {kind} com {feat_a.method} e Brute Force", fontsize=IMAGE_SUPTITLE)
    save_figure(fig, path)


def plot_detectors(image_set: ImageSet, i: int, j: int, features: dict, drawn: dict[str, tuple[MatchResult, dict]],
                   ratio: float, path: Path) -> None:
    """Accepted matches of every detector on the same pair (one panel per detector)."""
    canvas, offset = pair_canvas(image_set.images[i], image_set.images[j])
    h, w = canvas.shape[:2]
    fig, axes = plt.subplots(len(DETECTORS), 1, figsize=(14, len(DETECTORS) * 14 * h / w + 1.4),
                             layout="constrained")
    for ax, d in zip(axes, DETECTORS):
        show_canvas(ax, canvas)
        result, stats = drawn[d]
        draw_lines(ax, features[d][i], features[d][j], sample(result.good), offset, ACCEPTED_COLOR)
        ax.set_title(f"{d}: {pair_title(stats)}", fontsize=IMAGE_TITLE)
    fig.suptitle(f"Imagens {drawn[DETECTORS[0]][1]['par']}: Matches Aceitos por Detector (Brute Force, limiar {ratio})",
                 fontsize=IMAGE_SUPTITLE)
    save_figure(fig, path)


def plot_metrics(summary: list[dict], group: str, path: Path) -> None:
    panels = [("tempo_ms", "Tempo de Emparelhamento", "tempo por par (ms)", "{:.1f} ms", None),
              ("aceitos", "Matches Aceitos pelo Ratio Test", "matches por par", "{:.0f}", None),
              ("inliers", "Inliers RANSAC", "inliers por par", "{:.0f}", None),
              ("precisao_%", "Precisão (Inliers / Aceitos)", "precisão (%)", "{:.1f}%", 100),
              ("falsos_aceitos", "Matches em Pares sem Sobreposição", "matches aceitos por par", "{:.1f}", None)]
    plot_metric_bars(summary, "detector", DETECTOR_COLORS, panels,
                     f"Métricas de Emparelhamento com Brute Force por Detector: Conjunto {group}", path, ncols=3)


def plot_ratio_curve(curve: list[dict], ratio: float, group: str, path: Path) -> None:
    """Accepted matches and precision as the ratio threshold grows; the dashed line marks the chosen threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    panels = [("aceitos", "Matches Aceitos por Par", "matches por par"),
              ("precisao_%", "Precisão (Inliers / Aceitos)", "precisão (%)")]
    for ax, (key, title, ylabel) in zip(axes, panels):
        for d in DETECTORS:
            points = [r for r in curve if r["detector"] == d]
            ax.plot([r["ratio"] for r in points], [r[key] for r in points], color=DETECTOR_COLORS[d], lw=2,
                    marker="o", ms=5, label=d)
        ax.axvline(ratio, color=INK_SECONDARY, ls="--", lw=1.2)
        ax.set_xticks(RATIOS, [f"{r:g}" for r in RATIOS])
        ax.set_xlabel("limiar do ratio test", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="x", color="#e1e0d9", lw=0.6)
    axes[1].set_ylim(0, 105)
    handles = axes[0].get_legend_handles_labels()[0]
    handles.append(Line2D([], [], color=INK_SECONDARY, ls="--", lw=1.2, label=f"limiar escolhido ({ratio})"))
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles), fontsize=10, frameon=False)
    fig.suptitle(f"Curva do Ratio Test de Lowe (Brute Force): Conjunto {group}", fontsize=14)
    save_figure(fig, path)
