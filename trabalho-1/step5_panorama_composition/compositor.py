from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from common import log
from step4_homography_connection.alignment import AlignedImages

DEGHOST = ("seam", "none")
BLENDS = ("feather", "multiband", "none")
FEATHER_SIGMA = 15  # default: px of smoothing across the seam (small keeps moving objects sharp)
MULTIBAND_LEVELS = 5  # default number of pyramid levels
SEAM_BLUR = 5  # default: px of blur of the difference image (seam keeps away from object borders)
OUTSIDE_OVERLAP_COST = 1e4  # keeps the seam inside the overlap
CROP_TO_CONTENT = True  # cut the panorama to the largest rectangle without empty (black) pixels
RECTANGLE_SCALE = 4  # the rectangle search runs on the mask downscaled 4x (speed)


@dataclass
class CompositionResult:
    """Everything already cropped to `rect` (x, y, w, h in canvas coordinates)."""

    panorama: np.ndarray  # final BGR panorama
    naive: np.ndarray  # plain average (reference "without deghosting")
    labels: np.ndarray | None  # entry of `aligned` that supplies each pixel (-1 = empty); None without seams
    mask: np.ndarray  # bool, pixels covered by at least one image
    aligned: AlignedImages  # aligned images cropped to the same rectangle (used by the metrics)
    rect: tuple[int, int, int, int]
    time_ms: float


def largest_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Largest axis-aligned rectangle (x, y, w, h) whose pixels are all covered: the classic
    'maximal rectangle in a histogram', row by row, on the mask downscaled by RECTANGLE_SCALE."""
    h, w = mask.shape
    s = RECTANGLE_SCALE
    small = cv2.resize(mask.astype(np.uint8) * 255, (w // s, h // s), interpolation=cv2.INTER_AREA) == 255
    heights = np.zeros(small.shape[1], np.int64)
    best = (0, 0, 0, 0, 0)  # area, x0, y0, x1, y1 (small scale)
    for r, row in enumerate(small):
        heights = np.where(row, heights + 1, 0)
        stack: list[tuple[int, int]] = []  # (start column, height)
        for c, height in enumerate([*heights, 0]):
            start = c
            while stack and stack[-1][1] >= height:
                start, top = stack.pop()
                if top * (c - start) > best[0]:
                    best = (top * (c - start), start, r - top + 1, c, r + 1)
            stack.append((start, height))
    _, x0, y0, x1, y1 = best
    return x0 * s, y0 * s, (x1 - x0) * s, (y1 - y0) * s


def crop_aligned(aligned: AlignedImages, rect: tuple[int, int, int, int]) -> AlignedImages:
    x, y, w, h = rect
    shift = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]], dtype=np.float64)
    return AlignedImages([img[y:y + h, x:x + w] for img in aligned.warped],
                         [m[y:y + h, x:x + w] for m in aligned.masks], aligned.indices,
                         {k: shift @ T for k, T in aligned.transforms.items()}, aligned.parents, (w, h))


def naive_composite(aligned: AlignedImages) -> np.ndarray:
    total = np.zeros((*aligned.masks[0].shape, 3), np.float32)
    count = np.zeros(aligned.masks[0].shape, np.float32)
    for image, mask in zip(aligned.warped, aligned.masks):
        valid = mask > 0
        total[valid] += image[valid]
        count[valid] += 1
    return (total / np.maximum(count, 1)[..., None]).astype(np.uint8)


def min_cost_seam(cost: np.ndarray) -> np.ndarray:
    """Top-to-bottom path of minimum accumulated cost (one column per row, 8-connected)."""
    h, w = cost.shape
    acc = cost.astype(np.float64).copy()
    step = np.zeros((h, w), np.int64)
    for r in range(1, h):
        prev = acc[r - 1]
        options = np.stack([np.r_[np.inf, prev[:-1]], prev, np.r_[prev[1:], np.inf]])  # from col-1, col, col+1
        best = options.argmin(axis=0)
        acc[r] += options[best, np.arange(w)]
        step[r] = best - 1
    seam = np.zeros(h, np.int64)
    seam[-1] = int(acc[-1].argmin())
    for r in range(h - 1, 0, -1):
        seam[r - 1] = seam[r] + step[r, seam[r]]
    return seam


def centroid(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()])


def seam_labels(aligned: AlignedImages) -> np.ndarray:
    """Adds the images in composition order; each new image enters the mosaic through the optimal seam
    of its overlap. Returns, for every canvas pixel, the entry that supplies it (-1 = empty)."""
    labels = np.full(aligned.masks[0].shape, -1, np.int32)
    mosaic = np.zeros_like(aligned.warped[0])
    position = {k: n for n, k in enumerate(aligned.indices)}
    for n, (image, mask) in enumerate(zip(aligned.warped, aligned.masks)):
        new = mask > 0
        covered = labels >= 0
        overlap = new & covered
        take_new = new & ~covered  # outside the overlap the new image is the only source
        if overlap.any():
            ys, xs = np.nonzero(overlap)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            diff = np.abs(mosaic[y0:y1, x0:x1].astype(np.float32) - image[y0:y1, x0:x1]).sum(axis=2)
            cost = cv2.GaussianBlur(diff, (0, 0), SEAM_BLUR)
            cost[~overlap[y0:y1, x0:x1]] = OUTSIDE_OVERLAP_COST

            # Seam direction from the position of the new image relative to its tree neighbor.
            parent = aligned.parents[aligned.indices[n]]
            dx, dy = centroid(new) - centroid(aligned.masks[position[parent]] > 0)
            horizontal = abs(dx) >= abs(dy)
            grid = cost if horizontal else cost.T
            seam = min_cost_seam(grid)
            columns = np.arange(grid.shape[1])[None, :]
            new_side = columns > seam[:, None] if (dx if horizontal else dy) > 0 else columns < seam[:, None]
            if not horizontal:
                new_side = new_side.T
            window = np.zeros_like(new)
            window[y0:y1, x0:x1] = new_side
            take_new |= overlap & window
        labels[take_new] = n
        mosaic[take_new] = image[take_new]
    return labels


def weighted_average(aligned: AlignedImages, weights: list[np.ndarray]) -> np.ndarray:
    total = np.zeros((*aligned.masks[0].shape, 3), np.float32)
    weight_sum = np.zeros(aligned.masks[0].shape, np.float32)
    for image, weight in zip(aligned.warped, weights):
        total += image * weight[..., None]
        weight_sum += weight
    return (total / np.maximum(weight_sum, 1e-6)[..., None]).clip(0, 255).astype(np.uint8)


def feather_blend(aligned: AlignedImages, labels: np.ndarray | None) -> np.ndarray:
    """With seams: the one-hot label maps are blurred, so only a narrow band around each seam mixes.
    Without seams: classic feathering, weights grow with the distance to the image border."""
    weights = []
    for n, mask in enumerate(aligned.masks):
        valid = (mask > 0).astype(np.float32)
        if labels is None:
            weights.append(cv2.distanceTransform(mask, cv2.DIST_L2, 5) * valid)
        else:
            weights.append(cv2.GaussianBlur((labels == n).astype(np.float32), (0, 0), FEATHER_SIGMA) * valid)
    return weighted_average(aligned, weights)


def multiband_blend(aligned: AlignedImages, labels: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """Laplacian pyramid of every image blended with the Gaussian pyramid of its label mask: low
    frequencies mix over a wide band, details only right at the seam. Empty pixels of each image are
    filled with `fill` (feathered panorama) so the pyramids do not pull black borders into the result."""
    h, w = labels.shape
    result = None
    weight_levels = None
    for n, (image, mask) in enumerate(zip(aligned.warped, aligned.masks)):
        source = np.where(mask[..., None] > 0, image, fill).astype(np.float32)
        weight = (labels == n).astype(np.float32)
        gauss_img, gauss_w = [source], [weight]
        for _ in range(MULTIBAND_LEVELS):
            gauss_img.append(cv2.pyrDown(gauss_img[-1]))
            gauss_w.append(cv2.pyrDown(gauss_w[-1]))
        laplace = [gauss_img[i] - cv2.pyrUp(gauss_img[i + 1], dstsize=gauss_img[i].shape[1::-1])
                   for i in range(MULTIBAND_LEVELS)] + [gauss_img[-1]]
        blended = [lap * gw[..., None] for lap, gw in zip(laplace, gauss_w)]
        result = blended if result is None else [r + b for r, b in zip(result, blended)]
        weight_levels = gauss_w if weight_levels is None else [a + b for a, b in zip(weight_levels, gauss_w)]
    levels = [r / np.maximum(wl, 1e-6)[..., None] for r, wl in zip(result, weight_levels)]
    panorama = levels[-1]
    for level in reversed(levels[:-1]):
        panorama = cv2.pyrUp(panorama, dstsize=level.shape[1::-1]) + level
    return panorama[:h, :w].clip(0, 255).astype(np.uint8)


class PanoramaCompositor:
    """deghost: 'seam' (optimal seams) or 'none'; blend: 'feather', 'multiband' or 'none'."""

    def __init__(self, deghost: str = "seam", blend: str = "feather"):
        if deghost not in DEGHOST or blend not in BLENDS:
            raise ValueError(f"Opções válidas: deghost {DEGHOST}, blend {BLENDS}")
        if deghost == "none" and blend == "multiband":
            raise ValueError("O blending multibanda precisa das costuras (use --deghost seam)")
        self.deghost, self.blend = deghost, blend

    @property
    def name(self) -> str:
        return f"{self.deghost}_{self.blend}"

    def compose(self, aligned: AlignedImages) -> CompositionResult:
        start = time.perf_counter()
        naive = naive_composite(aligned)
        mask = np.any([m > 0 for m in aligned.masks], axis=0)
        labels = seam_labels(aligned) if self.deghost == "seam" else None

        if self.blend == "none":
            panorama = naive if labels is None else weighted_average(
                aligned, [(labels == n).astype(np.float32) for n in range(len(aligned.masks))])
        elif self.blend == "feather":
            panorama = feather_blend(aligned, labels)
        else:
            panorama = multiband_blend(aligned, labels, feather_blend(aligned, labels))
        panorama[~mask] = 0
        H, W = mask.shape
        x, y, w, h = largest_rectangle(mask) if CROP_TO_CONTENT else (0, 0, W, H)
        cut = (slice(y, y + h), slice(x, x + w))
        time_ms = (time.perf_counter() - start) * 1000
        log("Passo 5", f"composição {self.deghost} + {self.blend}: {time_ms:.0f} ms; recorte de {W}x{H} para {w}x{h} px")
        return CompositionResult(panorama[cut], naive[cut], None if labels is None else labels[cut], mask[cut],
                                 crop_aligned(aligned, (x, y, w, h)), (x, y, w, h), time_ms)
