"""Máscara e medidas de cada candidato: correlação local (ZNCC), refino pelo ruído e medidas para o verificador."""
from __future__ import annotations

import cv2
import numpy as np

from .detector import Config, dense_masks, noise_refine, noise_residual, prepare, residual_ncc
from .panelfilter import off_panels
from .peak import peak_sharpness, sign_peak
from .textfilter import on_text

ESCALAS = (Config(2048, 32000), Config(1536, 16000), Config(1024, 8000))


def _passa(st: dict, diag: float) -> bool:
    if st["inliers"] < (2 if st.get("vseed", 0.0) else 4) or not st["valid_geom"]:
        return False
    return not (st.get("aniso", 1.0) > 2.0 or st["shift"] < 0.03 * diag)


def remask(full: np.ndarray, cands: list, cfg: Config, rfull: np.ndarray | None, text: np.ndarray | None, panels: np.ndarray | None):
    """Candidatos de detect_candidates -> [dict(stats, src, dst)] com máscara e medidas; devolve também o ruído da figura."""
    H, W = full.shape
    gray, sc = prepare(full, cfg)
    out, gsuave, gradx = [], None, None
    for c in cands:
        st = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in c["stats"].items()}
        if on_text(text, c["S"], c["T"], sc, (H, W)) or off_panels(panels, c["S"], c["T"], sc, (H, W)):
            continue
        if not _passa(st, np.hypot(H, W)):
            continue
        A = np.asarray(c["A"]).astype(np.float64)
        src, dst = dense_masks(gray, A, c["S"], cfg)
        if src.shape != (H, W):
            src = cv2.resize(src, (W, H), interpolation=cv2.INTER_NEAREST)
        An = A.copy()
        An[:, 2] /= sc
        if rfull is None:
            rfull = noise_residual(full)
        pre = residual_ncc(rfull, An, src.astype(bool))
        st["resid_ncc"] = pre[0]
        src = noise_refine(rfull, An, src, pre)
        dst = cv2.warpAffine(src.astype(np.uint8), An, (W, H), flags=cv2.INTER_NEAREST) > 0
        st["src_area"], st["dst_area"] = float(src.mean()), float(dst.mean())
        st["src_dst_iou"] = float((src & dst).sum()) / max(float((src | dst).sum()), 1.0)
        pk = sign_peak(rfull, An, src)
        st.update(s_margem=pk["s_pico"] - pk["s_lado_max"], s_frac=pk["s_frac"], s_pico=pk["s_pico"], s_lado_med=pk["s_lado_med"])
        ys, xs = np.nonzero(src)
        if len(ys):
            if gsuave is None:
                gsuave = cv2.GaussianBlur(full.astype(np.float32), (0, 0), 1.0)
                gradx = np.abs(cv2.Sobel(full, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(full, cv2.CV_32F, 0, 1))
            sub = src
            if len(ys) > 20000:
                sub = np.zeros_like(src)
                k = int(np.ceil(np.sqrt(len(ys) / 20000)))
                sub[::k, ::k] = src[::k, ::k]
            v = full[ys, xs].astype(np.float32)
            st.update(cz=peak_sharpness(gsuave, An, sub, W=2, guard=2)["pico"], tx_std=float(v.std()), tx_media=float(v.mean()),
                      tx_sat=float(((v < 8) | (v > 247)).mean()), tx_grad=float(gradx[ys, xs].mean()), area_px=float(len(ys)),
                      caixa_asp=float((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)),
                      cheio=float(len(ys) / ((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))),
                      borda=float(min(xs.min(), ys.min(), W - 1 - xs.max(), H - 1 - ys.max()) / max(H, W)))
        out.append(dict(stats=st, src=src, dst=dst))
    return out, rfull
