"""Procura de cópias dentro de UMA figura, de ponta a ponta: detecção em 3 ampliações, máscara densa (ZNCC 7 + 21),
refino pelo ruído e seleção dos candidatos.
"""
from __future__ import annotations

import time
from dataclasses import replace

import cv2
import numpy as np

from . import official
from .detector import Selection, _prepare, compose, dense_masks, detect_candidates, noise_refine, noise_residual, \
    passing, residual_ncc
from .panelfilter import off_panels
from .textfilter import on_text
from .variants import VARIANTS
from . import detector as _det
from .detector import _cron

SCALES = ("f3_2048", "f3_1536", "f3_1024")  # ordem do merge_caches (campo "origem")
NOISE = dict(win=5, thr=0.5, low=0.2, grow_frac=0.5)
SELECTION = Selection(min_inliers=4, min_inlier_ratio=0.0, max_src_dst_iou=0.1, max_aniso=1.3,
                      require_valid_geom=True, min_shift_frac=0.03, min_area_frac=0.0001, instances="union",
                      deep_min_inliers=1000, min_rel_inliers=0.1, field_min_support=1.0, max_degree=99,
                      min_resid_ncc=0.6, resid_bypass_inliers=80, merge_iou=0.0, merge_frac=0.3, dedup_iou=0.7,
                      vseed_min_inliers=1000)


def _clean(stats: dict) -> dict:
    """Como detect.py grava as estatísticas (bool ou float)."""
    return {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in stats.items()}


def _could_pass(st: dict, shape, sel: Selection, **override) -> bool:
    """O candidato ainda pode passar na seleção? Usa os MESMOS filtros de passing(), com as
    estatísticas que dependem da máscara trocadas por valores que sempre passam (ou por override)."""
    st2 = dict(st, src_area=1.0, dst_area=1.0, src_dst_iou=0.0, resid_ncc=1.0)
    st2.update(override)
    return bool(passing(dict(cands=[dict(stats=st2)], shape=shape), sel))


def remask(full: np.ndarray, res: dict, variant: str, rfull: np.ndarray | None,
           sel: Selection | None = None, text: np.ndarray | None = None, panels: np.ndarray | None = None):
    """Máscara densa (ZNCC 7 + 21) e borda pelo ruído, candidato a candidato.

    sel dado: o candidato que já falhou num filtro que não depende da máscara (pares, geometria,
    deslocamento) ou na correlação do ruído é descartado aqui, sem desenhar a máscara. passing()
    o descartaria de qualquer jeito, e a ordem dos que ficam se mantém, então a anotação final é a mesma.

    text dado (máscara de texto, cmfd.textfilter): o candidato com os pontos casados em cima de texto sai.
    panels dado (máscara de painéis, cmfd.panelfilter): o candidato com os pontos casados fora dos painéis sai.
    """
    H, W = full.shape
    cfg = replace(VARIANTS[variant], mask_mode="dense", zncc_win2=21)
    gray, sc = _prepare(full, cfg)
    out, rsig, gsuave, gradx = [], None, None, None
    for c in res["cands"]:
        st = _clean(c["stats"])
        if text is not None and "S" in c and on_text(text, c["S"], c["T"], sc, shape=(H, W)):
            continue
        if panels is not None and "S" in c and off_panels(panels, c["S"], c["T"], sc, (H, W)):
            continue
        if "A" not in c:
            out.append(dict(stats=st, src=c["src"], dst=c["dst"]))
            continue
        if sel is not None and not _could_pass(st, (H, W), sel):
            continue
        md = dict(A=np.asarray(c["A"]).astype(np.float64), S=c["S"], T=c["T"])
        tc = time.time()
        src, dst = dense_masks(gray, md, cfg)
        _cron("regiao_zncc", tc)
        _det.TEMPOS["n_regioes"] = _det.TEMPOS.get("n_regioes", 0.0) + 1
        if src.shape != (H, W):
            src = cv2.resize(src, (W, H), interpolation=cv2.INTER_NEAREST)
            dst = cv2.resize(dst, (W, H), interpolation=cv2.INTER_NEAREST)
        src, dst = src.astype(bool), dst.astype(bool)
        st["src_area"], st["dst_area"] = float(src.mean()), float(dst.mean())
        st["src_dst_iou"] = float((src & dst).sum()) / max(float((src | dst).sum()), 1.0)
        An = md["A"].copy()
        An[:, 2] /= sc
        if rfull is None:
            rfull = noise_residual(full)
        rcand = rfull
        if st.get("vseed", 0.0) and cfg.vlinks_sigma != 1.0:  # sementes confirmadas na textura fina: o mesmo resíduo vale aqui
            if rsig is None:
                rsig = noise_residual(full, cfg.vlinks_sigma)
            rcand = rsig
        tc = time.time()
        pre = residual_ncc(full, An, src, r=rcand, return_offset=True)  # (correlação, deslocamento subpixel)
        _cron("regiao_ruido", tc)
        st["resid_ncc"] = pre[0]
        if sel is not None and not _could_pass(st, (H, W), sel, resid_ncc=st["resid_ncc"]):
            continue
        tc = time.time()
        src = noise_refine(full, An, src, r=rcand, pre=pre, **NOISE)
        _cron("regiao_borda", tc)
        dst = cv2.warpAffine(src.astype(np.uint8), An, (W, H), flags=cv2.INTER_NEAREST) > 0
        st["src_area"], st["dst_area"] = float(src.mean()), float(dst.mean())
        st["src_dst_iou"] = float((src & dst).sum()) / max(float((src | dst).sum()), 1.0)
        if sel is not None and sel.pico_todos:  # medidas para o verificador treinado (scripts/verif_cache.py)
            from .peak import peak_sharpness, sign_peak
            pk = sign_peak(rcand, An, src)
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
        elif sel is not None and ((st.get("vseed", 0.0) and sel.vseed_min_margin >= 0) or (st.get("tile", 0.0) and sel.tile_min_margin >= 0)):
            from .peak import sign_peak
            tc = time.time()
            pk = sign_peak(rcand, An, src)
            _cron("regiao_pico", tc)
            st["s_margem"], st["s_frac"] = pk["s_pico"] - pk["s_lado_max"], pk["s_frac"]
            if not passing(dict(cands=[dict(stats=st)], shape=(H, W)), sel):
                continue
        out.append(dict(stats=st, src=src, dst=dst, An=An))
    return out, rfull


MATCHER = "exact"  # cmfd/knn.py; "bf" = cv2.BFMatcher, o da avaliação por cache (mesma resposta, ~4x mais lento)


def detecta(full: np.ndarray, scales=SCALES, prune: bool = True, matcher: str = MATCHER) -> dict:
    """Só a detecção de candidatos de cada escala (não usa texto nem painéis), para adiantar trabalho enquanto um
    modelo auxiliar ainda carrega. O resultado entra em predict(..., pre=...), que dá a mesma resposta de sempre.
    Qualquer erro sobe (como predict com strict=True)."""
    _det.TEMPOS.clear()
    pre = {}
    for v in scales:
        t0 = time.time()
        pre[v] = (detect_candidates(full, replace(VARIANTS[v], matcher=matcher), masks=not prune), time.time() - t0)
    return pre


def predict(full: np.ndarray, scales=SCALES, sel: Selection = SELECTION, prune: bool = True,
            matcher: str = MATCHER, text=None, strict: bool = False, panels=None, return_masks: bool = False,
            pre: dict | None = None):
    """Imagem em tons de cinza (cmfd.data.load_gray) -> (anotação no formato de submissão, tempos por etapa).

    text: máscara de texto (H, W) booleana, None (filtro desligado) ou uma função sem argumentos que a
    devolve; a função só é chamada depois da primeira detecção, para o OCR poder rodar em paralelo.
    panels: máscara reduzida de painéis (cmfd.panelfilter.load_panels), None ou função, como text.
    strict: qualquer erro numa escala derruba a execução, em vez de a escala ser pulada (uso na submissão).
    return_masks: devolve a lista de máscaras no lugar da anotação (uso em testes por recorte).
    """
    H, W = full.shape
    if pre is None:  # com a detecção adiantada (detecta), os cronômetros dela continuam valendo
        _det.TEMPOS.clear()
    cands, rfull, times, errors = [], None, {}, []
    for v in scales:
        vi = list(scales).index(v)  # com as escalas padrão, é o mesmo índice de SCALES
        t0 = time.time()
        if pre is not None:  # detecção já feita por detecta(): a mesma chamada, só que antes
            res, feito = pre[v]
            t0 = time.time() - feito
        else:
            try:
                res = detect_candidates(full, replace(VARIANTS[v], matcher=matcher), masks=not prune)
            except Exception as e:  # como detect.py: a escala problemática não derruba a imagem
                if strict:
                    raise
                res = dict(cands=[])
                errors.append(f"{v}: {e!r}")
        times[f"det_{v}"] = time.time() - t0
        t0 = time.time()
        if callable(text):
            text = text()
            times["text_wait"] = time.time() - t0
            t0 = time.time()
        if callable(panels):
            panels = panels()
            times["panel_wait"] = time.time() - t0
            t0 = time.time()
        part, rfull = remask(full, res, v, rfull, sel if prune else None, text, panels)
        for c in part:
            c["stats"]["origem"] = float(vi)
        cands += part
        times[f"mask_{v}"] = time.time() - t0
    t0 = time.time()
    res = dict(cands=cands, shape=(H, W))
    masks = compose(res, passing(res, sel), sel)
    ann = masks if return_masks else official.encode_prediction(masks)
    times["compose"] = time.time() - t0
    times.update({"t_" + k: v for k, v in _det.TEMPOS.items()})
    times["errors"] = "; ".join(errors)
    return ann, times
