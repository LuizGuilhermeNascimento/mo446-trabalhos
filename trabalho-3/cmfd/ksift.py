"""SIFT do Kornia (PyTorch), para rodar na GPU."""
from __future__ import annotations

import numpy as np

_MODELOS: dict = {}


def sift_kornia(gray: np.ndarray, max_keypoints: int, backend: str = "pyramid"):
    """Imagem uint8 (H, W) -> (pontos (N, 2) em x, y; descritores RootSIFT (N, 128)), como detector._sift."""
    import kornia.feature as KF
    import torch

    from .knn import _torch_device
    tb = _torch_device()
    dev = tb[1] if tb is not None else torch.device("cpu")
    chave = (max_keypoints, backend, str(dev))
    if chave not in _MODELOS:
        _MODELOS[chave] = KF.SIFTFeatureScaleSpace(max_keypoints, rootsift=True, device=dev, descriptor_backend=backend,
                                                   descriptor_layout="opencv").eval()
    img = torch.from_numpy(np.ascontiguousarray(gray)).to(dev).float()[None, None] / 255.0
    with torch.no_grad():
        lafs, _, des = _MODELOS[chave](img)
        ok = KF.laf_is_filled(lafs)[0]
        pts = KF.get_laf_center(lafs)[0][ok]
        des = des[0][ok]
    out = (np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)) if len(pts) == 0 else \
        (pts.float().cpu().numpy().astype(np.float32), des.float().cpu().numpy().astype(np.float32))
    del lafs, des, pts, ok, img
    if dev.type == "cuda":  # devolve a memória reservada: com vários processos na mesma placa ela enchia e tudo ficava 5x mais lento
        torch.cuda.empty_cache()
    return out
