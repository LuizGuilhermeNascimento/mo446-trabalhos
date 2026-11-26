"""k vizinhos mais próximos EXATOS para o g2NN, no lugar de cv2.BFMatcher.knnMatch."""
from __future__ import annotations

import os

import numpy as np

_TORCH = None


def _torch_device():
    """torch + CUDA, se disponíveis e não desligados por CMFD_KNN=cpu; senão None."""
    global _TORCH
    if _TORCH is None:
        _TORCH = False
        if os.environ.get("CMFD_KNN", "auto") != "cpu":
            try:
                import torch
                if torch.cuda.is_available():
                    torch.backends.cuda.matmul.allow_tf32 = False
                    _TORCH = (torch, torch.device("cuda"))
            except Exception:
                _TORCH = False
    return _TORCH or None


def _candidates(q: np.ndarray, d: np.ndarray, m: int, chunk: int = 1024) -> np.ndarray:
    tb = _torch_device()
    if tb is not None:
        torch, dev = tb
        D = torch.from_numpy(np.ascontiguousarray(d, np.float32)).to(dev)
        dn = (D * D).sum(1)
        out = []
        for s in range(0, len(q), 4 * chunk):
            Q = torch.from_numpy(np.ascontiguousarray(q[s:s + 4 * chunk], np.float32)).to(dev)
            out.append(torch.topk(dn[None, :] - 2.0 * (Q @ D.T), m, dim=1, largest=False).indices.cpu().numpy())
        return np.concatenate(out).astype(np.int64)
    d32 = np.ascontiguousarray(d, np.float32)
    dn = (d32 * d32).sum(1)
    out = np.empty((len(q), m), np.int64)
    for s in range(0, len(q), chunk):
        S = dn[None, :] - 2.0 * (np.ascontiguousarray(q[s:s + chunk], np.float32) @ d32.T)
        out[s:s + chunk] = np.argpartition(S, m - 1, axis=1)[:, :m]
    return out


def _knn_gpu(q, d, k, m, torch, dev, chunk: int = 4096):
    """Os dois passos de knn() na GPU: os mesmos candidatos de _candidates e a mesma reordenação exata em float64, sem trazer os candidatos para a CPU."""
    D = torch.from_numpy(np.ascontiguousarray(d, np.float32)).to(dev)
    dn = (D * D).sum(1)
    D64 = torch.from_numpy(np.ascontiguousarray(d)).to(dev).double()
    d64c = None  # cópia em float64 na CPU: só é montada se houver empate
    I, Dk = [], []
    # lote limitado pela memória: a tabela de distâncias (lote x n, float32) e os candidatos em float64 (lote x m x dim)
    chunk = int(max(64, min(chunk, (192 << 20) // (4 * len(d)), (96 << 20) // (8 * m * d.shape[1]))))
    for s in range(0, len(q), chunk):
        qs = np.ascontiguousarray(q[s:s + chunk])
        Q = torch.from_numpy(qs.astype(np.float32, copy=False)).to(dev)
        cand = torch.topk(dn[None, :] - 2.0 * (Q @ D.T), m, dim=1, largest=False).indices
        cand = cand.sort(dim=1).values  # empate de distância -> menor índice (ordenação estável abaixo)
        diff = D64[cand] - torch.from_numpy(qs).to(dev).double()[:, None, :]
        dist = torch.sqrt((diff * diff).sum(2))
        of = torch.argsort(dist, dim=1, stable=True)[:, :min(k + 1, m)]
        ds = torch.gather(dist, 1, of)
        o = of[:, :k]
        Ic, Dc = torch.gather(cand, 1, o).cpu().numpy(), ds[:, :k].cpu().numpy()
        # Empates: descritores binários (AKAZE, ORB) têm muitas distâncias iguais, e aí a ordem sai do arredondamento
        # da soma, que não é o mesmo na GPU e na CPU. Essas linhas são refeitas com a conta da CPU, para a resposta
        # ser a mesma do caminho antigo.
        emp = ((ds[:, 1:] - ds[:, :-1]) < 1e-9).any(1).cpu().numpy() if ds.shape[1] > 1 else np.zeros(len(qs), bool)
        if emp.any():
            cc = cand[torch.from_numpy(emp).to(dev)].cpu().numpy()
            if d64c is None:
                d64c = d.astype(np.float64)
            df = d64c[cc] - qs[emp][:, None, :].astype(np.float64)
            dc = np.sqrt(np.einsum("ijk,ijk->ij", df, df))
            oc = np.argsort(dc, axis=1, kind="stable")[:, :k]
            Ic[emp], Dc[emp] = np.take_along_axis(cc, oc, 1), np.take_along_axis(dc, oc, 1)
        I.append(Ic)
        Dk.append(Dc)
    return np.concatenate(I).astype(np.int64), np.concatenate(Dk).astype(np.float32)


def knn(q: np.ndarray, d: np.ndarray, k: int, margin: int = 8, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Para cada linha de q: índices (em d) e distâncias L2 dos k mais próximos, em ordem (distância, índice)."""
    n = len(d)
    k = min(k, n)
    m = min(n, k + margin)
    tb = _torch_device()
    if tb is not None and m < n and len(q) and os.environ.get("CMFD_KNN_EXATO", "gpu") != "cpu":
        return _knn_gpu(q, d, k, m, *tb)
    cand = np.tile(np.arange(n), (len(q), 1)) if m == n else _candidates(q, d, m, chunk)
    cand.sort(axis=1)  # empate de distância -> menor índice (argsort estável abaixo)
    dist = np.empty(cand.shape, np.float64)
    d64 = d.astype(np.float64)
    for s in range(0, len(q), chunk):
        diff = d64[cand[s:s + chunk]] - q[s:s + chunk, None, :].astype(np.float64)
        dist[s:s + chunk] = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    o = np.argsort(dist, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(cand, o, 1), np.take_along_axis(dist, o, 1).astype(np.float32)
