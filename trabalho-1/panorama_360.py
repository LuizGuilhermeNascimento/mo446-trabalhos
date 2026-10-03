"""Panorama esférico periódico: SIFT, ajuste robusto de rotações e composição 360°."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


def estimate(files, work_side=1200):
    features, shapes = [], []
    for path in files:
        im = cv.imread(str(path))
        if im is None:
            raise ValueError(f"Imagem ilegível: {path}")
        h, w = im.shape[:2]
        im = cv.resize(im, (round(w * work_side / max(h, w)), round(h * work_side / max(h, w))))
        shapes.append(im.shape[:2])
        features.append(cv.detail.computeImageFeatures2(cv.SIFT_create(5000), im))
    matcher = cv.detail.BestOf2NearestMatcher_create(False, .3)
    matches = matcher.apply2(features)
    matcher.collectGarbage()
    kept = cv.detail.leaveBiggestComponent(features, matches, 1.0)
    if len(kept) != len(files):
        raise ValueError("As imagens não formam um único componente; verifique a coleta.")
    ok, cameras = cv.detail_HomographyBasedEstimator().apply(features, matches, None)
    if not ok:
        raise RuntimeError("Falha na estimativa inicial das câmeras")
    rotations = cv.detail.waveCorrect([c.R.astype(np.float32) for c in cameras], cv.detail.WAVE_CORRECT_HORIZ)
    rows = [dict(file=p.name, R=r.tolist(), K=c.K().tolist(), shape=list(s), focal=c.focal,
                 keypoints=len(f.keypoints), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p, c, r, s, f in zip(files, cameras, rotations, shapes, features)]
    edges = []
    for pair in matches:
        i, j = pair.src_img_idx, pair.dst_img_idx
        if i < 0 or j <= i:
            continue
        accepted = [m for m, valid in zip(pair.getMatches(), pair.getInliers()) if valid]
        edges.append(dict(source=files[i].name, target=files[j].name,
                          inliers=pair.num_inliers, matches=len(pair.getMatches()), confidence=pair.confidence,
                          points1=[features[i].keypoints[m.queryIdx].pt for m in accepted],
                          points2=[features[j].keypoints[m.trainIdx].pt for m in accepted]))
    return dict(cameras=rows, pairs=edges)


def connected(n, edges):
    seen, pending = set(), [0]
    adjacency = [set() for _ in range(n)]
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    while pending:
        i = pending.pop()
        if i not in seen:
            seen.add(i)
            pending.extend(adjacency[i] - seen)
    return len(seen) == n


def fit_rotations(cameras, pairs, max_nfev):
    index = {c['file']: i for i, c in enumerate(cameras)}
    rotations = []
    for c in cameras:
        u, _, v = np.linalg.svd(c['R'])
        u[:, -1] *= np.linalg.det(u @ v)
        rotations.append(u @ v)
    initial = Rotation.from_matrix(rotations).as_rotvec()
    focal0 = float(np.median([c['focal'] for c in cameras]))
    aa, bb, ii, jj, slices = [], [], [], [], []
    for pair in pairs:
        i, j = index[pair['source']], index[pair['target']]
        a, b = np.asarray(pair['points1']), np.asarray(pair['points2'])
        selection = np.linspace(0, len(a)-1, min(120, len(a))).astype(int)
        a = a[selection] - np.asarray(cameras[i]['shape'][::-1]) / 2
        b = b[selection] - np.asarray(cameras[j]['shape'][::-1]) / 2
        slices.append(slice(len(ii), len(ii) + len(a)))
        aa.extend(a); bb.extend(b); ii.extend([i] * len(a)); jj.extend([j] * len(a))
    if not connected(len(cameras), zip(ii, jj)):
        raise ValueError("A rejeição de pares desconectou o grafo")
    a, b, ii, jj = map(np.asarray, (aa, bb, ii, jj))

    def residual(x):
        r = Rotation.from_rotvec(np.vstack([initial[0], x[:-1].reshape(-1, 3)])).as_matrix()
        f = np.exp(x[-1])
        ra, rb = np.c_[a, np.full(len(a), f)], np.c_[b, np.full(len(b), f)]
        ra /= np.linalg.norm(ra, axis=1)[:, None]
        rb /= np.linalg.norm(rb, axis=1)[:, None]
        return (np.einsum('nij,nj->ni', r[ii], ra) - np.einsum('nij,nj->ni', r[jj], rb)) * focal0

    x0 = np.r_[initial[1:].ravel(), np.log(focal0)]
    sparsity = lil_matrix((len(a) * 3, len(x0)), dtype=int)
    for k, (i, j) in enumerate(zip(ii, jj)):
        for q in (i, j):
            if q:
                sparsity[3*k:3*k+3, 3*(q-1):3*q] = 1
        sparsity[3*k:3*k+3, -1] = 1
    lo, hi = np.full(len(x0), -np.inf), np.full(len(x0), np.inf)
    lo[-1], hi[-1] = np.log(450), np.log(1100)
    result = least_squares(lambda x: residual(x).ravel(), x0, jac_sparsity=sparsity.tocsr(),
                           loss='soft_l1', f_scale=2, bounds=(lo, hi), max_nfev=max_nfev, ftol=1e-6)
    matrices = Rotation.from_rotvec(np.vstack([initial[0], result.x[:-1].reshape(-1, 3)])).as_matrix()
    matrices = cv.detail.waveCorrect([r.astype(np.float32) for r in matrices], cv.detail.WAVE_CORRECT_HORIZ)
    focal = float(np.exp(result.x[-1]))
    errors = np.linalg.norm(residual(result.x), axis=1)
    for c, r in zip(cameras, matrices):
        c.update(R=r.tolist(), focal=focal, yaw=float(np.degrees(np.arctan2(r[0, 2], r[2, 2]))))
        c['K'][0][0] = c['K'][1][1] = focal
    metrics = [dict(**{k: e[k] for k in ('source', 'target', 'confidence', 'inliers')},
                    median_ray_error_px=float(np.median(errors[s]))) for e, s in zip(pairs, slices)]
    return metrics, dict(success=bool(result.success), message=result.message, nfev=result.nfev,
                         focal=focal, median_ray_error_px=float(np.median(errors)),
                         residual_scale_px=focal0, sampled_matches=len(errors))


def filter_rotation_matches(data):
    """RANSAC em raios calibrados elimina homografias incompatíveis com rotação."""
    cameras = {c['file']: c for c in data['cameras']}
    focal = float(np.median([c['focal'] for c in data['cameras']]))
    rng = np.random.default_rng(0)
    for edge in data['pairs']:
        if edge['confidence'] <= 1 or edge['inliers'] < 25:
            continue
        a = np.asarray(edge['points1']) - np.asarray(cameras[edge['source']]['shape'][::-1])/2
        b = np.asarray(edge['points2']) - np.asarray(cameras[edge['target']]['shape'][::-1])/2
        a, b = np.c_[a, np.full(len(a), focal)], np.c_[b, np.full(len(b), focal)]
        a /= np.linalg.norm(a, axis=1)[:, None]
        b /= np.linalg.norm(b, axis=1)[:, None]
        best = np.zeros(len(a), bool)
        for _ in range(500):
            selected = rng.choice(len(a), 3, replace=False)
            u, _, v = np.linalg.svd(a[selected].T @ b[selected])
            u[:, -1] *= np.linalg.det(u @ v)
            valid = np.linalg.norm(a @ (u @ v) - b, axis=1)*focal < 4
            if valid.sum() > best.sum():
                best = valid
        edge['homography_inliers'] = edge['inliers']
        edge['inliers'] = int(best.sum())
        edge['points1'] = np.asarray(edge['points1'])[best].tolist()
        edge['points2'] = np.asarray(edge['points2'])[best].tolist()
    return data


def optimize(data):
    candidates = [e for e in data['pairs'] if e['confidence'] > 1 and e['inliers'] >= 25]
    first, first_status = fit_rotations(data['cameras'], candidates, 150)
    # Inconsistent edges can arise from repeated facade elements or unrelated textures.
    rejected = [m for m in first if m['median_ray_error_px'] > 15]
    accepted = [p for p, m in zip(candidates, first) if m['median_ray_error_px'] <= 15]
    metrics, status = fit_rotations(data['cameras'], accepted, 400)
    if not status['success']:
        raise RuntimeError(f"Ajuste final não convergiu: {status}")
    return dict(cameras=data['cameras'], edges=metrics, rejected_edges=rejected,
                initial_optimizer=first_status, optimizer=status)


def spherical_rays(width, y0=0, y1=None):
    if y1 is None:
        y1 = width // 2
    theta = (np.arange(width, dtype=np.float32) + .5) / width * 2 * np.pi - np.pi
    phi = ((np.arange(y0, y1, dtype=np.float32) + .5) / (width / 2) - .5) * np.pi
    return np.stack(np.broadcast_arrays(np.cos(phi[:, None]) * np.sin(theta),
                    np.sin(phi[:, None]), np.cos(phi[:, None]) * np.cos(theta)), axis=-1)


def inverse_maps(camera, rays, shape):
    v = rays @ np.asarray(camera['R'], np.float32)
    h, w = camera['shape']
    f = camera['focal']
    z = v[:, :, 2]
    u = ((v[:, :, 0] / np.maximum(z, 1e-6)*f + w/2 + .5)*(shape[1]/w)-.5).astype(np.float32)
    v = ((v[:, :, 1] / np.maximum(z, 1e-6)*f + h/2 + .5)*(shape[0]/h)-.5).astype(np.float32)
    mask = ((z > 0) & (u >= 2) & (u < shape[1]-3) & (v >= 2) & (v < shape[0]-3)).astype(np.uint8)*255
    return u, v, mask


def full_coverage_band(coverage, margin=2):
    rows = np.flatnonzero(np.all(coverage > 0, axis=1))
    if not len(rows):
        raise ValueError("Não existe faixa com cobertura contínua dos 360 graus")
    band = max(np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1), key=len)
    start, stop = int(band[0]+margin), int(band[-1]+1-margin)
    if stop <= start:
        raise ValueError("Faixa de cobertura insuficiente")
    return start, stop


def render(data, source, output, width=6000):
    cameras = data['cameras']
    seam_width = 1800
    rays = spherical_rays(seam_width)
    coverage = np.zeros(rays.shape[:2], np.uint16)
    for c in cameras:
        coverage += inverse_maps(c, rays, c['shape'])[2] > 0
    y0, y1 = full_coverage_band(coverage)
    rays = spherical_rays(seam_width, y0, y1)
    images, masks = [], []
    for c in cameras:
        im = cv.imread(str(source / c['file']))
        im = cv.resize(im, tuple(c['shape'][::-1]))
        u, v, mask = inverse_maps(c, rays, im.shape)
        images.append(cv.remap(im, u, v, cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)); masks.append(mask)
    compensator = cv.detail_BlocksChannelsCompensator(64, 64, 2)
    compensator.feed([(0, 0)]*len(images), images, masks)
    images = [compensator.apply(i, (0, 0), im.copy(), mask)
              for i, (im, mask) in enumerate(zip(images, masks))]
    gain_maps = [g.tolist() for g in compensator.getMatGains()]
    tripled = [cv.UMat(np.tile(im.astype(np.float32), (1, 3, 1))) for im in images]
    seam_masks = [cv.UMat(np.tile(m, (1, 3))) for m in masks]
    cv.detail_DpSeamFinder('COLOR_GRAD').find(tripled, [(0, 0)]*len(images), seam_masks)
    seams = [m.get()[:, seam_width:2*seam_width] for m in seam_masks]
    np.savez_compressed(output/'seams.npz', **{str(i): m for i, m in enumerate(seams)})
    del tripled, seam_masks, images
    top, bottom = int(np.ceil(y0*width/seam_width)), int(np.floor(y1*width/seam_width))
    rays = spherical_rays(width, top, bottom)
    height, pad = bottom-top, 512
    blender = cv.detail_MultiBandBlender(); blender.setNumBands(8)
    blender.prepare((0, 0, width+2*pad, height))
    for i, (c, seam) in enumerate(zip(cameras, seams)):
        im = cv.imread(str(source/c['file']))
        u, v, mask = inverse_maps(c, rays, im.shape)
        warped = cv.remap(im, u, v, cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)
        warped = compensator.apply(i, (0, 0), warped, mask).astype(np.int16)
        seam = cv.resize(cv.dilate(seam, np.ones((5, 5), np.uint8)), (width, height), interpolation=cv.INTER_NEAREST)
        seam = cv.bitwise_and(seam, mask)
        blender.feed(np.pad(warped, ((0, 0), (pad, pad), (0, 0)), mode='wrap'),
                     np.pad(seam, ((0, 0), (pad, pad)), mode='wrap'), (0, 0))
    result, mask = blender.blend(None, None)
    result = np.clip(result[:, pad:pad+width], 0, 255).astype(np.uint8)
    mask = mask[:, pad:pad+width]
    if not np.all(mask > 0):
        raise ValueError("A composição deixou lacunas; saída 360 rejeitada")
    cv.imwrite(str(output/'panorama_360.jpg'), result, [cv.IMWRITE_JPEG_QUALITY, 96])
    cv.imwrite(str(output/'preview.jpg'), cv.resize(result, (1800, round(height*1800/width))))
    span = width//10
    cv.imwrite(str(output/'fechamento.jpg'), np.concatenate([result[:, -span:], result[:, :span]], axis=1))
    cv.imwrite(str(output/'mascara.png'), mask)
    return dict(width=width, height=height, full_sphere_height=width//2, crop_top=top,
                horizontal_degrees=360, vertical_degrees=height/width*360,
                covered_fraction=float(np.mean(mask > 0)), source_count=len(cameras), exposure_block_size=64, exposure_feeds=2, gain_maps=gain_maps,
                seam_method="dynamic programming COLOR_GRAD", blending_bands=8,
                seam_boundary_mean_abs_difference=float(np.mean(np.abs(result[:, 0].astype(float)-result[:, -1].astype(float)))))


def write_viewer(output, metadata):
    template = (Path(__file__).parent/'docs/viewer360_template.html').read_text()
    small = {k: metadata[k] for k in ('crop_top', 'full_sphere_height', 'height')}
    html = template.replace('__META__', json.dumps(small)).replace('__JPEG__',
                base64.b64encode((output/'panorama_360.jpg').read_bytes()).decode())
    (output/'visualizador_360.html').write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, default=6000)
    args = parser.parse_args()
    if args.width < 1800 or args.width % 2:
        parser.error('--width deve ser par e >= 1800')
    cv.setNumThreads(4); cv.setRNGSeed(0)
    files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if len(files) < 3:
        parser.error('São necessárias ao menos três imagens sobrepostas')
    random.Random(0).shuffle(files)
    args.output.mkdir(parents=True, exist_ok=True)
    data = estimate(files)
    (args.output/'correspondencias.json').write_text(json.dumps(data, indent=2))
    data = filter_rotation_matches(data)
    (args.output/'correspondencias_rotacao.json').write_text(json.dumps(data, indent=2))
    data = optimize(data)
    (args.output/'cameras_ajustadas.json').write_text(json.dumps(data, indent=2))
    meta = render(data, args.input, args.output, args.width)
    meta.update(opencv=cv.__version__, seed=0, manual_correspondences=0, manual_source_regions=0)
    (args.output/'render.json').write_text(json.dumps(meta, indent=2))
    write_viewer(args.output, meta)
    print(json.dumps({k:v for k,v in meta.items() if k != "gain_maps"}, indent=2))


if __name__ == '__main__':
    main()
