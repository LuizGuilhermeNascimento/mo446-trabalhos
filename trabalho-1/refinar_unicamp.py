"""Refina a composição da Unicamp usando a geometria já estimada pela branch.

Não substitui os estudos de detecção/matching/alinhamento. O JSON registra as
homografias congeladas, hashes das 42 entradas e duas prioridades MANUAIS de
costura. Inclui um ajuste afim manual, limitado ao corrimão. Não há geração de conteúdo
ou fluxo óptico.

Execute da raiz: python trabalho-1/refinar_unicamp.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from common import ROOT, resize_max_side, save_image, write_csv
from step4_homography_connection.alignment import AlignedImages, AlignmentResult, cylindrical_image
from step5_panorama_composition import compositor as c, evaluation as e

METHOD = 'Graph cut assistido'


def original_maps(transform, shape, original_shape, focal, corner, size, scale):
    """Canvas -> cilindro -> foto original, com uma única reamostragem.

    `transform` mapeia o cilindro de trabalho para o canvas de trabalho.
    A convenção de centros de pixels coincide com cv2.resize.
    """
    width, height = size
    xx, yy = np.meshgrid((np.arange(width) + .5) / scale - .5 + corner[0],
                         (np.arange(height) + .5) / scale - .5 + corner[1])
    inv = np.linalg.inv(transform)
    den = inv[2, 0] * xx + inv[2, 1] * yy + inv[2, 2]
    cx = (inv[0, 0] * xx + inv[0, 1] * yy + inv[0, 2]) / den
    cy = (inv[1, 0] * xx + inv[1, 1] * yy + inv[1, 2]) / den
    h, w = shape
    theta = (cx - w / 2) / focal
    px = focal * np.tan(theta) + w / 2
    py = (cy - h / 2) / np.cos(theta) + h / 2
    return (((px + .5) * original_shape[1] / w - .5).astype(np.float32),
            ((py + .5) * original_shape[0] / h - .5).astype(np.float32))


def prioritize(seams, masks, names, priorities, rect):
    """Restrict overrides to valid source pixels; never open holes in coverage."""
    x, y, _, _ = rect
    for item in priorities:
        k = names.index(item['name'])
        region = np.zeros_like(seams[k])
        cv2.fillPoly(region, [np.asarray(item['polygon'], np.int32) + [x, y]], 255)
        region &= cv2.erode(masks[k], np.ones((7, 7), np.uint8))
        for j, seam in enumerate(seams):
            seam[region > 0] = 255 if j == k else 0


def run(config_path, input_dir, output_dir):
    start = time.perf_counter()
    cv2.setRNGSeed(0)
    cv2.setNumThreads(4)
    cfg = json.loads(config_path.read_text())
    source = cfg['sources']
    names = [s['name'] for s in source]
    if len(names) != 42 or len(set(names)) != 42:
        raise ValueError('A calibração deve conter exatamente as 42 fotos distintas.')
    paths = [input_dir / s['filename'] for s in source]
    actual = {p.name for p in input_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg'}}
    if actual != {p.name for p in paths}:
        raise ValueError('Conjunto de JPGs diferente da calibração; reestime a geometria.')
    for path, item in zip(paths, source):
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Entrada alterada: {path.name}; reestime a geometria.')
    canvas = tuple(cfg['canvas_size'])
    rect = tuple(cfg['rect']); x, y, w, h = rect
    transforms = {k: np.array(s['transform'], np.float64) for k, s in enumerate(source)}
    adjust = cfg.get('foreground_adjustment')
    if adjust:
        k = names.index(adjust['name'])
        affine = np.eye(3)
        affine[:2] = cv2.getAffineTransform(np.float32(adjust['from']), np.float32(adjust['to']))
        shift = np.array([[1, 0, x], [0, 1, y], [0, 0, 1.]])
        transforms[k] = shift @ affine @ np.linalg.inv(shift) @ transforms[k]
    parents = {k: None if s['parent'] is None else names.index(s['parent']) for k, s in enumerate(source)}
    projected, warped, masks = [], [], []
    for k, path in enumerate(paths):
        original = cv2.imread(str(path))
        small = resize_max_side(original, 1024)
        if list(small.shape[:2]) != source[k]['shape']:
            raise ValueError(f'Dimensões divergentes: {path.name}')
        im, mask = cylindrical_image(small, cfg['focal'])
        projected.append(im)
        warped.append(cv2.warpPerspective(im, transforms[k], canvas))
        masks.append(cv2.erode(cv2.warpPerspective(mask, transforms[k], canvas,
                     flags=cv2.INTER_NEAREST), np.ones((5, 5), np.uint8)))
    if adjust:
        k = names.index(adjust['name'])
        allowed = np.zeros_like(masks[k])
        cv2.fillPoly(allowed, [np.int32(adjust['polygon']) + [x, y]], 255)
        masks[k] &= allowed
    corners, images, tight_masks = [], [], []
    for im, mask in zip(warped, masks):
        cx, cy, cw, ch = cv2.boundingRect(mask)
        corners.append((cx, cy)); images.append(im[cy:cy+ch, cx:cx+cw].copy())
        tight_masks.append(mask[cy:cy+ch, cx:cx+cw].copy())
    compensator = cv2.detail_BlocksChannelsCompensator(64, 64, 2)
    compensator.feed(corners, images, tight_masks)
    corrected = [compensator.apply(k, p, im.copy(), m) for k, (p, im, m)
                 in enumerate(zip(corners, images, tight_masks))]
    tight_seams = cv2.detail_GraphCutSeamFinder('COST_COLOR_GRAD').find(
        [im.astype(np.float32) for im in corrected], corners,
        [cv2.UMat(m.copy()) for m in tight_masks])
    seams = []
    for k, (im, sm, (cx, cy)) in enumerate(zip(corrected, tight_seams, corners)):
        sm = sm.get(); ch, cw = sm.shape
        warped[k][:] = 0; warped[k][cy:cy+ch, cx:cx+cw] = im
        full = np.zeros_like(masks[k]); full[cy:cy+ch, cx:cx+cw] = sm; seams.append(full)
    prioritize(seams, masks, names, cfg['seam_priorities'], rect)
    union = np.any(np.array(masks) > 0, axis=0)
    coverage = np.any(np.array(seams) > 0, axis=0)
    if not np.all(coverage[union]) or not union[y:y+h, x:x+w].all():
        raise ValueError('Costura/recorte com pixels sem cobertura.')
    aligned = AlignedImages(warped, masks, list(range(42)), transforms, parents, canvas)
    labels = np.full(union.shape, -1, np.int32)
    blend = cv2.detail_MultiBandBlender(0, 4); blend.prepare((0, 0, *canvas))
    for k, (im, mask) in enumerate(zip(warped, seams)):
        labels[mask > 0] = k
        blend.feed(im.astype(np.int16), mask, (0, 0))
    low, _ = blend.blend(None, None); low = low.clip(0, 255).astype(np.uint8)
    result = c.CompositionResult(low[y:y+h, x:x+w], c.naive_composite(aligned)[y:y+h, x:x+w],
             labels[y:y+h, x:x+w], union[y:y+h, x:x+w], c.crop_aligned(aligned, rect), rect,
             (time.perf_counter()-start)*1000)
    step_dir = output_dir / 'step5_panorama_composition/unicamp'; step_dir.mkdir(parents=True, exist_ok=True)
    alignment = AlignmentResult(aligned, projected, {}, {'bundle': transforms}, 'cylindrical', cfg['focal'])
    e.save_composition_outputs(names, alignment, METHOD, result, 'unicamp', step_dir, (90, 140, 410, 168))
    results = {f'{d}_{b}': c.PanoramaCompositor(d, b).compose(result.aligned) for d, b in e.STUDY}
    results[METHOD] = result
    rows = [e.method_row(name, r, result.labels) for name, r in results.items()]
    write_csv(rows, step_dir/'metricas_metodos_composicao.csv')
    e.plot_metrics(rows, 'unicamp', step_dir/'metricas_metodos_composicao.png')
    e.plot_methods(results, (90, 140, 410, 168), 'unicamp', step_dir/'comparacao_metodos.jpg')
    scale = int(cfg['render_scale'])
    blender = cv2.detail_MultiBandBlender(0, 4 + round(np.log2(scale)))
    blender.prepare((0, 0, w*scale, h*scale))
    contributions = []
    for k, (path, (cx, cy), mask) in enumerate(zip(paths, corners, tight_masks)):
        ch, cw = mask.shape
        original = cv2.imread(str(path))
        mx, my = original_maps(transforms[k], source[k]['shape'], original.shape, cfg['focal'],
                               (cx, cy), (cw*scale, ch*scale), scale)
        im = cv2.remap(original, mx, my, cv2.INTER_CUBIC)
        valid = cv2.resize(mask, (cw*scale, ch*scale), interpolation=cv2.INTER_NEAREST)
        im = compensator.apply(k, (cx*scale, cy*scale), im, valid)
        sx0, sy0 = max(x, cx), max(y, cy)
        sx1, sy1 = min(x+w, cx+cw), min(y+h, cy+ch)
        if sx1 <= sx0 or sy1 <= sy0:
            contributions.append({'imagem': names[k], 'pixels_costura': 0}); continue
        region = seams[k][sy0:sy1, sx0:sx1]
        contributions.append({'imagem': names[k], 'pixels_costura': int(np.count_nonzero(region))*scale**2})
        crop = im[(sy0-cy)*scale:(sy1-cy)*scale, (sx0-cx)*scale:(sx1-cx)*scale]
        seam = cv2.resize(region, ((sx1-sx0)*scale, (sy1-sy0)*scale), interpolation=cv2.INTER_NEAREST)
        blender.feed(crop.astype(np.int16), seam, ((sx0-x)*scale, (sy0-y)*scale))
    final, final_mask = blender.blend(None, None)
    if not np.all(final_mask > 0):
        raise ValueError('Renderização final contém pixels sem cobertura.')
    final = final.clip(0, 255).astype(np.uint8)
    final_dir = output_dir/'final/unicamp'; final_dir.mkdir(parents=True, exist_ok=True)
    save_image(final_dir/'panorama.jpg', final)
    write_csv(contributions, step_dir/'contribuicao_fontes.csv')
    report = {'method': METHOD, 'input_images': 42, 'output_size': [w*scale, h*scale],
              'metrics_resolution': [w, h], 'metrics': rows, 'opencv': cv2.__version__,
              'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
              'elapsed_seconds': time.perf_counter()-start,
              'manual_seam_priorities': cfg['seam_priorities'],
              'foreground_adjustment': adjust,
              'limitations': 'Geometria base com ajuste afim manual de primeiro plano; paralaxe residual nos degraus/objetos próximos. Métricas em baixa resolução e sobre fontes compensadas não são diretamente comparáveis às antigas.'}
    (step_dir/'refinamento.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print('FINAL', final.shape, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/unicamp_refinamento.json')
    parser.add_argument('--input', type=Path, default=ROOT/'input/unicamp')
    parser.add_argument('--output', type=Path, default=ROOT/'output')
    args = parser.parse_args()
    run(args.config, args.input, args.output)
