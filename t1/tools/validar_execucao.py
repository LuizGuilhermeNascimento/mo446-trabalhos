"""Recalcula o panorama em cinco pastas sem nomes ou metadados informativos."""

import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from executar import carregar_funcoes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entrada', type=Path, default=ROOT/'images/v1')
    parser.add_argument('--saida', type=Path, default=ROOT/'outputs')
    args = parser.parse_args()
    p = carregar_funcoes()
    medidas = json.loads((args.saida/'metricas.json').read_text())
    cfg = p.Config(**medidas['configuracao'])
    base = p.carregar_imagens(args.entrada, cfg)
    assert [q.uid for q in base] == [q['uid_pixels'] for q in medidas['inventario']]
    esperado = np.asarray(Image.open(args.saida/'panorama.png'))
    temporarios = args.saida/'validacao_temporaria'
    temporarios.mkdir(exist_ok=True)
    relatorio = {'concluida':False, 'configuracao':medidas['configuracao'], 'embaralhamentos':[]}
    for seed in range(446, 451):
        inicio = time.perf_counter()
        rng = np.random.default_rng(seed)
        with tempfile.TemporaryDirectory(prefix='imagens-', dir=temporarios) as pasta:
            for idx in rng.permutation(len(base)):
                nome = rng.bytes(8).hex()+'.png'
                Image.fromarray(base[idx].rgb).save(Path(pasta)/nome)
            quadros = p.carregar_imagens(pasta, cfg)
            assert all(a.uid == b.uid and np.array_equal(a.rgb,b.rgb) for a,b in zip(base,quadros))
            fs = p.extrair(quadros, 'SIFT', cfg)
            pares = p.emparelhar(quadros, fs, 'SIFT', cfg)
            resultado = p.montar(quadros, pares, cfg)
            atual = np.round(p.linear_srgb(p.recortar(resultado['final'],resultado['caixa']))*255).astype(np.uint8)
            mesma_matriz = bool(np.array_equal(resultado['W'],medidas['matriz_inliers']))
            ordem = [quadros[i].uid for i in resultado['ordem']]
            mesma_ordem = ordem == medidas['ordem_uids']
            mesmo_tamanho = atual.shape == esperado.shape
            diferenca = int(np.max(np.abs(atual.astype(np.int16)-esperado.astype(np.int16)))) if mesmo_tamanho else None
            linha = {'seed':seed, 'mesma_matriz':mesma_matriz, 'mesma_ordem':mesma_ordem,
                     'mesmo_tamanho':mesmo_tamanho, 'diferenca_max_8bit':diferenca,
                     'tempo_s':round(time.perf_counter()-inicio,2)}
            relatorio['embaralhamentos'].append(linha)
            (args.saida/'validacao.json').write_text(json.dumps(relatorio,ensure_ascii=False,indent=2))
            print(json.dumps(linha,ensure_ascii=False),flush=True)
            assert mesma_matriz and mesma_ordem and mesmo_tamanho and diferenca <= 1
            del resultado, fs, pares, atual, quadros
            gc.collect()
    relatorio['concluida'] = True
    (args.saida/'validacao.json').write_text(json.dumps(relatorio,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
