import numpy as np
import pytest
from executar import carregar_funcoes


@pytest.fixture(scope="module")
def p():
    return carregar_funcoes()


def test_dlt_recupera_transformacao_e_rejeita_colinearidade(p):
    points = np.array([[0, 0], [400, 0], [400, 300], [0, 300], [200, 120], [80, 250]], float)
    expected = np.array([[1.1, .06, 40], [-.02, .95, 12], [.0002, -.0001, 1.]])
    target = p.projetar(expected, points)
    actual = p.dlt_normalizado(points, target)
    np.testing.assert_allclose(p.projetar(actual, points), target, atol=1e-8)
    with pytest.raises(ValueError):
        p.dlt_normalizado(np.c_[np.arange(5), np.arange(5)], np.c_[np.arange(5), np.arange(5)])


def test_costura_encontra_caminho_conhecido_e_respeita_mascara(p):
    cost = np.full((7, 6), 50., dtype=float)
    expected = np.array([1, 1, 2, 3, 3, 2, 1])
    cost[np.arange(7), expected] = 0
    seam, accumulated = p.costura_minima(cost)
    np.testing.assert_array_equal(seam, expected)
    assert accumulated[-1, seam[-1]] == 0
    cost[3, :] = np.inf
    with pytest.raises(ValueError):
        p.costura_minima(cost)


def test_piramide_reconstroi_dimensoes_impares(p):
    rgb = np.random.default_rng(2).random((127, 185, 3)).astype(np.float32)
    levels = p.piramide_laplaciana(rgb, 5)
    np.testing.assert_allclose(p.reconstruir(levels), rgb, atol=2e-7)


def test_mascara_preserva_pixels_pretos_e_fonte_unica(p):
    images = [np.zeros((15, 21, 3), np.float32), np.full((15, 21, 3), .8, np.float32)]
    masks = [np.ones((15, 21), bool), np.zeros((15, 21), bool)]
    result = p.media_valida(images, masks)
    assert np.all(result == 0)
    np.testing.assert_allclose(p.multibanda(images, masks, 3), result, atol=1e-6)


def test_matching_ransac_recupera_geometria_com_outliers(p):
    rng = np.random.default_rng(446)
    origem = rng.uniform(15, 380, (80, 2)).astype(np.float32)
    H = np.array([[.98, .015, 8], [-.01, 1.02, 4], [.00003, -.00002, 1.]])
    destino = p.projetar(H, origem).astype(np.float32)
    destino[:20] = rng.uniform(0, 400, (20, 2))
    descritores = rng.random((80, 128)).astype(np.float32)
    quadros = [p.Quadro(str(i), str(i), np.zeros((420, 420, 3), np.uint8), 'rgb', (420, 420), 1) for i in range(2)]
    features = [p.Caracteristicas([p.cv.KeyPoint(float(x), float(y), 3) for x, y in pts], descritores, 0)
                for pts in [origem, destino]]
    par = p.emparelhar(quadros, features, 'SIFT', p.Config())[0]
    assert par.aceito and par.n == 60
    assert not par.inliers[:20].any()
    np.testing.assert_allclose(p.projetar(par.H, origem[20:]), destino[20:], atol=1e-3)


def test_sem_descritores_nao_inventa_conexao(p):
    quadros = [p.Quadro(str(i), str(i), np.zeros((40, 40, 3), np.uint8), 'rgb', (40, 40), 1) for i in range(2)]
    features = [p.Caracteristicas([], None, 0) for _ in quadros]
    pares = p.emparelhar(quadros, features, 'SIFT', p.Config())
    assert pares[0].n == 0 and not pares[0].aceito
    with pytest.raises(ValueError, match='Nenhum par'):
        p.alinhar_grafo(quadros, pares)


def test_recorte_maximo_contra_busca_exaustiva(p):
    rng = np.random.default_rng(3)
    for _ in range(12):
        mask = rng.random((6, 8)) > .3
        x0, y0, x1, y1 = p.maior_retangulo(mask, passo=1)
        assert mask[y0:y1, x0:x1].all()
        areas = [(b-a)*(d-c) for a in range(8) for b in range(a+1, 9)
                 for c in range(6) for d in range(c+1, 7) if mask[c:d, a:b].all()]
        assert (x1-x0)*(y1-y0) == max(areas)
    with pytest.raises(ValueError):
        p.maior_retangulo(np.zeros((10, 10), bool))


def test_costura_e_multibanda_preservam_objeto_em_uma_fonte(p):
    a = np.full((100, 160, 3), .2, np.float32)
    b = a.copy()
    b[40:60, 105:125] = .8
    masks = [np.ones(a.shape[:2], bool), np.ones(a.shape[:2], bool)]
    selecoes, fontes, _ = p.escolher_fontes([a, b], masks, [0, 1])
    assert (fontes[40:60, 105:125] == 1).all()
    result = p.multibanda([a, b], selecoes, 4)
    np.testing.assert_allclose(result[45:55, 110:120], .8, atol=1e-4)
    np.testing.assert_allclose(p.media_valida([a, b], masks)[45:55, 110:120], .5, atol=1e-6)
