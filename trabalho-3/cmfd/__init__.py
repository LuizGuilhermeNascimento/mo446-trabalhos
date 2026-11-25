"""Detecção de copy-move por auto-correspondência de pontos-chave (MO446 - T3).

Ideias centrais:
  1. a imagem é casada com ela mesma, dentro e entre fotos;
  2. os descritores também são extraídos da imagem espelhada;
  3. a máscara sai da sobreposição densa (ZNCC local) após o movimento estimado.
"""
