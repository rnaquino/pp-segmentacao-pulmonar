"""Validação da geometria de referência e construção de elementos em milímetros."""

import numpy as np

from src.preprocessing import carregar_imagem, verificar_alinhamento, obter_metadados


def elemento_em_mm(raio_mm, spacing):
    """Discretiza um disco físico; admite pixels retangulares no plano.

    Cada deslocamento (i, j) entra se (i*sx)^2 + (j*sy)^2 <= raio_mm^2.
    Não arredonda o raio para um único inteiro nem usa o espaçamento entre fatias.
    """
    spacing = np.asarray(spacing, dtype=float)
    if not np.isfinite(raio_mm) or raio_mm < 0:
        raise ValueError("O raio em mm deve ser finito e não negativo.")
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Informe dois espaçamentos positivos e finitos em mm.")
    sx, sy = spacing
    nx, ny = np.floor(raio_mm / spacing).astype(int)
    x, y = np.ogrid[-nx:nx + 1, -ny:ny + 1]
    return np.hypot(x * sx, y * sy) <= raio_mm


def espacamento_referencia(caminho_imagem, caminho_mascara, todos_eixos=False):
    """Lê apenas a geometria da referência, sem usar seus rótulos."""
    img = carregar_imagem(caminho_imagem)
    ref = carregar_imagem(caminho_mascara)
    verificar_alinhamento(img, ref)
    meta = obter_metadados(ref)
    if meta["unidade_espacial"] != "mm":
        raise ValueError("A máscara de referência precisa declarar unidade espacial mm.")
    spacing = np.asarray(meta["spacing"])
    eixos = meta["affine"][:3, :3]
    if not np.allclose(np.linalg.norm(eixos, axis=0), spacing, atol=1e-6, rtol=0):
        raise ValueError("Espaçamento da referência inconsistente com a affine.")
    direcoes = eixos / spacing
    if not np.allclose(direcoes.T @ direcoes, np.eye(3), atol=1e-5, rtol=0):
        raise ValueError("A grade possui cisalhamento; requer tratamento geométrico específico.")
    return tuple(spacing if todos_eixos else spacing[:2])
