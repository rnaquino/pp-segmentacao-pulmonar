"""Medidas geométricas dos segmentos do Adaptive Border Marching (ABM).

Cada segmento deve ser enviado como uma sequência ordenada de pontos do
contorno, incluindo o ponto inicial e o ponto final. As coordenadas seguem
a ordem dos eixos da matriz: (linha, coluna).
"""

from __future__ import annotations

import numpy as np


def medir_track(pontos, spacing=(1.0, 1.0)):
    """Calcula W, Hmax e lambda para um segmento do contorno.

    Parameters
    ----------
    pontos : array-like, shape (N, 2)
        Pontos do contorno entre o início e o fim do track, inclusive.
    spacing : tuple[float, float]
        Espaçamento físico dos eixos da matriz (linha, coluna). Use (1, 1)
        para trabalhar em pixels. Com pixels isotrópicos, lambda é igual
        usando pixels ou unidades físicas.

    Returns
    -------
    dict
        W, Hmax, lambda e o índice do ponto que alcançou Hmax.
    """
    pontos = np.asarray(pontos, dtype=float)
    espacamento = np.asarray(spacing, dtype=float)

    if pontos.ndim != 2 or pontos.shape[1] != 2 or len(pontos) < 2:
        raise ValueError("O track deve ter pelo menos dois pontos (N, 2).")
    if not np.isfinite(pontos).all():
        raise ValueError("O track contém coordenadas não finitas.")
    if espacamento.shape != (2,) or not np.isfinite(espacamento).all():
        raise ValueError("spacing deve conter dois valores finitos.")
    if np.any(espacamento <= 0):
        raise ValueError("Os espaçamentos devem ser positivos.")

    # Converte (linha, coluna) para coordenadas físicas antes das distâncias.
    fisicos = pontos * espacamento
    inicio = fisicos[0]
    fim = fisicos[-1]
    vetor = fim - inicio
    W = float(np.linalg.norm(vetor))

    if W <= np.finfo(float).eps:
        raise ValueError("W é zero: os pontos inicial e final coincidem.")

    intermediarios = fisicos[1:-1]
    if len(intermediarios) == 0:
        Hmax = 0.0
        indice_hmax = None
    else:
        relativos = intermediarios - inicio
        # Em 2D, o módulo do produto vetorial dividido por |vetor|
        # fornece a distância perpendicular à reta que contém o segmento.
        produtos = vetor[0] * relativos[:, 1] - vetor[1] * relativos[:, 0]
        distancias = np.abs(produtos) / W
        posicao = int(np.argmax(distancias))
        Hmax = float(distancias[posicao])
        indice_hmax = posicao + 1  # índice dentro do array original do track

    razao = Hmax / W
    return {
        "W": W,
        "Hmax": Hmax,
        "lambda": float(razao),
        "indice_hmax": indice_hmax,
        "quantidade_pontos": int(len(pontos)),
    }


def medir_tracks(tracks, spacing=(1.0, 1.0)):
    """Calcula as medidas para uma lista de tracks, preservando sua ordem."""
    resultados = []
    for indice, track in enumerate(tracks):
        medida = medir_track(track, spacing=spacing)
        medida["track"] = indice
        resultados.append(medida)
    return resultados


def imprimir_resumo(resultados):
    """Imprime uma tabela textual simples das medidas calculadas."""
    if not resultados:
        print("Nenhum track para medir.")
        return

    print("\n=== MEDIDAS GEOMÉTRICAS DOS TRACKS ===")
    print(f"{'Track':>7} {'Pontos':>8} {'W':>10} {'Hmax':>10} {'lambda':>10}")
    for item in resultados:
        print(
            f"{item['track']:>7d} "
            f"{item['quantidade_pontos']:>8d} "
            f"{item['W']:>10.3f} "
            f"{item['Hmax']:>10.3f} "
            f"{item['lambda']:>10.4f}"
        )

    valores = np.asarray([item["lambda"] for item in resultados], dtype=float)
    print("\nResumo de lambda:")
    print(f"Mínimo: {valores.min():.4f}")
    print(f"Média:  {valores.mean():.4f}")
    print(f"Máximo: {valores.max():.4f}")
