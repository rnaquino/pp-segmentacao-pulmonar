
"""
Separação pulmonar por erosão 2D e dilatação condicional.

Implementação experimental independente dos métodos originais.

Etapas:
1. Identificar os componentes da máscara pulmonar.
2. Aplicar erosão progressiva se os pulmões estiverem conectados.
3. Selecionar duas sementes principais.
4. Reconstruir as regiões por crescimento geodésico condicionado.
5. Calcular Dice e IoU quando houver máscara de referência.

Não utiliza watershed.
"""

import argparse
import heapq

import matplotlib.pyplot as plt
import numpy as np

from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, disk

from avaliar_segmentacao import calcular_dice, calcular_iou
from segmentacao_tres_metodos import (
    segmentar_pulmao,
    inferir_caminho_mascara,
)


def encontrar_sementes(mascara, raio_maximo=20):
    """
    Procura dois componentes principais.

    Se a máscara já estiver separada em dois componentes,
    não é necessário erodir.

    Caso contrário, aplica erosão 2D progressiva.
    """
    mascara = np.asarray(mascara, dtype=bool)

    if mascara.ndim != 2 or not mascara.any():
        return None, None, None

    for raio in range(raio_maximo + 1):

        if raio == 0:
            candidata = mascara.copy()
        else:
            candidata = binary_erosion(
                mascara,
                footprint=disk(raio)
            )

        rotulos = label(candidata, connectivity=2)

        regioes = sorted(
            regionprops(rotulos),
            key=lambda regiao: regiao.area,
            reverse=True
        )

        if len(regioes) < 2:
            continue

        primeira, segunda = regioes[:2]

        # Evita tratar pequenos ruídos como um pulmão.
        if segunda.area < max(20, 0.05 * candidata.sum()):
            continue

        # Exige separação espacial entre as sementes.
        diferenca_x = abs(
            primeira.centroid[0] - segunda.centroid[0]
        )

        if diferenca_x < 0.05 * mascara.shape[0]:
            continue

        if primeira.centroid[0] < segunda.centroid[0]:
            esquerda, direita = primeira, segunda
        else:
            esquerda, direita = segunda, primeira

        semente_esquerda = rotulos == esquerda.label
        semente_direita = rotulos == direita.label

        return semente_esquerda, semente_direita, raio

    return None, None, None


def dilatacao_condicional(
    mascara_original,
    semente_esquerda,
    semente_direita
):
    """
    Crescimento geodésico competitivo das duas sementes.

    Cada região cresce apenas dentro da máscara original.

    A expansão ocorre por caminhos de pixels vizinhos,
    sem utilizar watershed ou intensidades da imagem.
    """

    mascara = np.asarray(mascara_original, dtype=bool)
    semente_esquerda = np.asarray(
        semente_esquerda, dtype=bool
    )
    semente_direita = np.asarray(
        semente_direita, dtype=bool
    )

    if (
        mascara.shape != semente_esquerda.shape
        or mascara.shape != semente_direita.shape
    ):
        raise ValueError("As máscaras devem ter a mesma forma.")

    if np.any(semente_esquerda & semente_direita):
        raise ValueError("As sementes não podem se sobrepor.")

    if not semente_esquerda.any() or not semente_direita.any():
        raise ValueError("As duas sementes precisam existir.")

    if np.any((semente_esquerda | semente_direita) & ~mascara):
        raise ValueError("Sementes fora da máscara original.")

    rotulos = np.zeros(mascara.shape, dtype=np.uint8)

    distancias = np.full(
        mascara.shape, np.inf, dtype=float
    )

    fila = []

    for lado, semente in (
        (1, semente_esquerda),
        (2, semente_direita),
    ):
        for x, y in np.argwhere(semente):
            x, y = int(x), int(y)

            rotulos[x, y] = lado
            distancias[x, y] = 0.0

            heapq.heappush(
                fila, (0.0, lado, x, y)
            )

    vizinhos = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, np.sqrt(2)),
        (-1, 1, np.sqrt(2)),
        (1, -1, np.sqrt(2)),
        (1, 1, np.sqrt(2)),
    ]

    altura, largura = mascara.shape
    tolerancia = 1e-10

    while fila:
        distancia, lado, x, y = heapq.heappop(fila)

        if distancia > distancias[x, y] + tolerancia:
            continue

        if rotulos[x, y] != lado:
            continue

        for dx, dy, custo in vizinhos:
            nx = x + dx
            ny = y + dy

            if not (0 <= nx < altura and 0 <= ny < largura):
                continue

            if not mascara[nx, ny]:
                continue

            nova_distancia = distancia + custo

            if (
                nova_distancia
                < distancias[nx, ny] - tolerancia
            ):
                distancias[nx, ny] = nova_distancia
                rotulos[nx, ny] = lado

                heapq.heappush(
                    fila,
                    (nova_distancia, lado, nx, ny)
                )

    esquerda = rotulos == 1
    direita = rotulos == 2

    return esquerda, direita


def separar_pulmoes(mascara, raio_maximo=20):
    """
    Executa erosão 2D e reconstrução condicional.

    Retorna:
        esquerda, direita, raio_utilizado
    """

    esquerda_seed, direita_seed, raio = encontrar_sementes(
        mascara,
        raio_maximo=raio_maximo
    )

    if esquerda_seed is None:
        return None, None, None

    esquerda, direita = dilatacao_condicional(
        mascara,
        esquerda_seed,
        direita_seed
    )

    return esquerda, direita, raio


def avaliar(esquerda, direita, referencia):
    """Calcula Dice e IoU usando a melhor correspondência de rótulos."""

    ref1 = referencia == 1
    ref2 = referencia == 2

    if not ref1.any() or not ref2.any():
        print("Referência sem os dois rótulos pulmonares.")
        return

    opcao_a = (
        calcular_dice(esquerda, ref1)
        + calcular_dice(direita, ref2)
    )

    opcao_b = (
        calcular_dice(esquerda, ref2)
        + calcular_dice(direita, ref1)
    )

    if opcao_a >= opcao_b:
        ref_esquerda, ref_direita = ref1, ref2
    else:
        ref_esquerda, ref_direita = ref2, ref1

    print("\n=== AVALIAÇÃO ===")
    print(
        f"Dice esquerdo: "
        f"{calcular_dice(esquerda, ref_esquerda):.4f}"
    )
    print(
        f"Dice direito:  "
        f"{calcular_dice(direita, ref_direita):.4f}"
    )
    print(
        f"IoU esquerdo:  "
        f"{calcular_iou(esquerda, ref_esquerda):.4f}"
    )
    print(
        f"IoU direito:   "
        f"{calcular_iou(direita, ref_direita):.4f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Separação pulmonar por erosão e dilatação condicional."
    )

    parser.add_argument("imagem")
    parser.add_argument("--fatia", type=int, default=None)
    parser.add_argument("--raio-maximo", type=int, default=20)
    parser.add_argument("--referencia", default=None)

    args = parser.parse_args()

    caminho_referencia = (
        args.referencia
        or inferir_caminho_mascara(args.imagem)
    )

    mascara, k, referencia = segmentar_pulmao(
        args.imagem,
        k=args.fatia,
        caminho_mascara=caminho_referencia
    )

    esquerda, direita, raio = separar_pulmoes(
        mascara,
        raio_maximo=args.raio_maximo
    )

    if esquerda is None:
        print("Não foi possível encontrar duas sementes válidas.")
        return

    print(f"\nFatia avaliada: {k}")
    print(f"Raio de erosão utilizado: {raio}")

    # Verifica se a máscara foi reconstruída integralmente.
    reconstruida = esquerda | direita

    pixels_nao_atribuidos = np.count_nonzero(
        mascara & ~reconstruida
    )

    print(
        f"Pixels não atribuídos: {pixels_nao_atribuidos}"
    )

    if referencia is not None:
        avaliar(esquerda, direita, referencia)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    imagens = [
        (mascara, "Máscara inicial"),
        (esquerda, "Pulmão 1"),
        (direita, "Pulmão 2"),
    ]

    for ax, (imagem, titulo) in zip(axes, imagens):
        ax.imshow(imagem.T, cmap="gray", origin="lower")
        ax.set_title(titulo)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
