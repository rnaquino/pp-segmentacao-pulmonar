"""Mostra a remoção dos componentes de borda de uma máscara obtida por Otsu.

Exibe somente a fatia preparada em 0–255 e a máscara após a remoção.
A conectividade inclui lados e diagonais. Regiões internas indesejadas
podem permanecer: ainda não há seleção anatômica ou morfologia.
Não importa outros scripts de segmentação.

Exemplo:
    python segmentacao_componentes.py data/raw/radiopaedia_10_85902_1.nii.gz
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing import (
    preparar_fatia, CENTRO_JANELA_PULMAO, LARGURA_JANELA_PULMAO,
)
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.segmentation import clear_border
from avaliar_segmentacao import calcular_dice, calcular_iou


def remover_componentes_da_borda(mascara):
    # Cada região conectada recebe um número; zero representa o fundo.
    # Em 2D, connectivity=2 considera lados e diagonais como conexões.
    rotulos = label(mascara, connectivity=2)

    # Remove todas as regiões que alcançam qualquer borda da fatia.
    rotulos_sem_borda = clear_border(rotulos)

    # Volta para uma máscara booleana: qualquer rótulo positivo é incluído.
    mascara_sem_borda = rotulos_sem_borda > 0
    return rotulos, mascara_sem_borda


def main():
    parser = argparse.ArgumentParser(
        description="Remove componentes de borda da máscara de Otsu."
    )
    parser.add_argument("imagem", help="Tomografia NIfTI Coronacases ou Radiopaedia")
    parser.add_argument("--fatia", type=int, default=None,
                        help="Opcional; sem este argumento, usa a fatia central")
    parser.add_argument("--fonte", default=None,
                        choices=["Coronacases", "Radiopaedia"],
                        help="Opcional; detectada pelos prefixos coronacases_ e radiopaedia_")
    parser.add_argument("--centro", type=float, default=CENTRO_JANELA_PULMAO,
                        help="Centro da janela em HU, somente para Coronacases")
    parser.add_argument("--largura", type=float, default=LARGURA_JANELA_PULMAO,
                        help="Largura da janela em HU, somente para Coronacases")
    args = parser.parse_args()

    try:
        dados = preparar_fatia(args.imagem, args.fonte, k=args.fatia,
                               centro=args.centro, largura=args.largura)
        k = dados["indice_fatia"]
        fatia = dados["imagem_255"]
    except (OSError, ValueError) as erro:
        parser.error(str(erro))
    if fatia.min() == fatia.max():
        parser.error("A fatia é constante: escolha outra para aplicar Otsu.")

    limiar = float(threshold_otsu(fatia))
    candidata = fatia < limiar
    _, sem_borda = remover_componentes_da_borda(candidata)

    print(f"Fonte: {dados['source']}; escala de trabalho: 0–255")
    print("Fatia:", k)
    print(f"Limiar Otsu: {limiar:.2f} (0–255)")
    print("Pixels antes da remoção:", np.count_nonzero(candidata))
    print("Pixels após remover a borda:", np.count_nonzero(sem_borda))
    print("Regiões restantes ainda precisam de seleção anatômica e refinamento.")

    sx, sy = dados["metadados"]["spacing"][:2]
    aspecto = float(sy) / float(sx)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(
        fatia.T, cmap="gray", origin="lower", aspect=aspecto,
        interpolation="nearest", vmin=0, vmax=255,
    )
    axes[0].set_title(f"{dados['source']} — fatia {k} (0–255)")
    axes[1].imshow(
        sem_borda.T, cmap="gray", origin="lower", aspect=aspecto,
        interpolation="nearest", vmin=0, vmax=1,
    )
    axes[1].set_title("Otsu: componentes de borda removidos")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
