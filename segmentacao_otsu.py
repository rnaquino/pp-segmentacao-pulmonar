"""Mostra uma fatia e sua máscara por limiar automático de Otsu em 0–255.

Usa imagem_255 do pré-processamento para Coronacases e Radiopaedia.
Seleciona intensidades abaixo do limiar calculado na própria fatia.
Não remove componentes nem aplica morfologia: o ar externo também pode
ser selecionado. Fatias constantes são rejeitadas.

Exemplo:
    python segmentacao_otsu.py data/raw/coronacases_001.nii.gz
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from avaliar_segmentacao import calcular_dice, calcular_iou

from src.preprocessing import (
    preparar_fatia, CENTRO_JANELA_PULMAO, LARGURA_JANELA_PULMAO,
)
from skimage.filters import threshold_otsu


def main():
    # Recebe o caminho da tomografia e os parâmetros pelo terminal.
    parser = argparse.ArgumentParser(
        description="Mostra uma fatia e sua máscara por Otsu."
    )
    parser.add_argument("imagem", help="Caminho da tomografia NIfTI")
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
    mascara = fatia < limiar

    print(f"Fonte: {dados['source']}; escala de trabalho: 0–255")
    print("Fatia:", k)
    print("Intensidade mínima:", fatia.min())
    print("Intensidade máxima:", fatia.max())
    print("Limiar Otsu (0–255):", limiar)
    print("Pixels selecionados:", np.count_nonzero(mascara))

    # Mantém as proporções físicas da imagem.
    sx, sy = dados["metadados"]["spacing"][:2]
    aspecto = float(sy) / float(sx)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(
        fatia.T,
        cmap="gray",
        origin="lower",
        vmin=0,
        vmax=255,
        aspect=aspecto,
        interpolation="nearest",
    )
    axes[0].set_title(f"{dados['source']} — fatia {k} (0–255)")

    axes[1].imshow(
        mascara.T,
        cmap="gray",
        origin="lower",
        vmin=0,
        vmax=1,
        aspect=aspecto,
        interpolation="nearest",
    )
    axes[1].set_title(f"Otsu: intensidade < {limiar:.2f} (0–255)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
