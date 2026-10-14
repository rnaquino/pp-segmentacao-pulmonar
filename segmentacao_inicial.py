"""Cria uma máscara candidata por limiar fixo em uma fatia da tomografia.

Carrega e extrai a fatia usando src.preprocessing. Cada pixel com
intensidade menor que --limiar recebe True na máscara; os demais recebem
False. Mostra a tomografia e a máscara lado a lado e imprime a quantidade
de pixels selecionados. Sem --fatia, usa a fatia central.

Exemplo para Coronacases em HU:
    python segmentacao_inicial.py data/raw/coronacases_001.nii.gz --fatia 150 --limiar -400

O limiar padrão é -400 HU. O ar externo e outros espaços de ar também
podem ser selecionados: não há remoção de componentes nem morfologia.
Esses limiares em HU não se aplicam diretamente aos Radiopaedia.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing import (
    carregar_imagem, obter_indice_fatia, extrair_fatia, obter_metadados,
)


def main():
    # Recebe o caminho da tomografia e os parâmetros pelo terminal.
    parser = argparse.ArgumentParser(
        description="Experimenta uma segmentação por limiar fixo."
    )
    parser.add_argument("imagem", help="Caminho da tomografia NIfTI")
    parser.add_argument("--fatia", type=int, default=None)
    parser.add_argument("--limiar", type=float, default=-400)
    args = parser.parse_args()

    try:
        img = carregar_imagem(args.imagem)
        k = obter_indice_fatia(img, args.fatia)
        fatia = extrair_fatia(img, k)
    except (OSError, ValueError) as erro:
        parser.error(str(erro))

    # Compara todos os pixels com o limiar.
    mascara = fatia < args.limiar

    print("Fatia:", k)
    print("Intensidade mínima:", fatia.min())
    print("Intensidade máxima:", fatia.max())
    print("Limiar:", args.limiar)
    print("Pixels selecionados:", np.count_nonzero(mascara))

    # Mantém as proporções físicas da imagem.
    sx, sy = obter_metadados(img)["spacing"][:2]
    aspecto = float(sy) / float(sx)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(
        fatia.T,
        cmap="gray",
        origin="lower",
        vmin=-1000,
        vmax=400,
        aspect=aspecto,
        interpolation="nearest",
    )
    axes[0].set_title(f"Tomografia — fatia {k}")

    axes[1].imshow(
        mascara.T,
        cmap="gray",
        origin="lower",
        vmin=0,
        vmax=1,
        aspect=aspecto,
        interpolation="nearest",
    )
    axes[1].set_title(f"Intensidade < {args.limiar:g} HU")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()