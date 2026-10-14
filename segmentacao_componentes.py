"""Compara limiar fixo e Otsu e remove componentes que tocam a borda.

Exemplo, para um exame Coronacases em HU:
    python segmentacao_componentes.py data/raw/coronacases_001.nii.gz --fatia 150

As regiões restantes são candidatas: ainda não representam uma máscara
pulmonar final. Este exercício usa componentes conexos em uma fatia 2D.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing import (
    carregar_imagem, obter_indice_fatia, extrair_fatia, obter_metadados,
)
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.segmentation import clear_border


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
        description="Compara limiares e remove o ar conectado à borda."
    )
    parser.add_argument("imagem", help="Tomografia NIfTI em HU (Coronacases)")
    parser.add_argument("--fatia", type=int, default=None)
    parser.add_argument("--limiar", type=float, default=-400)
    args = parser.parse_args()

    try:
        img = carregar_imagem(args.imagem)
        k = obter_indice_fatia(img, args.fatia)
        fatia = extrair_fatia(img, k)
    except (OSError, ValueError) as erro:
        parser.error(str(erro))
    if not np.isfinite(args.limiar):
        parser.error("O limiar precisa ser finito.")
    if fatia.min() == fatia.max():
        parser.error("A fatia é constante: escolha outra para comparar os métodos.")

    # Duas estratégias aplicadas à mesma imagem original, sem janelamento numérico.
    mascara_fixa = fatia < args.limiar
    limiar_otsu = float(threshold_otsu(fatia))
    mascara_otsu = fatia < limiar_otsu

    _, fixa_sem_borda = remover_componentes_da_borda(mascara_fixa)
    rotulos_otsu, otsu_sem_borda = remover_componentes_da_borda(mascara_otsu)

    print("Fatia:", k)
    print(f"Limiar fixo: {args.limiar:g} HU")
    print(f"Limiar Otsu desta fatia: {limiar_otsu:.2f} HU")
    for nome, antes, depois in [
        ("Fixo", mascara_fixa, fixa_sem_borda),
        ("Otsu", mascara_otsu, otsu_sem_borda),
    ]:
        print(f"{nome}: {np.count_nonzero(antes)} pixels antes; "
              f"{np.count_nonzero(depois)} depois da remoção da borda.")
    print("Regiões restantes ainda precisam de seleção anatômica e refinamento.")

    sx, sy = obter_metadados(img)["spacing"][:2]
    aspecto = float(sy) / float(sx)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    axes[0, 0].imshow(
        fatia.T, cmap="gray", origin="lower", aspect=aspecto,
        interpolation="nearest", vmin=-1000, vmax=400,
    )
    axes[0, 0].set_title(f"Tomografia — fatia {k}")

    for ax, mascara, titulo in [
        (axes[0, 1], mascara_fixa, f"Fixo: < {args.limiar:g} HU"),
        (axes[0, 2], mascara_otsu, f"Otsu: < {limiar_otsu:.1f} HU"),
        (axes[1, 1], fixa_sem_borda, "Fixo: componentes de borda removidos"),
        (axes[1, 2], otsu_sem_borda, "Otsu: componentes de borda removidos"),
    ]:
        ax.imshow(mascara.T, cmap="gray", origin="lower", aspect=aspecto,
                  interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(titulo, fontsize=10)

    # As cores representam componentes conectados, não classes anatômicas.
    cores = plt.get_cmap("tab20").copy()
    cores.set_bad("black")
    componentes = np.ma.masked_where(rotulos_otsu == 0, rotulos_otsu)
    axes[1, 0].imshow(componentes.T, cmap=cores, origin="lower",
                      aspect=aspecto, interpolation="nearest")
    axes[1, 0].set_title("Componentes de Otsu antes da remoção", fontsize=10)

    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
