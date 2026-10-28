
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.measure import label

from src.preprocessing import preparar_fatia
from segmentacao_componentes import remover_componentes_da_borda
from segmentacao_maior_componente import manter_maior_componente


def quantidade_componentes(mascara):
    return int(label(mascara, connectivity=2).max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagem")
    parser.add_argument("--fatia", type=int, required=True)
    args = parser.parse_args()

    dados = preparar_fatia(args.imagem, k=args.fatia)
    imagem = dados["imagem_255"]

    limiar = threshold_otsu(imagem)
    mascara_otsu = imagem < limiar

    _, sem_borda = remover_componentes_da_borda(mascara_otsu)

    mascara_maior, _, _ = manter_maior_componente(sem_borda)

    mascara_final = binary_fill_holes(mascara_maior)

    etapas = [
        ("Otsu", mascara_otsu),
        ("Remocao da borda", sem_borda),
        ("Componentes principais", mascara_maior),
        ("Preenchimento de buracos", mascara_final),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(17, 5))

    for ax, (titulo, mascara) in zip(axes, etapas):
        componentes = quantidade_componentes(mascara)

        print(f"{titulo}: {componentes} componentes")

        ax.imshow(
            mascara.T,
            cmap="gray",
            origin="lower"
        )

        ax.set_title(
            f"{titulo}\n{componentes} componentes"
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
