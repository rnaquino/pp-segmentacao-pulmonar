
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu

from src.preprocessing import preparar_fatia
from segmentacao_componentes import remover_componentes_da_borda
from segmentacao_maior_componente import manter_maior_componente
from avaliar_segmentacao import calcular_dice, calcular_iou
from segmentacao_tres_metodos import inferir_caminho_mascara


def main():
    parser = argparse.ArgumentParser(
        description="Avalia preenchimento de buracos na máscara pulmonar."
    )
    parser.add_argument("imagem")
    parser.add_argument("--fatia", type=int, default=100)
    parser.add_argument("--referencia", default=None)
    args = parser.parse_args()

    caminho_referencia = (
        args.referencia
        or inferir_caminho_mascara(args.imagem)
    )

    dados = preparar_fatia(
        args.imagem,
        k=args.fatia,
        caminho_mascara=caminho_referencia
    )

    imagem = dados["imagem_255"]
    rotulos = dados["rotulos_referencia"]

    # Mesmas etapas de isolamento do código original
    limiar = threshold_otsu(imagem)
    mascara_otsu = imagem < limiar

    _, sem_borda = remover_componentes_da_borda(mascara_otsu)
    mascara_original, _, _ = manter_maior_componente(sem_borda)

    # Preenchimento 2D dos buracos internos
    mascara_preenchida = binary_fill_holes(mascara_original)

    # Máscara pulmonar de referência
    referencia = np.isin(rotulos, [1, 2])

    # Calcula Dice e IoU antes e depois
    print("\n=== COMPARAÇÃO DO PREENCHIMENTO ===")

    for nome, mascara in [
        ("Original", mascara_original),
        ("Preenchida", mascara_preenchida)
    ]:
        print(f"\nMáscara: {nome}")
        print(f"Dice: {calcular_dice(mascara, referencia):.4f}")
        print(f"IoU:  {calcular_iou(mascara, referencia):.4f}")

    pixels_adicionados = mascara_preenchida & ~mascara_original
    print(f"\nPixels adicionados: {np.count_nonzero(pixels_adicionados)}")

    # Visualização
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    imagens = [
        (mascara_original, "Máscara original"),
        (mascara_preenchida, "Após preenchimento"),
        (referencia, "Ground truth"),
        (pixels_adicionados, "Pixels recuperados")
    ]

    for ax, (matriz, titulo) in zip(axes, imagens):
        ax.imshow(matriz.T, cmap="gray", origin="lower")
        ax.set_title(titulo)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
