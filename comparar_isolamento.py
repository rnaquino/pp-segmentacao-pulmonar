
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu

from src.preprocessing import preparar_fatia
from segmentacao_componentes import remover_componentes_da_borda
from segmentacao_maior_componente import manter_maior_componente
from avaliar_segmentacao import calcular_dice, calcular_iou


def main():
    parser = argparse.ArgumentParser(
        description="Compara dois estágios de isolamento pulmonar."
    )

    parser.add_argument("imagem", help="Caminho da tomografia NIfTI")
    parser.add_argument("referencia", help="Caminho da máscara ground truth")
    parser.add_argument("--fatia", type=int, default=None)

    args = parser.parse_args()

    # 1. Carrega imagem e referência da mesma fatia
    dados = preparar_fatia(
        args.imagem,
        k=args.fatia,
        caminho_mascara=args.referencia
    )

    fatia = dados["imagem_255"]
    referencia = dados["mascara_pulmonar"]
    k = dados["indice_fatia"]

    if referencia is None:
        raise ValueError("Máscara de referência não disponível.")

    # 2. Aplica Otsu como etapa inicial comum
    limiar = threshold_otsu(fatia)
    mascara_otsu = fatia < limiar

    # 3. Método 1: remover componentes da borda
    _, mascara_borda = remover_componentes_da_borda(mascara_otsu)

    # 4. Método 2: selecionar componentes principais
    # a partir do resultado do método 1
    mascara_maior, _, _ = manter_maior_componente(mascara_borda)

    # 5. Calcula as métricas
    resultados = {
        "Remoção da borda": mascara_borda,
        "Seleção de componentes": mascara_maior
    }

    print(f"\nExame: {Path(args.imagem).name}")
    print(f"Fatia: {k}")
    print("\n=== COMPARAÇÃO DOS MÉTODOS ===")

    for nome, mascara in resultados.items():
        dice = calcular_dice(mascara, referencia)
        iou = calcular_iou(mascara, referencia)

        print(f"\nMétodo: {nome}")
        print(f"Dice: {dice:.4f}")
        print(f"IoU:  {iou:.4f}")

    # 6. Exibe as máscaras para comparação visual
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    imagens = [
        (referencia, "Ground Truth"),
        (mascara_borda, "Remoção da borda"),
        (mascara_maior, "Seleção de componentes")
    ]

    for ax, (imagem, titulo) in zip(axes, imagens):
        ax.imshow(imagem.T, cmap="gray", origin="lower")
        ax.set_title(titulo)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
