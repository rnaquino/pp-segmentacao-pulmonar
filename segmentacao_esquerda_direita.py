import argparse

import matplotlib.pyplot as plt
import numpy as np

from avaliar_segmentacao import calcular_dice, calcular_iou
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, disk
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt

from src.preprocessing import preparar_fatia
from segmentacao_componentes import remover_componentes_da_borda
from segmentacao_maior_componente import manter_maior_componente

def segmentar_pulmao(caminho_imagem, k=None, caminho_mascara=None):
    """Roda a Tarefa 1 (Otsu + maior componente) e devolve a máscara do pulmão."""
    preparada = preparar_fatia(caminho_imagem, k=k, caminho_mascara=caminho_mascara)
    fatia = preparada["imagem_255"]
    limiar = float(threshold_otsu(fatia))
    mascara_otsu = fatia < limiar
    _, sem_borda = remover_componentes_da_borda(mascara_otsu)
    mascara_pulmao, _, _ = manter_maior_componente(sem_borda)
    return mascara_pulmao, preparada["indice_fatia"], preparada["rotulos_referencia"]

def separar_esquerdo_direito(mascara_pulmao):
    """Separa pulmão esquerdo/direito e distribui pedaços pequenos pro lado mais próximo."""
    rotulos = label(mascara_pulmao, connectivity=2)
    regioes = regionprops(rotulos)

    if len(regioes) < 2:
        return None, None

    regioes_ordenadas = sorted(regioes, key=lambda r: r.area, reverse=True)
    r1, r2 = regioes_ordenadas[0], regioes_ordenadas[1]

    if r1.centroid[0] < r2.centroid[0]:
        ancora_esquerda, ancora_direita = r1, r2
    else:
        ancora_esquerda, ancora_direita = r2, r1

    mascara_esquerdo = rotulos == ancora_esquerda.label
    mascara_direito = rotulos == ancora_direita.label

    dist_para_esquerda = distance_transform_edt(~mascara_esquerdo)
    dist_para_direita = distance_transform_edt(~mascara_direito)


    for r in regioes_ordenadas:
        if r.label in (ancora_esquerda.label, ancora_direita.label):
            continue
        coords = r.coords
        media_esquerda = dist_para_esquerda[coords[:, 0], coords[:, 1]].mean()
        media_direita = dist_para_direita[coords[:, 0], coords[:, 1]].mean()
        if media_esquerda <= media_direita:
            mascara_esquerdo = mascara_esquerdo | (rotulos == r.label)
        else:
            mascara_direito = mascara_direito | (rotulos == r.label)

    return mascara_esquerdo, mascara_direito


def inferir_caminho_mascara(caminho_imagem):
    return caminho_imagem.replace("raw", "masks/Lung_Mask")


def main():
    parser = argparse.ArgumentParser(
        description="Separa pulmão esquerdo e direito."
    )
    parser.add_argument("imagem", help="Caminho da tomografia NIfTI")
    parser.add_argument("--fatia", type=int, default=None)
    args = parser.parse_args()

    caminho_mascara = inferir_caminho_mascara(args.imagem)
    try:
        mascara_pulmao, k, rotulos_referencia = segmentar_pulmao(
            args.imagem, k=args.fatia, caminho_mascara=caminho_mascara
        )
    except (OSError, ValueError) as erro:
        parser.error(str(erro))

    print("TIPO:", type(rotulos_referencia), "SHAPE:", rotulos_referencia.shape, "VALORES UNICOS:", np.unique(rotulos_referencia))

    mascara_esquerdo, mascara_direito = separar_esquerdo_direito(mascara_pulmao)

    if mascara_esquerdo is None:
        print("Os pulmões estão conectados nessa fatia - precisa do método de fatia vizinha aqui.")
        return

    print("Fatias:", k)
    print("Pixels pulmão esquerdo:", int(mascara_esquerdo.sum()))
    print("Pixels pulmao direito:", int(mascara_direito.sum()))

    if rotulos_referencia is not None:
        dice_esquerdo, dice_direito = validar_com_gabarito(
            mascara_esquerdo, mascara_direito, rotulos_referencia
        )
        print(f"Dice pulmão esquerdo: {dice_esquerdo:.3f}")
        print(f"Dice pulmão direito: {dice_direito:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(mascara_esquerdo.T, cmap="gray", origin="lower")
    axes[0].set_title("Pulmão esquerdo")
    axes[1].imshow(mascara_direito.T, cmap="gray", origin="lower")
    axes[1].set_title("Pulmão direito")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def calcular_dice(mascara_a, mascara_b):
    intersecao = np.logical_and(mascara_a, mascara_b).sum()
    total = mascara_a.sum() + mascara_b.sum()
    return 1.0 if total == 0 else 2 * intersecao / total


def validar_com_gabarito(mascara_esquerdo, mascara_direito, rotulos_referencia):
    gabarito_1 = rotulos_referencia == 1
    gabarito_2 = rotulos_referencia == 2

    opcao_a = calcular_dice(mascara_esquerdo, gabarito_1) + calcular_dice(mascara_direito, gabarito_2)
    opcao_b = calcular_dice(mascara_esquerdo, gabarito_2) + calcular_dice(mascara_direito, gabarito_1)

    if opcao_a >= opcao_b:
        return calcular_dice(mascara_esquerdo, gabarito_1), calcular_dice(mascara_direito, gabarito_2)
    return calcular_dice(mascara_esquerdo, gabarito_2), calcular_dice(mascara_direito, gabarito_1)


if __name__ == "__main__":
    main()