import argparse

import matplotlib.pyplot as plt
from avaliar_segmentacao import calcular_dice, calcular_iou

import numpy as np 
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, disk

from src.preprocessing import carregar_imagem, obter_indice_fatia, preparar_fatia
from segmentacao_componentes import remover_componentes_da_borda

NUM_COMPONENTES_PULMAO = 2
RAIO_PROXIMIDADE = 3

def manter_maior_componente(mascara):
    rotulos = label(mascara, connectivity=2)
    regioes = regionprops(rotulos)

    if not regioes:
        return mascara, regioes, None

    regioes_ordenadas = sorted(regioes, key=lambda r: r.area, reverse=True)
    maior_regiao = regioes_ordenadas[0]
    principais = regioes_ordenadas[:NUM_COMPONENTES_PULMAO]
    rotulos_mantidos = {r.label for r in principais}

    mascara_principal = np.isin(rotulos, list(rotulos_mantidos))
    area_proxima = binary_dilation(mascara_principal, disk(RAIO_PROXIMIDADE))

    for r in regioes_ordenadas[NUM_COMPONENTES_PULMAO:]:
        coords = r.coords
        if area_proxima[coords[:, 0], coords[:, 1]].any():
            rotulos_mantidos.add(r.label)

    mascara_final = np.isin(rotulos, list(rotulos_mantidos))
    return mascara_final, regioes, maior_regiao

def main():
    parser = argparse.ArgumentParser(
        description="Mantém somente o maior componente conexo da máscara."
    )
    parser.add_argument("imagem", help="Caminho da tomografia NIfTI")
    parser.add_argument("--fatia", type=int, default=None)
    args = parser.parse_args()

    try:
        preparada = preparar_fatia(args.imagem, k=args.fatia)
    except (OSError, ValueError) as erro:
        parser.error(str(erro))

    k = preparada["indice_fatia"]
    fatia = preparada["imagem_255"]

    from skimage.filters import threshold_otsu
    limiar = float(threshold_otsu(fatia))
    mascara_otsu = fatia < limiar
    _, sem_borda = remover_componentes_da_borda(mascara_otsu)

    mascara_final, regioes, maior_regiao = manter_maior_componente(sem_borda)

    print("Fatia:", k)
    print("Quantidade de componentes antes da seleção:", len(regioes))
    for r in sorted(regioes, key=lambda r: r.area, reverse=True):
        print(f" componente {r.label}: {r.area} pixels")
    if maior_regiao is not None:
        print(f"Maior componente escolhido: {maior_regiao.label} "
                f"({maior_regiao.area} pixels)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(sem_borda.T, cmap="gray", origin="lower")
    axes[0].set_title("Antes: todos os componentes sem borda")
    axes[1].imshow(mascara_final.T, cmap="gray", origin="lower")
    axes[1].set_title("Depois: só o maior componente")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()