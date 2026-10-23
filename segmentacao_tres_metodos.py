import argparse

import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion, disk
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt

from src.preprocessing import preparar_fatia
from segmentacao_componentes import remover_componentes_da_borda
from segmentacao_maior_componente import manter_maior_componente


def inferir_caminho_mascara(caminho_imagem):
    return caminho_imagem.replace("raw", "masks/Lung_Mask")


def segmentar_pulmao(caminho_imagem, k=None, caminho_mascara=None):
    """Roda a Tarefa 1 (Otsu + maior componente) e devolve a máscara do pulmão."""
    preparada = preparar_fatia(caminho_imagem, k=k, caminho_mascara=caminho_mascara)
    fatia = preparada["imagem_255"]
    limiar = float(threshold_otsu(fatia))
    mascara_otsu = fatia < limiar
    _, sem_borda = remover_componentes_da_borda(mascara_otsu)
    mascara_pulmao, _, _ = manter_maior_componente(sem_borda)
    return mascara_pulmao, preparada["indice_fatia"], preparada["rotulos_referencia"]


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


def completar_pedacos_soltos(rotulos_finais, mascara_pulmao):
    """Garante que pedacinhos soltos (não conectados a nenhuma semente) também sejam atribuídos a um lado."""
    faltando = mascara_pulmao & (rotulos_finais == 0)
    if not faltando.any():
        return rotulos_finais

    dist_para_esquerda = distance_transform_edt(rotulos_finais != 1)
    dist_para_direita = distance_transform_edt(rotulos_finais != 2)

    rotulos_completos = rotulos_finais.copy()
    for regiao in regionprops(label(faltando, connectivity=2)):
        coords = regiao.coords
        media_esquerda = dist_para_esquerda[coords[:, 0], coords[:, 1]].mean()
        media_direita = dist_para_direita[coords[:, 0], coords[:, 1]].mean()
        valor = 1 if media_esquerda <= media_direita else 2
        rotulos_completos[coords[:, 0], coords[:, 1]] = valor

    return rotulos_completos


def metodo_juncoes(mascara_pulmao, raio_maximo=15):
    """Método 1: erode a máscara até achar o ponto mais fino (junção ant/post) e quebrar em 2."""
    rotulos = label(mascara_pulmao, connectivity=2)
    regioes = regionprops(rotulos)

    if len(regioes) >= 2:
        mascara_erodida = mascara_pulmao
    else:
        mascara_erodida = mascara_pulmao
        raio = 1
        while raio <= raio_maximo:
            candidata = binary_erosion(mascara_pulmao, disk(raio))
            regioes_candidata = regionprops(label(candidata, connectivity=2))
            if len(regioes_candidata) >= 2:
                mascara_erodida = candidata
                break
            raio += 1

    rotulos_erodidos = label(mascara_erodida, connectivity=2)
    regioes_erodidas = regionprops(rotulos_erodidos)

    if len(regioes_erodidas) < 2:
        return None, None

    regioes_ordenadas = sorted(regioes_erodidas, key=lambda r: r.area, reverse=True)
    r1, r2 = regioes_ordenadas[0], regioes_ordenadas[1]

    if r1.centroid[0] < r2.centroid[0]:
        ancora_esquerda, ancora_direita = r1, r2
    else:
        ancora_esquerda, ancora_direita = r2, r1

    marcadores = np.zeros(mascara_pulmao.shape, dtype=np.int32)
    marcadores[rotulos_erodidos == ancora_esquerda.label] = 1
    marcadores[rotulos_erodidos == ancora_direita.label] = 2

    distancia = distance_transform_edt(mascara_pulmao)
    rotulos_finais = watershed(-distancia, markers=marcadores, mask=mascara_pulmao)
    rotulos_finais = completar_pedacos_soltos(rotulos_finais, mascara_pulmao)

    mascara_esquerdo = rotulos_finais == 1
    mascara_direito = rotulos_finais == 2
    return mascara_esquerdo, mascara_direito


def metodo_dilatacao_condicional(mascara_pulmao):
    """Método 2: planta uma semente no extremo esquerdo e outra no extremo direito, e cresce cada uma só dentro da máscara."""
    coords = np.argwhere(mascara_pulmao)
    if coords.size == 0:
        return None, None

    indice_esquerda = np.argmin(coords[:, 0])
    indice_direita = np.argmax(coords[:, 0])

    semente_esquerda = coords[indice_esquerda]
    semente_direita = coords[indice_direita]

    marcadores = np.zeros(mascara_pulmao.shape, dtype=np.int32)
    marcadores[semente_esquerda[0], semente_esquerda[1]] = 1
    marcadores[semente_direita[0], semente_direita[1]] = 2

    distancia = distance_transform_edt(mascara_pulmao)
    rotulos_finais = watershed(-distancia, markers=marcadores, mask=mascara_pulmao)
    rotulos_finais = completar_pedacos_soltos(rotulos_finais, mascara_pulmao)

    mascara_esquerdo = rotulos_finais == 1
    mascara_direito = rotulos_finais == 2
    return mascara_esquerdo, mascara_direito


def metodo_fatia_vizinha(mascara_atual, mascara_esquerda_vizinha, mascara_direita_vizinha):
    """Método 3: usa a separação já resolvida de uma fatia vizinha como semente pra dividir a fatia atual."""
    marcadores = np.zeros(mascara_atual.shape, dtype=np.int32)
    marcadores[mascara_esquerda_vizinha & mascara_atual] = 1
    marcadores[mascara_direita_vizinha & mascara_atual] = 2

    if not (marcadores == 1).any() or not (marcadores == 2).any():
        return None, None

    distancia = distance_transform_edt(mascara_atual)
    rotulos_finais = watershed(-distancia, markers=marcadores, mask=mascara_atual)
    rotulos_finais = completar_pedacos_soltos(rotulos_finais, mascara_atual)

    mascara_esquerdo = rotulos_finais == 1
    mascara_direito = rotulos_finais == 2
    return mascara_esquerdo, mascara_direito


def main():
    parser = argparse.ArgumentParser(
        description="Compara 3 métodos de separação esquerdo/direito."
    )
    parser.add_argument("imagem", help="Caminho da tomografia NIfTI")
    parser.add_argument("--fatia", type=int, default=None)
    args = parser.parse_args()

    caminho_mascara = inferir_caminho_mascara(args.imagem)

    try:
        mascara_pulmao, k, rotulos_referencia = segmentar_pulmao(
            args.imagem, k=args.fatia, caminho_mascara=caminho_mascara
        )
        mascara_vizinha, k_vizinha, _ = segmentar_pulmao(
            args.imagem, k=k - 1, caminho_mascara=caminho_mascara
        )
    except (OSError, ValueError) as erro:
        parser.error(str(erro))

    esquerdo_vizinho, direito_vizinho = metodo_dilatacao_condicional(mascara_vizinha)

    resultados = {
        "Junções ant/post": metodo_juncoes(mascara_pulmao),
        "Dilatação condicional": metodo_dilatacao_condicional(mascara_pulmao),
        "Fatia vizinha": (
            metodo_fatia_vizinha(mascara_pulmao, esquerdo_vizinho, direito_vizinho)
            if esquerdo_vizinho is not None else (None, None)
        ),
    }

    print("Fatia:", k, "| Fatia vizinha usada:", k_vizinha)
    print()

    fig, axes = plt.subplots(len(resultados), 2, figsize=(8, 4 * len(resultados)))

    for i, (nome, (mascara_esquerdo, mascara_direito)) in enumerate(resultados.items()):
        if mascara_esquerdo is None:
            print(f"{nome}: não conseguiu separar")
            continue

        if rotulos_referencia is not None:
            dice_esquerdo, dice_direito = validar_com_gabarito(
                mascara_esquerdo, mascara_direito, rotulos_referencia
            )
            print(f"{nome}: Dice esquerdo={dice_esquerdo:.3f}  Dice direito={dice_direito:.3f}")

        axes[i, 0].imshow(mascara_esquerdo.T, cmap="gray", origin="lower")
        axes[i, 0].set_title(f"{nome} - esquerdo")
        axes[i, 1].imshow(mascara_direito.T, cmap="gray", origin="lower")
        axes[i, 1].set_title(f"{nome} - direito")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()