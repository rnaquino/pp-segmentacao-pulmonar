
"""
Separação pulmonar 2D por heurística de maior intensidade.

Método experimental:
1. Recebe a máscara pulmonar inicial.
2. Verifica se os pulmões já estão separados.
3. Localiza duas regiões iniciais quando estão conectados.
4. Procura pixels de maior intensidade na região central.
5. Constrói uma linha separadora por decisões locais.
6. Calcula Dice e IoU utilizando a referência.

Não utiliza watershed nem programação dinâmica.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops

from avaliar_segmentacao import calcular_dice, calcular_iou
from src.preprocessing import preparar_fatia
from segmentacao_tres_metodos import (
    segmentar_pulmao,
    inferir_caminho_mascara,
)
from separacao_erosao_condicional import encontrar_sementes


def localizar_componentes(mascara):
    """Verifica se já existem dois pulmões separados."""

    rotulos = label(mascara, connectivity=2)

    regioes = sorted(
        regionprops(rotulos),
        key=lambda r: r.area,
        reverse=True
    )

    if len(regioes) < 2:
        return None

    primeira, segunda = regioes[:2]

    # Desconsidera componentes muito pequenos.
    if segunda.area < 0.05 * mascara.sum():
        return None

    if primeira.centroid[0] > segunda.centroid[0]:
        primeira, segunda = segunda, primeira

    return (
        rotulos == primeira.label,
        rotulos == segunda.label
    )


def encontrar_linha_maior_intensidade(
    imagem,
    mascara,
    centro_x,
    largura_corredor=25,
    deslocamento_maximo=3,
    penalidade_deslocamento=2.0,
    penalidade_centro=0.5,
):
    """
    Encontra uma linha por seleção local de pixels intensos.

    A cada coluna da imagem, escolhe um pixel próximo
    da posição anterior, favorecendo alta intensidade.

    Não calcula custos acumulados nem utiliza
    programação dinâmica.
    """

    imagem = np.asarray(imagem, dtype=float)
    mascara = np.asarray(mascara, dtype=bool)

    if imagem.ndim != 2 or imagem.shape != mascara.shape:
        raise ValueError("Imagem e máscara devem ser 2D e alinhadas.")

    if not np.all(np.isfinite(imagem)):
        raise ValueError("A imagem contém intensidades inválidas.")

    if largura_corredor <= 0 or deslocamento_maximo <= 0:
        raise ValueError("Os parâmetros de busca devem ser positivos.")

    altura, largura = imagem.shape

    coords = np.argwhere(mascara)

    if coords.size == 0:
        return None

    y_inicio = int(coords[:, 1].min())
    y_fim = int(coords[:, 1].max())

    x_min = max(0, int(centro_x - largura_corredor))
    x_max = min(altura - 1, int(centro_x + largura_corredor))

    # Normalização robusta das intensidades para 0 a 255.
    p2, p98 = np.percentile(imagem, [2, 98])
    amplitude = max(float(p98 - p2), 1e-8)

    intensidade = 255.0 * np.clip(
        (imagem - p2) / amplitude, 0, 1
    )

    caminho = np.full(largura, int(round(centro_x)), dtype=int)

    posicao_atual = int(
        np.clip(round(centro_x), x_min, x_max)
    )

    for y in range(y_inicio, y_fim + 1):

        candidatos = range(
            max(x_min, posicao_atual - deslocamento_maximo),
            min(x_max, posicao_atual + deslocamento_maximo) + 1
        )

        melhor_posicao = posicao_atual
        melhor_pontuacao = -np.inf

        for x in candidatos:

            # Favorece pixels de maior intensidade.
            pontuacao = intensidade[x, y]

            # Evita mudanças bruscas de direção.
            pontuacao -= (
                penalidade_deslocamento
                * abs(x - posicao_atual)
            )

            # Mantém a procura próxima da região central.
            pontuacao -= (
                penalidade_centro
                * abs(x - centro_x)
            )

            if pontuacao > melhor_pontuacao:
                melhor_pontuacao = pontuacao
                melhor_posicao = x

        posicao_atual = melhor_posicao
        caminho[y] = posicao_atual

    return caminho


def separar_pulmoes(
    imagem,
    mascara,
    largura_corredor=25,
):
    """
    Separa os pulmões pela heurística de intensidade.

    Retorna:
        esquerda, direita, caminho
    """

    mascara = np.asarray(mascara, dtype=bool)

    if mascara.ndim != 2 or not mascara.any():
        return None, None, None

    if np.asarray(imagem).shape != mascara.shape:
        raise ValueError("Imagem e máscara não possuem a mesma forma.")

    # Se os pulmões já estão separados, preserva o resultado.
    componentes = localizar_componentes(mascara)

    if componentes is not None:
        return componentes[0], componentes[1], None

    # Encontra duas regiões para estimar a faixa central.
    # A erosão é usada apenas para localizar a região,
    # não para efetuar a separação final.
    semente_e, semente_d, _ = encontrar_sementes(
        mascara,
        raio_maximo=30
    )

    if semente_e is None or semente_d is None:
        return None, None, None

    centro_e = np.argwhere(semente_e)[:, 0].mean()
    centro_d = np.argwhere(semente_d)[:, 0].mean()

    centro_x = (centro_e + centro_d) / 2.0

    caminho = encontrar_linha_maior_intensidade(
        imagem,
        mascara,
        centro_x,
        largura_corredor=largura_corredor
    )

    if caminho is None:
        return None, None, None

    # Divide a máscara seguindo a linha encontrada.
    posicoes_x = np.arange(mascara.shape[0])[:, None]
    linha = caminho[None, :]

    esquerda = mascara & (posicoes_x <= linha)
    direita = mascara & (posicoes_x > linha)

    if not esquerda.any() or not direita.any():
        return None, None, caminho

    return esquerda, direita, caminho


def avaliar(esquerda, direita, referencia):
    """Calcula Dice e IoU para ambos os pulmões."""

    ref1 = referencia == 1
    ref2 = referencia == 2

    if not ref1.any() or not ref2.any():
        print("Referência sem os dois rótulos pulmonares.")
        return

    opcao_a = (
        calcular_dice(esquerda, ref1)
        + calcular_dice(direita, ref2)
    )

    opcao_b = (
        calcular_dice(esquerda, ref2)
        + calcular_dice(direita, ref1)
    )

    if opcao_a >= opcao_b:
        ref_e, ref_d = ref1, ref2
    else:
        ref_e, ref_d = ref2, ref1

    print("\n=== AVALIAÇÃO ===")
    print(f"Dice esquerdo: {calcular_dice(esquerda, ref_e):.4f}")
    print(f"Dice direito:  {calcular_dice(direita, ref_d):.4f}")
    print(f"IoU esquerdo:  {calcular_iou(esquerda, ref_e):.4f}")
    print(f"IoU direito:   {calcular_iou(direita, ref_d):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Separação pulmonar por maior intensidade."
    )

    parser.add_argument("imagem")
    parser.add_argument("--fatia", type=int, default=None)
    parser.add_argument("--referencia", default=None)
    parser.add_argument(
        "--largura-corredor",
        type=int,
        default=25
    )

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

    mascara, k, referencia = segmentar_pulmao(
        args.imagem,
        k=args.fatia,
        caminho_mascara=caminho_referencia
    )

    imagem = dados["imagem_255"]

    esquerda, direita, caminho = separar_pulmoes(
        imagem,
        mascara,
        largura_corredor=args.largura_corredor
    )

    if esquerda is None:
        print("Não foi possível separar os pulmões.")
        return

    print(f"\nFatia analisada: {k}")

    if caminho is None:
        print("Os pulmões já estavam separados.")
    else:
        print("Separação realizada por maior intensidade.")

    if referencia is not None:
        avaliar(esquerda, direita, referencia)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(imagem.T, cmap="gray", origin="lower")
    axes[0].set_title("Tomografia")

    axes[1].imshow(mascara.T, cmap="gray", origin="lower")

    if caminho is not None:
        axes[1].plot(
            caminho,
            np.arange(len(caminho)),
            color="red",
            linewidth=1.5
        )

    axes[1].set_title("Máscara e linha de separação")

    axes[2].imshow(esquerda.T, cmap="gray", origin="lower")
    axes[2].set_title("Pulmão 1")

    axes[3].imshow(direita.T, cmap="gray", origin="lower")
    axes[3].set_title("Pulmão 2")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
