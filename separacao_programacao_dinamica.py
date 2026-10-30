
"""
Separação pulmonar por programação dinâmica 2D.

Implementação experimental inspirada em Hu et al. (2001).

Utiliza a intensidade da CT para buscar um caminho
de separação entre os pulmões.

Não altera os algoritmos originais e não usa watershed.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from skimage.measure import label, regionprops

from src.preprocessing import preparar_fatia
from avaliar_segmentacao import calcular_dice, calcular_iou
from segmentacao_tres_metodos import inferir_caminho_mascara
from separacao_erosao_condicional import encontrar_sementes


def localizar_pulmoes(mascara):
    """Identifica as duas regiões principais já separadas."""

    rotulos = label(mascara, connectivity=2)

    regioes = sorted(
        regionprops(rotulos),
        key=lambda r: r.area,
        reverse=True
    )

    if len(regioes) < 2:
        return None

    r1, r2 = regioes[:2]

    if r1.centroid[0] > r2.centroid[0]:
        r1, r2 = r2, r1

    return rotulos == r1.label, rotulos == r2.label


def encontrar_caminho_dinamico(
    imagem,
    mascara,
    centro_x,
    largura_corredor=35,
    penalidade_movimento=0.25,
    peso_centro=0.02
):
    """
    Busca caminho vertical de custo mínimo.

    Pixels de maior intensidade têm menor custo.
    Restringe a busca à região central dos pulmões.
    """

    imagem = np.asarray(imagem, dtype=float)
    mascara = np.asarray(mascara, dtype=bool)

    if imagem.shape != mascara.shape or imagem.ndim != 2:
        raise ValueError("Imagem e máscara devem ser 2D e alinhadas.")

    if not np.all(np.isfinite(imagem)):
        raise ValueError("A imagem contém valores inválidos.")

    altura, largura = imagem.shape

    x_min = max(0, int(centro_x - largura_corredor))
    x_max = min(altura, int(centro_x + largura_corredor + 1))

    if x_max - x_min < 2:
        raise ValueError("Corredor de busca muito estreito.")

    # Normalização para trabalhar com diferentes escalas.
    p2, p98 = np.percentile(imagem, [2, 98])
    amplitude = max(float(p98 - p2), 1e-8)

    imagem_normalizada = np.clip(
        (imagem - p2) / amplitude, 0, 1
    )

    # Maior intensidade = menor custo.
    custo_intensidade = 1.0 - imagem_normalizada

    corredor = custo_intensidade[x_min:x_max, :].copy()

    # Penaliza caminhos muito distantes do centro estimado.
    posicoes_x = np.arange(x_min, x_max)

    penalidade_centro = (
        peso_centro * np.abs(posicoes_x - centro_x)
    )

    corredor += penalidade_centro[:, None]

    # Penaliza levemente caminhos fora da máscara,
    # mantendo a possibilidade de atravessar pequenos
    # espaços já separados.
    corredor += 0.15 * (~mascara[x_min:x_max, :])

    largura_busca = corredor.shape[0]

    # Programação dinâmica: cada coluna corresponde
    # a uma etapa do caminho.
    acumulado = np.full(
        corredor.shape, np.inf, dtype=float
    )

    predecessores = np.full(
        corredor.shape, -1, dtype=int
    )

    acumulado[:, 0] = corredor[:, 0]

    for y in range(1, largura):
        for x in range(largura_busca):

            candidatos = []

            for deslocamento in (-1, 0, 1):
                anterior = x + deslocamento

                if 0 <= anterior < largura_busca:
                    valor = (
                        acumulado[anterior, y - 1]
                        + penalidade_movimento
                        * abs(deslocamento)
                    )
                    candidatos.append((valor, anterior))

            melhor_custo, melhor_anterior = min(candidatos)

            acumulado[x, y] = (
                corredor[x, y] + melhor_custo
            )

            predecessores[x, y] = melhor_anterior

    # Reconstrói o caminho ótimo de trás para frente.
    caminho = np.zeros(largura, dtype=int)

    posicao = int(np.argmin(acumulado[:, -1]))

    for y in range(largura - 1, -1, -1):
        caminho[y] = posicao + x_min

        if y > 0:
            posicao = predecessores[posicao, y]

    return caminho


def separar_pulmoes(imagem, mascara, largura_corredor=35):
    """
    Separa os pulmões por programação dinâmica.

    Se a máscara já possui dois componentes principais,
    apenas identifica os componentes.

    Se estiver conectada, busca uma linha separadora.
    """

    mascara = np.asarray(mascara, dtype=bool)

    if not mascara.any():
        return None, None, None

    rotulos = label(mascara, connectivity=2)
    regioes = sorted(
        regionprops(rotulos),
        key=lambda r: r.area,
        reverse=True
    )

    # Verifica se já existem duas regiões relevantes.
    if len(regioes) >= 2:
        segunda_area = regioes[1].area

        if segunda_area >= 0.05 * mascara.sum():
            par = localizar_pulmoes(mascara)

            if par is not None:
                return par[0], par[1], None

    # Se há conexão, a erosão fornece apenas uma
    # estimativa inicial da localização dos pulmões.
    semente_e, semente_d, raio = encontrar_sementes(
        mascara, raio_maximo=30
    )

    if semente_e is None:
        print("Não foi possível localizar duas regiões iniciais.")
        return None, None, None

    centro_e = np.argwhere(semente_e)[:, 0].mean()
    centro_d = np.argwhere(semente_d)[:, 0].mean()

    centro_x = (centro_e + centro_d) / 2.0

    # Encontra o caminho separador usando intensidades.
    caminho = encontrar_caminho_dinamico(
        imagem,
        mascara,
        centro_x,
        largura_corredor=largura_corredor
    )

    # Divide a máscara de acordo com o caminho.
    posicoes_x = np.arange(mascara.shape[0])[:, None]
    separador = caminho[None, :]

    esquerda = mascara & (posicoes_x <= separador)
    direita = mascara & (posicoes_x > separador)

    if not esquerda.any() or not direita.any():
        return None, None, caminho

    return esquerda, direita, caminho


def avaliar(esquerda, direita, referencia):
    """Avalia Dice e IoU dos dois pulmões."""

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
        description="Separação pulmonar por programação dinâmica."
    )

    parser.add_argument("imagem")
    parser.add_argument("--fatia", type=int, default=None)
    parser.add_argument("--referencia", default=None)
    parser.add_argument(
        "--largura-corredor", type=int, default=35
    )

    args = parser.parse_args()

    caminho_referencia = (
        args.referencia
        or inferir_caminho_mascara(args.imagem)
    )

    # Usa o mesmo pré-processamento dos outros métodos.
    dados = preparar_fatia(
        args.imagem,
        k=args.fatia,
        caminho_mascara=caminho_referencia
    )

    from segmentacao_tres_metodos import segmentar_pulmao

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
        print("Separação realizada por programação dinâmica.")

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

    axes[1].set_title("Máscara e caminho calculado")

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
