"""Inspeção visual dos métodos de separação pulmonar em 2D."""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.ndimage import distance_transform_edt
from skimage.measure import label

from avaliar_segmentacao import calcular_dice, calcular_iou
from src.preprocessing import preparar_fatia
from pipeline_base import preparar_mascara_base
from segmentacao_morfologia import espacamento_referencia
from separacao_erosao_condicional import separar_pulmoes as erosao
from separacao_programacao_dinamica import separar_pulmoes as dinamica
from separacao_maior_intensidade import separar_pulmoes as intensidade
from reinclusao_rolling_ball import rolling_ball


def ler_argumentos():
    parser = argparse.ArgumentParser(
        description="Inspeção visual dos métodos de separação pulmonar."
    )
    parser.add_argument("imagem", help="Arquivo NIfTI da tomografia.")
    parser.add_argument("--referencia", required=True)
    parser.add_argument("--fatias", nargs="+", type=int, required=True)
    parser.add_argument("--saida", default="resultados/inspecao_separacao.png")
    parser.add_argument(
        "--raio",
        type=float,
        default=3.0,
        help="Raio em mm do rolling ball aplicado após separar os lados.",
    )
    parser.add_argument("--mostrar", action="store_true")
    return parser.parse_args()


def normalizar_resultado(resultado):
    """Obtém as duas máscaras dos diferentes formatos de retorno."""
    if not isinstance(resultado, (tuple, list)):
        raise ValueError("Retorno inesperado do método.")

    if len(resultado) not in (2, 3):
        raise ValueError("O método deve retornar 2 ou 3 valores.")

    esquerda, direita = resultado[:2]

    if esquerda is None or direita is None:
        raise ValueError("O método não identificou os dois pulmões.")

    return (
        np.asarray(esquerda, dtype=bool),
        np.asarray(direita, dtype=bool),
    )


def correspondencia_referencia(esquerda, direita, referencia):
    referencia_1 = referencia == 1
    referencia_2 = referencia == 2

    if not referencia_1.any() or not referencia_2.any():
        return None, None

    direta = (
        calcular_dice(esquerda, referencia_1)
        + calcular_dice(direita, referencia_2)
    )
    invertida = (
        calcular_dice(esquerda, referencia_2)
        + calcular_dice(direita, referencia_1)
    )

    if direta >= invertida:
        return referencia_1, referencia_2

    return referencia_2, referencia_1


def imprimir_metricas(nome, etapa, esquerda, direita, ref_esquerda, ref_direita):
    valores = {
        "Dice A": calcular_dice(esquerda, ref_esquerda),
        "Dice B": calcular_dice(direita, ref_direita),
        "IoU A": calcular_iou(esquerda, ref_esquerda),
        "IoU B": calcular_iou(direita, ref_direita),
    }

    print(
        f"  {nome} — {etapa}: "
        f"Dice A/B={valores['Dice A']:.4f}/{valores['Dice B']:.4f}; "
        f"IoU A/B={valores['IoU A']:.4f}/{valores['IoU B']:.4f}"
    )
    return valores


def imprimir_resumo_metricas(metricas_por_metodo):
    print("\nResumo entre fatias: média ± desvio-padrão amostral")

    for nome, etapas in metricas_por_metodo.items():
        for etapa, registros in etapas.items():
            if not registros:
                continue

            print(f"  {nome} — {etapa}:")

            for metrica in ("Dice A", "Dice B", "IoU A", "IoU B"):
                valores = np.asarray(
                    [registro[metrica] for registro in registros],
                    dtype=float,
                )
                desvio = (
                    np.std(valores, ddof=1)
                    if valores.size > 1
                    else 0.0
                )
                print(
                    f"    {metrica}: "
                    f"{np.mean(valores):.4f} ± {desvio:.4f}"
                )


def reincluir_apos_separacao(esquerda, direita, raio_mm, spacing):
    """Aplica o fechamento a cada lado e resolve eventuais sobreposições."""
    sx, sy = spacing
    esquerda_base = np.asarray(esquerda, dtype=bool)
    direita_base = np.asarray(direita, dtype=bool)

    esquerda_final, novos_esquerda = rolling_ball(
        esquerda_base, raio_mm, sx, sy
    )
    direita_final, novos_direita = rolling_ball(
        direita_base, raio_mm, sx, sy
    )

    sobreposicao = esquerda_final & direita_final
    quantidade_sobreposta = int(sobreposicao.sum())

    if quantidade_sobreposta:
        distancia_esquerda = distance_transform_edt(~esquerda_base)
        distancia_direita = distance_transform_edt(~direita_base)
        atribuir_direita = (
            distancia_direita[sobreposicao]
            < distancia_esquerda[sobreposicao]
        )
        esquerda_final[sobreposicao] = ~atribuir_direita
        direita_final[sobreposicao] = atribuir_direita

    return (
        esquerda_final,
        direita_final,
        int(novos_esquerda.sum()),
        int(novos_direita.sum()),
        quantidade_sobreposta,
    )


def criar_rotulos(esquerda, direita):
    """Cria uma imagem com 0=fundo, 1=região A, 2=região B."""
    rotulos = np.zeros(esquerda.shape, dtype=np.uint8)
    rotulos[esquerda] = 1
    rotulos[direita] = 2
    return rotulos


def desenhar(ax, imagem, rotulos=None, referencia=None):
    ax.imshow(
        imagem.T,
        cmap="gray",
        origin="lower",
        vmin=0,
        vmax=255,
        interpolation="nearest",
    )

    if rotulos is not None:
        cores = ListedColormap(
            [
                (0, 0, 0, 0),
                (0, 0.9, 0.95, 0.55),
                (1, 0.55, 0, 0.55),
            ]
        )

        ax.imshow(
            rotulos.T,
            cmap=cores,
            origin="lower",
            vmin=0,
            vmax=2,
            interpolation="nearest",
        )

    if referencia is not None:
        for valor in (1, 2):
            regiao = referencia == valor
            if regiao.any():
                ax.contour(
                    regiao.T.astype(float),
                    levels=[0.5],
                    colors="lime",
                    linewidths=0.8,
                    linestyles="dashed",
                    origin="lower",
                )

    ax.axis("off")


def diferenca(mascaras_a, mascaras_b):
    esquerda_a, direita_a = mascaras_a
    esquerda_b, direita_b = mascaras_b

    direta = (
        np.count_nonzero(esquerda_a != esquerda_b)
        + np.count_nonzero(direita_a != direita_b)
    )
    invertida = (
        np.count_nonzero(esquerda_a != direita_b)
        + np.count_nonzero(direita_a != esquerda_b)
    )

    return min(direta, invertida)


def main():
    args = ler_argumentos()
    spacing = espacamento_referencia(args.imagem, args.referencia)

    fig, axes = plt.subplots(
        len(args.fatias),
        4,
        figsize=(19, 4.5 * len(args.fatias)),
        squeeze=False,
    )

    metodos = {
        "Erosão condicional": lambda img, mascara: erosao(mascara),
        "Programação dinâmica": lambda img, mascara: dinamica(img, mascara),
        "Maior intensidade": lambda img, mascara: intensidade(img, mascara),
    }

    metricas_por_metodo = {
        nome: {"antes": [], "depois": []}
        for nome in metodos
    }

    for linha, k in enumerate(args.fatias):
        dados = preparar_fatia(
            args.imagem,
            k=k,
            caminho_mascara=args.referencia,
        )

        imagem = dados["imagem_255"]
        referencia = dados["rotulos_referencia"]

        mascara = preparar_mascara_base(
            imagem,
            spacing=spacing,
            raio_mm=0.0,
        )

        n_componentes = int(label(mascara, connectivity=2).max())
        print(f"\nFatia {k}: {n_componentes} componentes na entrada")

        desenhar(axes[linha, 0], imagem, referencia=referencia)
        axes[linha, 0].set_title(f"Fatia {k} — Tomografia e referência")

        resultados = {}

        for coluna, (nome, metodo) in enumerate(metodos.items(), start=1):
            ax = axes[linha, coluna]

            try:
                esquerda_base, direita_base = normalizar_resultado(
                    metodo(imagem, mascara)
                )

                ref_a, ref_b = correspondencia_referencia(
                    esquerda_base, direita_base, referencia
                )

                if ref_a is not None:
                    metricas_por_metodo[nome]["antes"].append(
                        imprimir_metricas(
                            nome,
                            "antes",
                            esquerda_base,
                            direita_base,
                            ref_a,
                            ref_b,
                        )
                    )

                (
                    esquerda,
                    direita,
                    pixels_esquerda,
                    pixels_direita,
                    sobrepostos,
                ) = reincluir_apos_separacao(
                    esquerda_base,
                    direita_base,
                    args.raio,
                    spacing,
                )

                if ref_a is not None:
                    metricas_por_metodo[nome]["depois"].append(
                        imprimir_metricas(
                            nome,
                            "depois",
                            esquerda,
                            direita,
                            ref_a,
                            ref_b,
                        )
                    )

                print(
                    f"  {nome}: acrescentados "
                    f"{pixels_esquerda} px à esquerda e "
                    f"{pixels_direita} px à direita; "
                    f"{sobrepostos} px sobrepostos antes da atribuição."
                )

                resultados[nome] = (esquerda, direita)
                rotulos = criar_rotulos(esquerda, direita)

                desenhar(
                    ax,
                    imagem,
                    rotulos=rotulos,
                    referencia=referencia,
                )
                ax.set_title(nome)

            except (ValueError, RuntimeError) as erro:
                desenhar(ax, imagem, referencia=referencia)
                ax.set_title(f"{nome} — Falha")
                print(f"  {nome}: {erro}")

        print("Diferenças entre os métodos (pixels):")

        for (nome_a, mascaras_a), (nome_b, mascaras_b) in combinations(
            resultados.items(), 2
        ):
            print(
                f"  {nome_a} x {nome_b}: "
                f"{diferenca(mascaras_a, mascaras_b)}"
            )

    imprimir_resumo_metricas(metricas_por_metodo)

    fig.suptitle(
        f"Métodos + rolling ball pós-separação "
        f"(raio {args.raio:g} mm) — {Path(args.imagem).name}",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    destino = Path(args.saida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(destino, dpi=180, bbox_inches="tight")
    print(f"\nFigura salva em: {destino}")

    if args.mostrar:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
