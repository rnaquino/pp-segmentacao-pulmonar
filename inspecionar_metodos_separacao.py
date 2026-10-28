
"""Inspeção visual dos métodos de separação pulmonar em 2D."""

from pathlib import Path
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from skimage.measure import label

from src.preprocessing import preparar_fatia
from pipeline_base import preparar_mascara_base
from segmentacao_morfologia import espacamento_referencia

from separacao_erosao_condicional import separar_pulmoes as erosao
from separacao_programacao_dinamica import separar_pulmoes as dinamica
from separacao_maior_intensidade import separar_pulmoes as intensidade


IMAGEM = "data/raw/coronacases_002.nii.gz"
REFERENCIA = "data/masks/Lung_Mask/coronacases_002.nii.gz"
FATIAS = [129, 130, 133, 134]


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


def criar_rotulos(esquerda, direita):
    """Cria uma imagem com 0=fundo, 1=região A, 2=região B."""
    rotulos = np.zeros(esquerda.shape, dtype=np.uint8)
    rotulos[esquerda] = 1
    rotulos[direita] = 2
    return rotulos


def desenhar(ax, imagem, rotulos=None, referencia=None):
    ax.imshow(
        imagem.T, cmap="gray", origin="lower",
        vmin=0, vmax=255, interpolation="nearest"
    )

    if rotulos is not None:
        cores = ListedColormap([
            (0, 0, 0, 0),
            (0, 0.9, 0.95, 0.55),
            (1, 0.55, 0, 0.55),
        ])

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
    ea, da = mascaras_a
    eb, db = mascaras_b

    direta = (
        np.count_nonzero(ea != eb)
        + np.count_nonzero(da != db)
    )
    invertida = (
        np.count_nonzero(ea != db)
        + np.count_nonzero(da != eb)
    )

    return min(direta, invertida)


def main():
    spacing = espacamento_referencia(IMAGEM, REFERENCIA)

    fig, axes = plt.subplots(
        len(FATIAS), 4,
        figsize=(19, 4.5 * len(FATIAS)),
        squeeze=False,
    )

    metodos = {
        "Erosão condicional": lambda img, m: erosao(m),
        "Programação dinâmica": lambda img, m: dinamica(img, m),
        "Maior intensidade": lambda img, m: intensidade(img, m),
    }

    for linha, k in enumerate(FATIAS):
        dados = preparar_fatia(
            IMAGEM,
            k=k,
            caminho_mascara=REFERENCIA,
        )

        imagem = dados["imagem_255"]
        referencia = dados["rotulos_referencia"]

        mascara = preparar_mascara_base(
            imagem,
            spacing=spacing,
            raio_mm=3.0,
        )

        n_componentes = int(
            label(mascara, connectivity=2).max()
        )

        print(f"\nFatia {k}: {n_componentes} componentes na entrada")

        desenhar(axes[linha, 0], imagem, referencia=referencia)
        axes[linha, 0].set_title(f"Fatia {k} — Tomografia e referência")

        resultados = {}

        for coluna, (nome, metodo) in enumerate(
            metodos.items(), start=1
        ):
            ax = axes[linha, coluna]

            try:
                esquerda, direita = normalizar_resultado(
                    metodo(imagem, mascara)
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

        for (nome_a, masks_a), (nome_b, masks_b) in combinations(
            resultados.items(), 2
        ):
            print(
                f"  {nome_a} x {nome_b}: "
                f"{diferenca(masks_a, masks_b)}"
            )

    fig.suptitle(
        "Comparação visual dos métodos de separação — Coronacases 002",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    destino = Path("resultados/inspecao_separacao_002.png")
    destino.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(destino, dpi=180, bbox_inches="tight")

    print(f"\nFigura salva em: {destino}")

    plt.show()


if __name__ == "__main__":
    main()
