
import argparse
import matplotlib.pyplot as plt
import numpy as np

from skimage.filters import threshold_otsu
from skimage.measure import label

from src.preprocessing import preparar_fatia
from segmentacao_componentes import remover_componentes_da_borda
from segmentacao_maior_componente import manter_maior_componente
from segmentacao_tres_metodos import inferir_caminho_mascara


def inspecionar(caminho_imagem, caminho_referencia, fatias):

    fig, axes = plt.subplots(
        len(fatias), 4,
        figsize=(16, 4 * len(fatias)),
        squeeze=False
    )

    for i, k in enumerate(fatias):

        dados = preparar_fatia(
            caminho_imagem,
            k=k,
            caminho_mascara=caminho_referencia
        )

        imagem = dados["imagem_255"]
        referencia = dados["rotulos_referencia"]

        # Segmentação inicial por Otsu
        limiar = threshold_otsu(imagem)
        mascara_otsu = imagem < limiar

        # Etapa 1: remoção dos componentes da borda
        _, sem_borda = remover_componentes_da_borda(
            mascara_otsu
        )

        # Etapa 2: seleção dos componentes principais
        mascara_final, _, _ = manter_maior_componente(
            sem_borda
        )

        componentes_antes = label(sem_borda, connectivity=2).max()
        componentes_depois = label(mascara_final, connectivity=2).max()

        imagens = [
            (imagem, "Tomografia"),
            (sem_borda, f"Sem borda ({componentes_antes} comp.)"),
            (mascara_final, f"Seleção ({componentes_depois} comp.)"),
            (referencia, "Ground truth")
        ]

        for j, (matriz, titulo) in enumerate(imagens):
            if matriz is None:
                axes[i, j].text(
                    0.5, 0.5, "Indisponível",
                    ha="center", va="center"
                )
            else:
                axes[i, j].imshow(
                    matriz.T,
                    cmap="gray",
                    origin="lower",
                    interpolation="nearest"
                )

            axes[i, j].set_title(f"Fatia {k} — {titulo}")
            axes[i, j].axis("off")

        removidos = np.count_nonzero(sem_borda & ~mascara_final)

        print(
            f"Fatia {k}: {componentes_antes} -> "
            f"{componentes_depois} componentes | "
            f"{removidos} pixels removidos"
        )

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Inspeção visual das etapas de isolamento pulmonar"
    )
    parser.add_argument("imagem")
    parser.add_argument("--referencia", default=None)
    parser.add_argument(
        "--fatias",
        type=int,
        nargs="+",
        default=[65, 100, 207]
    )

    args = parser.parse_args()

    referencia = (
        args.referencia
        or inferir_caminho_mascara(args.imagem)
    )

    inspecionar(args.imagem, referencia, args.fatias)


if __name__ == "__main__":
    main()
