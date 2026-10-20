"""Cria uma máscara por limiar fixo sobre imagem_255 do pré-processamento.

Identifica a fonte pelo prefixo do arquivo. Seleciona intensidades
menores que --limiar, sempre na escala 0–255. O padrão 144.5 corresponde
à conversão de −400 HU pela janela Coronacases padrão (−1250 a 250 HU).
É um ponto de partida, não um limiar validado para ambas as fontes.
O ar externo e outros espaços de ar também podem ser selecionados.

Exemplo:
    python segmentacao_limiar_fixo.py data/raw/coronacases_001.nii.gz --limiar 144.5
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing import (
    preparar_fatia, CENTRO_JANELA_PULMAO, LARGURA_JANELA_PULMAO,
)


def main():
    # Recebe o caminho da tomografia e os parâmetros pelo terminal.
    parser = argparse.ArgumentParser(
        description="Experimenta uma segmentação por limiar fixo."
    )
    parser.add_argument("imagem", help="Caminho da tomografia NIfTI")
    parser.add_argument("--fatia", type=int, default=None,
                        help="Opcional; sem este argumento, usa a fatia central")
    parser.add_argument("--limiar", type=float, default=144.5,
                        help="Limiar na escala 0–255; padrão: 144.5")
    parser.add_argument("--fonte", default=None,
                        choices=["Coronacases", "Radiopaedia"],
                        help="Opcional; detectada pelos prefixos coronacases_ e radiopaedia_")
    parser.add_argument("--centro", type=float, default=CENTRO_JANELA_PULMAO,
                        help="Centro da janela em HU, somente para Coronacases")
    parser.add_argument("--largura", type=float, default=LARGURA_JANELA_PULMAO,
                        help="Largura da janela em HU, somente para Coronacases")
    args = parser.parse_args()

    try:
        dados = preparar_fatia(args.imagem, args.fonte, k=args.fatia,
                               centro=args.centro, largura=args.largura)
        k = dados["indice_fatia"]
        fatia = dados["imagem_255"]
    except (OSError, ValueError) as erro:
        parser.error(str(erro))

    if not np.isfinite(args.limiar) or not 0 <= args.limiar <= 255:
        parser.error("O limiar deve ser finito e estar entre 0 e 255.")

    # Compara todos os pixels com o limiar.
    mascara = fatia < args.limiar

    print(f"Fonte: {dados['source']}; escala de trabalho: 0–255")
    print("Fatia:", k)
    print("Intensidade mínima:", fatia.min())
    print("Intensidade máxima:", fatia.max())
    print("Limiar:", args.limiar)
    print("Pixels selecionados:", np.count_nonzero(mascara))

    # Mantém as proporções físicas da imagem.
    sx, sy = dados["metadados"]["spacing"][:2]
    aspecto = float(sy) / float(sx)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(
        fatia.T,
        cmap="gray",
        origin="lower",
        vmin=0,
        vmax=255,
        aspect=aspecto,
        interpolation="nearest",
    )
    axes[0].set_title(f"{dados['source']} — fatia {k} (0–255)")

    axes[1].imshow(
        mascara.T,
        cmap="gray",
        origin="lower",
        vmin=0,
        vmax=1,
        aspect=aspecto,
        interpolation="nearest",
    )
    axes[1].set_title(f"Intensidade < {args.limiar:g} (0–255)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
