"""Mostra uma fatia de uma tomografia NIfTI.

Exemplo:
    python visualizar_fatia.py data/raw/coronacases_001.nii.gz --fatia 150

Os limites padrão são para imagens em HU. Para dados normalizados,
ajuste --vmin e --vmax à escala encontrada no arquivo.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing import (
    carregar_imagem, obter_indice_fatia, extrair_fatia, obter_metadados,
)


def main():
    # Recebe o arquivo e as opções pelo terminal.
    parser = argparse.ArgumentParser(description="Mostra uma fatia da tomografia.")
    parser.add_argument("imagem", help="Caminho do arquivo .nii ou .nii.gz")
    parser.add_argument("--fatia", type=int, help="Índice da fatia; padrão: central")
    parser.add_argument("--vmin", type=float, default=-1000)
    parser.add_argument("--vmax", type=float, default=400)
    args = parser.parse_args()

    try:
        img = carregar_imagem(args.imagem)
        k = obter_indice_fatia(img, args.fatia)
        fatia = extrair_fatia(img, k)
    except (OSError, ValueError) as erro:
        parser.error(str(erro))
    if not np.isfinite([args.vmin, args.vmax]).all() or args.vmin >= args.vmax:
        parser.error("vmin e vmax devem ser finitos, com vmin menor que vmax.")

    sx, sy = obter_metadados(img)["spacing"][:2]

    plt.figure(figsize=(6, 6))
    plt.imshow(
        fatia.T, cmap="gray", origin="lower", interpolation="nearest",
        vmin=args.vmin, vmax=args.vmax, aspect=float(sy) / float(sx),
    )
    plt.title(f"Tomografia — fatia {k}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
