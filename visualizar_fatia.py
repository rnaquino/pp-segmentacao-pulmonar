"""Mostra uma fatia NIfTI preparada na escala 0–255, para ambas as fontes.

A fonte é identificada automaticamente pelo prefixo do arquivo.
Sem --fatia, usa o corte central. --centro e --largura configuram a janela
HU aplicada somente aos Coronacases. --vmin e --vmax ajustam a exibição
da imagem já preparada; seus padrões são 0 e 255.

Exemplo:
    python visualizar_fatia.py data/raw/coronacases_001.nii.gz
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing import (
    preparar_fatia, CENTRO_JANELA_PULMAO, LARGURA_JANELA_PULMAO,
)


def main():
    # Recebe o arquivo e as opções pelo terminal.
    parser = argparse.ArgumentParser(description="Mostra uma fatia da tomografia.")
    parser.add_argument("imagem", help="Caminho do arquivo .nii ou .nii.gz")
    parser.add_argument("--fatia", type=int, default=None,
                        help="Opcional; sem este argumento, usa a fatia central")
    parser.add_argument("--vmin", type=float, default=0)
    parser.add_argument("--vmax", type=float, default=255)
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
    if not np.isfinite([args.vmin, args.vmax]).all() or not 0 <= args.vmin < args.vmax <= 255:
        parser.error("Use limites finitos com 0 <= vmin < vmax <= 255.")

    sx, sy = dados["metadados"]["spacing"][:2]

    plt.figure(figsize=(6, 6))
    plt.imshow(
        fatia.T, cmap="gray", origin="lower", interpolation="nearest",
        vmin=args.vmin, vmax=args.vmax, aspect=float(sy) / float(sx),
    )
    plt.title(f"{dados['source']} — fatia {k} (0–255)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
