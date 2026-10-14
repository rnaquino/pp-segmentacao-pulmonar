"""Mostra os efeitos do fechamento e do preenchimento de buracos em uma fatia.

Exemplo para Coronacases em HU:
    python segmentacao_morfologia.py data/raw/coronacases_001.nii.gz --fatia 150 --raio 2

O raio é em PIXELS neste exercício. A máscara ainda não é uma segmentação
pulmonar final: componentes internos indesejados podem permanecer.
O arquivo segmentacao_componentes.py deve estar na mesma pasta.
"""

import argparse

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from src.preprocessing import (
    carregar_imagem, obter_indice_fatia, extrair_fatia, obter_metadados,
)
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import disk

from segmentacao_componentes import remover_componentes_da_borda


def refinar_mascara(mascara, raio):
    """Devolve resultados separados para observar cada operação."""
    if raio < 0:
        raise ValueError("O raio não pode ser negativo.")
    if raio == 0:
        fechada = mascara.copy()
    else:
        # disk cria a vizinhança circular usada no fechamento.
        elemento = disk(raio)
        fechada = ndi.binary_closing(mascara, structure=elemento)

    # Inclui regiões de fundo totalmente cercadas pela máscara.
    preenchida = ndi.binary_fill_holes(fechada)
    return fechada, preenchida


def main():
    parser = argparse.ArgumentParser(
        description="Experimenta fechamento e preenchimento de buracos em 2D."
    )
    parser.add_argument("imagem", help="Tomografia NIfTI em HU (Coronacases)")
    parser.add_argument("--fatia", type=int, default=None)
    parser.add_argument("--metodo", choices=["fixo", "otsu"], default="otsu")
    parser.add_argument("--limiar", type=float, default=-400,
                        help="Limiar em HU, usado somente no método fixo")
    parser.add_argument("--raio", type=int, default=2,
                        help="Raio do disco em pixels; 0 desliga o fechamento")
    args = parser.parse_args()

    if args.raio < 0:
        parser.error("O raio deve ser um inteiro não negativo.")
    if not np.isfinite(args.limiar):
        parser.error("O limiar precisa ser finito.")

    try:
        img = carregar_imagem(args.imagem)
        k = obter_indice_fatia(img, args.fatia)
        fatia = extrair_fatia(img, k)
    except (OSError, ValueError) as erro:
        parser.error(str(erro))
    if args.metodo == "otsu" and fatia.min() == fatia.max():
        parser.error("A fatia é constante: escolha outra para experimentar Otsu.")
    limiar = float(threshold_otsu(fatia)) if args.metodo == "otsu" else args.limiar

    candidata = fatia < limiar
    _, sem_borda = remover_componentes_da_borda(candidata)
    fechada, preenchida = refinar_mascara(sem_borda, args.raio)

    print(f"Fatia {k}; método {args.metodo}; limiar {limiar:.2f} HU")
    print(f"Raio do fechamento: {args.raio} pixels")
    print("Pixels após remover a borda:", np.count_nonzero(sem_borda))
    print("Pixels após fechamento:", np.count_nonzero(fechada))
    print("Pixels após preencher buracos:", np.count_nonzero(preenchida))
    print("Incluídos pelo fechamento:", np.count_nonzero(fechada & ~sem_borda))
    print("Removidos pelo fechamento:", np.count_nonzero(sem_borda & ~fechada))
    print("Incluídos ao preencher buracos:", np.count_nonzero(preenchida & ~fechada))

    sx, sy = obter_metadados(img)["spacing"][:2]
    aspecto = float(sy) / float(sx)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax in [axes[0, 0], axes[1, 2]]:
        ax.imshow(fatia.T, cmap="gray", origin="lower", aspect=aspecto,
                  interpolation="nearest", vmin=-1000, vmax=400)
    axes[0, 0].set_title(f"Tomografia — fatia {k}")
    axes[1, 2].set_title("Máscara refinada sobre a tomografia", fontsize=10)

    for ax, mascara, titulo in [
        (axes[0, 1], candidata, f"1. Limiarização: {limiar:.1f} HU"),
        (axes[0, 2], sem_borda, "2. Componentes de borda removidos"),
        (axes[1, 0], fechada, f"3. Fechamento: raio {args.raio} pixels"),
        (axes[1, 1], preenchida, "4. Buracos preenchidos"),
    ]:
        ax.imshow(mascara.T, cmap="gray", origin="lower", aspect=aspecto,
                  interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(titulo, fontsize=10)

    # Torna o fundo transparente para ver onde a máscara inclui a imagem.
    sobreposicao = np.ma.masked_where(~preenchida.T, preenchida.T)
    axes[1, 2].imshow(sobreposicao, cmap=ListedColormap(["red"]),
                      origin="lower", aspect=aspecto, interpolation="nearest",
                      vmin=0, vmax=1, alpha=0.35)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
