"""Mostra uma fatia e a máscara final após refinamento morfológico em 0–255.

Usa imagem_255 de src.preprocessing para Coronacases e Radiopaedia.
Otsu é o método padrão.
--limiar é expresso em 0–255; 144.5 equivale a −400 HU somente na janela
Coronacases padrão. O raio é em milímetros; zero desliga apenas o fechamento.
O espaçamento vem do cabeçalho da máscara pulmonar alinhada, em mm.
Busca a referência em data/masks/Lung_Mask com o mesmo nome da imagem;
use --mascara para outro caminho. Os rótulos não participam da segmentação.
A máscara ainda pode conter componentes internos indesejados.
A figura contém apenas a tomografia e a máscara final, lado a lado.

Exemplo:
    python segmentacao_morfologia.py data/raw/coronacases_001.nii.gz --raio 2
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing import (
    preparar_fatia, CENTRO_JANELA_PULMAO, LARGURA_JANELA_PULMAO,
    carregar_imagem, verificar_alinhamento, obter_metadados,
)
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

from skimage.measure import label
from skimage.segmentation import clear_border


def remover_componentes_da_borda(mascara):
    """Remove regiões conectadas à borda, incluindo conexões diagonais."""
    rotulos = label(mascara, connectivity=2)
    return rotulos, clear_border(rotulos) > 0


def elemento_em_mm(raio_mm, spacing):
    """Discretiza um disco físico; admite pixels retangulares no plano.

    Cada deslocamento (i, j) entra se (i*sx)^2 + (j*sy)^2 <= raio_mm^2.
    Não arredonda o raio para um único inteiro nem usa o espaçamento entre fatias.
    """
    spacing = np.asarray(spacing, dtype=float)
    if not np.isfinite(raio_mm) or raio_mm < 0:
        raise ValueError("O raio em mm deve ser finito e não negativo.")
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Informe dois espaçamentos positivos e finitos em mm.")
    sx, sy = spacing
    nx, ny = np.floor(raio_mm / spacing).astype(int)
    x, y = np.ogrid[-nx:nx + 1, -ny:ny + 1]
    return np.hypot(x * sx, y * sy) <= raio_mm


def espacamento_referencia(caminho_imagem, caminho_mascara):
    """Lê apenas a geometria da referência, sem usar seus rótulos."""
    img = carregar_imagem(caminho_imagem)
    ref = carregar_imagem(caminho_mascara)
    verificar_alinhamento(img, ref)
    meta = obter_metadados(ref)
    if meta["unidade_espacial"] != "mm":
        raise ValueError("A máscara de referência precisa declarar unidade espacial mm.")
    spacing = np.asarray(meta["spacing"])
    eixos = meta["affine"][:3, :3]
    if not np.allclose(np.linalg.norm(eixos, axis=0), spacing, atol=1e-6, rtol=0):
        raise ValueError("Espaçamento da referência inconsistente com a affine.")
    direcoes = eixos / spacing
    if not np.allclose(direcoes.T @ direcoes, np.eye(3), atol=1e-5, rtol=0):
        raise ValueError("A grade possui cisalhamento; requer tratamento geométrico específico.")
    return tuple(spacing[:2])


def refinar_mascara(mascara, raio_mm, spacing):
    """Devolve resultados separados para observar cada operação."""
    elemento = elemento_em_mm(raio_mm, spacing)
    if raio_mm == 0:
        fechada = mascara.copy()
    else:
        fechada = ndi.binary_closing(mascara, structure=elemento)

    # Inclui regiões de fundo totalmente cercadas pela máscara.
    preenchida = ndi.binary_fill_holes(fechada)
    return fechada, preenchida


def main():
    parser = argparse.ArgumentParser(
        description="Experimenta fechamento e preenchimento de buracos em 2D."
    )
    parser.add_argument("imagem", help="Tomografia NIfTI Coronacases ou Radiopaedia")
    parser.add_argument("--fatia", type=int, default=None,
                        help="Opcional; sem este argumento, usa a fatia central")
    parser.add_argument("--metodo", choices=["fixo", "otsu"], default="otsu")
    parser.add_argument("--limiar", type=float, default=144.5,
                        help="Limiar em 0–255, usado somente no método fixo")
    parser.add_argument("--raio", type=float, default=2.0,
                        help="Raio em milímetros (padrão: 2 mm); 0 desliga o fechamento")
    parser.add_argument("--mascara", type=Path,
                        help="Referência pulmonar; padrão: ../masks/Lung_Mask/nome da imagem")
    parser.add_argument("--fonte", default=None,
                        choices=["Coronacases", "Radiopaedia"],
                        help="Opcional; detectada pelos prefixos coronacases_ e radiopaedia_")
    parser.add_argument("--centro", type=float, default=CENTRO_JANELA_PULMAO,
                        help="Centro da janela em HU, somente para Coronacases")
    parser.add_argument("--largura", type=float, default=LARGURA_JANELA_PULMAO,
                        help="Largura da janela em HU, somente para Coronacases")
    args = parser.parse_args()

    if not np.isfinite(args.raio) or args.raio < 0:
        parser.error("O raio em mm deve ser finito e não negativo.")
    if not np.isfinite(args.limiar) or not 0 <= args.limiar <= 255:
        parser.error("O limiar deve ser finito e estar entre 0 e 255.")

    try:
        imagem = Path(args.imagem)
        caminho_mascara = args.mascara or (
            imagem.parent.parent / "masks" / "Lung_Mask" / imagem.name
        )
        if not caminho_mascara.is_file():
            raise ValueError(
                f"Referência não encontrada: {caminho_mascara}. "
                "Informe o caminho com --mascara."
            )
        spacing = espacamento_referencia(imagem, caminho_mascara)
        dados = preparar_fatia(args.imagem, args.fonte, k=args.fatia,
                               centro=args.centro, largura=args.largura)
        k = dados["indice_fatia"]
        fatia = dados["imagem_255"]
    except (OSError, ValueError) as erro:
        parser.error(str(erro))
    if args.metodo == "otsu" and fatia.min() == fatia.max():
        parser.error("A fatia é constante: escolha outra para experimentar Otsu.")
    limiar = float(threshold_otsu(fatia)) if args.metodo == "otsu" else args.limiar

    candidata = fatia < limiar
    _, sem_borda = remover_componentes_da_borda(candidata)
    fechada, preenchida = refinar_mascara(sem_borda, args.raio, spacing)

    print(f"Fonte: {dados['source']}; escala de trabalho: 0–255")
    print(f"Fatia {k}; método {args.metodo}; limiar {limiar:.2f} (0–255)")
    print(f"Referência de geometria: {caminho_mascara}")
    print(f"Espaçamento no plano: {spacing[0]:.6g} × {spacing[1]:.6g} mm/pixel")
    print(f"Raio do fechamento: {args.raio:g} mm")
    print(f"Raio equivalente por eixo: {args.raio / spacing[0]:.3f} × "
          f"{args.raio / spacing[1]:.3f} pixels")
    print("Pixels após remover a borda:", np.count_nonzero(sem_borda))
    print("Pixels após fechamento:", np.count_nonzero(fechada))
    print("Pixels após preencher buracos:", np.count_nonzero(preenchida))
    print("Incluídos pelo fechamento:", np.count_nonzero(fechada & ~sem_borda))
    print("Removidos pelo fechamento:", np.count_nonzero(sem_borda & ~fechada))
    print("Incluídos ao preencher buracos:", np.count_nonzero(preenchida & ~fechada))

    sx, sy = dados["metadados"]["spacing"][:2]
    aspecto = float(sy) / float(sx)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(
        fatia.T, cmap="gray", origin="lower", aspect=aspecto,
        interpolation="nearest", vmin=0, vmax=255,
    )
    axes[0].set_title(f"{dados['source']} — fatia {k} (0–255)")
    axes[1].imshow(
        preenchida.T, cmap="gray", origin="lower", aspect=aspecto,
        interpolation="nearest", vmin=0, vmax=1,
    )
    axes[1].set_title("Máscara final após morfologia")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
