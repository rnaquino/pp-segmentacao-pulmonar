
"""Pipeline compartilhado de preparação da máscara pulmonar em 2D."""

import numpy as np
from skimage.filters import threshold_otsu

from segmentacao_componentes import remover_componentes_da_borda
from segmentacao_maior_componente import manter_maior_componente
from segmentacao_morfologia import (
    espacamento_referencia,
    refinar_mascara,
)


def preparar_mascara_base(imagem_255, spacing, raio_mm=3.0):
    """
    Prepara uma máscara pulmonar para os métodos de separação.

    Etapas:
    1. Otsu
    2. Remoção dos componentes da borda
    3. Seleção dos componentes principais
    4. Fechamento morfológico
    5. Preenchimento de buracos

    A referência não participa da segmentação.
    """

    imagem_255 = np.asarray(imagem_255)

    if np.ptp(imagem_255) == 0:
        return np.zeros(imagem_255.shape, dtype=bool)

    limiar = threshold_otsu(imagem_255)
    mascara_otsu = imagem_255 < limiar

    _, sem_borda = remover_componentes_da_borda(mascara_otsu)

    mascara_isolada, _, _ = manter_maior_componente(sem_borda)

    _, mascara_final = refinar_mascara(
        mascara_isolada,
        raio_mm=raio_mm,
        spacing=spacing,
    )

    return mascara_final.astype(bool)
