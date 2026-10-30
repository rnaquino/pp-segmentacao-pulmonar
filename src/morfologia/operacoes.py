"""Operações morfológicas da máscara pulmonar e sua ordem de aplicação."""

from scipy import ndimage as ndi
from skimage.measure import label
from skimage.segmentation import clear_border

from .geometria import elemento_em_mm
from .traqueia import remover_traqueia


def remover_componentes_da_borda(mascara):
    """Remove regiões conectadas à borda, incluindo conexões diagonais."""
    rotulos = label(mascara, connectivity=2)
    return rotulos, clear_border(rotulos) > 0


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


def finalizar_mascara(sem_borda, exclusao, raio_mm, spacing):
    """A exclusão não pode alterar a morfologia fora de sua própria região."""
    fechada, preenchida = refinar_mascara(sem_borda, raio_mm, spacing)
    return fechada, preenchida, remover_traqueia(preenchida, exclusao)
