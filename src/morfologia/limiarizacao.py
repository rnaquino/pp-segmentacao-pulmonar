"""Seleção de regiões de ar por limiar fixo ou Otsu na escala 0–255."""

from skimage.filters import threshold_otsu


def limiarizar(fatia, metodo, limiar_fixo):
    """Usa o mesmo critério em todas as fatias, na escala preparada 0–255."""
    if metodo == "otsu":
        if fatia.min() == fatia.max():
            raise ValueError("Fatia constante: Otsu não define uma região de ar útil.")
        limiar = float(threshold_otsu(fatia))
    else:
        limiar = limiar_fixo
    return fatia < limiar, limiar
