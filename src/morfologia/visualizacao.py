"""Exibição da tomografia e da máscara final em uma única figura."""

import matplotlib.pyplot as plt


def exibir_resultado(dados, final, exclusao):
    """Mostra somente os dois painéis finais, sem resultados intermediários."""
    k = dados["indice_fatia"]
    fatia = dados["imagem_255"]
    sx, sy = dados["metadados"]["spacing"][:2]
    aspecto = float(sy) / float(sx)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(
        fatia.T, cmap="gray", origin="lower", aspect=aspecto,
        interpolation="nearest", vmin=0, vmax=255,
    )
    axes[0].set_title(f"{dados['source']} — fatia {k} (0–255)")
    axes[1].imshow(
        final.T, cmap="gray", origin="lower", aspect=aspecto,
        interpolation="nearest", vmin=0, vmax=1,
    )
    axes[1].set_title("Máscara final" if exclusao.any()
                      else "Máscara final — sem exclusão neste corte", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    plt.show()
