import numpy as np
from scipy import ndimage as ndi


def disco_fisico(raio_mm, sx, sy):
    """Disco em mm: vale mesmo com pixels retangulares."""
    ri = int(np.ceil(raio_mm / sx))
    rj = int(np.ceil(raio_mm / sy))
    i, j = np.mgrid[-ri:ri + 1, -rj:rj + 1]
    return (i * sx) ** 2 + (j * sy) ** 2 <= raio_mm ** 2


def rolling_ball(mascara, raio_mm, sx, sy):
    """Fechamento morfológico = bola rolando por fora da borda.
    Devolve (máscara fechada, só os pixels reincluídos)."""
    mascara = mascara.astype(bool)
    se = disco_fisico(raio_mm, sx, sy)

    pad = max(se.shape)  # sem a moldura, a erosão "come" a borda da imagem
    m = np.pad(mascara, pad)
    fechada = ndi.binary_closing(m, structure=se)[pad:-pad, pad:-pad]

    reincluida = fechada & ~mascara
    return fechada, reincluida


def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return (a & b).sum() / (a | b).sum()


if __name__ == "__main__":
    # Só roda quando você executa este arquivo diretamente.
    import matplotlib.pyplot as plt

    sx = sy = 0.7  # mm por pixel (valor típico destes exames)

    # "Pulmão" ideal: um círculo de raio 70 px.
    yy, xx = np.mgrid[0:200, 0:200]
    pulmao = (xx - 100) ** 2 + (yy - 100) ** 2 <= 70 ** 2

    # "Nódulo excluído": um buraco circular mordendo a borda de cima.
    buraco = (xx - 100) ** 2 + (yy - 35) ** 2 <= 12 ** 2
    mascara = pulmao & ~buraco  # o que a limiarização nos daria

    print(f"sem reinclusão: Dice={dice(mascara, pulmao):.4f}  IoU={iou(mascara, pulmao):.4f}")
    for raio in (3, 6, 9, 12):
        fechada, reinc = rolling_ball(mascara, raio, sx, sy)
        print(f"raio {raio:>2} mm: reincluídos={reinc.sum():>4} px  "
              f"Dice={dice(fechada, pulmao):.4f}  IoU={iou(fechada, pulmao):.4f}")

    fechada, reinc = rolling_ball(mascara, 12, sx, sy)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(mascara, cmap="gray"); ax[0].set_title("Máscara com o buraco")
    ax[1].imshow(fechada, cmap="gray"); ax[1].set_title("Fechada (raio 12 mm)")
    ax[2].imshow(reinc, cmap="gray"); ax[2].set_title("Reincluído")
    plt.savefig("teste_rolling_ball.png", dpi=100)
    plt.show()