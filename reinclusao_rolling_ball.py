import argparse
from pathlib import Path
import csv

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from src.preprocessing import preparar_fatia, carregar_imagem
from segmentacao_morfologia import (
    espacamento_referencia,
    limiarizar,
    identificar_semente_traqueia,
    rastrear_traqueia,
    remover_componentes_da_borda,
)


def disco_fisico(raio_mm, sx, sy):
    """Cria um disco físico considerando o espaçamento dos pixels."""
    ri = int(np.ceil(raio_mm / sx))
    rj = int(np.ceil(raio_mm / sy))

    i, j = np.mgrid[-ri:ri + 1, -rj:rj + 1]

    return (i * sx) ** 2 + (j * sy) ** 2 <= raio_mm ** 2


def rolling_ball(mascara, raio_mm, sx, sy):
    """
    Aproximação por fechamento morfológico com disco físico.

    Retorna:
        fechada: máscara após o fechamento.
        reincluida: pixels acrescentados pelo método.
    """
    mascara = np.asarray(mascara, dtype=bool)

    if raio_mm <= 0:
        raise ValueError("O raio deve ser maior que zero.")

    estrutura = disco_fisico(raio_mm, sx, sy)

    # Moldura para evitar perda artificial nas bordas da imagem.
    pad = max(estrutura.shape)
    ampliada = np.pad(mascara, pad, mode="constant")

    fechada = ndi.binary_closing(
        ampliada,
        structure=estrutura
    )[pad:-pad, pad:-pad]

    # Preserva todos os pixels que já existiam na entrada.
    fechada |= mascara

    reincluida = fechada & ~mascara

    return fechada, reincluida


def separar_pulmoes(mascara):
    """
    Identifica os dois maiores componentes e cria regiões de atuação
    independentes, atribuindo os fragmentos à região mais próxima.

    Os fragmentos originais são preservados.
    """
    rotulos, quantidade = ndi.label(
        mascara,
        structure=np.ones((3, 3), dtype=bool)
    )

    if quantidade < 2:
        raise ValueError(
            "Não foram encontrados dois componentes pulmonares principais."
        )

    tamanhos = np.bincount(rotulos.ravel())
    tamanhos[0] = 0

    principais = np.argsort(tamanhos)[-2:]
    ids_principais = [int(v) for v in principais if tamanhos[v] > 0]

    if len(ids_principais) != 2:
        raise ValueError("Não foi possível identificar dois pulmões.")

    sementes = np.isin(rotulos, ids_principais)

    # Cada pixel é associado à semente pulmonar mais próxima.
    _, indices = ndi.distance_transform_edt(
        ~sementes,
        return_indices=True
    )

    rotulos_proximos = rotulos[tuple(indices)]

    regioes = []
    for ident in ids_principais:
        regiao = mascara & (rotulos_proximos == ident)
        regioes.append(regiao)

    return regioes, ids_principais


def reincluir(mascara, raio_mm, sx, sy):
    """Aplica o método separadamente nas duas regiões pulmonares."""
    regioes, ids = separar_pulmoes(mascara)

    resultado = np.zeros_like(mascara, dtype=bool)

    for regiao in regioes:
        fechada, _ = rolling_ball(regiao, raio_mm, sx, sy)
        resultado |= fechada

    # Garante que nenhum pixel da máscara inicial seja perdido.
    resultado |= mascara

    pixels_novos = resultado & ~mascara

    return resultado, pixels_novos, ids


def dice(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)

    denominador = a.sum() + b.sum()
    if denominador == 0:
        return 1.0

    return 2 * np.count_nonzero(a & b) / denominador


def iou(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)

    uniao = np.count_nonzero(a | b)
    if uniao == 0:
        return 1.0

    return np.count_nonzero(a & b) / uniao


def main():
    parser = argparse.ArgumentParser(
        description="Experimento de reinclusão pulmonar por rolling ball."
    )

    parser.add_argument(
        "imagem",
        help="Caminho do exame NIfTI, por exemplo data/raw/coronacases_001.nii.gz"
    )
    parser.add_argument("--fatia", type=int, default=100)
    parser.add_argument(
        "--raios",
        type=float,
        nargs="+",
        default=[3, 6, 9, 12]
    )
    parser.add_argument(
        "--metodo",
        choices=["otsu", "fixo"],
        default="otsu"
    )
    parser.add_argument("--limiar", type=float, default=144.5)
    parser.add_argument("--mascara", type=Path, default=None)
    parser.add_argument(
        "--semente",
        type=int,
        nargs=3,
        metavar=("I", "J", "K"),
        default=None
    )

    args = parser.parse_args()

    imagem = Path(args.imagem)

    caminho_ref = args.mascara or (
        imagem.parent.parent / "masks" / "Lung_Mask" / imagem.name
    )

    if not caminho_ref.is_file():
        parser.error(f"Máscara de referência não encontrada: {caminho_ref}")

    if any(not np.isfinite(r) or r <= 0 for r in args.raios):
        parser.error("Todos os raios devem ser positivos e finitos.")

    # 1. Preparação da imagem e obtenção do espaçamento.
    spacing = espacamento_referencia(imagem, caminho_ref)

    dados = preparar_fatia(
        str(imagem),
        k=args.fatia
    )

    k = dados["indice_fatia"]
    fatia = dados["imagem_255"]

    img = carregar_imagem(imagem)

    # 2. Etapas já implementadas pelos colegas.
    candidata, limiar = limiarizar(
        fatia,
        args.metodo,
        args.limiar
    )

    if args.semente is None:
        semente, deteccao = identificar_semente_traqueia(
            img,
            dados["source"],
            k,
            spacing,
            metodo=args.metodo,
            limiar_fixo=args.limiar
        )
    else:
        semente = tuple(args.semente)
        deteccao = {"origem": "manual"}

    traqueia, historico = rastrear_traqueia(
        img,
        dados["source"],
        semente,
        k,
        spacing,
        metodo=args.metodo,
        limiar_fixo=args.limiar
    )

    _, sem_borda = remover_componentes_da_borda(candidata)
    sem_traqueia = sem_borda & ~traqueia

    # 3. Carrega a referência apenas para avaliação.
    ref_img = nib.as_closest_canonical(nib.load(str(caminho_ref)))
    ref = np.asanyarray(ref_img.dataobj)[:, :, k] > 0

    if ref.shape != sem_traqueia.shape:
        parser.error("A máscara de referência e a máscara segmentada têm formas diferentes.")

    sx, sy = spacing

    # 4. Avaliação da máscara inicial.
    faltavam = np.count_nonzero(ref & ~sem_traqueia)

    print("\n========== ENTRADA ==========")
    print(f"Exame: {imagem.name}")
    print(f"Fatia: {k}")
    print(f"Método de limiarização: {args.metodo}")
    print(f"Limiar: {limiar:.2f}")
    print(f"Espaçamento: {sx:.6f} x {sy:.6f} mm/pixel")
    print(f"Pixels na máscara inicial: {sem_traqueia.sum()}")
    print(f"Pixels de traqueia removidos: {(sem_borda & traqueia).sum()}")
    print(f"Pixels pulmonares ausentes na entrada: {faltavam}")
    print(f"Dice inicial: {dice(sem_traqueia, ref):.4f}")
    print(f"IoU inicial: {iou(sem_traqueia, ref):.4f}")

    # 5. Testa os raios sem modificar a máscara inicial.
    resultados = []

    print("\n========== RESULTADOS ==========")

    for raio in args.raios:
        final, novos, ids = reincluir(
            sem_traqueia,
            raio,
            sx,
            sy
        )

        acertos = np.count_nonzero(novos & ref)

        precisao = (
            acertos / novos.sum()
            if novos.sum() else float("nan")
        )

        recuperacao = (
            acertos / faltavam
            if faltavam else float("nan")
        )

        resultados.append((raio, final, novos))

        print(f"\nRaio: {raio:g} mm")
        print(f"Pixels reincluídos: {novos.sum()}")
        print(f"Dice: {dice(final, ref):.4f}")
        print(f"IoU: {iou(final, ref):.4f}")
        print(f"Precisão da reinclusão: {precisao:.4f}")
        print(f"Recuperação: {recuperacao:.4f}")

            # 6. Salvamento dos resultados para comparação futura.
    saida = Path("resultados/rolling_ball")
    pasta_mascaras = saida / "mascaras"

    pasta_mascaras.mkdir(parents=True, exist_ok=True)

    nome_exame = imagem.name.replace(".nii.gz", "").replace(".nii", "")

    # Tabela de métricas: inclui também a máscara inicial (baseline).
    registros = [{
        "metodo": "rolling_ball",
        "exame": nome_exame,
        "fatia": k,
        "raio_mm": "",
        "spacing_x_mm": sx,
        "spacing_y_mm": sy,
        "pixels_entrada": int(sem_traqueia.sum()),
        "pixels_reincluidos": 0,
        "pixels_saida": int(sem_traqueia.sum()),
        "dice": dice(sem_traqueia, ref),
        "iou": iou(sem_traqueia, ref),
        "precisao": "",
        "recuperacao": "",
    }]

    for raio, final, novos in resultados:
        acertos = np.count_nonzero(novos & ref)

        precisao = (
            acertos / novos.sum()
            if novos.sum() else ""
        )

        recuperacao = (
            acertos / faltavam
            if faltavam else ""
        )

        registros.append({
            "metodo": "rolling_ball",
            "exame": nome_exame,
            "fatia": k,
            "raio_mm": raio,
            "spacing_x_mm": sx,
            "spacing_y_mm": sy,
            "pixels_entrada": int(sem_traqueia.sum()),
            "pixels_reincluidos": int(novos.sum()),
            "pixels_saida": int(final.sum()),
            "dice": dice(final, ref),
            "iou": iou(final, ref),
            "precisao": precisao,
            "recuperacao": recuperacao,
        })

        # Salva as máscaras binárias sem transposição ou perda de informação.
        caminho_npz = pasta_mascaras / (
            f"{nome_exame}_f{k}_raio_{raio:g}mm.npz"
        )

        np.savez_compressed(
            caminho_npz,
            mascara_entrada=sem_traqueia,
            mascara_referencia=ref,
            mascara_final=final,
            pixels_reincluidos=novos,
            raio_mm=raio,
            spacing_mm=np.array([sx, sy]),
            exame=nome_exame,
            fatia=k,
        )

    # Salva a tabela desta execução.
    caminho_csv = saida / f"metricas_{nome_exame}_f{k}.csv"

    with open(caminho_csv, "w", newline="", encoding="utf-8") as arquivo:
        campos = list(registros[0].keys())
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(registros)

    print(f"\nMétricas salvas em: {caminho_csv}")
    print(f"Máscaras salvas em: {pasta_mascaras}")

    # 6. Visualização comparativa.
    fig, axes = plt.subplots(
        1,
        len(resultados) + 2,
        figsize=(4 * (len(resultados) + 2), 5)
    )

    axes[0].imshow(ref.T, cmap="gray", origin="lower")
    axes[0].set_title("Referência")

    axes[1].imshow(sem_traqueia.T, cmap="gray", origin="lower")
    axes[1].set_title("Máscara de entrada")

    for ax, (raio, _, novos) in zip(axes[2:], resultados):
        rgb = np.zeros(sem_traqueia.shape + (3,))
        rgb[sem_traqueia] = (1, 1, 1)
        rgb[novos] = (1, 0, 0)

        ax.imshow(rgb.transpose(1, 0, 2), origin="lower")
        ax.set_title(f"Reincluído: {raio:g} mm")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()

    saida = Path("resultados/rolling_ball")
    saida.mkdir(parents=True, exist_ok=True)

    caminho_figura = saida / f"rolling_ball_{imagem.stem}_f{k}.png"
    fig.savefig(caminho_figura, dpi=150, bbox_inches="tight")

    print(f"\nFigura salva em: {caminho_figura}")

    plt.show()


if __name__ == "__main__":
    main()
