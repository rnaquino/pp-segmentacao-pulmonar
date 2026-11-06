
"""
Gera a máscara pulmonar final de uma fatia.

Fluxo:
1. Preparação da tomografia
2. Limiarização de Otsu
3. Remoção dos componentes da borda
4. Isolamento e refinamento morfológico
5. Separação dos pulmões
6. Reinclusão por Rolling Ball
7. Exportação da máscara final

A referência é usada para obter o espaçamento e avaliar
a saída, nunca para escolher pixels da segmentação.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from skimage.filters import threshold_otsu

from src.preprocessing import preparar_fatia
from segmentacao_morfologia import espacamento_referencia
from segmentacao_componentes import remover_componentes_da_borda
from separacao_final import segmentar_fatia
from reinclusao_rolling_ball import rolling_ball
from avaliar_segmentacao import calcular_dice, calcular_iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imagem",
        default="data/raw/coronacases_001.nii.gz"
    )
    parser.add_argument(
        "--referencia",
        default="data/masks/Lung_Mask/coronacases_001.nii.gz"
    )
    parser.add_argument("--fatia", type=int, default=207)
    parser.add_argument("--raio-base", type=float, default=3.0)
    parser.add_argument("--raio-rb", type=float, default=6.0)
    parser.add_argument(
        "--saida",
        default="resultados/mascara_final"
    )
    args = parser.parse_args()

    imagem_path = Path(args.imagem)
    ref_path = Path(args.referencia)

    if not imagem_path.is_file() or not ref_path.is_file():
        parser.error("Tomografia ou referência não encontrada.")

    if args.raio_base <= 0 or args.raio_rb <= 0:
        parser.error("Os raios devem ser positivos.")

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    # 1. Preparação da fatia e espaçamento.
    dados = preparar_fatia(
        str(imagem_path),
        k=args.fatia,
        caminho_mascara=str(ref_path),
    )

    ct = dados["imagem_255"]
    ref = dados["mascara_pulmonar"]
    k = dados["indice_fatia"]

    sx, sy = espacamento_referencia(
        str(imagem_path),
        str(ref_path)
    )

    # 2. Otsu e remoção dos componentes da borda
    # para registrar as etapas intermediárias.
    limiar = threshold_otsu(ct)
    mascara_otsu = ct < limiar

    _, sem_borda = remover_componentes_da_borda(
        mascara_otsu
    )

    # 3. Máscara-base e separação dos pulmões.
    resultado = segmentar_fatia(
        ct,
        spacing=(sx, sy),
        raio_mm=args.raio_base
    )

    status = resultado["status"]

    if status not in (
        "separacao_aplicada",
        "componentes_preservados_heuristica"
    ):
        raise RuntimeError(
            f"Separação não concluída: {status}"
        )

    esquerda = resultado["pulmao_esquerdo"].astype(bool)
    direita = resultado["pulmao_direito"].astype(bool)
    base = esquerda | direita

    if not esquerda.any() or not direita.any():
        raise RuntimeError(
            "A separação não produziu dois pulmões."
        )

    # 4. Rolling Ball em cada pulmão, separadamente.
    rb_esquerda, _ = rolling_ball(
        esquerda, args.raio_rb, sx, sy
    )
    rb_direita, _ = rolling_ball(
        direita, args.raio_rb, sx, sy
    )

    # 5. Preservar as máscaras originais e impedir
    # sobreposição das novas regiões entre os lados.
    novas_esquerda = rb_esquerda & ~base
    novas_direita = rb_direita & ~base

    ambiguas = novas_esquerda & novas_direita

    novas_esquerda &= ~ambiguas
    novas_direita &= ~ambiguas

    final_esquerda = esquerda | novas_esquerda
    final_direita = direita | novas_direita

    if np.any(final_esquerda & final_direita):
        raise RuntimeError(
            "Sobreposição entre os pulmões na máscara final."
        )

    mascara_final = final_esquerda | final_direita
    adicoes = mascara_final & ~base

    assert np.all(mascara_final[base])

    # 6. Salvar máscaras NumPy.
    prefixo = f"coronacases_001_fatia_{k}"

    np.save(
        saida / f"{prefixo}_FINAL.npy",
        mascara_final.astype(np.uint8)
    )

    np.savez_compressed(
        saida / f"{prefixo}_FINAL.npz",
        mascara_base=base,
        pulmao_esquerdo=final_esquerda,
        pulmao_direito=final_direita,
        mascara_final=mascara_final,
        pixels_reincluidos=adicoes,
        fatia=k,
        raio_base_mm=args.raio_base,
        raio_rolling_ball_mm=args.raio_rb,
        metodo="separacao_dinamica_rolling_ball",
    )

    # 7. Salvar NIfTI de uma única fatia, com
    # affine ajustada para a posição do corte 207.
    nii = nib.as_closest_canonical(
        nib.load(str(imagem_path))
    )

    affine = nii.affine.copy()
    centro_fatia = nii.affine @ np.array(
        [0, 0, k, 1.0]
    )
    affine[:3, 3] = centro_fatia[:3]

    header = nii.header.copy()
    header.set_data_dtype(np.uint8)

    nifti_final = nib.Nifti1Image(
        mascara_final.astype(np.uint8)[:, :, None],
        affine,
        header=header
    )

    nib.save(
        nifti_final,
        str(saida / f"{prefixo}_FINAL.nii.gz")
    )

    # 8. Figura com as etapas completas.
  
    # 8. Visualização em uma única linha (5 imagens).
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    figuras = [
        (ct, "1. Tomografia preparada"),
        (mascara_otsu, "2. Limiarização Otsu"),
        (sem_borda, "3. Sem componentes da borda"),
        (resultado["mascara_base"], "4. Refinamento morfológico"),
        (mascara_final, "7. Máscara final"),
    ]

    for ax, (imagem, titulo) in zip(axes, figuras):
        ax.imshow(
            imagem.T,
            cmap="gray",
            origin="lower",
            interpolation="nearest"
        )
        ax.set_title(titulo, fontsize=11)
        ax.axis("off")

    fig.suptitle(
        f"Coronacases 001 — Fatia {k} — Rolling Ball {args.raio_rb:g} mm",
        fontsize=15
    )

    fig.tight_layout()

    figura_path = saida / f"{prefixo}_FLUXO_COMPLETO.png"

    fig.savefig(
        figura_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)


    # 9. Salvar também a máscara isoladamente.
    plt.imsave(
        saida / f"{prefixo}_MASCARA.png",
        mascara_final.T.astype(np.uint8),
        cmap="gray",
        origin="lower",
        vmin=0,
        vmax=1
    )

    # 10. Avaliação (sem alterar a predição).
    dice_base = calcular_dice(base, ref)
    dice_final = calcular_dice(mascara_final, ref)
    iou_base = calcular_iou(base, ref)
    iou_final = calcular_iou(mascara_final, ref)

    recuperados = int(np.count_nonzero(adicoes & ref))
    falsos_positivos = int(
        np.count_nonzero(adicoes & ~ref)
    )

    metricas = {
        "exame": "coronacases_001",
        "fatia": k,
        "raio_base_mm": args.raio_base,
        "raio_rolling_ball_mm": args.raio_rb,
        "pixels_base": int(base.sum()),
        "pixels_final": int(mascara_final.sum()),
        "pixels_adicionados": int(adicoes.sum()),
        "recuperados_corretamente": recuperados,
        "falsos_positivos_adicionados": falsos_positivos,
        "dice_base": float(dice_base),
        "dice_final": float(dice_final),
        "iou_base": float(iou_base),
        "iou_final": float(iou_final),
    }

    with open(
        saida / f"{prefixo}_METRICAS.json",
        "w",
        encoding="utf-8"
    ) as arquivo:
        json.dump(
            metricas, arquivo, indent=2,
            ensure_ascii=False
        )

    print("\n=== MÁSCARA FINAL — FATIA 207 ===")
    print("Status da separação:", status)
    print(f"Rolling Ball: {args.raio_rb:g} mm")
    print(f"Pixels na base: {base.sum()}")
    print(f"Pixels reincluídos: {adicoes.sum()}")
    print(f"Pixels na máscara final: {mascara_final.sum()}")
    print(f"Recuperados corretamente: {recuperados}")
    print(f"Falsos positivos adicionados: {falsos_positivos}")
    print(f"Dice: {dice_base:.4f} -> {dice_final:.4f}")
    print(f"IoU: {iou_base:.4f} -> {iou_final:.4f}")
    print(f"\nArquivos salvos em: {saida.resolve()}")


if __name__ == "__main__":
    main()
