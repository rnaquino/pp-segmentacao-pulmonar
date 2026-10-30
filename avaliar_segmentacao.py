
"""
Avaliação quantitativa da segmentação pulmonar.

Métricas:
- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU / Jaccard)

Compara uma máscara gerada pelo algoritmo com uma máscara
de referência (ground truth).

Formatos aceitos: .nii, .nii.gz e .npy

Exemplo:
python avaliar_segmentacao.py \
    data/resultados/mascara_otsu.nii.gz \
    data/masks/Lung_Mask/coronacases_001.nii.gz
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def calcular_dice(mascara_predita, mascara_referencia):
    """Calcula o coeficiente Dice entre duas máscaras binárias."""

    predita = np.asarray(mascara_predita, dtype=bool)
    referencia = np.asarray(mascara_referencia, dtype=bool)

    if predita.shape != referencia.shape:
        raise ValueError("As máscaras devem ter as mesmas dimensões.")

    intersecao = np.count_nonzero(predita & referencia)
    total = np.count_nonzero(predita) + np.count_nonzero(referencia)

    # Duas máscaras vazias são consideradas concordantes.
    if total == 0:
        return 1.0

    return 2.0 * intersecao / total


def calcular_iou(mascara_predita, mascara_referencia):
    """Calcula Intersection over Union (índice de Jaccard)."""

    predita = np.asarray(mascara_predita, dtype=bool)
    referencia = np.asarray(mascara_referencia, dtype=bool)

    if predita.shape != referencia.shape:
        raise ValueError("As máscaras devem ter as mesmas dimensões.")

    intersecao = np.count_nonzero(predita & referencia)
    uniao = np.count_nonzero(predita | referencia)

    if uniao == 0:
        return 1.0

    return intersecao / uniao


def carregar_mascara(caminho):
    """Carrega uma máscara NIfTI ou NumPy."""

    caminho = Path(caminho)
    nome = caminho.name.lower()

    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    if nome.endswith((".nii", ".nii.gz")):
        imagem = nib.load(str(caminho))

        # Padroniza a orientação espacial para RAS.
        imagem = nib.as_closest_canonical(imagem)

        dados = np.asanyarray(imagem.dataobj)
        affine = imagem.affine

    elif nome.endswith(".npy"):
        dados = np.load(caminho, allow_pickle=False)
        affine = None

    else:
        raise ValueError("Formato não suportado. Use NIfTI ou NPY.")

    if not np.issubdtype(dados.dtype, np.number) and dados.dtype != np.bool_:
        raise ValueError("A máscara deve conter valores numéricos ou booleanos.")

    if not np.all(np.isfinite(dados)):
        raise ValueError("A máscara contém valores NaN ou infinitos.")

    # Todo valor diferente de zero representa a região segmentada.
    mascara = dados != 0

    return mascara, affine


def avaliar(mascara_predita, mascara_referencia, fatia=None):
    """Calcula Dice e IoU para um volume ou uma fatia específica."""

    predita, affine_predita = carregar_mascara(mascara_predita)
    referencia, affine_referencia = carregar_mascara(mascara_referencia)

    if predita.shape != referencia.shape:
        raise ValueError(
            f"Dimensões diferentes: {predita.shape} e {referencia.shape}"
        )

    # Se as duas máscaras têm geometria NIfTI, verificar alinhamento.
    if affine_predita is not None and affine_referencia is not None:
        if not np.allclose(
            affine_predita, affine_referencia, rtol=0, atol=1e-4
        ):
            raise ValueError(
                "As máscaras possuem geometrias espaciais diferentes."
            )

    if fatia is not None:
        if predita.ndim != 3:
            raise ValueError("A opção --fatia exige máscaras 3D.")

        if not 0 <= fatia < predita.shape[2]:
            raise ValueError("Índice de fatia fora dos limites.")

        predita = predita[:, :, fatia]
        referencia = referencia[:, :, fatia]

    dice = calcular_dice(predita, referencia)
    iou = calcular_iou(predita, referencia)

    print("\n=== Avaliação da Segmentação Pulmonar ===")
    print(f"Dimensões avaliadas: {predita.shape}")
    print(f"Dice: {dice:.4f}")
    print(f"IoU (Jaccard): {iou:.4f}")
    print("========================================\n")

    return {"dice": dice, "iou": iou}


def main():
    parser = argparse.ArgumentParser(
        description="Calcula Dice e IoU para segmentação pulmonar."
    )

    parser.add_argument(
        "predita",
        help="Caminho da máscara gerada pelo algoritmo."
    )

    parser.add_argument(
        "referencia",
        help="Caminho da máscara de referência."
    )

    parser.add_argument(
        "--fatia",
        type=int,
        default=None,
        help="Índice da fatia a avaliar. Padrão: volume inteiro."
    )

    args = parser.parse_args()

    avaliar(args.predita, args.referencia, args.fatia)


if __name__ == "__main__":
    main()
