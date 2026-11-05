
"""
Avaliação da reinclusão pulmonar pelo ABM.

Compara:
- Máscaras originais;
- Máscaras após a reinclusão;
- Ground truth do Coronacases.

Métricas:
- Dice;
- IoU;
- Pixels corretamente reincluídos;
- Pixels indevidamente acrescentados.
"""

import argparse
from pathlib import Path

import numpy as np

from src.preprocessing import preparar_fatia
from avaliar_segmentacao import calcular_dice, calcular_iou


def carregar_npz(caminho, chaves):
    caminho = Path(caminho)

    if not caminho.is_file():
        raise FileNotFoundError(caminho)

    with np.load(caminho, allow_pickle=False) as dados:
        return {
            chave: np.asarray(dados[chave], dtype=bool)
            for chave in chaves
        }


def avaliar_lado(nome, original, reincluida, referencia):
    """Compara a máscara antes e depois da reinclusão."""

    if not (
        original.shape == reincluida.shape == referencia.shape
    ):
        raise ValueError(
            f"Dimensões incompatíveis no pulmão {nome}"
        )

    if np.any(original & ~reincluida):
        raise ValueError(
            f"A reinclusão removeu pixels do pulmão {nome}"
        )

    adicionados = reincluida & ~original

    recuperados = adicionados & referencia
    falsos_positivos = adicionados & ~referencia

    dice_antes = calcular_dice(original, referencia)
    dice_depois = calcular_dice(reincluida, referencia)

    iou_antes = calcular_iou(original, referencia)
    iou_depois = calcular_iou(reincluida, referencia)

    print(f"\n=== PULMÃO {nome.upper()} ===")

    print(f"Dice antes:  {dice_antes:.4f}")
    print(f"Dice depois: {dice_depois:.4f}")
    print(f"Variação Dice: {dice_depois - dice_antes:+.4f}")

    print(f"IoU antes:  {iou_antes:.4f}")
    print(f"IoU depois: {iou_depois:.4f}")
    print(f"Variação IoU: {iou_depois - iou_antes:+.4f}")

    print(f"Pixels acrescentados: {int(adicionados.sum())}")
    print(f"Pixels recuperados corretamente: {int(recuperados.sum())}")
    print(f"Pixels falsos positivos: {int(falsos_positivos.sum())}")

    if adicionados.any():
        precisao = recuperados.sum() / adicionados.sum()
        print(f"Precisão dos acréscimos: {precisao:.2%}")

    return original, reincluida, referencia


def main():
    parser = argparse.ArgumentParser(
        description="Avalia a reinclusão ABM contra o ground truth."
    )

    parser.add_argument(
        "--imagem",
        default="data/raw/coronacases_001.nii.gz"
    )

    parser.add_argument(
        "--referencia",
        default="data/masks/Lung_Mask/coronacases_001.nii.gz"
    )

    parser.add_argument(
        "--fatia",
        type=int,
        default=100
    )

    parser.add_argument(
        "--passo",
        type=float,
        default=50
    )

    args = parser.parse_args()

    # Carrega o ground truth pelo mesmo pré-processamento
    # utilizado no pipeline de segmentação.
    preparada = preparar_fatia(
        args.imagem,
        k=args.fatia,
        caminho_mascara=args.referencia
    )

    referencia = preparada["rotulos_referencia"]

    if referencia is None:
        raise ValueError("Ground truth não foi carregado.")

    referencia = np.asarray(referencia)

    if referencia.ndim != 2:
        raise ValueError("Referência deve ser 2D.")

    print("\n=== GROUND TRUTH ===")
    print("Fatia:", preparada["indice_fatia"])
    print("Dimensões:", referencia.shape)
    print("Rótulos:", np.unique(referencia))

    rotulos = set(np.unique(referencia).tolist())

    if not {1, 2}.issubset(rotulos):
        raise ValueError(
            "Esperados os rótulos pulmonares 1 e 2."
        )

    # Máscaras originais da etapa de separação.
    base = carregar_npz(
        f"resultados/mascaras_coronacases_001_fatia_{args.fatia}.npz",
        ["pulmao_esquerdo", "pulmao_direito"]
    )

    prefixo = f"resultados/abm_{{}}_passo_{args.passo:g}.npz"

    esquerdo = carregar_npz(
        prefixo.format("esquerdo"),
        ["mascara_original", "mascara_reincluida"]
    )

    direito = carregar_npz(
        prefixo.format("direito"),
        ["mascara_original", "mascara_reincluida"]
    )

    # Garante que o ABM partiu das máscaras esperadas.
    for lado, resultado in (
        ("esquerdo", esquerdo),
        ("direito", direito),
    ):
        if not np.array_equal(
            base[f"pulmao_{lado}"],
            resultado["mascara_original"]
        ):
            raise ValueError(
                f"A máscara original do ABM ({lado}) "
                "é diferente da máscara de separação."
            )

    # Escolhe a correspondência entre os rótulos da referência
    # e os pulmões usando somente as máscaras originais.
    # O mesmo mapeamento é preservado na avaliação posterior.
    ref1 = referencia == 1
    ref2 = referencia == 2

    e0 = base["pulmao_esquerdo"]
    d0 = base["pulmao_direito"]

    opcao_a = (
        calcular_dice(e0, ref1)
        + calcular_dice(d0, ref2)
    )

    opcao_b = (
        calcular_dice(e0, ref2)
        + calcular_dice(d0, ref1)
    )

    if opcao_a >= opcao_b:
        ref_e, ref_d = ref1, ref2
        print("Mapeamento: esquerdo=1, direito=2")
    else:
        ref_e, ref_d = ref2, ref1
        print("Mapeamento: esquerdo=2, direito=1")

    avaliar_lado(
        "esquerdo",
        e0,
        esquerdo["mascara_reincluida"],
        ref_e
    )

    avaliar_lado(
        "direito",
        d0,
        direito["mascara_reincluida"],
        ref_d
    )

    # Avaliação da união dos dois pulmões.
    original_total = e0 | d0

    reincluida_total = (
        esquerdo["mascara_reincluida"]
        | direito["mascara_reincluida"]
    )

    referencia_total = ref_e | ref_d

    print("\n=== AVALIAÇÃO COMBINADA ===")

    print(
        "Dice antes:",
        f"{calcular_dice(original_total, referencia_total):.4f}"
    )

    print(
        "Dice depois:",
        f"{calcular_dice(reincluida_total, referencia_total):.4f}"
    )

    print(
        "IoU antes:",
        f"{calcular_iou(original_total, referencia_total):.4f}"
    )

    print(
        "IoU depois:",
        f"{calcular_iou(reincluida_total, referencia_total):.4f}"
    )


if __name__ == "__main__":
    main()
