
"""
Comparação experimental dos três métodos de separação pulmonar.

Usa a mesma entrada para os três métodos.
Calcula Dice e IoU para cada pulmão.
Registra resultados em CSV.

Não modifica os algoritmos originais.
"""

import argparse
import csv
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

from src.preprocessing import preparar_fatia
from segmentacao_maior_componente import manter_maior_componente
from segmentacao_componentes import remover_componentes_da_borda
from pipeline_base import preparar_mascara_base
from segmentacao_morfologia import espacamento_referencia

from avaliar_segmentacao import calcular_dice, calcular_iou

from separacao_erosao_condicional import (
    separar_pulmoes as metodo_erosao
)
from separacao_programacao_dinamica import (
    separar_pulmoes as metodo_dinamico
)
from separacao_maior_intensidade import (
    separar_pulmoes as metodo_intensidade
)


def preparar_mascara(imagem):
    """Reproduz o isolamento usado na comparação anterior."""

    if np.max(imagem) == np.min(imagem):
        return np.zeros(imagem.shape, dtype=bool)

    limiar = threshold_otsu(imagem)
    mascara_otsu = imagem < limiar

    # Remoção dos componentes conectados à borda
    _, sem_borda = remover_componentes_da_borda(mascara_otsu)

    # Seleção dos principais componentes conexos
    mascara, _, _ = manter_maior_componente(sem_borda)

    # Preenchimento de buracos — etapa fixa
    return binary_fill_holes(mascara).astype(bool)


def contar_componentes(mascara):
    rotulos = label(mascara, connectivity=2)

    regioes = sorted(
        regionprops(rotulos),
        key=lambda r: r.area,
        reverse=True
    )

    return len(regioes), [r.area for r in regioes]


def calcular_metricas(esquerda, direita, referencia):
    """Escolhe a melhor correspondência com os rótulos 1 e 2."""

    if esquerda is None or direita is None:
        return None

    ref1 = referencia == 1
    ref2 = referencia == 2

    if not ref1.any() or not ref2.any():
        return None

    opcao_a = (
        calcular_dice(esquerda, ref1)
        + calcular_dice(direita, ref2)
    )

    opcao_b = (
        calcular_dice(esquerda, ref2)
        + calcular_dice(direita, ref1)
    )

    if opcao_a >= opcao_b:
        ref_e, ref_d = ref1, ref2
    else:
        ref_e, ref_d = ref2, ref1

    return {
        "dice_esquerdo": calcular_dice(esquerda, ref_e),
        "dice_direito": calcular_dice(direita, ref_d),
        "iou_esquerdo": calcular_iou(esquerda, ref_e),
        "iou_direito": calcular_iou(direita, ref_d),
    }


def diferenca_entre_metodos(par_a, par_b):
    """Conta pixels divergentes, ignorando troca dos nomes dos lados."""

    if any(m is None for m in (*par_a, *par_b)):
        return None

    a_e, a_d = par_a
    b_e, b_d = par_b

    normal = (
        np.count_nonzero(a_e != b_e)
        + np.count_nonzero(a_d != b_d)
    )

    invertido = (
        np.count_nonzero(a_e != b_d)
        + np.count_nonzero(a_d != b_e)
    )

    return min(normal, invertido)

def normalizar_resultado(resultado):
    """Aceita métodos que retornam duas ou três informações."""

    if not isinstance(resultado, (tuple, list)):
        raise ValueError("O método não retornou uma tupla ou lista.")

    if len(resultado) == 2:
        esquerda, direita = resultado

    elif len(resultado) == 3:
        esquerda, direita, _ = resultado

    else:
        raise ValueError(
            f"Quantidade inesperada de retornos: {len(resultado)}"
        )

    if esquerda is not None:
        esquerda = np.asarray(esquerda, dtype=bool)

    if direita is not None:
        direita = np.asarray(direita, dtype=bool)

    return esquerda, direita

def executar(imagem, fatias, saida, referencia=None):
    registros = []

    if referencia is None:
        raise ValueError("Informe o caminho da máscara de referência.")

    spacing = espacamento_referencia(imagem, referencia)

    for k in fatias:
        print(f"\n========== FATIA {k} ==========")

        dados = preparar_fatia(
            imagem,
            k=k,
            caminho_mascara=referencia
        )

        imagem_255 = dados["imagem_255"]
        rotulos_ref = dados["rotulos_referencia"]
        indice = dados["indice_fatia"]

        mascara = preparar_mascara_base(
            imagem_255,
            spacing=spacing,
            raio_mm=3.0
        )

        quantidade, areas = contar_componentes(mascara)

        print(f"Componentes iniciais: {quantidade}")
        print(f"Áreas dos maiores: {areas[:3]}")

        # Identifica se já existem dois componentes relevantes.
        ja_separados = (
            len(areas) >= 2
            and areas[1] >= 0.05 * np.count_nonzero(mascara)
        )

        resultados = {}

        metodos = {
            "Erosao_condicional": lambda: metodo_erosao(mascara),
            "Programacao_dinamica": lambda: metodo_dinamico(
                imagem_255, mascara
            ),
            "Maior_intensidade": lambda: metodo_intensidade(
                imagem_255, mascara
            ),
        }

        for nome, funcao in metodos.items():
            try:
                esquerda, direita = normalizar_resultado(funcao())
                status = "resultado_obtido"

                if esquerda is None or direita is None:
                    status = "falha"

            except (ValueError, RuntimeError) as erro:
                print(f"{nome}: erro - {erro}")
                esquerda, direita = None, None
                status = "falha"

            resultados[nome] = (esquerda, direita)

            metricas = (
                calcular_metricas(esquerda, direita, rotulos_ref)
                if rotulos_ref is not None else None
            )

            registro = {
                "exame": Path(imagem).name,
                "fatia": indice,
                "metodo": nome,
                "componentes_iniciais": quantidade,
                "status": status,
                "dice_esquerdo": "",
                "dice_direito": "",
                "iou_esquerdo": "",
                "iou_direito": "",
            }

            if metricas is not None:
                registro.update(metricas)

                print(
                    f"{nome}: "
                    f"Dice E={metricas['dice_esquerdo']:.4f}, "
                    f"Dice D={metricas['dice_direito']:.4f}, "
                    f"IoU E={metricas['iou_esquerdo']:.4f}, "
                    f"IoU D={metricas['iou_direito']:.4f}"
                )
            else:
                print(f"{nome}: avaliação indisponível")

            registros.append(registro)

        print("\nDiferenças entre máscaras:")

        for (nome_a, par_a), (nome_b, par_b) in combinations(
            resultados.items(), 2
        ):
            diferenca = diferenca_entre_metodos(par_a, par_b)

            print(f"{nome_a} x {nome_b}: {diferenca} pixels")

    destino = Path(saida)
    destino.parent.mkdir(parents=True, exist_ok=True)

    with destino.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(
            arquivo,
            fieldnames=list(registros[0].keys())
        )
        escritor.writeheader()
        escritor.writerows(registros)

    print(f"\nResultados salvos em: {destino}")


def main():
    parser = argparse.ArgumentParser(
        description="Compara os três métodos de separação pulmonar."
    )

    parser.add_argument("imagem")

    parser.add_argument(
        "--fatias",
        type=int,
        nargs="+",
        required=True
    )

    parser.add_argument(
        "--referencia",
        default=None
    )

    parser.add_argument(
        "--saida",
        default="resultados/comparacao_separacao.csv"
    )

    args = parser.parse_args()

    # Mantém a mesma convenção de caminhos do projeto.
    referencia = args.referencia or args.imagem.replace(
        "raw", "masks/Lung_Mask"
    )

    executar(
        args.imagem,
        args.fatias,
        args.saida,
        referencia
    )


if __name__ == "__main__":
    main()
