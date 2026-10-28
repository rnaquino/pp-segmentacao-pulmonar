
import argparse

import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

from src.preprocessing import (
    carregar_imagem,
    extrair_fatia,
    preparar_intensidades,
    identificar_fonte,
    verificar_alinhamento,
)

from segmentacao_componentes import remover_componentes_da_borda
from segmentacao_maior_componente import manter_maior_componente
from segmentacao_tres_metodos import inferir_caminho_mascara


def identificar_fatias_conectadas(caminho_imagem, caminho_referencia):
    # Abre os arquivos apenas uma vez
    imagem = carregar_imagem(caminho_imagem)
    referencia = carregar_imagem(caminho_referencia)

    print(
    "Rótulos presentes na referência:",
    np.unique(np.asanyarray(referencia.dataobj))
    )

    verificar_alinhamento(imagem, referencia)

    fonte = identificar_fonte(caminho_imagem)
    total_fatias = imagem.shape[2]

    candidatas = []

    print(f"\nTotal de fatias: {total_fatias}")
    print("=== ANÁLISE DAS FATIAS ===")

    for k in range(total_fatias):

        # Mostra o andamento a cada 25 fatias
        if k % 25 == 0:
            print(
                f"Processando fatia {k}/{total_fatias - 1}",
                flush=True
            )

        # Extrai somente a fatia necessária
        fatia = extrair_fatia(imagem, k)
        rotulos_ref = extrair_fatia(referencia, k)

        # Avalia apenas fatias com ambos os pulmões
        if not (
            np.any(rotulos_ref == 1)
            and np.any(rotulos_ref == 2)
        ):
            continue

        # Mesmo preparo de intensidades do projeto
        dados = preparar_intensidades(fatia, fonte)
        imagem_255 = dados["imagem_255"]

        # Segmentação inicial por Otsu
        limiar = threshold_otsu(imagem_255)
        mascara_otsu = imagem_255 < limiar

        # Etapas originais de isolamento pulmonar
        _, sem_borda = remover_componentes_da_borda(
            mascara_otsu
        )

        mascara_pulmao, _, _ = manter_maior_componente(
            sem_borda
        )


        # Compara os componentes antes e depois da seleção
        componentes_antes = label(sem_borda, connectivity=2)
        componentes_depois = label(mascara_pulmao, connectivity=2)

        quantidade_antes = componentes_antes.max()
        quantidade_depois = componentes_depois.max()

        if quantidade_antes != quantidade_depois:
            print(
                f"Fatia {k}: "
                f"antes={quantidade_antes} componentes | "
                f"depois={quantidade_depois} componentes"
            )

        if not np.any(mascara_pulmao):
            continue

        # Analisa os componentes conexos
        rotulos = label(mascara_pulmao, connectivity=2)

        regioes = sorted(
            regionprops(rotulos),
            key=lambda regiao: regiao.area,
            reverse=True
        )

        if not regioes:
            continue

        area_total = np.count_nonzero(mascara_pulmao)
        maior_area = regioes[0].area

        proporcao = maior_area / area_total


        # Mostra a quantidade de componentes encontrados
        if len(regioes) <= 2:
            print(
                f"Fatia {k}: "
                f"{len(regioes)} componentes | "
                f"Maior componente: {proporcao:.1%}"
            )

        # Identifica possíveis pulmões conectados
        if proporcao >= 0.95:
            candidatas.append(
                (k, len(regioes), proporcao)
            )


            print(
                f"Fatia {k}: "
                f"{len(regioes)} componentes | "
                f"Maior componente: {proporcao:.1%}"
            )

    print("\n=== RESULTADO FINAL ===")

    if candidatas:
        print(f"Total de fatias candidatas: {len(candidatas)}")
        print(
            "Índices:",
            [item[0] for item in candidatas]
        )
    else:
        print("Nenhuma fatia candidata encontrada.")

    return candidatas


def main():
    parser = argparse.ArgumentParser(
        description="Localiza possíveis conexões entre pulmões."
    )

    parser.add_argument(
        "imagem",
        help="Caminho da tomografia NIfTI"
    )

    parser.add_argument(
        "--referencia",
        default=None,
        help="Caminho da máscara de referência"
    )

    args = parser.parse_args()

    caminho_referencia = (
        args.referencia
        or inferir_caminho_mascara(args.imagem)
    )

    identificar_fatias_conectadas(
        args.imagem,
        caminho_referencia
    )


if __name__ == "__main__":
    main()

