
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from src.preprocessing import (
    carregar_imagem,
    extrair_fatia,
    preparar_intensidades,
    identificar_fonte,
    verificar_alinhamento,
)

from diagnosticos.diagnosticar_conexao_pulmonar import (
    preparar_mascara,
    diagnosticar,
)

from segmentacao_tres_metodos import inferir_caminho_mascara


def main():
    parser = argparse.ArgumentParser(
        description="Diagnóstico de todas as fatias de um exame."
    )
    parser.add_argument("imagem")
    parser.add_argument("--referencia", default=None)
    parser.add_argument(
        "--saida",
        default="resultados/diagnostico_exame_completo.csv"
    )
    args = parser.parse_args()

    caminho_ref = (
        args.referencia
        or inferir_caminho_mascara(args.imagem)
    )

    # Carrega os volumes uma única vez
    imagem = carregar_imagem(args.imagem)
    referencia = carregar_imagem(caminho_ref)

    verificar_alinhamento(imagem, referencia)

    fonte = identificar_fonte(args.imagem)
    total_fatias = imagem.shape[2]

    fatias_pulmonares = [
    k for k in range(total_fatias)
    if np.any(np.isin(extrair_fatia(referencia, k), [1, 2]))
]

    grupos = np.array_split(fatias_pulmonares, 3)

    regiao_por_fatia = {}

    for nome, grupo in zip(
        ["inicial", "central", "final"],
        grupos
    ):
        for k in grupo:
            regiao_por_fatia[int(k)] = nome

        registros = []

        print(f"\nTotal de fatias: {total_fatias}")

        for k in range(total_fatias):
            fatia = extrair_fatia(imagem, k)
            rotulos_ref = extrair_fatia(referencia, k)

            if not np.any(np.isin(rotulos_ref, [1, 2])):
                continue

            # Mesmo preparo de intensidade usado no projeto
            dados = preparar_intensidades(fatia, fonte)
            imagem_255 = dados["imagem_255"]

            mascara = preparar_mascara(imagem_255)
            resultado = diagnosticar(mascara, rotulos_ref)

            if k not in regiao_por_fatia:
                print(f"ATENÇÃO: fatia {k} sem região definida")
                
            registro = {

                "exame": Path(args.imagem).name,
                "fatia": k,
                "situacao": resultado["situacao"],
                "componentes": resultado["componentes"],
                "componente_1": resultado["componente_1"],
                "componente_2": resultado["componente_2"],
                "cobertura_1": resultado["cobertura_1"],
                "cobertura_2": resultado["cobertura_2"],
                "concentracao_1": resultado["concentracao_1"],
                "concentracao_2": resultado["concentracao_2"],
                "regiao": regiao_por_fatia.get(k, "nao_classificada"),
            }

            registros.append(registro)

            if resultado["situacao"] == "possivel_conexao":
                print(f"Possível conexão na fatia {k}")

            if k % 25 == 0:
                print(f"Processando fatia {k}/{total_fatias - 1}")

        if not registros:
            print("Nenhuma fatia com referência pulmonar encontrada.")
            return

        destino = Path(args.saida)
    destino.parent.mkdir(parents=True, exist_ok=True)

    with destino.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(
            arquivo, fieldnames=list(registros[0].keys())
        )
        escritor.writeheader()
        escritor.writerows(registros)

    contagem = Counter(r["situacao"] for r in registros)

    print("\n=== RESUMO DO EXAME ===")
    print(f"Fatias avaliadas: {len(registros)}")

    for situacao, quantidade in sorted(contagem.items()):
        print(f"{situacao}: {quantidade}")

    candidatas = [
        r["fatia"] for r in registros
        if r["situacao"] == "possivel_conexao"
    ]

    print("\nFatias com possível conexão:", candidatas)
    print(f"\nCSV salvo em: {destino}")


if __name__ == "__main__":
    main()
