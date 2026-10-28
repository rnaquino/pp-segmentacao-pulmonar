
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.measure import label

from src.preprocessing import preparar_fatia
from segmentacao_componentes import remover_componentes_da_borda
from segmentacao_maior_componente import manter_maior_componente
from segmentacao_tres_metodos import inferir_caminho_mascara
from src.preprocessing import carregar_imagem


def preparar_mascara(imagem):
    """Reproduz a entrada provisória usada no último comparador."""
    if np.ptp(imagem) == 0:
        return np.zeros(imagem.shape, dtype=bool)

    limiar = threshold_otsu(imagem)
    mascara_otsu = imagem < limiar

    _, sem_borda = remover_componentes_da_borda(mascara_otsu)
    mascara, _, _ = manter_maior_componente(sem_borda)

    return binary_fill_holes(mascara).astype(bool)


def diagnosticar(mascara, referencia):
    """
    Usa o ground truth apenas para diagnóstico.

    Verifica qual componente da máscara recupera a maior
    parte de cada pulmão da referência.
    """
    ref1 = referencia == 1
    ref2 = referencia == 2

    if not ref1.any() or not ref2.any():
        return {
            "situacao": "referencia_incompleta",
            "componentes": 0,
            "cobertura_1": 0.0,
            "cobertura_2": 0.0,
            "concentracao_1": 0.0,
            "concentracao_2": 0.0,
            "componente_1": 0,
            "componente_2": 0,
            "rotulos": np.zeros(mascara.shape, dtype=int),
        }

    rotulos = label(mascara, connectivity=2)
    quantidade = int(rotulos.max())

    def analisar_lado(ref):
        area_ref = np.count_nonzero(ref)
        intersecao = mascara & ref

        cobertura = np.count_nonzero(intersecao) / area_ref

        indices, contagens = np.unique(
            rotulos[intersecao], return_counts=True
        )

        validos = indices > 0
        indices = indices[validos]
        contagens = contagens[validos]

        if len(indices) == 0:
            return 0, cobertura, 0.0

        posicao = int(np.argmax(contagens))
        principal = int(indices[posicao])

        # Fração dos pixels recuperados desse pulmão
        # que pertence ao componente dominante.
        concentracao = (
            float(contagens[posicao])
            / max(1, np.count_nonzero(intersecao))
        )

        return principal, cobertura, concentracao

    comp1, cobertura1, concentracao1 = analisar_lado(ref1)
    comp2, cobertura2, concentracao2 = analisar_lado(ref2)

    # Limiares exploratórios de triagem.
    # Não são critérios clínicos validados.
    if min(cobertura1, cobertura2) < 0.20:
        situacao = "pulmao_pouco_recuperado"

    elif comp1 == comp2:
        situacao = "possivel_conexao"

    elif min(concentracao1, concentracao2) < 0.80:
        situacao = "fragmentacao"

    else:
        situacao = "dois_componentes_distintos"

    return {
        "situacao": situacao,
        "componentes": quantidade,
        "cobertura_1": cobertura1,
        "cobertura_2": cobertura2,
        "concentracao_1": concentracao1,
        "concentracao_2": concentracao2,
        "componente_1": comp1,
        "componente_2": comp2,
        "rotulos": rotulos,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostica possíveis conexões e fragmentação pulmonar."
    )
    parser.add_argument("imagem")
    parser.add_argument("--referencia", default=None)
    parser.add_argument(
    "--fatias",
    type=int,
    nargs="+",
    default=None,
    help="Fatias específicas. Se omitido, analisa o exame inteiro."
    )

    parser.add_argument(
        "--saida",
        default="resultados/diagnostico_conexao.csv"
    )
    args = parser.parse_args()

    if args.fatias is None:
        volume = carregar_imagem(args.imagem)
        args.fatias = list(range(volume.shape[2]))

    caminho_referencia = (
        args.referencia
        or inferir_caminho_mascara(args.imagem)
    )

    registros = []
    fig, axes = plt.subplots(
        len(args.fatias), 3,
        figsize=(13, 4 * len(args.fatias)),
        squeeze=False
    )

    for i, k in enumerate(args.fatias):
        dados = preparar_fatia(
            args.imagem,
            k=k,
            caminho_mascara=caminho_referencia
        )

        imagem = dados["imagem_255"]
        referencia = dados["rotulos_referencia"]

        if referencia is None:
            raise ValueError("Máscara de referência indisponível.")

        mascara = preparar_mascara(imagem)
        resultado = diagnosticar(mascara, referencia)

        print(f"\n=== FATIA {k} ===")
        print(f"Situação: {resultado['situacao']}")
        print(f"Componentes totais: {resultado['componentes']}")
        print(f"Componente do pulmão 1: {resultado['componente_1']}")
        print(f"Componente do pulmão 2: {resultado['componente_2']}")
        print(f"Cobertura pulmão 1: {resultado['cobertura_1']:.1%}")
        print(f"Cobertura pulmão 2: {resultado['cobertura_2']:.1%}")
        print(f"Concentração pulmão 1: {resultado['concentracao_1']:.1%}")
        print(f"Concentração pulmão 2: {resultado['concentracao_2']:.1%}")

        registros.append({
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
        })

        imagens = [
            (imagem, "Tomografia", "gray"),
            (resultado["rotulos"], "Componentes conexos", "nipy_spectral"),
            (referencia, "Ground truth", "viridis"),
        ]

        for j, (matriz, titulo, cores) in enumerate(imagens):
            axes[i, j].imshow(
                matriz.T, cmap=cores,
                origin="lower", interpolation="nearest"
            )
            axes[i, j].set_title(f"Fatia {k} — {titulo}")
            axes[i, j].axis("off")

    destino = Path(args.saida)
    destino.parent.mkdir(parents=True, exist_ok=True)

    with destino.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(
            arquivo, fieldnames=list(registros[0].keys())
        )
        escritor.writeheader()
        escritor.writerows(registros)

    print(f"\nDiagnóstico salvo em: {destino}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
