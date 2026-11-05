from pathlib import Path
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours


# ============================================================
# 1. GEOMETRIA DOS CONTORNOS
# ============================================================

def calcular_area_assinada(contorno):
    """Calcula a área assinada de um contorno fechado em coordenadas (x, y)."""
    pontos = np.asarray(contorno, dtype=float)
    x = pontos[:, 0]
    y = pontos[:, 1]

    return 0.5 * np.sum(
        x[:-1] * y[1:] - x[1:] * y[:-1]
    )


def orientar_contorno_anti_horario(contorno):
    """Fecha e orienta o contorno no sentido anti-horário."""
    contorno = np.asarray(contorno, dtype=float)

    if contorno.ndim != 2 or contorno.shape[1] != 2:
        raise ValueError("O contorno precisa ter formato Nx2.")

    if len(contorno) > 1 and np.allclose(contorno[0], contorno[-1]):
        contorno = contorno[:-1]

    if len(contorno) < 3:
        raise ValueError("Contorno com poucos pontos.")

    contorno = np.vstack((contorno, contorno[0]))

    if calcular_area_assinada(contorno) < 0:
        contorno = contorno[::-1].copy()

    return contorno


# ============================================================
# 2. EXTRAÇÃO DOS CONTORNOS
# ============================================================

def extrair_contornos(mascara):
    """Extrai contornos da máscara e os converte para coordenadas (x, y)."""
    mascara = np.asarray(mascara).astype(bool)

    if mascara.ndim != 2:
        raise ValueError("A máscara precisa ser bidimensional.")

    print("\n=== EXTRAÇÃO DE CONTORNOS ABM ===")
    print("Dimensões:", mascara.shape)
    print("Pixels pulmonares:", np.count_nonzero(mascara))

    contornos_skimage = find_contours(
        mascara.astype(float),
        level=0.5,
    )

    contornos = []

    for contorno in contornos_skimage:
        # find_contours retorna (linha, coluna).
        # Convertemos para (x, y), com y orientado para cima.
        pontos = np.column_stack((
            contorno[:, 1],
            -contorno[:, 0],
        ))

        pontos = orientar_contorno_anti_horario(pontos)
        area = abs(calcular_area_assinada(pontos))

        contornos.append({
            "pontos": pontos,
            "area": area,
            "quantidade_pontos": len(pontos) - 1,
        })

    contornos.sort(
        key=lambda item: item["area"],
        reverse=True,
    )

    print("Contornos encontrados:", len(contornos))

    for i, item in enumerate(contornos):
        print(f"\nContorno {i + 1}")
        print("Quantidade de pontos:", item["quantidade_pontos"])
        print(f"Área: {item['area']:.2f} pixels²")

    return contornos


# ============================================================
# 3. VERIFICAÇÃO DO LADO DIREITO DE UMA RETA
# ============================================================

def ponto_a_direita(ponto_inicio, ponto_referencia, candidato):
    """Retorna True quando candidato está à direita da reta direcionada."""
    vetor_referencia = ponto_referencia - ponto_inicio
    vetor_candidato = candidato - ponto_inicio

    produto_vetorial = (
        vetor_referencia[0] * vetor_candidato[1]
        - vetor_referencia[1] * vetor_candidato[0]
    )

    return produto_vetorial < -1e-10


# ============================================================
# 4. NON-ADAPTIVE BORDER MARCHING
# ============================================================

def marchar_contorno_nao_adaptativo(contorno, passo=50.0):
    """Executa o Non-adaptive Border Marching e retorna tracks e contorno."""
    contorno = np.asarray(contorno, dtype=float)

    if contorno.ndim != 2 or contorno.shape[1] != 2:
        raise ValueError("O contorno precisa ter formato Nx2.")

    if len(contorno) < 4:
        raise ValueError("Contorno insuficiente para marching.")

    if not np.isfinite(passo) or passo <= 0:
        raise ValueError("O passo precisa ser positivo e finito.")

    contorno = orientar_contorno_anti_horario(contorno)

    pontos = contorno[:-1]
    quantidade = len(pontos)

    pontos_ext = np.vstack((
        pontos,
        pontos[0],
    ))

    distancias = np.linalg.norm(
        np.diff(pontos_ext, axis=0),
        axis=1,
    )

    pontos_marchados = [pontos[0]]
    segmentos = []

    indice_atual = 0

    while indice_atual < quantidade:

        # Determina a janela de busca pelo comprimento acumulado.
        limite = indice_atual
        distancia_percorrida = 0.0

        while (
            limite < quantidade
            and distancia_percorrida < passo
        ):
            distancia_percorrida += distancias[limite]
            limite += 1

        ponto_inicio = pontos_ext[indice_atual]

        # Os dois primeiros pontos formam a reta inicial.
        indice_extremo = indice_atual + 1
        ponto_extremo = pontos_ext[indice_extremo]

        # Procura o ponto mais à direita da reta de referência.
        for indice in range(
            indice_atual + 2,
            limite + 1,
        ):
            candidato = pontos_ext[indice]

            if ponto_a_direita(
                ponto_inicio,
                ponto_extremo,
                candidato,
            ):
                indice_extremo = indice
                ponto_extremo = candidato

        if indice_extremo <= indice_atual:
            raise RuntimeError(
                "O marching não avançou no contorno."
            )

        trecho_original = pontos_ext[
            indice_atual:indice_extremo + 1
        ].copy()

        largura = float(
            np.linalg.norm(
                ponto_extremo - ponto_inicio
            )
        )

        comprimento_original = float(
            np.sum(
                distancias[indice_atual:indice_extremo]
            )
        )

        segmentos.append({
            "inicio": ponto_inicio.copy(),
            "fim": ponto_extremo.copy(),
            "trecho_original": trecho_original,
            "largura": largura,
            "comprimento_original": comprimento_original,
        })

        if indice_extremo == quantidade:
            break

        indice_atual = indice_extremo
        pontos_marchados.append(ponto_extremo)

    pontos_marchados.append(pontos[0])

    return np.asarray(pontos_marchados), segmentos


# ============================================================
# 5. MÉTRICAS GEOMÉTRICAS DAS CONVEX TRACKS
# ============================================================

def medir_track(segmento, spacing_x=1.0, spacing_y=1.0):
    """
    Calcula W, Hmax e lambda para uma convex track.

    W: distância euclidiana entre início e fim.
    Hmax: maior distância dos pontos do trecho original
          ao segmento de reta entre início e fim.
    lambda: Hmax / W.

    A projeção dos pontos é limitada ao segmento:
    quando cai fora dele, considera-se a distância até
    o extremo mais próximo.

    spacing_x e spacing_y permitem converter as coordenadas
    para uma unidade física, como milímetros.
    Com ambos iguais a 1, as medidas são expressas em pixels.
    """

    if (
        not np.isfinite(spacing_x)
        or spacing_x <= 0
        or not np.isfinite(spacing_y)
        or spacing_y <= 0
    ):
        raise ValueError(
            "Os espaçamentos devem ser positivos e finitos."
        )

    inicio = np.asarray(segmento["inicio"], dtype=float)
    fim = np.asarray(segmento["fim"], dtype=float)
    trecho = np.asarray(
        segmento["trecho_original"],
        dtype=float
    )

    if inicio.shape != (2,) or fim.shape != (2,):
        raise ValueError(
            "Início e fim da track precisam ter duas coordenadas."
        )

    if trecho.ndim != 2 or trecho.shape[1] != 2 or len(trecho) < 2:
        raise ValueError(
            "O trecho original precisa ter pelo menos dois pontos Nx2."
        )

    if (
        not np.isfinite(trecho).all()
        or not np.isfinite(inicio).all()
        or not np.isfinite(fim).all()
    ):
        raise ValueError(
            "A track contém coordenadas não finitas."
        )

    # Converte as coordenadas para a unidade física escolhida.
    escala = np.array(
        [spacing_x, spacing_y],
        dtype=float
    )

    inicio_fisico = inicio * escala
    fim_fisico = fim * escala
    trecho_fisico = trecho * escala

    # --------------------------------------------------------
    # 1. Calcula W: distância entre os extremos da track.
    # --------------------------------------------------------

    vetor = fim_fisico - inicio_fisico
    W = float(np.linalg.norm(vetor))

    if W <= 1e-12:
        raise ValueError(
            "A track tem W igual a zero; lambda não pode ser calculado."
        )

    # --------------------------------------------------------
    # 2. Calcula a projeção de cada ponto sobre o segmento.
    # --------------------------------------------------------

    deslocamentos = trecho_fisico - inicio_fisico

    # Parâmetro da projeção sobre a reta.
    # Valores entre 0 e 1 correspondem ao segmento.
    parametros = (
        deslocamentos @ vetor
    ) / (W ** 2)

    # Limita a projeção aos extremos do segmento.
    parametros = np.clip(parametros, 0.0, 1.0)

    # Calcula as coordenadas dos pontos projetados.
    projecoes = (
        inicio_fisico
        + parametros[:, np.newaxis] * vetor
    )

    # --------------------------------------------------------
    # 3. Calcula a distância de cada ponto ao segmento.
    # --------------------------------------------------------

    diferencas = trecho_fisico - projecoes

    distancias = np.linalg.norm(
        diferencas,
        axis=1
    )

    Hmax = float(np.max(distancias))

    # --------------------------------------------------------
    # 4. Calcula lambda.
    # --------------------------------------------------------

    lambda_abm = float(Hmax / W)

    return {
        "W": W,
        "Hmax": Hmax,
        "lambda": lambda_abm,
        "quantidade_pontos_trecho": len(trecho),
    }


def medir_tracks(
    segmentos,
    spacing_x=1.0,
    spacing_y=1.0,
    lambda0=0.33,
):
    """Acrescenta W, Hmax, lambda e classificação a cada track."""

    if not np.isfinite(lambda0) or lambda0 <= 0:
        raise ValueError(
            "lambda0 precisa ser positivo e finito."
        )

    segmentos_medidos = []

    for indice, segmento in enumerate(segmentos, start=1):
        metricas = medir_track(
            segmento,
            spacing_x,
            spacing_y,
        )

        segmento_medido = dict(segmento)
        segmento_medido.update(metricas)

        segmento_medido["numero"] = indice
        segmento_medido["abaixo_lambda0"] = (
            metricas["lambda"] < lambda0
        )

        segmentos_medidos.append(segmento_medido)

    return segmentos_medidos


# ============================================================
# 6. EXPORTAÇÃO DAS MÉTRICAS
# ============================================================

def salvar_metricas_csv(segmentos, caminho, lambda0, unidade):
    """Salva uma linha por convex track com as métricas calculadas."""

    campos = [
        "numero",
        "inicio_x_px",
        "inicio_y_px",
        "fim_x_px",
        "fim_y_px",
        f"W_{unidade}",
        f"Hmax_{unidade}",
        "lambda",
        "lambda0",
        "quantidade_pontos_trecho",
        "abaixo_lambda0",
    ]

    with caminho.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as arquivo:

        escritor = csv.DictWriter(
            arquivo,
            fieldnames=campos,
        )

        escritor.writeheader()

        for segmento in segmentos:
            inicio = segmento["inicio"]
            fim = segmento["fim"]

            escritor.writerow({
                "numero": segmento["numero"],
                "inicio_x_px": float(inicio[0]),
                "inicio_y_px": float(inicio[1]),
                "fim_x_px": float(fim[0]),
                "fim_y_px": float(fim[1]),
                f"W_{unidade}": segmento["W"],
                f"Hmax_{unidade}": segmento["Hmax"],
                "lambda": segmento["lambda"],
                "lambda0": lambda0,
                "quantidade_pontos_trecho": (
                    segmento["quantidade_pontos_trecho"]
                ),
                "abaixo_lambda0": segmento["abaixo_lambda0"],
            })


def imprimir_resumo_metricas(segmentos, lambda0, unidade):
    """Imprime estatísticas de W, Hmax e lambda."""

    if not segmentos:
        print("Nenhuma convex track foi gerada.")
        return

    larguras = np.array(
        [s["W"] for s in segmentos],
        dtype=float,
    )

    alturas = np.array(
        [s["Hmax"] for s in segmentos],
        dtype=float,
    )

    lambdas = np.array(
        [s["lambda"] for s in segmentos],
        dtype=float,
    )

    abaixo = int(
        np.count_nonzero(lambdas < lambda0)
    )

    acima_ou_igual = len(segmentos) - abaixo

    print("\n=== MÉTRICAS DAS CONVEX TRACKS ===")
    print(f"Unidade de W e Hmax: {unidade}")
    print(f"Limiar lambda0: {lambda0:.3f}")
    print(f"Tracks analisadas: {len(segmentos)}")

    print(f"W médio: {np.mean(larguras):.3f} {unidade}")
    print(f"W mínimo: {np.min(larguras):.3f} {unidade}")
    print(f"W máximo: {np.max(larguras):.3f} {unidade}")

    print(f"Hmax médio: {np.mean(alturas):.3f} {unidade}")
    print(f"Hmax mínimo: {np.min(alturas):.3f} {unidade}")
    print(f"Hmax máximo: {np.max(alturas):.3f} {unidade}")

    print(f"lambda médio: {np.mean(lambdas):.4f}")
    print(f"lambda mínimo: {np.min(lambdas):.4f}")
    print(f"lambda máximo: {np.max(lambdas):.4f}")

    print(f"Tracks com lambda < {lambda0:.3f}: {abaixo}")
    print(f"Tracks com lambda >= {lambda0:.3f}: {acima_ou_igual}")

    print("\nObservação: esta etapa apenas mede e classifica as tracks.")
    print("A redução adaptativa do passo ainda não foi aplicada.")


# ============================================================
# 7. VISUALIZAÇÃO DOS RESULTADOS
# ============================================================

def visualizar_contornos(
    mascara,
    original,
    marchado,
    lado,
    passo,
    saida,
):
    """Salva a comparação entre o contorno original e o marching."""

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.imshow(
        mascara,
        cmap="gray",
        origin="upper",
        interpolation="nearest",
    )

    ax.plot(
        original[:, 0],
        -original[:, 1],
        color="deepskyblue",
        linewidth=1.2,
        label="Contorno original",
    )

    ax.plot(
        marchado[:, 0],
        -marchado[:, 1],
        color="red",
        linewidth=1.5,
        label="Border Marching",
    )

    ax.set_title(
        f"Non-adaptive Border Marching — "
        f"Pulmão {lado} (passo={passo:g} px)"
    )

    ax.legend()
    ax.axis("off")

    plt.tight_layout()

    fig.savefig(
        saida,
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(f"\nImagem salva em: {saida}")


# ============================================================
# 8. CARREGAMENTO DAS MÁSCARAS
# ============================================================

def carregar_mascara(arquivo, lado):
    """Carrega a máscara do pulmão escolhido a partir do arquivo NPZ."""

    chaves = {
        "esquerdo": "pulmao_esquerdo",
        "direito": "pulmao_direito",
    }

    if lado not in chaves:
        raise ValueError(
            "Lado precisa ser 'esquerdo' ou 'direito'."
        )

    if not arquivo.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {arquivo}"
        )

    with np.load(arquivo) as dados:

        print("\n=== CARREGAMENTO DA MÁSCARA ===")
        print("Arquivo:", arquivo)
        print("Chaves disponíveis:", list(dados.keys()))

        chave = chaves[lado]

        if chave not in dados:
            raise KeyError(
                f"A chave '{chave}' não existe no arquivo NPZ."
            )

        mascara = dados[chave]

    if mascara.ndim != 2:
        raise ValueError(
            f"A máscara {chave} deveria ser 2D, "
            f"mas possui formato {mascara.shape}."
        )

    return mascara.astype(bool)


# ============================================================
# 9. EXECUÇÃO PRINCIPAL
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Executa Non-adaptive Border Marching e calcula "
            "W, Hmax e lambda para cada convex track."
        )
    )

    parser.add_argument(
        "arquivo",
        nargs="?",
        default=(
            "resultados/"
            "mascaras_coronacases_001_fatia_100.npz"
        ),
        help="Caminho do arquivo NPZ com as máscaras.",
    )

    parser.add_argument(
        "--lado",
        choices=["esquerdo", "direito"],
        required=True,
        help="Pulmão que será processado.",
    )

    parser.add_argument(
        "--passo",
        type=float,
        default=50.0,
        help="Comprimento da janela de busca em pixels. Padrão: 50.",
    )

    parser.add_argument(
        "--lambda0",
        type=float,
        default=0.33,
        help="Limiar lambda0 para classificar as tracks. Padrão: 0.33.",
    )

    parser.add_argument(
        "--spacing-x-mm",
        type=float,
        default=None,
        help="Espaçamento do eixo x em mm/pixel.",
    )

    parser.add_argument(
        "--spacing-y-mm",
        type=float,
        default=None,
        help="Espaçamento do eixo y em mm/pixel.",
    )

    parser.add_argument(
        "--saida-dir",
        default="resultados",
        help="Diretório para salvar imagens e tabela CSV.",
    )

    args = parser.parse_args()

    if (
        (args.spacing_x_mm is None)
        != (args.spacing_y_mm is None)
    ):
        parser.error(
            "Informe os dois espaçamentos: "
            "--spacing-x-mm e --spacing-y-mm."
        )

    if args.spacing_x_mm is None:
        spacing_x = spacing_y = 1.0
        unidade = "px"
    else:
        if args.spacing_x_mm <= 0 or args.spacing_y_mm <= 0:
            parser.error(
                "Os espaçamentos em mm/pixel precisam ser positivos."
            )

        spacing_x = args.spacing_x_mm
        spacing_y = args.spacing_y_mm
        unidade = "mm"

    if args.lambda0 <= 0 or not np.isfinite(args.lambda0):
        parser.error(
            "--lambda0 precisa ser positivo e finito."
        )

    arquivo = Path(args.arquivo)
    saida_dir = Path(args.saida_dir)

    saida_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Carrega a máscara.
    mascara = carregar_mascara(
        arquivo,
        args.lado,
    )

    # Extrai os contornos.
    contornos = extrair_contornos(mascara)

    if len(contornos) == 0:
        raise RuntimeError(
            "Nenhum contorno foi encontrado na máscara."
        )

    # Utiliza o maior contorno, correspondente à região pulmonar principal.
    contorno_original = contornos[0]["pontos"]

    # Executa o Border Marching não adaptativo.
    contorno_marchado, segmentos = marchar_contorno_nao_adaptativo(
        contorno_original,
        passo=args.passo,
    )

    # Mede cada track sem alterar o resultado do marching.
    segmentos = medir_tracks(
        segmentos,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        lambda0=args.lambda0,
    )

    quantidade_original = len(contorno_original) - 1
    quantidade_marchada = len(contorno_marchado) - 1

    reducao = (
        1 - quantidade_marchada / quantidade_original
    ) * 100

    print("\n=== NON-ADAPTIVE BORDER MARCHING ===")
    print("Lado:", args.lado)
    print(f"Passo: {args.passo:.2f} pixels")
    print("Pontos originais:", quantidade_original)
    print("Pontos após marching:", quantidade_marchada)
    print("Segmentos gerados:", len(segmentos))
    print(f"Redução de pontos: {reducao:.2f}%")

    if segmentos:
        larguras = [
            segmento["largura"]
            for segmento in segmentos
        ]

        print(
            f"Largura média das tracks: "
            f"{np.mean(larguras):.2f} pixels"
        )

        print(
            f"Largura mínima das tracks: "
            f"{np.min(larguras):.2f} pixels"
        )

        print(
            f"Largura máxima das tracks: "
            f"{np.max(larguras):.2f} pixels"
        )

    # Exibe o resumo das métricas.
    imprimir_resumo_metricas(
        segmentos,
        args.lambda0,
        unidade,
    )

    # Exporta as métricas para CSV.
    sufixo_passo = f"{args.passo:g}"

    saida_csv = (
        saida_dir
        / f"metricas_abm_{args.lado}_passo_{sufixo_passo}.csv"
    )

    salvar_metricas_csv(
        segmentos,
        saida_csv,
        args.lambda0,
        unidade,
    )

    print(f"Tabela CSV salva em: {saida_csv}")

    # Salva a visualização.
    saida_png = (
        saida_dir
        / f"marching_{args.lado}_passo_{sufixo_passo}.png"
    )

    visualizar_contornos(
        mascara,
        contorno_original,
        contorno_marchado,
        args.lado,
        args.passo,
        saida_png,
    )


if __name__ == "__main__":
    main()