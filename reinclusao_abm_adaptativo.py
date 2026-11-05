
"""
ABM experimental 2D inspirado em Pu et al. (2008).

Entradas: arquivo NPZ gerado por separacao_final.py com
'pulmao_esquerdo' e 'pulmao_direito'.

Este módulo NÃO altera reinclusao_abm.py. Importa dele a extração
original, o marching não adaptativo e a métrica geométrica.

Para reinclusão, usa união (OR) com a máscara original, evitando
remover regiões previamente segmentadas. A máscara do polígono ABM
puro também é salva para fins de auditoria.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon

from reinclusao_abm import (
    carregar_mascara,
    extrair_contornos,
    orientar_contorno_anti_horario,
    ponto_a_direita,
    medir_track,
    marchar_contorno_nao_adaptativo,
)


# ============================================================
# 1. BUSCA DO EXTREMO DE UMA TRACK
# ============================================================

def encontrar_extremo_local(pontos_ext, distancias, inicio_idx, passo):
    """
    Encontra a convex track a partir de um índice,
    dentro de uma janela medida em pixels.
    """

    n = len(distancias)
    limite = inicio_idx
    acumulado = 0.0

    while limite < n and acumulado < passo:
        acumulado += float(distancias[limite])
        limite += 1

    inicio = pontos_ext[inicio_idx]
    extremo_idx = inicio_idx + 1
    extremo = pontos_ext[extremo_idx]

    for i in range(inicio_idx + 2, limite + 1):
        candidato = pontos_ext[i]

        if ponto_a_direita(inicio, extremo, candidato):
            extremo_idx = i
            extremo = candidato

    return extremo_idx


# ============================================================
# 2. ADAPTIVE BORDER MARCHING
# ============================================================

def marchar_contorno_adaptativo(
    contorno,
    passo_inicial=50.0,
    lambda0=0.33,
    delta=0.9,
    passo_minimo=1.0,
    max_ajustes=80,
):
    """
    Aplica busca local de convex tracks com redução
    adaptativa do passo.

    Se lambda < lambda0, diminui o passo de busca
    por delta e repete.

    A redução termina quando:
        - lambda >= lambda0;
        - a busca chegou ao próximo ponto;
        - passo_minimo foi atingido;
        - max_ajustes foi atingido.

    O passo é reiniciado a cada nova track,
    permitindo adaptação local.
    """

    if not np.isfinite(passo_inicial) or passo_inicial <= 0:
        raise ValueError("passo_inicial deve ser positivo e finito")

    if not np.isfinite(lambda0) or lambda0 <= 0:
        raise ValueError("lambda0 deve ser positivo e finito")

    if not np.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta deve estar no intervalo aberto (0, 1)")

    if not np.isfinite(passo_minimo) or passo_minimo <= 0:
        raise ValueError("passo_minimo deve ser positivo e finito")

    if not isinstance(max_ajustes, int) or max_ajustes < 0:
        raise ValueError("max_ajustes deve ser inteiro não negativo")

    contorno = orientar_contorno_anti_horario(contorno)

    pontos = contorno[:-1]
    n = len(pontos)

    pontos_ext = np.vstack((pontos, pontos[0]))

    distancias = np.linalg.norm(
        np.diff(pontos_ext, axis=0),
        axis=1
    )

    if np.any(distancias <= 0):
        raise ValueError(
            "O contorno possui pontos consecutivos repetidos"
        )

    indices_selecionados = [0]
    tracks = []

    atual = 0

    # Percorre todo o contorno.
    while atual < n:

        # Reinicia o passo para cada nova região.
        passo_local = float(passo_inicial)
        ajustes = 0

        while True:

            # Encontra o extremo da track com o passo atual.
            fim = encontrar_extremo_local(
                pontos_ext,
                distancias,
                atual,
                passo_local
            )

            # Monta a track candidata.
            track = {
                "inicio": pontos_ext[atual].copy(),
                "fim": pontos_ext[fim].copy(),
                "trecho_original": pontos_ext[
                    atual:fim + 1
                ].copy(),
                "comprimento_original": float(
                    np.sum(distancias[atual:fim])
                ),
            }

            # Calcula W, Hmax e lambda.
            medida = medir_track(track)
            track.update(medida)

            # Decide se encerra a adaptação local.
            if (
                track["lambda"] >= lambda0
                or fim == atual + 1
                or passo_local <= passo_minimo
                or ajustes >= max_ajustes
            ):
                break

            # Reduz o passo e tenta novamente.
            novo_passo = max(
                passo_minimo,
                passo_local * delta
            )

            if novo_passo >= passo_local:
                break

            passo_local = novo_passo
            ajustes += 1

        # Registra o segmento final.
        track.update({
            "numero": len(tracks) + 1,
            "indice_inicio": atual,
            "indice_fim": fim,
            "passo_final": passo_local,
            "ajustes": ajustes,
            "acima_lambda0": track["lambda"] >= lambda0,
        })

        tracks.append(track)

        # O extremo final passa a ser o próximo início.
        atual = fim
        indices_selecionados.append(atual)

    # Verifica o fechamento do contorno.
    assert indices_selecionados[-1] == n

    contorno_adaptado = pontos_ext[
        indices_selecionados
    ].copy()

    assert np.allclose(
        contorno_adaptado[0],
        contorno_adaptado[-1]
    )

    return contorno_adaptado, tracks


# ============================================================
# 3. CONVERTER O CONTORNO EM MÁSCARA BINÁRIA
# ============================================================

def rasterizar_contorno(contorno, shape):
    """
    Converte o polígono ABM em uma máscara binária.

    Contorno:
        x = coordenada horizontal
        y = coordenada matemática, orientada para cima

    Máscara:
        linha = -y
        coluna = x
    """

    contorno = np.asarray(contorno, dtype=float)

    if not np.allclose(contorno[0], contorno[-1]):
        raise ValueError("Contorno precisa estar fechado")

    linhas = -contorno[:, 1]
    colunas = contorno[:, 0]

    rr, cc = polygon(
        linhas,
        colunas,
        shape=shape
    )

    mascara = np.zeros(shape, dtype=bool)
    mascara[rr, cc] = True

    return mascara


# ============================================================
# 4. COMPARAR MÁSCARAS
# ============================================================

def comparar_mascaras(original, poligono):
    """
    Calcula:
        pixels acrescentados pelo ABM;
        pixels que o polígono excluiria;
        máscara de reinclusão por união.

    A união preserva todos os pixels da máscara inicial.
    """

    adicionados = poligono & ~original

    removidos = original & ~poligono

    reinclusao = original | poligono

    return reinclusao, adicionados, removidos


# ============================================================
# 5. SALVAR AS MÉTRICAS DAS TRACKS EM CSV
# ============================================================

def salvar_tracks_csv(tracks, caminho):
    """Salva as métricas e os ajustes de cada track."""

    campos = [
        "numero",
        "indice_inicio",
        "indice_fim",
        "inicio_x",
        "inicio_y",
        "fim_x",
        "fim_y",
        "passo_final_px",
        "ajustes",
        "W_px",
        "Hmax_px",
        "lambda",
        "acima_lambda0",
    ]

    with open(
        caminho,
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        escritor = csv.DictWriter(
            arquivo,
            fieldnames=campos
        )

        escritor.writeheader()

        for t in tracks:
            escritor.writerow({
                "numero": t["numero"],
                "indice_inicio": t["indice_inicio"],
                "indice_fim": t["indice_fim"],
                "inicio_x": t["inicio"][0],
                "inicio_y": t["inicio"][1],
                "fim_x": t["fim"][0],
                "fim_y": t["fim"][1],
                "passo_final_px": t["passo_final"],
                "ajustes": t["ajustes"],
                "W_px": t["W"],
                "Hmax_px": t["Hmax"],
                "lambda": t["lambda"],
                "acima_lambda0": t["acima_lambda0"],
            })


# ============================================================
# 6. VISUALIZAÇÃO COMPARATIVA
# ============================================================

def visualizar(
    original,
    contorno_original,
    nao_adaptado,
    adaptado,
    adicionados,
    removidos,
    lado,
    passo,
    destino,
):
    """
    Gera uma figura comparando:
        1. Contorno original;
        2. Marching não adaptativo;
        3. Marching adaptativo;
        4. Diferenças entre o polígono ABM e a máscara original.
    """

    fig, eixos = plt.subplots(
        2, 2,
        figsize=(13, 11)
    )

    fig.suptitle(
        f"ABM experimental — {lado} — "
        f"passo inicial {passo:g} px"
    )

    for ax in eixos.flat:

        ax.imshow(
            original,
            cmap="gray",
            origin="upper"
        )

        ax.set_xlim(0, original.shape[1] - 1)
        ax.set_ylim(original.shape[0] - 1, 0)

        ax.axis("off")

    def desenhar(ax, pontos, cor, legenda):

        ax.plot(
            pontos[:, 0],
            -pontos[:, 1],
            color=cor,
            linewidth=1.5,
            label=legenda
        )

    # Original
    desenhar(
        eixos[0, 0],
        contorno_original,
        "deepskyblue",
        "Original"
    )

    eixos[0, 0].set_title(
        "Máscara e contorno original"
    )

    # Não adaptativo
    desenhar(
        eixos[0, 1],
        contorno_original,
        "deepskyblue",
        "Original"
    )

    desenhar(
        eixos[0, 1],
        nao_adaptado,
        "red",
        "Não adaptativo"
    )

    eixos[0, 1].set_title(
        "Marching não adaptativo"
    )

    eixos[0, 1].legend(loc="upper right")

    # Adaptativo
    desenhar(
        eixos[1, 0],
        contorno_original,
        "deepskyblue",
        "Original"
    )

    desenhar(
        eixos[1, 0],
        adaptado,
        "lime",
        "Adaptativo"
    )

    eixos[1, 0].set_title(
        "Marching adaptativo"
    )

    eixos[1, 0].legend(loc="upper right")

    # Alterações produzidas pelo polígono.
    alteracoes = np.zeros(
        (*original.shape, 4),
        dtype=float
    )

    # Laranja: pixels acrescentados.
    alteracoes[adicionados] = (
        1, 0.3, 0, 0.95
    )

    # Roxo: pixels que seriam excluídos.
    alteracoes[removidos] = (
        0.7, 0, 1, 0.9
    )

    eixos[1, 1].imshow(
        alteracoes,
        origin="upper"
    )

    eixos[1, 1].set_title(
        "Polígono ABM: adições (laranja), remoções (roxo)"
    )

    fig.tight_layout()

    fig.savefig(
        destino,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 7. EXECUÇÃO PRINCIPAL
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "ABM adaptativo experimental, aplicado "
            "a uma máscara pulmonar 2D."
        )
    )

    parser.add_argument(
        "arquivo",
        nargs="?",
        default=(
            "resultados/"
            "mascaras_coronacases_001_fatia_100.npz"
        )
    )

    parser.add_argument(
        "--lado",
        choices=["esquerdo", "direito"],
        required=True
    )

    parser.add_argument(
        "--passo",
        type=float,
        default=50.0
    )

    parser.add_argument(
        "--lambda0",
        type=float,
        default=0.33
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=0.9
    )

    parser.add_argument(
        "--passo-minimo",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--max-ajustes",
        type=int,
        default=80
    )

    parser.add_argument(
        "--saida-dir",
        default="resultados"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Carregar a máscara.
    # --------------------------------------------------------

    original = carregar_mascara(
        Path(args.arquivo),
        args.lado
    )

    # --------------------------------------------------------
    # Extrair o contorno original.
    # --------------------------------------------------------

    contornos = extrair_contornos(original)

    if not contornos:
        parser.error(
            "A máscara não possui contorno fechado"
        )

    contorno_original = contornos[0]["pontos"]

    # --------------------------------------------------------
    # Executar o método não adaptativo.
    # --------------------------------------------------------

    nao_adaptado, tracks_base = (
        marchar_contorno_nao_adaptativo(
            contorno_original,
            passo=args.passo
        )
    )

    # --------------------------------------------------------
    # Executar o método adaptativo.
    # --------------------------------------------------------

    adaptado, tracks = marchar_contorno_adaptativo(
        contorno_original,
        passo_inicial=args.passo,
        lambda0=args.lambda0,
        delta=args.delta,
        passo_minimo=args.passo_minimo,
        max_ajustes=args.max_ajustes,
    )

    # --------------------------------------------------------
    # Construir a máscara binária.
    # --------------------------------------------------------

    poligono_abm = rasterizar_contorno(
        adaptado,
        original.shape
    )

    reinclusao, adicionados, removidos = (
        comparar_mascaras(
            original,
            poligono_abm
        )
    )

    # --------------------------------------------------------
    # Preparar caminhos de saída.
    # --------------------------------------------------------

    saida_dir = Path(args.saida_dir)

    saida_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    prefixo = (
        f"abm_{args.lado}_passo_{args.passo:g}"
    )

    # --------------------------------------------------------
    # Salvar NPZ com todos os resultados.
    # --------------------------------------------------------

    npz = saida_dir / f"{prefixo}.npz"

    np.savez_compressed(
        npz,
        mascara_original=original,
        mascara_poligono_abm=poligono_abm,
        mascara_reincluida=reinclusao,
        pixels_adicionados=adicionados,
        pixels_removidos_pelo_poligono=removidos,
        contorno_original=contorno_original,
        contorno_nao_adaptativo=nao_adaptado,
        contorno_adaptativo=adaptado,
    )

    # --------------------------------------------------------
    # Salvar máscara de reinclusão em NPY.
    # --------------------------------------------------------

    npy = (
        saida_dir
        / f"{prefixo}_mascara_reincluida.npy"
    )

    np.save(
        npy,
        reinclusao
    )

    # --------------------------------------------------------
    # Salvar CSV com as tracks.
    # --------------------------------------------------------

    csv_path = (
        saida_dir
        / f"{prefixo}_tracks.csv"
    )

    salvar_tracks_csv(
        tracks,
        csv_path
    )

    # --------------------------------------------------------
    # Salvar figura comparativa.
    # --------------------------------------------------------

    png = (
        saida_dir
        / f"{prefixo}_comparacao.png"
    )

    visualizar(
        original,
        contorno_original,
        nao_adaptado,
        adaptado,
        adicionados,
        removidos,
        args.lado,
        args.passo,
        png
    )

    # --------------------------------------------------------
    # Exibir estatísticas.
    # --------------------------------------------------------

    print("\n=== ABM ADAPTATIVO EXPERIMENTAL ===")

    print(f"Pulmão: {args.lado}")
    print(f"Passo inicial: {args.passo:g} px")
    print(f"lambda0: {args.lambda0:g}")
    print(f"delta: {args.delta:g}")

    print(
        f"Tracks não adaptativas: "
        f"{len(tracks_base)}"
    )

    print(
        f"Tracks adaptativas: "
        f"{len(tracks)}"
    )

    print(
        f"Tracks com lambda >= lambda0: "
        f"{sum(t['acima_lambda0'] for t in tracks)}"
    )

    print(
        f"Tracks com ajustes: "
        f"{sum(t['ajustes'] > 0 for t in tracks)}"
    )

    print(
        f"Número total de reduções do passo: "
        f"{sum(t['ajustes'] for t in tracks)}"
    )

    print(
        f"Pixels originais: "
        f"{int(original.sum())}"
    )

    print(
        f"Pixels acrescentados: "
        f"{int(adicionados.sum())}"
    )

    print(
        f"Pixels que o polígono excluiria: "
        f"{int(removidos.sum())}"
    )

    print(
        f"Pixels da reinclusão (união): "
        f"{int(reinclusao.sum())}"
    )

    print(
        f"Removidos pela máscara de reinclusão: "
        f"{int((original & ~reinclusao).sum())}"
    )

    if original.any():
        percentual = 100 * adicionados.sum() / original.sum()

        print(
            f"Acréscimo relativo à máscara original: "
            f"{percentual:.2f}%"
        )

        if percentual > 5:
            print(
                "ATENÇÃO: acréscimo superior a 5%. "
                "Inspecione possíveis inclusões indevidas."
            )

    print(f"\nNPZ: {npz}")
    print(f"NPY: {npy}")
    print(f"CSV: {csv_path}")
    print(f"PNG: {png}")

    print(
        "\nObservação: a união com a máscara original "
        "é uma adaptação para reinclusão, não o ABM puro."
    )


if __name__ == "__main__":
    main()
