
"""
Executa o ABM experimental em todas as fatias de um NIfTI.

Reutiliza:
- separacao_final.py
- reinclusao_abm.py
- reinclusao_abm_adaptativo.py

A referência é usada somente para avaliação.
Os volumes NIfTI só são exportados quando todas
as fatias solicitadas foram processadas sem erros
e o intervalo corresponde ao volume completo.
"""

import argparse
import contextlib
import csv
import io
import os
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
from skimage.measure import label

from reinclusao_abm import extrair_contornos
from reinclusao_abm_adaptativo import (
    marchar_contorno_adaptativo,
    rasterizar_contorno,
)


# ============================================================
# 1. MÉTRICAS
# ============================================================

def medir_binario(predicao, referencia):
    a = np.asarray(predicao, dtype=bool)
    b = np.asarray(referencia, dtype=bool)

    inter = int(np.count_nonzero(a & b))
    soma = int(np.count_nonzero(a) + np.count_nonzero(b))
    uniao = int(np.count_nonzero(a | b))

    dice = 1.0 if soma == 0 else 2 * inter / soma
    iou = 1.0 if uniao == 0 else inter / uniao

    return dice, iou, inter, soma, uniao


# ============================================================
# 2. APLICAR ABM A UM PULMÃO
# ============================================================

def processar_lado(mascara, args):
    """
    Aplica o ABM a cada componente relevante.

    Componentes pequenos são preservados.
    Caso um componente falhe, mantém a máscara original.
    """

    mascara = np.asarray(mascara, dtype=bool)
    corrigida = mascara.copy()

    componentes = label(mascara, connectivity=2)

    total_tracks = 0
    total_ajustes = 0
    falhas_componentes = 0

    for identificador in range(
        1, int(componentes.max()) + 1
    ):
        componente = componentes == identificador

        if np.count_nonzero(componente) < args.area_minima:
            continue

        try:
            # Evita imprimir diagnósticos repetidos.
            with contextlib.redirect_stdout(io.StringIO()):
                contornos = extrair_contornos(componente)

            if not contornos:
                falhas_componentes += 1
                continue

            pontos = contornos[0]["pontos"]

            adaptado, tracks = marchar_contorno_adaptativo(
                pontos,
                passo_inicial=args.passo,
                lambda0=args.lambda0,
                delta=args.delta,
                passo_minimo=args.passo_minimo,
                max_ajustes=args.max_ajustes,
            )

            poligono = rasterizar_contorno(
                adaptado,
                mascara.shape
            )

            corrigida |= poligono

            total_tracks += len(tracks)
            total_ajustes += sum(
                t["ajustes"] for t in tracks
            )

        except (ValueError, RuntimeError, AssertionError) as erro:
            print(
                f"Aviso: componente {identificador} "
                f"preservado ({erro})"
            )
            falhas_componentes += 1

    if not np.all(corrigida[mascara]):
        raise RuntimeError("ABM removeu pixels da entrada.")

    return (
        corrigida,
        total_tracks,
        total_ajustes,
        falhas_componentes,
    )


# ============================================================
# 3. CARREGAR MÁSCARAS DA SEPARAÇÃO
# ============================================================

def carregar_base(caminho, formato):
    with np.load(caminho, allow_pickle=False) as dados:

        for chave in ("pulmao_esquerdo", "pulmao_direito"):
            if chave not in dados:
                raise ValueError(
                    f"Chave ausente no NPZ: {chave}"
                )

        esquerda = np.asarray(
            dados["pulmao_esquerdo"],
            dtype=bool
        )

        direita = np.asarray(
            dados["pulmao_direito"],
            dtype=bool
        )

        if esquerda.shape != formato or direita.shape != formato:
            raise ValueError(
                "Dimensões das máscaras incompatíveis com o NIfTI."
            )

        if np.any(esquerda & direita):
            raise ValueError(
                "Máscaras-base dos pulmões estão sobrepostas."
            )

        if "mascara_combinada" in dados:
            combinada = np.asarray(
                dados["mascara_combinada"],
                dtype=bool
            )

            if not np.array_equal(
                combinada,
                esquerda | direita
            ):
                raise ValueError(
                    "mascara_combinada difere da união dos lados."
                )

        status = (
            str(dados["status"].item())
            if "status" in dados
            else "sem_status"
        )

    return esquerda, direita, status


# ============================================================
# 4. GERAR MÁSCARA-BASE QUANDO NECESSÁRIO
# ============================================================

def gerar_base(imagem, fatia, saida_esperada):
    """
    Reutiliza separacao_final.py sem modificar seu código.
    """

    comando = [
        sys.executable,
        "separacao_final.py",
        str(imagem),
        "--fatia",
        str(fatia),
    ]

    ambiente = os.environ.copy()
    ambiente["MPLBACKEND"] = "Agg"

    resultado = subprocess.run(
        comando,
        env=ambiente,
        text=True,
        capture_output=True,
        check=False,
    )

    if resultado.returncode != 0:
        mensagem = (
            resultado.stderr or resultado.stdout
        )[-600:]

        raise RuntimeError(
            f"Falha em separacao_final.py: {mensagem}"
        )

    if not saida_esperada.is_file():
        raise FileNotFoundError(
            "A separação terminou, mas não gerou o arquivo "
            f"esperado: {saida_esperada}"
        )


# ============================================================
# 5. SALVAR NIFTI
# ============================================================

def salvar_nifti(array, imagem_referencia, caminho):
    """
    Salva volume alinhado à tomografia canonicalizada.
    """

    header = imagem_referencia.header.copy()
    header.set_data_dtype(np.uint8)

    saida = nib.Nifti1Image(
        np.asarray(array, dtype=np.uint8),
        imagem_referencia.affine.copy(),
        header=header,
    )

    nib.save(saida, str(caminho))


# ============================================================
# 6. EXECUÇÃO PRINCIPAL
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Executa ABM em um volume pulmonar."
    )

    parser.add_argument(
        "imagem",
        help="Caminho da tomografia 3D NIfTI"
    )

    parser.add_argument(
        "--referencia",
        default=None,
        help="Ground truth para avaliação (opcional)"
    )

    parser.add_argument(
        "--bases-dir",
        default="resultados"
    )

    parser.add_argument(
        "--saida-dir",
        default="resultados/abm_lote"
    )

    parser.add_argument(
        "--gerar-base",
        action="store_true",
        help="Gera máscaras-base faltantes com separacao_final.py"
    )

    parser.add_argument(
        "--inicio",
        type=int,
        default=0
    )

    parser.add_argument(
        "--fim",
        type=int,
        default=None
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
        "--area-minima",
        type=int,
        default=20,
        help="Componentes menores são preservados sem ABM"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validação dos parâmetros.
    # --------------------------------------------------------

    if not np.isfinite(args.passo) or args.passo <= 0:
        parser.error("--passo deve ser positivo.")

    if not np.isfinite(args.lambda0) or args.lambda0 <= 0:
        parser.error("--lambda0 deve ser positivo.")

    if not np.isfinite(args.delta) or not 0 < args.delta < 1:
        parser.error("--delta deve estar entre 0 e 1.")

    if (
        not np.isfinite(args.passo_minimo)
        or args.passo_minimo <= 0
    ):
        parser.error("--passo-minimo deve ser positivo.")

    if args.max_ajustes < 0 or args.area_minima < 1:
        parser.error("Parâmetros de ajuste ou área inválidos.")

    # --------------------------------------------------------
    # Identificar caso.
    # --------------------------------------------------------

    imagem = Path(args.imagem)

    if not imagem.is_file():
        parser.error(f"Tomografia não encontrada: {imagem}")

    caso = imagem.name

    if caso.endswith(".nii.gz"):
        caso = caso[:-7]
    elif caso.endswith(".nii"):
        caso = caso[:-4]
    else:
        parser.error("Use um arquivo .nii ou .nii.gz.")

    # --------------------------------------------------------
    # Carregar volume em orientação canônica RAS.
    # --------------------------------------------------------

    nii = nib.as_closest_canonical(
        nib.load(str(imagem))
    )

    if len(nii.shape) != 3:
        parser.error("A tomografia precisa ser 3D.")

    shape = nii.shape

    fim = (
        shape[2] - 1
        if args.fim is None
        else args.fim
    )

    if not 0 <= args.inicio <= fim < shape[2]:
        parser.error(
            f"Intervalo inválido. Use fatias de 0 a {shape[2] - 1}."
        )

    completo = (
        args.inicio == 0
        and fim == shape[2] - 1
    )

    # --------------------------------------------------------
    # Ground truth: somente avaliação.
    # --------------------------------------------------------

    caminho_ref = (
        Path(args.referencia)
        if args.referencia
        else Path("data/masks/Lung_Mask") / f"{caso}.nii.gz"
    )

    referencia_nii = None

    if caminho_ref.is_file():

        referencia_nii = nib.as_closest_canonical(
            nib.load(str(caminho_ref))
        )

        if (
            referencia_nii.shape != shape
            or not np.allclose(
                referencia_nii.affine,
                nii.affine,
                rtol=0,
                atol=1e-4
            )
        ):
            parser.error(
                "Referência e tomografia não estão alinhadas."
            )

    elif args.referencia is not None:
        parser.error(
            f"Referência não encontrada: {caminho_ref}"
        )

    else:
        print(
            "Aviso: referência não encontrada. "
            "Dice e IoU serão omitidos."
        )

    # --------------------------------------------------------
    # Preparar diretórios.
    # --------------------------------------------------------

    bases_dir = Path(args.bases_dir)

    saida_dir = Path(args.saida_dir) / caso
    saida_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Alocar volumes de saída.
    # --------------------------------------------------------

    base_vol = np.zeros(
        shape,
        dtype=np.uint8
    )

    abm_vol = np.zeros(
        shape,
        dtype=np.uint8
    )

    rotulos_vol = np.zeros(
        shape,
        dtype=np.uint8
    )

    adicoes_vol = np.zeros(
        shape,
        dtype=np.uint8
    )

    linhas = []
    erros = 0
    fallback_total = 0

    # Interseção, soma das áreas, união.
    metricas_base = [0, 0, 0]
    metricas_abm = [0, 0, 0]

    # ========================================================
    # 7. PERCORRER AS FATIAS
    # ========================================================

    for k in range(args.inicio, fim + 1):

        base_npz = (
            bases_dir
            / f"mascaras_{caso}_fatia_{k}.npz"
        )

        linha = {
            "fatia": k,
            "status": "ok",
            "erro": "",
            "status_separacao": "",
        }

        try:
            # ------------------------------------------------
            # Obter máscara original da separação.
            # ------------------------------------------------

            if not base_npz.is_file():

                if not args.gerar_base:
                    raise FileNotFoundError(
                        f"Arquivo ausente: {base_npz}. "
                        "Use --gerar-base."
                    )

                gerar_base(
                    imagem,
                    k,
                    base_npz
                )

            e0, d0, status = carregar_base(
                base_npz,
                shape[:2]
            )

            linha["status_separacao"] = status

            # ------------------------------------------------
            # Aplicar ABM independentemente aos dois pulmões.
            # ------------------------------------------------

            e1, tracks_e, ajustes_e, falhas_e = (
                processar_lado(e0, args)
            )

            d1, tracks_d, ajustes_d, falhas_d = (
                processar_lado(d0, args)
            )

            # ------------------------------------------------
            # Prevenir adições sobre o outro pulmão.
            # ------------------------------------------------

            novos_e = e1 & ~e0 & ~d0
            novos_d = d1 & ~d0 & ~e0

            # Pixels reivindicados por ambos não são atribuídos.
            ambiguos = novos_e & novos_d

            novos_e &= ~ambiguos
            novos_d &= ~ambiguos

            e_final = e0 | novos_e
            d_final = d0 | novos_d

            if np.any(e_final & d_final):
                raise ValueError(
                    "Sobreposição entre pulmões após ABM."
                )

            original = e0 | d0
            final = e_final | d_final

            # ------------------------------------------------
            # Inserir a fatia nos volumes 3D.
            # ------------------------------------------------

            base_vol[:, :, k] = original.astype(np.uint8)
            abm_vol[:, :, k] = final.astype(np.uint8)

            rotulos_vol[:, :, k] = (
                e_final.astype(np.uint8)
                + 2 * d_final.astype(np.uint8)
            )

            adicoes_vol[:, :, k] = (
                final & ~original
            ).astype(np.uint8)

            # ------------------------------------------------
            # Registrar resultados por fatia.
            # ------------------------------------------------

            falhas = falhas_e + falhas_d

            fallback_total += falhas

            if falhas:
                linha["status"] = "fallback_parcial"

            linha.update({
                "pixels_base": int(original.sum()),
                "pixels_abm": int(final.sum()),
                "adicoes_esquerdo": int(novos_e.sum()),
                "adicoes_direito": int(novos_d.sum()),
                "adicoes_ambiguas_descartadas": int(
                    ambiguos.sum()
                ),
                "tracks_esquerdo": tracks_e,
                "tracks_direito": tracks_d,
                "ajustes_esquerdo": ajustes_e,
                "ajustes_direito": ajustes_d,
                "componentes_fallback": falhas,
            })

            # ------------------------------------------------
            # Avaliar contra o ground truth, se disponível.
            # ------------------------------------------------

            if referencia_nii is not None:

                rotulos_ref = np.asarray(
                    referencia_nii.dataobj[:, :, k]
                )

                referencia = (
                    (rotulos_ref == 1)
                    | (rotulos_ref == 2)
                )

                dbase, ibase, inter, soma, uniao = (
                    medir_binario(original, referencia)
                )

                metricas_base = [
                    a + b
                    for a, b in zip(
                        metricas_base,
                        (inter, soma, uniao)
                    )
                ]

                dabm, iabm, inter, soma, uniao = (
                    medir_binario(final, referencia)
                )

                metricas_abm = [
                    a + b
                    for a, b in zip(
                        metricas_abm,
                        (inter, soma, uniao)
                    )
                ]

                novos = final & ~original

                linha.update({
                    "dice_base": round(dbase, 6),
                    "dice_abm": round(dabm, 6),
                    "iou_base": round(ibase, 6),
                    "iou_abm": round(iabm, 6),
                    "pixels_recuperados_corretos": int(
                        (novos & referencia).sum()
                    ),
                    "pixels_adicionados_incorretos": int(
                        (novos & ~referencia).sum()
                    ),
                })

        except Exception as erro:
            linha["status"] = "erro"
            linha["erro"] = str(erro)
            erros += 1

            print(f"Fatia {k}: ERRO - {erro}")

        linhas.append(linha)

        if (
            (k - args.inicio) % 10 == 0
            or k == fim
        ):
            adicionados = (
                linha.get("pixels_abm", 0)
                - linha.get("pixels_base", 0)
            )

            print(
                f"Fatia {k}/{fim} | "
                f"status={linha['status']} | "
                f"adições={adicionados}"
            )

    # ========================================================
    # 8. EXPORTAR CSV
    # ========================================================

    csv_caminho = (
        saida_dir
        / f"{caso}_abm_por_fatia.csv"
    )

    campos = [
        "fatia",
        "status",
        "erro",
        "status_separacao",
        "pixels_base",
        "pixels_abm",
        "adicoes_esquerdo",
        "adicoes_direito",
        "adicoes_ambiguas_descartadas",
        "tracks_esquerdo",
        "tracks_direito",
        "ajustes_esquerdo",
        "ajustes_direito",
        "componentes_fallback",
        "dice_base",
        "dice_abm",
        "iou_base",
        "iou_abm",
        "pixels_recuperados_corretos",
        "pixels_adicionados_incorretos",
    ]

    with csv_caminho.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        escritor = csv.DictWriter(
            arquivo,
            fieldnames=campos
        )

        escritor.writeheader()
        escritor.writerows(linhas)

    print(f"\nCSV salvo: {csv_caminho}")
    print(f"Fatias com erro: {erros}")
    print(f"Componentes preservados por fallback: {fallback_total}")

    # --------------------------------------------------------
    # Não exportar volumes incompletos.
    # --------------------------------------------------------

    if erros:
        print(
            "Os volumes NIfTI não foram exportados. "
            "Corrija as fatias com erro e execute novamente."
        )
        sys.exit(1)

    if not completo:
        print(
            "Intervalo parcial: CSV gerado, "
            "mas nenhum volume 3D foi exportado."
        )
        return

    # ========================================================
    # 9. EXPORTAR MÁSCARAS NIFTI
    # ========================================================

    caminhos = {
        "base": (
            saida_dir / f"{caso}_BASE.nii.gz"
        ),
        "abm": (
            saida_dir / f"{caso}_ABM.nii.gz"
        ),
        "rotulos": (
            saida_dir / f"{caso}_ABM_ROTULOS.nii.gz"
        ),
        "adicoes": (
            saida_dir / f"{caso}_ABM_ADICOES.nii.gz"
        ),
    }

    salvar_nifti(
        base_vol,
        nii,
        caminhos["base"]
    )

    salvar_nifti(
        abm_vol,
        nii,
        caminhos["abm"]
    )

    salvar_nifti(
        rotulos_vol,
        nii,
        caminhos["rotulos"]
    )

    salvar_nifti(
        adicoes_vol,
        nii,
        caminhos["adicoes"]
    )

    print("\n=== VOLUMES ABM EXPORTADOS ===")

    for nome, caminho in caminhos.items():
        print(f"{nome}: {caminho}")

    print(
        "\nRótulos da máscara ABM: "
        "0=fundo, 1=chave pulmao_esquerdo, "
        "2=chave pulmao_direito"
    )

    print(
        "Total de pixels acrescentados:",
        int(adicoes_vol.sum())
    )

    # ========================================================
    # 10. MÉTRICAS VOLUMÉTRICAS
    # ========================================================

    if referencia_nii is not None:

        def pontuacao(totais):
            inter, soma, uniao = totais

            dice = (
                1.0 if soma == 0
                else 2 * inter / soma
            )

            iou = (
                1.0 if uniao == 0
                else inter / uniao
            )

            return dice, iou

        dice_base, iou_base = pontuacao(
            metricas_base
        )

        dice_abm, iou_abm = pontuacao(
            metricas_abm
        )

        print("\n=== AVALIAÇÃO VOLUMÉTRICA ===")

        print(
            f"Dice: {dice_base:.4f} -> "
            f"{dice_abm:.4f} "
            f"(Δ={dice_abm - dice_base:+.4f})"
        )

        print(
            f"IoU:  {iou_base:.4f} -> "
            f"{iou_abm:.4f} "
            f"(Δ={iou_abm - iou_base:+.4f})"
        )


if __name__ == "__main__":
    main()
