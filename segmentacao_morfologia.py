"""Mostra uma fatia e a máscara final após refinamento morfológico em 0–255.

Usa imagem_255 de src.preprocessing para Coronacases e Radiopaedia.
Otsu é o método padrão.
--limiar é expresso em 0–255; 144.5 equivale a −400 HU somente na janela
Coronacases padrão. O raio é em milímetros; zero desliga apenas o fechamento.
O espaçamento vem do cabeçalho da máscara pulmonar alinhada, em mm.
Busca a referência em data/masks/Lung_Mask com o mesmo nome da imagem;
use --mascara para outro caminho. Os rótulos não participam da segmentação.
A máscara ainda pode conter componentes internos indesejados.
A figura contém apenas a tomografia e a máscara final, lado a lado.
Por padrão, busca automaticamente um candidato nos cortes superiores e
confirma sua persistência em três fatias. Se não for único, pede uma semente.
--semente I J K substitui a busca com um ponto dentro da traqueia em uma fatia superior,
em índices RAS iniciando em zero. O rastreamento 2,5D acompanha até dois ramos até a fatia exibida.

Uso:
    python segmentacao_morfologia.py data/raw/coronacases_001.nii.gz --raio 2
"""

import argparse
import json
from pathlib import Path
from typing_extensions import final

import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing import (
    preparar_fatia, CENTRO_JANELA_PULMAO, LARGURA_JANELA_PULMAO,
    carregar_imagem, verificar_alinhamento, obter_metadados,
    extrair_fatia, preparar_intensidades, obter_indice_fatia,
)
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from avaliar_segmentacao import calcular_dice, calcular_iou


def remover_componentes_da_borda(mascara):
    """Remove regiões conectadas à borda, incluindo conexões diagonais."""
    rotulos = label(mascara, connectivity=2)
    return rotulos, clear_border(rotulos) > 0


def elemento_em_mm(raio_mm, spacing):
    """Discretiza um disco físico; admite pixels retangulares no plano.

    Cada deslocamento (i, j) entra se (i*sx)^2 + (j*sy)^2 <= raio_mm^2.
    Não arredonda o raio para um único inteiro nem usa o espaçamento entre fatias.
    """
    spacing = np.asarray(spacing, dtype=float)
    if not np.isfinite(raio_mm) or raio_mm < 0:
        raise ValueError("O raio em mm deve ser finito e não negativo.")
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Informe dois espaçamentos positivos e finitos em mm.")
    sx, sy = spacing
    nx, ny = np.floor(raio_mm / spacing).astype(int)
    x, y = np.ogrid[-nx:nx + 1, -ny:ny + 1]
    return np.hypot(x * sx, y * sy) <= raio_mm


def espacamento_referencia(caminho_imagem, caminho_mascara):
    """Lê apenas a geometria da referência, sem usar seus rótulos."""
    img = carregar_imagem(caminho_imagem)
    ref = carregar_imagem(caminho_mascara)
    verificar_alinhamento(img, ref)
    meta = obter_metadados(ref)
    if meta["unidade_espacial"] != "mm":
        raise ValueError("A máscara de referência precisa declarar unidade espacial mm.")
    spacing = np.asarray(meta["spacing"])
    eixos = meta["affine"][:3, :3]
    if not np.allclose(np.linalg.norm(eixos, axis=0), spacing, atol=1e-6, rtol=0):
        raise ValueError("Espaçamento da referência inconsistente com a affine.")
    direcoes = eixos / spacing
    if not np.allclose(direcoes.T @ direcoes, np.eye(3), atol=1e-5, rtol=0):
        raise ValueError("A grade possui cisalhamento; requer tratamento geométrico específico.")
    return tuple(spacing[:2])


def refinar_mascara(mascara, raio_mm, spacing):
    """Devolve resultados separados para observar cada operação."""
    elemento = elemento_em_mm(raio_mm, spacing)
    if raio_mm == 0:
        fechada = mascara.copy()
    else:
        fechada = ndi.binary_closing(mascara, structure=elemento)

    # Inclui regiões de fundo totalmente cercadas pela máscara.
    preenchida = ndi.binary_fill_holes(fechada)
    return fechada, preenchida


def limiarizar(fatia, metodo, limiar_fixo):
    """Usa o mesmo critério em todas as fatias, na escala preparada 0–255."""
    if metodo == "otsu":
        if fatia.min() == fatia.max():
            raise ValueError("Fatia constante: Otsu não define uma região de ar útil.")
        limiar = float(threshold_otsu(fatia))
    else:
        limiar = limiar_fixo
    return fatia < limiar, limiar


def candidatos_semente(fatia, spacing, metodo="otsu", limiar_fixo=144.5,
                       area_max=400.0):
    """Candidatos internos, compactos e centrais; somente intensidades da TC.

    A região corporal é o maior componente de tecido, com buracos preenchidos.
    Os critérios são heurísticos: área >= 30 mm², solidez >= 0.75,
    razão dos autovalores no plano físico >= 0.19 (evita regiões alongadas).
    Regiões acima de area_max também são retornadas como alternativas que
    impedem escolher arbitrariamente uma região pequena vizinha.
    """
    ar, _ = limiarizar(fatia, metodo, limiar_fixo)
    tecidos = label(~ar, connectivity=2)
    tamanhos = np.bincount(tecidos.ravel())
    if len(tamanhos) < 2:
        return []
    tamanhos[0] = 0
    corpo = ndi.binary_fill_holes(tecidos == tamanhos.argmax())
    pontos_corpo = np.argwhere(corpo)
    inferior, superior = pontos_corpo.min(axis=0), pontos_corpo.max(axis=0)
    centro_lateral = (inferior[0] + superior[0]) / 2
    margem_mm = min(40.0, (superior[0] - inferior[0]) * spacing[0] * 0.15)
    candidatos = []
    rotulos = clear_border(label(ar, connectivity=2))
    for regiao in regionprops(rotulos):
        area = float(regiao.area * np.prod(spacing))
        if area < 30 or regiao.solidity < 0.75:
            continue
        centroide = np.asarray(regiao.centroid)
        if abs(centroide[0] - centro_lateral) * spacing[0] > margem_mm:
            continue
        coords = regiao.coords
        if not corpo[coords[:, 0], coords[:, 1]].all():
            continue
        distancias = (coords - centroide) * np.asarray(spacing)
        autovalores = np.linalg.eigvalsh(distancias.T @ distancias / len(coords))
        if autovalores[-1] <= 0 or autovalores[0] / autovalores[-1] < 0.19:
            continue
        # O centroide geométrico pode cair no fundo: escolhe um pixel interno.
        ponto = coords[np.argmin(np.sum(distancias ** 2, axis=1))]
        candidatos.append({"ponto": tuple(int(v) for v in ponto),
                           "centroide": centroide, "area_mm2": area,
                           "area_permitida": area <= area_max,
                           "componente": rotulos == regiao.label})
    return candidatos


def identificar_semente_traqueia(img, fonte, fatia_final, spacing,
                                metodo="otsu", limiar_fixo=144.5,
                                centro=CENTRO_JANELA_PULMAO,
                                largura=LARGURA_JANELA_PULMAO,
                                area_max=400.0, crescimento_max=2.0,
                                deslocamento_max=5.0):
    """Busca do topo para baixo nos 40% superiores; confirma em três cortes.

    Exige um único candidato persistente. Ambiguidade ou ausência geram erro
    orientando --semente; não usam referências de rótulos ou coordenadas fixas.
    A semente retornada não fica abaixo da fatia solicitada. A confirmação
    pode consultar até duas fatias abaixo dela, sem alterar o alvo exibido.
    """
    elemento_em_mm(0, spacing)
    obter_indice_fatia(img, fatia_final)
    if obter_metadados(img)["orientacao"] != ("R", "A", "S"):
        raise ValueError("A busca automática exige orientação RAS.")
    if (not np.isfinite([area_max, crescimento_max, deslocamento_max]).all()
            or area_max < 30 or crescimento_max < 1 or deslocamento_max <= 0):
        raise ValueError("Busca automática exige área máxima >= 30 mm², crescimento >= 1 e deslocamento positivo.")
    limite = max(fatia_final, int(np.floor(img.shape[2] * 0.6)))
    cache = {}
    examinadas = 0

    def candidatos(k):
        if k not in cache:
            preparada = preparar_intensidades(extrair_fatia(img, k), fonte,
                                               centro, largura)["imagem_255"]
            if preparada.min() == preparada.max():
                cache[k] = []
            else:
                cache[k] = candidatos_semente(preparada, spacing, metodo,
                                               limiar_fixo, area_max)
        return cache[k]

    for k in range(img.shape[2] - 1, limite - 1, -1):
        if k < 2:
            break
        examinadas += 1
        confirmados = []
        for inicial in candidatos(k):
            if not inicial["area_permitida"]:
                continue
            anterior = inicial
            for seguinte in (k - 1, k - 2):
                continuacoes = [c for c in candidatos(seguinte)
                               if np.any(c["componente"] & anterior["componente"])]
                if len(continuacoes) != 1:
                    break
                atual = continuacoes[0]
                if not atual["area_permitida"]:
                    break
                distancia = np.linalg.norm((atual["centroide"] - anterior["centroide"]) * spacing)
                razao = atual["area_mm2"] / anterior["area_mm2"]
                if distancia > deslocamento_max or not 1 / crescimento_max <= razao <= crescimento_max:
                    break
                anterior = atual
            else:
                confirmados.append(inicial)
        if len(confirmados) > 1:
            raise ValueError(f"Semente automática ambígua na fatia {k}: múltiplos candidatos persistentes. Use --semente I J K.")
        if len(confirmados) == 1:
            if any(not c["area_permitida"] for c in candidatos(k)):
                raise ValueError(f"Semente automática ambígua na fatia {k}: há uma região central maior que o limite de área. Revise a imagem e use --semente I J K.")
            escolhido = confirmados[0]
            semente = (*escolhido["ponto"], k)
            return semente, {"origem": "automatica", "fatias_iniciais_examinadas": examinadas,
                              "intervalo_busca": [limite, img.shape[2] - 1],
                              "fatias_confirmacao": [k, k-1, k-2],
                              "area_inicial_mm2": escolhido["area_mm2"],
                              "criterios": {"area_min_mm2": 30, "area_max_mm2": area_max,
                                            "solidez_min": 0.75, "razao_autovalores_min": 0.19,
                                            "margem_lateral": "min(40 mm, 15% da largura corporal)"}}
        # Mantém no máximo as fatias necessárias à janela de confirmação.
        cache.pop(k, None)
    raise ValueError("Nenhuma semente automática confiável nos cortes superiores. Use --semente I J K.")


def rastrear_traqueia(img, fonte, semente, fatia_final, spacing,
                     metodo="otsu", limiar_fixo=144.5,
                     centro=CENTRO_JANELA_PULMAO, largura=LARGURA_JANELA_PULMAO,
                     area_max=400.0, crescimento_max=2.0, deslocamento_max=5.0,
                     roi_mm=20.0, ajuste_limiar_max=30.0):
    """Rastreamento 2,5D de uma traqueia e até dois ramos, fatia a fatia.

    A continuidade exige sobreposição; não pula fatias nem faz operações 3D.
    Usa ROI física, limiar adaptativo conservador e parada independente.
    Retorna a união dos ramos aceitos no alvo e eventos de cada ramo.
    """
    if (not np.isfinite([area_max, crescimento_max, deslocamento_max]).all()
            or area_max <= 0 or crescimento_max < 1 or deslocamento_max <= 0):
        raise ValueError("Limites de rastreamento inválidos: área/deslocamento positivos, crescimento >= 1.")
    spacing = np.asarray(spacing, dtype=float)
    elemento_em_mm(0, spacing)  # Valida os dois espaçamentos.
    if obter_metadados(img)["orientacao"] != ("R", "A", "S"):
        raise ValueError("O rastreamento exige imagem canonicalizada para RAS.")
    if len(semente) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in semente):
        raise ValueError("A semente deve conter três índices inteiros: I J K.")
    i, j, inicio = semente
    obter_indice_fatia(img, inicio)
    obter_indice_fatia(img, fatia_final)
    if not (0 <= i < img.shape[0] and 0 <= j < img.shape[1]):
        raise ValueError("Semente fora da imagem no plano.")
    if inicio < fatia_final:
        raise ValueError("A semente deve estar na fatia solicitada ou acima dela (K >= --fatia em RAS).")

    if not np.isfinite([roi_mm, ajuste_limiar_max]).all() or roi_mm <= 0 or not 0 <= ajuste_limiar_max <= 255:
        raise ValueError("ROI deve ser positiva e ajuste de limiar deve estar entre 0 e 255.")
    ativos = [{"id": 0, "geracao": 0, "centroide": np.array([i, j], float),
               "componente": None, "area_mm2": None}]
    proximo_id = 1
    alvo = np.zeros(img.shape[:2], dtype=bool)
    historico = []
    for k in range(inicio, fatia_final - 1, -1):
        fatia = preparar_intensidades(extrair_fatia(img, k), fonte, centro, largura)["imagem_255"]
        try:
            _, limiar = limiarizar(fatia, metodo, limiar_fixo)
        except ValueError as erro:
            if k == inicio:
                raise ValueError(f"Semente inválida: {erro}") from erro
            historico.append({"fatia": k, "estado": "parada", "motivo": str(erro)})
            break
        propostas = []
        for ramo in ativos:
            candidatos, usado, motivo = crescer_localmente(
                fatia, ramo, spacing, limiar, roi_mm, ajuste_limiar_max,
                area_max, crescimento_max, deslocamento_max)
            if not candidatos:
                if k == inicio:
                    raise ValueError(f"Semente inválida na fatia {k}: {motivo}")
                historico.append({"fatia": k, "ramo": ramo["id"], "estado": "parada", "motivo": motivo})
                continue
            for candidato in candidatos:
                dividido = len(candidatos) == 2
                ident = proximo_id if dividido else ramo["id"]
                if dividido:
                    proximo_id += 1
                candidato.update(id=ident, geracao=ramo["geracao"] + int(dividido))
                evento = {"fatia": k, "ramo": ident, "pai": ramo["id"] if dividido else None,
                          "estado": "aceita", "bifurcacao": dividido,
                          "limiar": usado, "limiar_base": limiar,
                          "semente_propagada": ramo["centroide"].tolist(),
                          "semente_interna": candidato.pop("semente_interna"),
                          "centroide": candidato["centroide"].tolist(),
                          "area_mm2": candidato["area_mm2"]}
                propostas.append((candidato, evento))
        # Dois rastros não podem reivindicar o mesmo componente na fatia atual.
        conflitos = set()
        for n, (c, _) in enumerate(propostas):
            for m, (d, _) in enumerate(propostas[:n]):
                if np.any(c["componente"] & d["componente"]):
                    conflitos.update((n, m))
        ativos = []
        for n, (candidato, evento) in enumerate(propostas):
            if n in conflitos:
                historico.append({"fatia": k, "ramo": candidato["id"], "estado": "parada",
                                  "motivo": "Fusão/ambiguidade entre ramos rastreados."})
            else:
                historico.append(evento)
                ativos.append(candidato)
                if k == fatia_final:
                    alvo |= candidato["componente"]
        if not ativos:
            break
    return alvo, historico


def crescer_localmente(fatia, anterior, spacing, limiar, roi_mm, ajuste_max,
                       area_max, crescimento_max, deslocamento_max):
    """Crescimento 8-conexo em ROI física; nunca aceita regiões cortadas pela ROI.

    Usa sobreposição para colocar sementes internas, próximas do centroide.
    Se houver vazamento, reduz o limiar em passos de 5 na escala 0–255.
    A bifurcação é uma hipótese geométrica, limitada a dois ramos e uma geração.
    """
    posicao = anterior["centroide"]
    raio = np.ceil(roi_mm / spacing).astype(int)
    lo = np.maximum(0, np.floor(posicao).astype(int) - raio)
    hi = np.minimum(fatia.shape, np.ceil(posicao).astype(int) + raio + 1)
    corte = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    local = fatia[corte]
    xx, yy = np.ogrid[lo[0]:hi[0], lo[1]:hi[1]]
    roi = ((xx-posicao[0])*spacing[0])**2 + ((yy-posicao[1])*spacing[1])**2 <= roi_mm**2
    borda = roi & ~ndi.binary_erosion(roi, structure=np.ones((3, 3)), border_value=0)
    previa = anterior["componente"]
    motivo = "Perda de continuidade com a fatia anterior."
    ajustes = np.unique(np.r_[np.arange(0, ajuste_max, 5), ajuste_max])
    for ajuste in ajustes:
        usado = max(0.0, float(limiar - ajuste))
        ar = (local < usado) & roi
        rotulos = label(ar, connectivity=2)
        if previa is None:
            ponto = tuple((np.rint(posicao).astype(int)-lo))
            ids = [int(rotulos[ponto])] if rotulos[ponto] else []
        else:
            ids = np.unique(rotulos[previa[corte] & ar]).tolist()
        if not ids:
            break
        candidatos = []
        motivo = None
        for ident in ids:
            componente = rotulos == ident
            area = float(componente.sum() * np.prod(spacing))
            pontos = np.argwhere(componente) + lo
            centroide = pontos.mean(axis=0)
            if np.any(componente & borda):
                motivo = "Região alcançou a borda da ROI/imagem; possível vazamento."
                break
            if area > area_max:
                motivo = "Área superior ao limite; possível conexão com o pulmão."
                break
            if previa is not None:
                if area > anterior["area_mm2"] * crescimento_max:
                    motivo = "Crescimento abrupto; possível vazamento."
                    break
                if np.linalg.norm((centroide-posicao)*spacing) > deslocamento_max:
                    motivo = "Deslocamento do centroide superior ao limite."
                    break
            completa = np.zeros(fatia.shape, bool)
            completa[corte] = componente
            ponto = pontos[np.argmin(np.sum(((pontos-posicao)*spacing)**2, axis=1))]
            candidatos.append({"componente": completa, "centroide": centroide,
                               "area_mm2": area, "semente_interna": ponto.tolist()})
        if motivo:
            continue
        if len(candidatos) > 1:
            areas = [c["area_mm2"] for c in candidatos]
            if (len(candidatos) != 2 or anterior["geracao"] > 0
                    or min(areas) < max(3.0, anterior["area_mm2"] * .15)
                    or not anterior["area_mm2"] / crescimento_max <= sum(areas) <= anterior["area_mm2"] * crescimento_max):
                motivo = "Divisão incompatível com dois ramos principais; rastreamento encerrado."
                continue
        return candidatos, usado, None
    return [], None, motivo or "Semente fora da região de ar."


def main():
    parser = argparse.ArgumentParser(
        description="Experimenta fechamento e preenchimento de buracos em 2D."
    )
    parser.add_argument("imagem", help="Tomografia NIfTI Coronacases ou Radiopaedia")
    parser.add_argument("--semente", type=int, nargs=3, default=None, metavar=("I", "J", "K"),
                        help="Opcional: substitui a busca automática; índices RAS a partir de zero")
    parser.add_argument("--area-max-traqueia", type=float, default=400.0,
                        help="Área máxima aceita, em mm² (heurística; padrão: 400)")
    parser.add_argument("--crescimento-max", type=float, default=2.0,
                        help="Razão máxima de área entre fatias (heurística; padrão: 2)")
    parser.add_argument("--deslocamento-max", type=float, default=5.0,
                        help="Deslocamento máximo por fatia, em mm (heurística; padrão: 5)")
    parser.add_argument("--roi-mm", type=float, default=20.0,
                        help="Raio da busca local por ramo, em mm (padrão: 20)")
    parser.add_argument("--ajuste-limiar-max", type=float, default=30.0,
                        help="Redução máxima do limiar de vias aéreas em 0–255 (padrão: 30; 0 desliga)")
    parser.add_argument("--registro", type=Path,
                        help="Opcional: salvar parâmetros e trajetória dos centroides em JSON")
    parser.add_argument("--fatia", type=int, default=None,
                        help="Opcional; sem este argumento, usa a fatia central")
    parser.add_argument("--metodo", choices=["fixo", "otsu"], default="otsu")
    parser.add_argument("--limiar", type=float, default=144.5,
                        help="Limiar em 0–255, usado somente no método fixo")
    parser.add_argument("--raio", type=float, default=2.0,
                        help="Raio em milímetros (padrão: 2 mm); 0 desliga o fechamento")
    parser.add_argument("--mascara", type=Path,
                        help="Referência pulmonar; padrão: ../masks/Lung_Mask/nome da imagem")
    parser.add_argument("--fonte", default=None,
                        choices=["Coronacases", "Radiopaedia"],
                        help="Opcional; detectada pelos prefixos coronacases_ e radiopaedia_")
    parser.add_argument("--centro", type=float, default=CENTRO_JANELA_PULMAO,
                        help="Centro da janela em HU, somente para Coronacases")
    parser.add_argument("--largura", type=float, default=LARGURA_JANELA_PULMAO,
                        help="Largura da janela em HU, somente para Coronacases")
    args = parser.parse_args()

    if not np.isfinite(args.raio) or args.raio < 0:
        parser.error("O raio em mm deve ser finito e não negativo.")
    if not np.isfinite(args.limiar) or not 0 <= args.limiar <= 255:
        parser.error("O limiar deve ser finito e estar entre 0 e 255.")

    try:
        imagem = Path(args.imagem)
        caminho_mascara = args.mascara or (
            imagem.parent.parent / "masks" / "Lung_Mask" / imagem.name
        )
        if not caminho_mascara.is_file():
            raise ValueError(
                f"Referência não encontrada: {caminho_mascara}. "
                "Informe o caminho com --mascara."
            )
        spacing = espacamento_referencia(imagem, caminho_mascara)
        dados = preparar_fatia(args.imagem, args.fonte, k=args.fatia,
                               centro=args.centro, largura=args.largura)
        k = dados["indice_fatia"]
        fatia = dados["imagem_255"]
    except (OSError, ValueError) as erro:
        parser.error(str(erro))
    try:
        candidata, limiar = limiarizar(fatia, args.metodo, args.limiar)
        img = carregar_imagem(args.imagem)
        semente = args.semente
        deteccao = {"origem": "manual"}
        if semente is None:
            semente, deteccao = identificar_semente_traqueia(
                img, dados["source"], k, spacing,
                metodo=args.metodo, limiar_fixo=args.limiar,
                centro=args.centro, largura=args.largura,
                area_max=args.area_max_traqueia, crescimento_max=args.crescimento_max,
                deslocamento_max=args.deslocamento_max,
            )
        traqueia, historico = rastrear_traqueia(
            img, dados["source"], semente, k, spacing,
            metodo=args.metodo, limiar_fixo=args.limiar,
            centro=args.centro, largura=args.largura,
            area_max=args.area_max_traqueia, crescimento_max=args.crescimento_max,
            deslocamento_max=args.deslocamento_max,
            roi_mm=args.roi_mm, ajuste_limiar_max=args.ajuste_limiar_max,
        )
    except (OSError, ValueError) as erro:
        parser.error(str(erro))
    _, sem_borda = remover_componentes_da_borda(candidata)
    sem_traqueia = sem_borda & ~traqueia
    fechada, preenchida = refinar_mascara(sem_traqueia, args.raio, spacing)
    # Impede que fechamento/preenchimento reintroduzam a região rastreada.
    final = preenchida & ~traqueia

    if args.registro is not None:
        registro = {"imagem": str(imagem), "referencia_geometria": str(caminho_mascara),
                    "fonte": dados["source"], "fatia_exibida": k, "semente": semente,
                    "deteccao_semente": deteccao, "rastreamento": "2.5D",
                    "roi_mm": args.roi_mm, "ajuste_limiar_max": args.ajuste_limiar_max,
                    "metodo": args.metodo, "limiar_fixo": args.limiar,
                    "centro_hu": args.centro, "largura_hu": args.largura,
                    "spacing_mm": spacing, "raio_mm": args.raio,
                    "area_max_mm2": args.area_max_traqueia,
                    "crescimento_max": args.crescimento_max,
                    "deslocamento_max_mm": args.deslocamento_max,
                    "alvo_rastreado": bool(traqueia.any()), "historico": historico}
        try:
            args.registro.parent.mkdir(parents=True, exist_ok=True)
            args.registro.write_text(json.dumps(registro, ensure_ascii=False, indent=2) + "\n")
        except OSError as erro:
            parser.error(f"Não foi possível salvar o registro: {erro}")

    print(f"Fonte: {dados['source']}; escala de trabalho: 0–255")
    print(f"Semente {deteccao['origem']}: {tuple(semente)} (índices RAS)")
    print(f"Fatia {k}; método {args.metodo}; limiar {limiar:.2f} (0–255)")
    print(f"Referência de geometria: {caminho_mascara}")
    print(f"Espaçamento no plano: {spacing[0]:.6g} × {spacing[1]:.6g} mm/pixel")
    print(f"Raio do fechamento: {args.raio:g} mm")
    print(f"Raio equivalente por eixo: {args.raio / spacing[0]:.3f} × "
          f"{args.raio / spacing[1]:.3f} pixels")
    print("Pixels após remover a borda:", np.count_nonzero(sem_borda))
    print("Pixels de traqueia removidos nesta fatia:", np.count_nonzero(sem_borda & traqueia))
    paradas = [e for e in historico if e["estado"] == "parada"]
    for evento in paradas:
        print(f"Parada na fatia {evento['fatia']}, ramo {evento.get('ramo', 'todos')}: {evento['motivo']}")
    if not traqueia.any():
        print("A fatia exibida não foi alcançada; nenhuma remoção de vias aéreas foi aplicada nela.")
    else:
        print("Rastreamento 2,5D chegou ao alvo; a remoção pode ser parcial se algum ramo parou antes.")
    print("Pixels após fechamento:", np.count_nonzero(fechada))
    print("Pixels após preencher buracos:", np.count_nonzero(preenchida))
    print("Incluídos pelo fechamento:", np.count_nonzero(fechada & ~sem_traqueia))
    print("Removidos pelo fechamento:", np.count_nonzero(sem_traqueia & ~fechada))
    print("Incluídos ao preencher buracos:", np.count_nonzero(preenchida & ~fechada))
    print("Pixels da máscara final:", np.count_nonzero(final))

    sx, sy = dados["metadados"]["spacing"][:2]
    aspecto = float(sy) / float(sx)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(
        fatia.T, cmap="gray", origin="lower", aspect=aspecto,
        interpolation="nearest", vmin=0, vmax=255,
    )
    axes[0].set_title(f"{dados['source']} — fatia {k} (0–255)")
    axes[1].imshow(
        final.T, cmap="gray", origin="lower", aspect=aspecto,
        interpolation="nearest", vmin=0, vmax=1,
    )
    axes[1].set_title("Máscara final" if traqueia.any()
                      else "Máscara final — alvo não rastreado", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
