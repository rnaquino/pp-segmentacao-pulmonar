"""Remoção de traqueia: sementes, crescimento local, rastreamento e exclusão.

A geometria e a limiarização são compartilhadas com as demais etapas.
A exclusão deve ser aplicada depois do fechamento e preenchimento pulmonar.
Os rótulos de referência não participam das decisões de segmentação.
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border

from src.preprocessing import (
    CENTRO_JANELA_PULMAO, LARGURA_JANELA_PULMAO, obter_indice_fatia,
    obter_metadados, preparar_intensidades, extrair_fatia,
)
from .geometria import elemento_em_mm
from .limiarizacao import limiarizar


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
                                deslocamento_max=5.0, sentido=-1, ajuste_limiar_max=60.0, passo_mm=1.0):
    """Busca nos 40% iniciais no sentido escolhido; confirma em três cortes.

    Ignora cortes ambíguos e procura um candidato inequívoco adiante na busca.
    Ajuste de limiar não pode fazer uma alternativa grande desaparecer.
    A semente não ultrapassa o alvo; a confirmação pode consultar dois cortes
    além dele. Só usa intensidades, nunca os rótulos da referência.
    """
    elemento_em_mm(0, spacing)
    if not np.isfinite(passo_mm) or passo_mm <= 0:
        raise ValueError("Espaçamento entre fatias deve ser positivo e finito, em mm.")
    if not np.isfinite(deslocamento_max) or deslocamento_max <= 0:
        raise ValueError("O deslocamento máximo deve ser positivo e finito.")
    deslocamento_max = max(deslocamento_max, passo_mm)
    obter_indice_fatia(img, fatia_final)
    if obter_metadados(img)["orientacao"] != ("R", "A", "S"):
        raise ValueError("A busca automática exige orientação RAS.")
    if (not np.isfinite([area_max, crescimento_max, deslocamento_max]).all()
            or area_max < 30 or crescimento_max < 1 or deslocamento_max <= 0):
        raise ValueError("Busca automática exige área máxima >= 30 mm², crescimento >= 1 e deslocamento positivo.")
    if not np.isfinite(ajuste_limiar_max) or not 0 <= ajuste_limiar_max <= 255:
        raise ValueError("Ajuste máximo deve estar entre 0 e 255.")
    if sentido not in (-1, 1):
        raise ValueError("Sentido deve ser -1 (decrescente) ou 1 (crescente).")
    limite = (max(fatia_final, int(np.floor(img.shape[2] * 0.6))) if sentido == -1
              else min(fatia_final, int(np.ceil(img.shape[2] * 0.4)) - 1))
    inicio = img.shape[2] - 1 if sentido == -1 else 0
    ambiguidades = []
    cache = {}
    examinadas = 0

    def candidatos(k):
        if k not in cache:
            preparada = preparar_intensidades(extrair_fatia(img, k), fonte,
                                               centro, largura)["imagem_255"]
            if preparada.min() == preparada.max():
                cache[k] = []
            else:
                base = candidatos_semente(preparada, spacing, metodo, limiar_fixo, area_max)
                _, limiar_base = limiarizar(preparada, metodo, limiar_fixo)
                for c in base:
                    c["limiar"] = limiar_base
                grandes = [c for c in base if not c["area_permitida"]]
                cache[k] = base
                # Uma região grande não pode simplesmente desaparecer e liberar uma pequena vizinha.
                for ajuste in np.unique(np.r_[np.arange(5, ajuste_limiar_max, 5), ajuste_limiar_max]):
                    if not grandes or ajuste <= 0:
                        break
                    novos = candidatos_semente(preparada, spacing, "fixo", max(0, limiar_base-ajuste), area_max)
                    if (novos and all(c["area_permitida"] for c in novos)
                            and all(any(np.count_nonzero(c["componente"] & g["componente"])
                                        >= .5 * np.count_nonzero(g["componente"]) for c in novos) for g in grandes)):
                        for c in novos:
                            c["limiar"] = max(0, limiar_base-ajuste)
                        cache[k] = novos
                        break
        return cache[k]

    for k in range(inicio, limite + sentido, sentido):
        if not 0 <= k + 2 * sentido < img.shape[2]:
            break
        examinadas += 1
        confirmados = []
        for inicial in candidatos(k):
            if not inicial["area_permitida"]:
                continue
            anterior = inicial
            for seguinte in (k + sentido, k + 2 * sentido):
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
            ambiguidades.append(f"Semente automática ambígua na fatia {k}: múltiplos candidatos persistentes.")
            cache.pop(k, None)
            continue
        if len(confirmados) == 1:
            if any(not c["area_permitida"] for c in candidatos(k)):
                ambiguidades.append(f"Semente automática ambígua na fatia {k}: região central maior que o limite de área.")
                cache.pop(k, None)
                continue
            escolhido = confirmados[0]
            semente = (*escolhido["ponto"], k)
            return semente, {"origem": "automatica", "fatias_iniciais_examinadas": examinadas,
                              "intervalo_busca": sorted([limite, inicio]), "sentido": sentido,
                              "ambiguidades_ignoradas": ambiguidades,
                              "fatias_confirmacao": [k, k + sentido, k + 2 * sentido],
                              "area_inicial_mm2": escolhido["area_mm2"], "limiar_semente": escolhido["limiar"],
                              "criterios": {"area_min_mm2": 30, "area_max_mm2": area_max,
                                            "solidez_min": 0.75, "razao_autovalores_min": 0.19,
                                            "margem_lateral": "min(40 mm, 15% da largura corporal)"}}
        # Mantém no máximo as fatias necessárias à janela de confirmação.
        cache.pop(k, None)
    if ambiguidades:
        raise ValueError(ambiguidades[-1] + " Nenhum corte inequívoco posterior na busca. Use --semente I J K.")
    raise ValueError("Nenhuma semente automática confiável nos cortes pesquisados. Revise --sentido ou use --semente I J K.")


def crescer_localmente(fatia, anterior, spacing, limiar, roi_mm, ajuste_max,
                       area_max, crescimento_max, deslocamento_max):
    """Crescimento 8-conexo em ROI física; nunca aceita regiões cortadas pela ROI.

    Usa sobreposição para colocar sementes internas, próximas do centroide.
    Se houver vazamento, reduz o limiar em passos de 5 na escala 0–255.
    A bifurcação é uma hipótese geométrica, limitada a dois ramos e uma geração.
    """
    posicao = anterior["centroide"]
    previa = anterior["componente"]
    if previa is not None:
        extensao = np.linalg.norm((np.argwhere(previa) - posicao) * spacing, axis=1).max()
        # A ROI não pode amputar uma região previamente aceita só porque ela alongou.
        roi_mm = min(2 * roi_mm, max(roi_mm, extensao + deslocamento_max))
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
        if previa is not None:
            # Sobreposições residuais não são sementes de novos brônquios.
            minimo = max(2.0, .05 * anterior["area_mm2"])
            ids = [ident for ident in ids
                   if np.count_nonzero((rotulos == ident) & previa[corte]) * np.prod(spacing) >= minimo]
            if not ids:
                motivo = "Perda de sobreposição significativa com a fatia anterior."
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
        # Na bifurcação, compare a união com o pai, não cada filho com o centro do pai.
        areas = np.array([c["area_mm2"] for c in candidatos])
        centro_conjunto = np.average([c["centroide"] for c in candidatos], axis=0, weights=areas)
        if previa is not None and np.linalg.norm((centro_conjunto-posicao)*spacing) > deslocamento_max:
            motivo = "Deslocamento do centroide conjunto superior ao limite."
            continue
        return candidatos, usado, None
    return [], None, motivo or "Semente fora da região de ar."


def proteger_borda(mascara, margem_mm, spacing):
    """Exclui apenas o núcleo interno; preserva uma faixa de incerteza em mm."""
    elemento = elemento_em_mm(margem_mm, spacing)
    return ndi.binary_erosion(mascara, structure=elemento) if margem_mm else mascara.copy()


def rastrear_traqueia(img, fonte, semente, fatia_final, spacing,
                     metodo="otsu", limiar_fixo=144.5,
                     centro=CENTRO_JANELA_PULMAO, largura=LARGURA_JANELA_PULMAO,
                     area_max=400.0, crescimento_max=2.0, deslocamento_max=5.0,
                     roi_mm=20.0, ajuste_limiar_max=60.0, observador=None,
                     margem_protecao=1.5, sentido=-1, passo_mm=1.0):
    """Rastreamento 2,5D de uma traqueia e até dois ramos, fatia a fatia.

    A continuidade exige sobreposição; não pula fatias nem faz operações 3D.
    Usa ROI física, limiar adaptativo conservador e parada independente.
    passo_mm vem da geometria de referência alinhada; define junto ao limite
    configurado a tolerância de deslocamento max(deslocamento_max, passo_mm).
    Retorna o núcleo protegido dos ramos no alvo e eventos de cada ramo.
    observador(k, exclusao, evento) permite auditoria por fatia sem montar 3D.
    """
    if (not np.isfinite([area_max, crescimento_max, deslocamento_max]).all()
            or area_max <= 0 or crescimento_max < 1 or deslocamento_max <= 0):
        raise ValueError("Limites de rastreamento inválidos: área/deslocamento positivos, crescimento >= 1.")
    if not np.isfinite(passo_mm) or passo_mm <= 0:
        raise ValueError("Espaçamento entre fatias deve ser positivo e finito, em mm.")
    deslocamento_max = max(deslocamento_max, passo_mm)
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
    if sentido not in (-1, 1):
        raise ValueError("Sentido deve ser -1 (decrescente) ou 1 (crescente).")
    if (fatia_final - inicio) * sentido < 0:
        raise ValueError("Semente e alvo incompatíveis com o sentido de rastreamento.")
    elemento_em_mm(margem_protecao, spacing)

    if not np.isfinite([roi_mm, ajuste_limiar_max]).all() or roi_mm <= 0 or not 0 <= ajuste_limiar_max <= 255:
        raise ValueError("ROI deve ser positiva e ajuste de limiar deve estar entre 0 e 255.")
    ativos = [{"id": 0, "geracao": 0, "centroide": np.array([i, j], float),
               "componente": None, "area_mm2": None}]
    proximo_id = 1
    alvo = np.zeros(img.shape[:2], dtype=bool)
    historico = []
    for k in range(inicio, fatia_final + sentido, sentido):
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
                exclusao = proteger_borda(candidato["componente"], margem_protecao, spacing)
                evento["area_excluida_mm2"] = float(exclusao.sum() * np.prod(spacing))
                if observador is not None:
                    observador(k, exclusao.copy(), evento.copy())
                if k == fatia_final:
                    alvo |= exclusao
        if not ativos:
            break
    return alvo, historico


def remover_traqueia(mascara, exclusao):
    """Subtrai somente a região protegida de exclusão, sem alterar o restante."""
    return mascara & ~exclusao
