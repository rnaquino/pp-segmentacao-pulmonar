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

import numpy as np

from src.preprocessing import (
    preparar_fatia, CENTRO_JANELA_PULMAO, LARGURA_JANELA_PULMAO, carregar_imagem,
    verificar_alinhamento, obter_metadados, extrair_fatia, preparar_intensidades,
    obter_indice_fatia,
)
# Reexporta os nomes anteriores para preservar os usos existentes em notebooks e testes.
# Novos consumidores devem importar diretamente o módulo responsável pela etapa.
from src.morfologia.geometria import elemento_em_mm, espacamento_referencia
from src.morfologia.limiarizacao import limiarizar
from src.morfologia.traqueia import (
    candidatos_semente, identificar_semente_traqueia, crescer_localmente,
    rastrear_traqueia, proteger_borda,
)
from src.morfologia.operacoes import (
    remover_componentes_da_borda, refinar_mascara, finalizar_mascara,
)
from src.morfologia.visualizacao import exibir_resultado
from src.mascaras import salvar_fatia_mascara


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
                        help="Raio mínimo da busca local em mm, expansível até o dobro (padrão: 20)")
    parser.add_argument("--ajuste-limiar-max", type=float, default=60.0,
                        help="Redução máxima do limiar de vias aéreas em 0–255 (padrão: 60; 0 desliga)")
    parser.add_argument("--margem-protecao", type=float, default=1.5,
                        help="Faixa interna preservada na borda das vias aéreas, em mm (padrão: 1.5)")
    parser.add_argument("--sentido", choices=["decrescente", "crescente"], default="decrescente",
                        help="Ordem dos índices desde a semente; revisar anatomia antes de alterar")
    parser.add_argument("--salvar-mascara", type=Path,
                        help="Salva a máscara final desta fatia em .npz, com índice e geometria")
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
    sentido = -1 if args.sentido == "decrescente" else 1

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
        spacing_xyz = espacamento_referencia(imagem, caminho_mascara, todos_eixos=True)
        spacing = spacing_xyz[:2]
        passo_mm = spacing_xyz[2]
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
                deslocamento_max=args.deslocamento_max, sentido=sentido,
                ajuste_limiar_max=args.ajuste_limiar_max, passo_mm=passo_mm,
            )
        traqueia, historico = rastrear_traqueia(
            img, dados["source"], semente, k, spacing,
            metodo=args.metodo, limiar_fixo=args.limiar,
            centro=args.centro, largura=args.largura,
            area_max=args.area_max_traqueia, crescimento_max=args.crescimento_max,
            deslocamento_max=args.deslocamento_max,
            roi_mm=args.roi_mm, ajuste_limiar_max=args.ajuste_limiar_max,
            margem_protecao=args.margem_protecao, sentido=sentido, passo_mm=passo_mm,
        )
    except (OSError, ValueError) as erro:
        parser.error(str(erro))
    _, sem_borda = remover_componentes_da_borda(candidata)
    fechada, preenchida, final = finalizar_mascara(sem_borda, traqueia, args.raio, spacing)

    if args.salvar_mascara is not None:
        try:
            salvar_fatia_mascara(args.salvar_mascara, final, img, k)
        except (OSError, ValueError) as erro:
            parser.error(str(erro))

    if args.registro is not None:
        registro = {"imagem": str(imagem), "referencia_geometria": str(caminho_mascara),
                    "fonte": dados["source"], "fatia_exibida": k, "semente": semente,
                    "deteccao_semente": deteccao, "rastreamento": "2.5D",
                    "roi_mm": args.roi_mm, "ajuste_limiar_max": args.ajuste_limiar_max,
                    "margem_protecao_mm": args.margem_protecao, "sentido": args.sentido,
                    "metodo": args.metodo, "limiar_fixo": args.limiar,
                    "centro_hu": args.centro, "largura_hu": args.largura,
                    "spacing_mm": spacing, "passo_mm": passo_mm, "raio_mm": args.raio,
                    "deslocamento_efetivo_mm": max(args.deslocamento_max, passo_mm),
                    "area_max_mm2": args.area_max_traqueia,
                    "crescimento_max": args.crescimento_max,
                    "deslocamento_max_mm": args.deslocamento_max,
                    "alvo_rastreado": bool(traqueia.any()),
                    "pixels_excluidos_alvo": int(np.count_nonzero(preenchida & traqueia)),
                    "fatias_aceitas": len({e["fatia"] for e in historico if e["estado"] == "aceita"}),
                    "historico": historico}
        try:
            args.registro.parent.mkdir(parents=True, exist_ok=True)
            args.registro.write_text(json.dumps(registro, ensure_ascii=False, indent=2) + "\n")
        except OSError as erro:
            parser.error(f"Não foi possível salvar o registro: {erro}")

    print(f"Fonte: {dados['source']}; escala de trabalho: 0–255")
    print(f"Sentido: {args.sentido}; margem de proteção: {args.margem_protecao:g} mm")
    print(f"Semente {deteccao['origem']}: {tuple(semente)} (índices RAS)")
    print(f"Fatia {k}; método {args.metodo}; limiar {limiar:.2f} (0–255)")
    print(f"Referência de geometria: {caminho_mascara}")
    print(f"Espaçamento no plano: {spacing[0]:.6g} × {spacing[1]:.6g} mm/pixel")
    print(f"Raio do fechamento: {args.raio:g} mm")
    print(f"Raio equivalente por eixo: {args.raio / spacing[0]:.3f} × "
          f"{args.raio / spacing[1]:.3f} pixels")
    print("Pixels após remover a borda:", np.count_nonzero(sem_borda))
    print("Pixels de vias aéreas excluídos da máscara final:", np.count_nonzero(preenchida & traqueia))
    paradas = [e for e in historico if e["estado"] == "parada"]
    for evento in paradas:
        print(f"Parada na fatia {evento['fatia']}, ramo {evento.get('ramo', 'todos')}: {evento['motivo']}")
    if not traqueia.any():
        print("Sem exclusão de vias aéreas neste corte; consulte a trajetória. Isso não prova ausência de via aérea.")
    else:
        print("Rastreamento 2,5D chegou ao alvo; a remoção pode ser parcial se algum ramo parou antes.")
    print("Pixels após fechamento:", np.count_nonzero(fechada))
    print("Pixels após preencher buracos:", np.count_nonzero(preenchida))
    print("Incluídos pelo fechamento:", np.count_nonzero(fechada & ~sem_borda))
    print("Removidos pelo fechamento:", np.count_nonzero(sem_borda & ~fechada))
    print("Incluídos ao preencher buracos:", np.count_nonzero(preenchida & ~fechada))
    print("Pixels da máscara final:", np.count_nonzero(final))

    exibir_resultado(dados, final, traqueia)


if __name__ == "__main__":
    main()
