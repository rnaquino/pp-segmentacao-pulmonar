"""Reúne funções reutilizáveis de leitura e preparação dos dados NIfTI.

carregar_imagem lê um exame 3D e aproxima seus eixos de RAS, sem corrigir
obliquidade por reamostragem. obter_indice_fatia valida ou escolhe o corte
central; extrair_fatia devolve os valores 2D em float32 na escala original.
obter_metadados reúne forma, espaçamento, orientação, unidades e affine.

carregar_mascara lê o volume de rótulos inteiros e pode conferir seu
alinhamento com a imagem por verificar_alinhamento. Unidades desconhecidas
não são automaticamente interpretadas como milímetros.

Exemplo de uso em outro script:
    from src.preprocessing import carregar_imagem, extrair_fatia
    img = carregar_imagem("data/raw/coronacases_001.nii.gz")
    fatia = extrair_fatia(img, 150)

aplicar_janela cria uma representação janelada e normalizada para 0–1
somente quando chamada explicitamente. Preserve os valores originais
para limiarização em HU. Este módulo não segmenta nem exibe imagens.

preparar_intensidades aceita arrays 2D/3D e produz imagem_255 em float32.
preparar_fatia organiza original, janelada, normalizada, imagem_255, referência opcional,
metadados e parâmetros em um dicionário com campos padronizados.
preparar_paciente aplica essa preparação a uma linha do manifest.csv.
Para Radiopaedia, a janela em HU não é aplicada: usa-se a escala informada.

Uso da escala comum (não salva nem sobrescreve o NIfTI):
    dados = preparar_fatia("data/raw/coronacases_001.nii.gz", "Coronacases", k=150)
    imagem = dados["imagem_255"]

Para um volume já carregado, use preparar_intensidades(array_3d, fonte).
Os resultados são arrays em memória; geometria e máscaras não são
redimensionadas. Este módulo depende apenas de bibliotecas, não dos scripts
de visualização ou segmentação do projeto.
"""

from pathlib import Path
import operator
import warnings

import nibabel as nib
import numpy as np


# Protocolo recomendado pelos autores do COVID-19-CT-Seg:
# https://github.com/JunMa11/COVID-19-CT-Seg-Benchmark#guidelines
CENTRO_JANELA_PULMAO = -500
LARGURA_JANELA_PULMAO = 1500  # Limites: -1250 a 250 HU.


def carregar_imagem(caminho):
    """Lê um NIfTI 3D e devolve o objeto NiBabel com eixos canonicalizados.

    Os dados podem continuar disponíveis por proxy; não materializamos aqui
    todo o volume. Unidades físicas desconhecidas permanecem desconhecidas.
    """
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    img = nib.as_closest_canonical(nib.load(caminho))
    if len(img.shape) != 3 or any(tamanho <= 0 for tamanho in img.shape):
        raise ValueError("A imagem precisa ser um volume 3D não vazio.")
    spacing = np.asarray(img.header.get_zooms()[:3], dtype=float)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("A imagem possui espaçamento inválido.")
    if not np.isfinite(img.affine).all():
        raise ValueError("A imagem possui affine com NaN ou infinito.")
    return img


def obter_indice_fatia(img, k=None):
    """Valida o índice; se k for None, escolhe a fatia central."""
    if k is None:
        return img.shape[2] // 2
    if isinstance(k, (bool, np.bool_)):
        raise ValueError("O índice da fatia deve ser um número inteiro.")
    try:
        k = operator.index(k)
    except TypeError as erro:
        raise ValueError("O índice da fatia deve ser um número inteiro.") from erro
    if not 0 <= k < img.shape[2]:
        raise ValueError(f"Escolha uma fatia entre 0 e {img.shape[2] - 1}.")
    return k


def extrair_fatia(img, k=None):
    """Devolve uma matriz 2D float32, sem transpor ou alterar sua escala.

    A leitura por dataobj aplica a escala NIfTI e solicita somente a fatia.
    O array devolvido é independente para não alterar dados já em memória.
    """
    k = obter_indice_fatia(img, k)
    fatia = np.array(img.dataobj[:, :, k], dtype=np.float32, copy=True)
    if not np.isfinite(fatia).all():
        raise ValueError("A fatia contém NaN ou infinito.")
    return fatia


def obter_metadados(img):
    """Reúne geometria e unidades sem atribuir milímetros automaticamente."""
    unidade_espacial, unidade_temporal = img.header.get_xyzt_units()
    return {
        "shape": tuple(img.shape),
        "spacing": tuple(float(s) for s in img.header.get_zooms()[:3]),
        "orientacao": nib.aff2axcodes(img.affine),
        "unidade_espacial": unidade_espacial,
        "unidade_temporal": unidade_temporal,
        "affine": img.affine.copy(),
    }


def verificar_alinhamento(img, referencia):
    """Confere forma, affine e unidades; não registra ou reamostra imagens.

    Uma unidade ausente gera aviso: a igualdade numérica da grade permite
    inspeção, mas não prova equivalência física em unidades conhecidas.
    """
    if img.shape != referencia.shape:
        raise ValueError("Imagem e referência têm formas diferentes.")
    unidade_img = img.header.get_xyzt_units()[0]
    unidade_ref = referencia.header.get_xyzt_units()[0]
    if "unknown" not in (unidade_img, unidade_ref) and unidade_img != unidade_ref:
        raise ValueError("Imagem e referência têm unidades espaciais diferentes.")
    if not np.allclose(img.affine, referencia.affine, atol=1e-4, rtol=0):
        raise ValueError("Imagem e referência têm grades espaciais diferentes.")
    if "unknown" in (unidade_img, unidade_ref):
        warnings.warn(
            f"Unidade espacial não informada: imagem={unidade_img}, "
            f"referência={unidade_ref}. Formas e affines numéricas coincidem. "
            "Confirme a unidade antes de calcular medidas físicas.",
            UserWarning,
            stacklevel=2,
        )


def carregar_mascara(caminho, imagem_referencia=None):
    """Devolve (objeto NiBabel, array 3D de rótulos int32).

    Diferentemente de carregar_imagem, esta função materializa todo o volume
    de rótulos. Não combina classes, normaliza ou escolhe lados do pulmão.
    """
    img = carregar_imagem(caminho)
    if imagem_referencia is not None:
        verificar_alinhamento(imagem_referencia, img)
    valores = img.get_fdata(dtype=np.float32, caching="unchanged")
    if not np.isfinite(valores).all():
        raise ValueError("A máscara contém NaN ou infinito.")
    inteiros = np.rint(valores)
    if not np.allclose(valores, inteiros, atol=1e-5, rtol=0):
        raise ValueError("A máscara deve conter rótulos inteiros, não probabilidades.")
    limites = np.iinfo(np.int32)
    if inteiros.min() < limites.min or inteiros.max() > limites.max:
        raise ValueError("Os rótulos não cabem em int32.")
    return img, inteiros.astype(np.int32)


def aplicar_janela_pulmao(dados, centro=CENTRO_JANELA_PULMAO, largura=LARGURA_JANELA_PULMAO):
    """Recorta intensidades HU aos limites da janela, sem normalizar.

    Os padrões seguem a janela [-1250, 250] HU recomendada pelo conjunto.
    Esta função não
    deve ser usada em Radiopaedia já transformado ou em máscaras de classes.
    Não modifica os dados de entrada.
    """
    if not np.isfinite([centro, largura]).all() or largura <= 0:
        raise ValueError("Centro e largura devem ser finitos e largura positiva.")
    dados = np.asarray(dados, dtype=np.float32)
    if not np.isfinite(dados).all():
        raise ValueError("Os dados contêm NaN ou infinito.")
    inferior = centro - largura / 2
    superior = centro + largura / 2
    return np.clip(dados, inferior, superior)


def normalizar(dados, limite_inferior, limite_superior):
    """Recorta à faixa explícita e mapeia para 0–1, devolvendo float32.

    Os limites são fixados por protocolo, não pelos extremos de cada fatia.
    Uma entrada constante é permitida porque a faixa não depende dela.
    """
    if (not np.isfinite([limite_inferior, limite_superior]).all()
            or limite_inferior >= limite_superior):
        raise ValueError("Os limites devem ser finitos, com inferior menor que superior.")
    dados = np.asarray(dados, dtype=np.float32)
    if not np.isfinite(dados).all():
        raise ValueError("Os dados contêm NaN ou infinito.")
    recortados = np.clip(dados, limite_inferior, limite_superior)
    return ((recortados - limite_inferior)
            / (limite_superior - limite_inferior)).astype(np.float32)


def aplicar_janela(dados, centro=CENTRO_JANELA_PULMAO, largura=LARGURA_JANELA_PULMAO):
    """Mantém a API anterior: janela pulmonar e normalização para 0–1."""
    janelada = aplicar_janela_pulmao(dados, centro, largura)
    return normalizar(janelada, centro - largura / 2, centro + largura / 2)


def preparar_intensidades(dados, fonte, centro=CENTRO_JANELA_PULMAO, largura=LARGURA_JANELA_PULMAO,
                         limites_radiopaedia=(0, 255)):
    """Prepara uma fatia 2D ou volume 3D para trabalhar na faixa 0–255.

    Coronacases: recorta HU à janela [centro-largura/2, centro+largura/2]
    e aplica 255 * (HU_recortado - inferior) / (superior - inferior).
    Radiopaedia: preserva 0–255; uma faixa diferente exige limites explícitos.
    Não estima os limites pelo mínimo/máximo de cada fatia ou paciente.

    imagem_255 é float32, sem quantização para uint8. A faixa é comum, mas
    não recupera HU de Radiopaedia nem garante contraste igual entre fontes.
    original é uma cópia; normalizada (0–1) é mantida por compatibilidade.
    Não altera forma, orientação, arquivos ou máscaras de referência.
    """
    fontes = {"coronacases": "Coronacases", "radiopaedia": "Radiopaedia"}
    if not isinstance(fonte, str) or fonte.lower() not in fontes:
        raise ValueError("Informe a fonte: Coronacases ou Radiopaedia.")
    fonte = fontes[fonte.lower()]
    original = np.array(dados, dtype=np.float32, copy=True)
    if original.ndim not in (2, 3) or original.size == 0:
        raise ValueError("Informe uma fatia 2D ou volume 3D não vazio.")
    if not np.isfinite(original).all():
        raise ValueError("Os dados contêm NaN ou infinito.")

    if fonte == "Coronacases":
        janelada = aplicar_janela_pulmao(original, centro, largura)
        limites = (centro - largura / 2, centro + largura / 2)
        normalizada = normalizar(janelada, *limites)
        parametros = {"janela_hu": {"centro": centro, "largura": largura},
                      "limites_normalizacao": limites, "escala_original": "HU"}
    else:
        if len(limites_radiopaedia) != 2:
            raise ValueError("Informe dois limites para a escala Radiopaedia.")
        limites = tuple(limites_radiopaedia)
        # Não tentar recuperar HU nem aplicar uma segunda janela pulmonar.
        janelada = None
        normalizada = normalizar(original, *limites)
        parametros = {"janela_hu": None, "limites_normalizacao": limites,
                      "escala_original": "intensidade transformada"}
        if original.min() < limites[0] or original.max() > limites[1]:
            warnings.warn(
                "A imagem Radiopaedia ultrapassa a faixa configurada. "
                "A normalização recortará esses valores; confira os limites.",
                UserWarning, stacklevel=2,
            )

    # Na escala nativa Radiopaedia, preservar exatamente cada intensidade.
    if fonte == "Radiopaedia" and limites == (0, 255):
        imagem_255 = np.clip(original, 0, 255)
    else:
        imagem_255 = (normalizada * np.float32(255)).astype(np.float32)
    parametros["escala_trabalho"] = (0, 255)
    parametros["dtype_trabalho"] = "float32"
    return {
        "source": fonte,
        "original": original,
        "janelada": janelada,
        "normalizada": normalizada,
        "imagem_255": imagem_255,
        "parametros": parametros,
    }


def identificar_fonte(caminho_imagem):
    """Identifica a fonte pelo prefixo do nome, sem inferir pela intensidade."""
    nome = Path(caminho_imagem).name.lower()
    if nome.startswith("coronacases_"):
        return "Coronacases"
    if nome.startswith("radiopaedia_"):
        return "Radiopaedia"
    raise ValueError(
        "Não foi possível identificar a fonte pelo nome do arquivo. "
        "Informe fonte='Coronacases' ou fonte='Radiopaedia' "
        "(nos scripts, use --fonte)."
    )


def preparar_fatia(caminho_imagem, fonte=None, k=None, caminho_mascara=None,
                  patient_id=None, centro=CENTRO_JANELA_PULMAO, largura=LARGURA_JANELA_PULMAO,
                  limites_radiopaedia=(0, 255), rotulos_pulmonares=(1, 2)):
    """Devolve um registro 2D padronizado, sem gerar uma previsão.

    Sem fonte, identifica Coronacases/Radiopaedia pelo prefixo do arquivo.
    Uma fonte explícita tem prioridade, inclusive para arquivos renomeados.
    Use registro["imagem_255"] nos próximos scripts para a escala 0–255.
    "normalizada" continua em 0–1 para compatibilidade com a API anterior.
    A original é sempre preservada. Em Radiopaedia, janelada é None.
    A referência é opcional e não participa das transformações da imagem.
    """
    if fonte is None:
        fonte = identificar_fonte(caminho_imagem)
    img = carregar_imagem(caminho_imagem)
    k = obter_indice_fatia(img, k)
    intensidades = preparar_intensidades(
        extrair_fatia(img, k), fonte, centro, largura, limites_radiopaedia,
    )

    rotulos = None
    referencia = None
    metadados = obter_metadados(img)
    metadados["referencia"] = None
    if caminho_mascara is not None:
        ref_img = carregar_imagem(caminho_mascara)
        verificar_alinhamento(img, ref_img)
        valores = extrair_fatia(ref_img, k)
        inteiros = np.rint(valores)
        limites_int32 = np.iinfo(np.int32)
        if (not np.allclose(valores, inteiros, atol=1e-5, rtol=0)
                or inteiros.min() < limites_int32.min
                or inteiros.max() > limites_int32.max):
            raise ValueError("A referência deve conter rótulos válidos para int32.")
        rotulos = inteiros.astype(np.int32)
        if not rotulos_pulmonares or 0 in rotulos_pulmonares:
            raise ValueError("Informe rótulos pulmonares sem incluir o fundo 0.")
        if any(isinstance(r, (bool, np.bool_))
               or not isinstance(r, (int, np.integer)) for r in rotulos_pulmonares):
            raise ValueError("Os rótulos pulmonares devem ser inteiros.")
        referencia = np.isin(rotulos, rotulos_pulmonares)
        metadados["referencia"] = obter_metadados(ref_img)

    nome = Path(caminho_imagem).name
    if patient_id is None:
        patient_id = nome[:-7] if nome.endswith(".nii.gz") else Path(nome).stem
    return {
        "patient_id": patient_id,
        "source": intensidades["source"],
        "image_path": str(caminho_imagem),
        "lung_mask_path": str(caminho_mascara) if caminho_mascara is not None else None,
        "indice_fatia": k,
        "original": intensidades["original"],
        "janelada": intensidades["janelada"],
        "normalizada": intensidades["normalizada"],
        "imagem_255": intensidades["imagem_255"],
        "rotulos_referencia": rotulos,
        "mascara_pulmonar": referencia,
        "metadados": metadados,
        "parametros": {**intensidades["parametros"],
                       "rotulos_pulmonares": tuple(rotulos_pulmonares)},
    }


def preparar_paciente(registro, raiz_projeto, k=None, **parametros):
    """Prepara uma fatia a partir de uma linha lida por csv.DictReader.

    Os caminhos do catálogo são relativos à raiz do projeto. Não guarda
    todos os pacientes na memória: o chamador pode processar um por vez.
    """
    obrigatorios = ("patient_id", "source", "image_path", "lung_mask_path")
    faltantes = [campo for campo in obrigatorios if not registro.get(campo)]
    if faltantes:
        raise ValueError(f"Campos ausentes no inventário: {', '.join(faltantes)}")
    raiz = Path(raiz_projeto)
    return preparar_fatia(
        raiz / registro["image_path"], fonte=registro["source"], k=k,
        caminho_mascara=raiz / registro["lung_mask_path"],
        patient_id=registro["patient_id"], **parametros,
    )
