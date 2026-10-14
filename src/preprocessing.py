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
"""

from pathlib import Path
import operator
import warnings

import nibabel as nib
import numpy as np


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


def aplicar_janela(dados, centro=-600, largura=1500):
    """Cria uma cópia janelada e normalizada para 0–1.

    Centro/largura usam a escala dos dados de entrada. Os padrões são para
    HU, não para Radiopaedia já normalizado. Preserve os dados originais
    para limiarização em HU; nunca aplique esta função a rótulos de máscara.
    """
    if not np.isfinite([centro, largura]).all() or largura <= 0:
        raise ValueError("Centro e largura devem ser finitos e largura positiva.")
    dados = np.asarray(dados, dtype=np.float32)
    if not np.isfinite(dados).all():
        raise ValueError("Os dados contêm NaN ou infinito.")
    inferior = centro - largura / 2
    superior = centro + largura / 2
    return (np.clip(dados, inferior, superior) - inferior) / largura
