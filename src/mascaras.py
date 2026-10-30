"""Leitura de cortes de máscaras NIfTI e exportação de máscaras 2D com geometria."""
from pathlib import Path
import warnings
import numpy as np
from .preprocessing import carregar_imagem, extrair_fatia, obter_indice_fatia, obter_metadados


def validar_rotulos(valores):
    # Trabalha em cópia contígua: fatias de volumes podem ter strides não contíguos.
    valores = np.array(valores, dtype=np.float64, order="C", copy=True)
    if (valores.ndim != 2 or not np.isfinite(valores).all()
            or np.any(valores < 0) or not np.allclose(valores, np.rint(valores), atol=1e-5, rtol=0)
            or np.any(valores > np.iinfo(np.int32).max)):
        raise ValueError("A máscara deve ser 2D, com rótulos inteiros não negativos; probabilidades não são aceitas.")
    return np.rint(valores).astype(np.int32)


def salvar_fatia_mascara(caminho, mascara, imagem, k):
    """Salva somente o corte calculado; não inventa máscaras para outras fatias."""
    caminho = Path(caminho)
    if caminho.suffix.lower() != '.npz':
        raise ValueError("Use extensão .npz para salvar a máscara de uma fatia.")
    k = obter_indice_fatia(imagem, k)
    mascara = validar_rotulos(mascara)
    if mascara.shape != imagem.shape[:2]:
        raise ValueError("A máscara não corresponde à forma da imagem.")
    meta = obter_metadados(imagem)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(caminho, mascara=mascara, fatia=k,
                        shape=meta['shape'], affine=meta['affine'],
                        spacing=meta['spacing'], unidade=meta['unidade_espacial'])


def carregar_fatia_mascara(caminho, k=None):
    """Retorna rótulos, índice e metadados RAS; NPZ usa o corte salvo por padrão."""
    caminho = Path(caminho)
    if caminho.suffix.lower() != '.npz':
        img = carregar_imagem(caminho)
        k = obter_indice_fatia(img, k)
        return validar_rotulos(extrair_fatia(img, k)), k, obter_metadados(img)
    with np.load(caminho, allow_pickle=False) as dados:
        if not {'mascara', 'fatia', 'shape', 'affine', 'spacing', 'unidade'} <= set(dados.files):
            raise ValueError("NPZ sem os campos de máscara, índice e geometria exigidos.")
        indice = np.asarray(dados['fatia'])
        shape = np.asarray(dados['shape'])
        affine = np.asarray(dados['affine'], dtype=float)
        spacing = np.asarray(dados['spacing'], dtype=float)
        if (indice.shape != () or indice.dtype.kind not in 'iu'
                or shape.shape != (3,) or shape.dtype.kind not in 'iu' or np.any(shape <= 0)
                or not 0 <= int(indice) < shape[2]
                or affine.shape != (4, 4) or not np.isfinite(affine).all()
                or spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0)):
            raise ValueError("Geometria ou índice inválido no NPZ.")
        indice = int(indice)
        if k is not None and k != indice:
            raise ValueError(f"Este NPZ contém somente a fatia {indice}, não a fatia {k}.")
        mascara = validar_rotulos(dados['mascara'])
        if mascara.shape != tuple(shape[:2]):
            raise ValueError("Forma da máscara incompatível com a geometria salva.")
        meta = {'shape': tuple(shape), 'affine': affine, 'spacing': tuple(spacing),
                'unidade_espacial': str(dados['unidade'].item())}
        return mascara, indice, meta


def conferir_grades(a, b):
    """Não reamostra: recusa comparação entre grades diferentes."""
    if a['shape'] != b['shape'] or not np.allclose(a['affine'], b['affine'], atol=1e-4, rtol=0):
        raise ValueError("As máscaras têm formas ou grades diferentes; não podem ser comparadas diretamente.")
    ua, ub = a['unidade_espacial'], b['unidade_espacial']
    if 'unknown' in (ua, ub):
        warnings.warn("Unidade espacial ausente; comparação baseada na igualdade numérica das grades.")
    elif ua != ub:
        raise ValueError("As máscaras têm unidades espaciais diferentes.")
