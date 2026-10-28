"""Interface estável de passagem: segmentação 2D ANTES da reincorporação.

As máscaras não são resultados clínicos validados. Coordenadas seguem a
orientação RAS preparada em src.preprocessing (eixo 0 crescente para a direita
anatômica). A identificação dos lados é heurística e deve ser auditada.
"""
import argparse
from pathlib import Path

import numpy as np

from src.preprocessing import preparar_fatia
from pipeline_base import preparar_mascara_base
from segmentacao_morfologia import espacamento_referencia
from separacao_programacao_dinamica import separar_pulmoes

RAIO_MM_PADRAO = 3.0


def segmentar_fatia(imagem_255, spacing, raio_mm=RAIO_MM_PADRAO):
    """Retorna máscaras booleanas na grade 2D da entrada.

    Sem usar rótulos de referência na segmentação. Não corrige ausência de pulmão
    nem reinclui regiões excluídas; isso pertence ao estágio seguinte.
    """
    imagem = np.asarray(imagem_255)
    if imagem.ndim != 2 or not np.isfinite(imagem).all():
        raise ValueError('A imagem precisa ser 2D e conter valores finitos.')

    base = preparar_mascara_base(imagem, spacing=spacing, raio_mm=raio_mm)
    vazia = np.zeros(base.shape, dtype=bool)
    resultado = {
        'mascara_base': base,
        'pulmao_esquerdo': vazia.copy(),
        'pulmao_direito': vazia.copy(),
        'mascara_combinada': vazia.copy(),
        'status': 'falha',
        'metodo': 'programacao_dinamica',
        'raio_mm': float(raio_mm),
    }
    if not base.any():
        resultado['status'] = 'mascara_vazia'
        return resultado

    esquerdo, direito, caminho = separar_pulmoes(imagem, base)
    if esquerdo is None or direito is None:
        return resultado
    esquerdo = np.asarray(esquerdo, dtype=bool)
    direito = np.asarray(direito, dtype=bool)
    if esquerdo.shape != base.shape or direito.shape != base.shape:
        raise ValueError('Método de separação retornou forma incompatível.')
    if np.any(esquerdo & direito):
        raise ValueError('Máscaras esquerda e direita se sobrepõem.')
    if not esquerdo.any() or not direito.any():
        return resultado
    if np.any((esquerdo | direito) & ~base):
        raise ValueError('A separação acrescentou pixels fora da máscara base.')

    resultado['pulmao_esquerdo'] = esquerdo
    resultado['pulmao_direito'] = direito
    resultado['mascara_combinada'] = esquerdo | direito
    resultado['status'] = ('separacao_aplicada' if caminho is not None
                            else 'componentes_preservados_heuristica')
    return resultado


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('imagem', help='Tomografia NIfTI em data/raw/')
    parser.add_argument('--fatia', type=int, required=True)
    parser.add_argument('--fonte', choices=('Coronacases', 'Radiopaedia'), default=None,
                        help='Fonte da TC; opcional quando o nome permite identificação')
    parser.add_argument('--referencia', default=None,
                        help='Referência NIfTI alinhada (somente geometria)')
    parser.add_argument('--raio', type=float, default=RAIO_MM_PADRAO)
    parser.add_argument('--saida', default=None,
                        help='Arquivo NPZ com as máscaras e o status')
    args = parser.parse_args()
    referencia = args.referencia or str(Path('data/masks/Lung_Mask') / Path(args.imagem).name)
    spacing = espacamento_referencia(args.imagem, referencia)
    dados = preparar_fatia(args.imagem, fonte=args.fonte, k=args.fatia)
    resultado = segmentar_fatia(dados['imagem_255'], spacing, args.raio)
    destino = Path(args.saida or f'resultados/mascaras_{Path(args.imagem).name.replace(".nii.gz", "")}_fatia_{args.fatia}.npz')
    destino.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destino, **resultado)
    print(f"Fatia {args.fatia}: {resultado['status']}; saída: {destino}")


if __name__ == '__main__':
    main()
