"""Mostra uma máscara ou compara duas na mesma fatia, sem executar segmentação.

Aceita volumes NIfTI (.nii/.nii.gz) e cortes .npz exportados pela morfologia.
NIfTI usa o corte central por padrão; NPZ usa seu único corte salvo.
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from src.mascaras import carregar_fatia_mascara, conferir_grades


def exibir_mascaras(mascara, referencia=None, k=None, rotulos=None, rotulos_referencia=None):
    # Se a referência é um corte salvo, seu índice determina também o corte NIfTI.
    if referencia and Path(referencia).suffix.lower() == '.npz' and k is None:
        _, k, _ = carregar_fatia_mascara(referencia)
    atual, k, meta = carregar_fatia_mascara(mascara, k)
    paineis = [(atual, Path(mascara).name, rotulos)]
    if referencia:
        ref, _, geometria = carregar_fatia_mascara(referencia, k)
        conferir_grades(meta, geometria)
        paineis.append((ref, 'Referência: ' + Path(referencia).name, rotulos_referencia))
    # Uma paleta compartilhada preserva o significado visual de cada rótulo.
    classes = sorted({0} | {int(v) for dados, _, _ in paineis for v in np.unique(dados)})
    cores = ['black'] + [plt.get_cmap('tab20')((n-1) % 20) for n in range(1, len(classes))]
    cmap = ListedColormap(cores)
    norm = BoundaryNorm(np.arange(len(classes)+1)-.5, cmap.N)
    fig, axs = plt.subplots(1, len(paineis), figsize=(6*len(paineis), 6), squeeze=False)
    sx, sy = meta['spacing'][:2]
    for ax, (dados, titulo, selecao) in zip(axs[0], paineis):
        if selecao is not None:
            imagem = np.isin(dados, selecao)
            ax.imshow(imagem.T, cmap='gray', vmin=0, vmax=1, origin='lower',
                      interpolation='nearest', aspect=sy/sx)
            titulo += '\nrótulos ' + ', '.join(map(str, selecao))
        else:
            imagem = np.searchsorted(classes, dados)
            im = ax.imshow(imagem.T, cmap=cmap, norm=norm, origin='lower',
                           interpolation='nearest', aspect=sy/sx)
            barra = fig.colorbar(im, ax=ax, ticks=range(len(classes)), shrink=.65)
            barra.ax.set_yticklabels(classes)
            barra.set_label('Rótulo')
        ax.set_title(f'{titulo}\nFatia {k}', fontsize=10)
        ax.axis('off')
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualiza máscaras e compara com a referência do dataset.')
    parser.add_argument('mascara', help='Máscara NIfTI ou fatia .npz gerada')
    parser.add_argument('--referencia', help='Máscara do dataset para comparar lado a lado')
    parser.add_argument('--fatia', type=int, help='Padrão: central no NIfTI ou índice salvo no NPZ')
    parser.add_argument('--rotulos', type=int, nargs='+', help='Classes da primeira máscara exibidas como binária')
    parser.add_argument('--rotulos-referencia', type=int, nargs='+', help='Classes da referência exibidas como binária')
    args = parser.parse_args()
    if args.rotulos_referencia is not None and not args.referencia:
        parser.error('--rotulos-referencia exige --referencia.')
    try:
        exibir_mascaras(args.mascara, args.referencia, args.fatia, args.rotulos, args.rotulos_referencia)
    except (OSError, ValueError, KeyError) as erro:
        parser.error(str(erro))
    plt.show()


if __name__ == '__main__':
    main()
