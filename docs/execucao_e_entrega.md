# Execução, artefatos e passagem de etapa

## Ambiente

Instale `requirements.txt` em um ambiente virtual. O projeto original registra Python 3.9.6; não há garantia de compatibilidade de todas as versões de bibliotecas com ambientes diferentes. Execute os comandos **na raiz** do repositório.

## Dados e geometria

Os volumes `.nii.gz` não são distribuídos neste pacote. Consulte `data/README.md` ou o catálogo do dataset para obter os dados e suas máscaras de referência. Se os arquivos de dados não estiverem disponíveis, é possível inspecionar o código e executar apenas verificações sintéticas, não a validação científica.

O espaçamento em milímetros é extraído da referência alinhada por `espacamento_referencia`; cabeçalhos das tomografias podem indicar unidade `unknown`. É necessário confirmar a compatibilidade geométrica. A referência NÃO é utilizada como máscara para segmentar.

## Produção de máscara por fatia

```bash
python separacao_final.py data/raw/coronacases_001.nii.gz --fatia 142 --referencia data/masks/Lung_Mask/coronacases_001.nii.gz --raio 3 --saida resultados/coronacases_001_fatia_142.npz
```

Abra os resultados em Python com `np.load('resultados/coronacases_001_fatia_142.npz', allow_pickle=False)`. As quatro máscaras são booleanas e têm a mesma grade da fatia de entrada. O retorno `status` distingue máscara vazia, falha, separação efetiva e preservação heurística de componentes. Não trate toda saída retornada como segmentação anatômica válida.

Para integração direta, use:

```python
from separacao_final import segmentar_fatia
resultado = segmentar_fatia(imagem_255, spacing=(0.8, 0.8), raio_mm=3.0)
esquerdo = resultado['pulmao_esquerdo']
direito = resultado['pulmao_direito']
```

A próxima etapa recebe estas máscaras e a combinada, e é responsável pela **reincorporação de regiões excluídas**; isso não é realizado aqui.

## Métricas, imagens e comparações

- `python -m experimentos.comparar_metodos_separacao --help` lista os parâmetros da comparação quantitativa em CSV entre erosão condicional, programação dinâmica e caminho de maior intensidade.
- `python avaliar_segmentacao.py --help` mostra como avaliar Dice e IoU com referência, quando disponíveis máscaras e volumes alinhados.
- `python -m experimentos.inspecionar_metodos_separacao` disponibiliza a inspeção visual de métodos e fatias selecionadas.
- `python -m diagnosticos.diagnostico_exame_completo --help` permite inspecionar casos ao longo do exame.

As figuras de scripts que usam `plt.show()` exigem sessão gráfica ou ajuste explícito do backend; **nem todos os scripts salvam figuras automaticamente**. Registre a revisão Git, o exame, a fatia, os parâmetros, os resultados e os erros em cada comparação. Não se deve afirmar que foi calculada distância de Hausdorff pelo comparador atual, que reporta Dice e IoU.

## Testes e critérios para entrega

1. `python -m unittest discover -s tests -v` — testes sintéticos da interface.
2. `python -m compileall -q .` — verificação sintática, sem executar a pesquisa.
3. Com volumes locais, executar casos representativos de Coronacases e Radiopaedia, inclusive fatias periféricas e com pulmões conexos, e verificar as saídas.
4. Revisar visualmente identidade dos lados e plausibilidade anatômica. Métricas por lado precisam respeitar o mapeamento real entre lados anatômicos e rótulos de referência.
5. Manter resultados CSV/NPZ/figuras em `resultados/` sem incluí-los automaticamente no Git.

**Limitações atuais:** não há teste integral com os 20 volumes nesta cópia; não há evidência de superioridade estatística de programação dinâmica; o pipeline pode perder regiões, falhar nas extremidades e preservar componentes incorretos. Não encaminhar como ferramenta clínica.
