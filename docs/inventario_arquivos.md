# Inventário e política de limpeza

O ZIP recebido tem 26 scripts `.py`, um módulo `src`, documentação e configuração. **Preservar os scripts essenciais e suas dependências:** há importações cruzadas entre módulos na raiz. Remover arquivos históricos sem revisar as dependências quebraria a reprodutibilidade.

## Caminho de execução e dependências diretas
- `separacao_final.py`: ponto de entrada para a colega.
- `pipeline_base.py`: preparo padronizado.
- `src/preprocessing.py`: NIfTI, RAS e intensidades.
- `segmentacao_componentes.py`, `segmentacao_maior_componente.py`, `segmentacao_morfologia.py`: estágios do preparo.
- `separacao_programacao_dinamica.py`: método adotado.
- `separacao_erosao_condicional.py`: **dependência** da programação dinâmica para encontrar sementes.
- `segmentacao_tres_metodos.py`, `avaliar_segmentacao.py`: dependências transitivas importadas por scripts experimentais.

## Avaliação e documentação reproduzíveis
- `experimentos/comparar_metodos_separacao.py`: comparação quantitativa.
- `experimentos/inspecionar_metodos_separacao.py`: figuras das fatias 129, 130, 133 e 134.
- `diagnosticos/diagnostico_exame_completo.py`, `diagnosticos/diagnosticar_conexao_pulmonar.py`, `diagnosticos/identificar_fatias_conectadas.py`: triagem de casos.
- `experimentos/comparar_isolamento.py`, `diagnosticos/inspecionar_isolamento.py`, `diagnosticos/verificar_etapas_isolamento.py`, `experimentos/experimento_preenchimento.py`: observações anteriores do pipeline.
- `segmentacao_otsu.py`, `segmentacao_limiar_fixo.py`, `segmentacao_esquerda_direita.py`, `visualizar_fatia.py`: etapas históricas/comparativas; preservar para artigo.

## Higiene segura
- `__pycache__` e `src/.DS_Store`: removíveis, não versionar.
- `.venv`, NIfTI, resultados gerados, PNGs e arquivos `.npz`: manter fora do Git.
- A pasta `data/` não foi incluída no ZIP, por isso não se testou execução com os volumes reais nesta revisão.
- Os scripts auxiliares já foram organizados em experimentos/ e diagnosticos/. As importações foram verificadas após a movimentação. Novas movimentações exigem testes de dependências.
