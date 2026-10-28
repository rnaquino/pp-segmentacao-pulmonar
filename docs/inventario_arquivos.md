# Inventário e política de limpeza

O ZIP recebido tem 26 scripts `.py`, um módulo `src`, documentação e configuração. **Não deletar nem mover** os scripts antigos nesta entrega: há importações cruzadas entre módulos na raiz. Remover arquivos históricos sem revisar as dependências quebraria a reprodutibilidade.

## Caminho de execução e dependências diretas
- `separacao_final.py`: ponto de entrada para a colega.
- `pipeline_base.py`: preparo padronizado.
- `src/preprocessing.py`: NIfTI, RAS e intensidades.
- `segmentacao_componentes.py`, `segmentacao_maior_componente.py`, `segmentacao_morfologia.py`: estágios do preparo.
- `separacao_programacao_dinamica.py`: método adotado.
- `separacao_erosao_condicional.py`: **dependência** da programação dinâmica para encontrar sementes.
- `segmentacao_tres_metodos.py`, `avaliar_segmentacao.py`: dependências transitivas importadas por scripts experimentais.

## Avaliação e documentação reproduzíveis
- `comparar_metodos_separacao.py`: comparação quantitativa.
- `inspecionar_metodos_separacao.py`: figuras das fatias 129, 130, 133 e 134.
- `diagnostico_exame_completo.py`, `diagnosticar_conexao_pulmonar.py`, `identificar_fatias_conectadas.py`: triagem de casos.
- `comparar_isolamento.py`, `inspecionar_isolamento.py`, `verificar_etapas_isolamento.py`, `experimento_preenchimento.py`: observações anteriores do pipeline.
- `segmentacao_otsu.py`, `segmentacao_limiar_fixo.py`, `segmentacao_esquerda_direita.py`, `visualizar_fatia.py`: etapas históricas/comparativas; preservar para artigo.

## Higiene segura
- `__pycache__` e `src/.DS_Store`: removíveis, não versionar.
- `.venv`, NIfTI, resultados gerados, PNGs e arquivos `.npz`: manter fora do Git.
- A pasta `data/` não foi incluída no ZIP, por isso não se testou execução com os volumes reais nesta revisão.
- Evitar reorganização física dos scripts antes de passar à colega. A reorganização em pacotes é uma tarefa posterior que exige testes de importação.
