# Registro das decisões — preparação e separação 2D

## Objetivo e etapa entregue
O projeto usa métodos clássicos para segmentar parênquima pulmonar. Esta entrega termina após a separação dos pulmões e **antes** da reincorporação de regiões excluídas. O estágio seguinte recebe as duas máscaras e a máscara binária combinada.

## Sequência efetivamente implementada
1. Leitura/reorientação RAS e preparação das intensidades em 0–255 por `src/preprocessing.py`.
2. Otsu 2D por fatia (`threshold_otsu`; valores inferiores ao limiar correspondem a candidatos de ar).
3. Exclusão de componentes conectados à borda (`clear_border`, conectividade 8).
4. Preservação dos dois maiores componentes e fragmentos próximos a até 3 px (`manter_maior_componente`). **Não é o mesmo que escolher isoladamente o maior componente do corpo.**
5. Fechamento morfológico 2D com disco físico de **raio de 3 mm** (não diâmetro), convertido pelo espaçamento no plano.
6. Preenchimento de buracos interiores (`binary_fill_holes`). Não recupera todo o tecido excluído, a cargo da etapa seguinte.
7. Separação condicional: programação dinâmica se necessário; caso contrário, preserva os dois componentes principais.

## Métodos de separação comparados
- `separacao_erosao_condicional.py`: erosão para sementes e reconstrução competitiva.
- `separacao_programacao_dinamica.py`: busca de caminho de custo mínimo com intensidade e restrição de corredor. Método **selecionado operacionalmente**, sem superioridade estatística demonstrada.
- `separacao_maior_intensidade.py`: caminho local favorecendo intensidades mais altas.

A implementação selecionada usa `encontrar_sementes` do arquivo de erosão para inicializar o centro nos casos conectados; por isso **esse arquivo continua sendo uma dependência do pipeline principal**.

## Amostra, resultados e limitações
Selecionar todas as fatias com qualquer rótulo pulmonar (1 ou 2) na referência, subdividindo a sequência por `np.array_split` nas regiões inicial, central e final. A referência seleciona a amostra e permite avaliação, mas nunca orienta os pixels da segmentação.

Diagnóstico de conexão (triagem): Coronacases 001 — 227 fatias, 198 com dois componentes distintos, 12 com recuperação insuficiente, 17 com referência bilateral incompleta; Coronacases 002 — 176 fatias, 169 com dois componentes distintos, 7 com recuperação insuficiente. Esses diagnósticos foram produzidos **antes da integração do fechamento de 3 mm**, e não devem ser confundidos com métricas do pipeline final.

Na comparação integrada do Coronacases 002, as fatias 129, 130, 133 e 134 apresentaram pequenas diferenças entre os três métodos; a inspeção qualitativa não mostrou vantagem anatômica clara. Existem falhas e saídas com Dice zero em fatias periféricas. Não declarar generalização para todos os 20 exames nem significância estatística.

**Pendências metodológicas registradas:** as tomografias apresentam unidade espacial `unknown` enquanto as máscaras alinhadas declaram `mm`; o uso do espaçamento físico depende da compatibilidade geométrica já verificada e de confirmação da unidade nos dados de origem. O fechamento de 3 mm é tratado neste código como **raio**. A regra de dois maiores componentes pode confundir região pulmonar com ruído, especialmente nas extremidades; o retorno `componentes_preservados_heuristica` não garante adequação anatômica. A orientação RAS e a correspondência dos nomes esquerdo/direito devem ser verificadas antes de métricas por lado. A avaliação final do projeto também requer distância de Hausdorff e os 20 volumes, após o refinamento seguinte.

## Para reprodução e artigo
- Fixar a revisão do Git, versões em `requirements.txt`, parâmetros e janela do preprocessing.
- Usar exatamente a mesma entrada e fatias para todos os três métodos.
- Reportar Dice/IoU por fatia e por lado, agregações por paciente e por região; falhas não devem ser escondidas ou confundidas com valores ausentes.
- Guardar CSVs e figuras geradas como artefatos, sem adicionar volumes NIfTI privados ao Git.
- Consultar `docs/preprocessing.md` para decisões de intensidade/geometria.
