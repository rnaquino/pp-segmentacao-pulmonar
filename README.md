# Segmentação pulmonar por métodos clássicos

Projeto acadêmico de segmentação pulmonar em tomografia computadorizada, usando limiarização, componentes conexos e operações morfológicas.

## Estado atual

- Leitura de NIfTI e padronização da orientação com NiBabel.
- Visualização de uma fatia.
- Experimentos separados de limiar fixo e Otsu na escala 0–255 para ambas as fontes.
- Remoção de componentes que tocam a borda da fatia.
- Experimentos com fechamento e preenchimento de buracos.
- Módulo compartilhado de pré-processamento.
- Inventário dos vinte volumes, com suas referências pulmonares.

Os scripts são experimentos em fatias 2D. Ainda faltam seleção anatômica robusta de componentes, separação dos lados, avaliação e ajuste dos métodos para Radiopaedia, processamento volumétrico, avaliação quantitativa e notebook final. As máscaras candidatas não são resultados clínicos validados.

## Instalação local

O ambiente de desenvolvimento foi verificado com Python 3.9.6. As dependências diretas e suas versões estão em `requirements.txt`.

Depois de obter o código, execute na raiz do projeto, no macOS ou Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

No Windows, pelo PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Se o ambiente já estiver criado e funcionando, basta ativá-lo. Os scripts que exibem figuras precisam de uma sessão local com interface gráfica. Não coloque arquivos do projeto dentro da `.venv`.

## Dados

Baixe os dados conforme [data/README.md](data/README.md). O conjunto usado é o [COVID-19 CT Lung and Infection Segmentation Dataset](https://zenodo.org/records/3757476).

As tomografias, máscaras e o ambiente virtual são excluídos pelo `.gitignore`. O catálogo `data/manifest.csv` e as instruções de download são versionados.

### Janela e escala de trabalho

O padrão segue a [recomendação dos autores do benchmark](https://github.com/JunMa11/COVID-19-CT-Seg-Benchmark#guidelines): Radiopaedia já foi janelado em **[−1250, 250] HU** e normalizado para **[0, 255]**. Aplicamos aos Coronacases:

```text
imagem_255 = (clip(original_HU, -1250, 250) + 1250) × 255 / 1500
```

O centro é **−500 HU** e a largura é **1500 HU**. Esses padrões são definidos em `src/preprocessing.py` e compartilhados pelos cinco scripts. Radiopaedia conserva suas intensidades 0–255, sem aplicar uma segunda janela HU. A saída é `float32`, sem quantização adicional, e os arquivos originais são preservados.

Os parâmetros `--centro` e `--largura` permitem experimentos diferentes, mas seus valores padrão seguem o protocolo acima. A configuração anterior [−1350, 150] foi substituída; resultados anteriores devem ser identificados pela janela utilizada. Com a janela nova, −400 HU corresponde a **144,5**, novo padrão do limiar fixo.

## Metadados dos cabeçalhos NIfTI

Inspeção local em **20/10/2026** dos vinte exames listados em `data/manifest.csv` e das vinte referências em `data/masks/Lung_Mask/`. Os dados abaixo descrevem os arquivos distribuídos, antes da canonicalização e da conversão de intensidades do pré-processamento. Nenhum cabeçalho foi alterado.

### Kernel de reconstrução e informações de aquisição

**O kernel de reconstrução não está informado nos arquivos inspecionados.** Em todas as tomografias, `descrip`, `aux_file` e `intent_name` estão vazios, `intent_code=0` e não há extensões NIfTI. A pasta de dados também não contém arquivos auxiliares JSON ou DICOM com parâmetros de aquisição. Não é possível atribuir um kernel específico, nem classificá-lo como suave ou de alta frequência, com base nesses cabeçalhos.

Também não foram encontrados fabricante/modelo do tomógrafo, tensão do tubo (kVp), corrente/exposição (mA/mAs), espessura nominal de reconstrução nos cabeçalhos; a janela originalmente aplicada aos Radiopaedia é informada pelos autores fora dos arquivos. Para documentar os demais parâmetros será necessário obter metadados de aquisição da fonte original. Não confundir o kernel de reconstrução do tomógrafo com o disco usado como elemento estruturante na morfologia do projeto.

### Resumo por fonte

| Propriedade | Coronacases (10 exames) | Radiopaedia (10 exames) |
|---|---|---|
| Formato | NIfTI-1, `.nii.gz`, 3D | NIfTI-1, `.nii.gz`, 3D |
| Tipo armazenado | `int16` (16 bits, com sinal) | `uint8` (8 bits, sem sinal) |
| Dimensões no plano | 512 × 512 | 630 × 630, exceto um exame 630 × 401 |
| Número de fatias | 200–301 | 39–418 |
| Espaçamento no plano, numérico | Aproximadamente 0,683594–0,810547 | Aproximadamente 0,683594 |
| Espaçamento no terceiro eixo, numérico | 1 ou 1,5 | 1, 2, 4 ou 6 |
| Unidade espacial da tomografia | `unknown` em todos | `unknown` em todos |
| Orientação original pela affine | LAS em todos | LAS em todos |
| Escala NIfTI no arquivo | `scl_slope=1`, `scl_inter=0` | `scl_slope=1`, `scl_inter=0` |
| Transformação espacial | `qform_code=0`, `sform_code=2` | `qform_code=0`, `sform_code=2` |
| Metadados de kernel | Ausentes | Ausentes |

A transformação de intensidade registrada no NIfTI é identidade (`valor × 1 + 0`), também confirmada no proxy do NiBabel. Isso não recupera HU nos Radiopaedia. Os campos `cal_min` e `cal_max` são ambos zero em todos os exames e não descrevem seus extremos reais. A auditoria anterior dos voxels, separada desta leitura dos cabeçalhos, encontrou Coronacases entre −1023 e 9567 no conjunto e Radiopaedia dentro de 0–255.

### Geometria por exame

As dimensões e os espaçamentos seguem a ordem dos três eixos do arquivo. Os espaçamentos estão arredondados a seis casas decimais e **não recebem unidade física nesta tabela**, pois as tomografias declaram `unknown`.

| Exame | Dimensões | Espaçamento numérico (eixo 1 × eixo 2 × eixo 3) |
|---|---|---|
| `coronacases_001` | 512 × 512 × 301 | 0.810547 × 0.810547 × 1 |
| `coronacases_002` | 512 × 512 × 200 | 0.683594 × 0.683594 × 1.5 |
| `coronacases_003` | 512 × 512 × 200 | 0.736328 × 0.736328 × 1.5 |
| `coronacases_004` | 512 × 512 × 270 | 0.683594 × 0.683594 × 1 |
| `coronacases_005` | 512 × 512 × 290 | 0.683594 × 0.683594 × 1 |
| `coronacases_006` | 512 × 512 × 213 | 0.759766 × 0.759766 × 1.5 |
| `coronacases_007` | 512 × 512 × 249 | 0.712891 × 0.712891 × 1 |
| `coronacases_008` | 512 × 512 × 301 | 0.724609 × 0.724609 × 1 |
| `coronacases_009` | 512 × 512 × 256 | 0.683594 × 0.683594 × 1 |
| `coronacases_010` | 512 × 512 × 301 | 0.683594 × 0.683594 × 1 |
| `radiopaedia_4_85506_1` | 630 × 630 × 39 | 0.683594 × 0.683594 × 6 |
| `radiopaedia_7_85703_0` | 630 × 630 × 45 | 0.683594 × 0.683594 × 6 |
| `radiopaedia_10_85902_1` | 630 × 630 × 39 | 0.683594 × 0.683594 × 6 |
| `radiopaedia_10_85902_3` | 630 × 630 × 418 | 0.683594 × 0.683594 × 1 |
| `radiopaedia_14_85914_0` | 630 × 401 × 110 | 0.683594 × 0.683594 × 2 |
| `radiopaedia_27_86410_0` | 630 × 630 × 66 | 0.683594 × 0.683594 × 6 |
| `radiopaedia_29_86490_1` | 630 × 630 × 42 | 0.683594 × 0.683594 × 6 |
| `radiopaedia_29_86491_1` | 630 × 630 × 42 | 0.683594 × 0.683594 × 6 |
| `radiopaedia_36_86526_0` | 630 × 630 × 45 | 0.683594 × 0.683594 × 6 |
| `radiopaedia_40_86625_0` | 630 × 630 × 93 | 0.683594 × 0.683594 × 4 |

### Consequências para o processamento e a avaliação

- **Orientação:** os arquivos originais estão em LAS; o pré-processamento os canonicaliza para RAS. As affines inspecionadas não apresentam obliquidade. Não interpretar índices do arquivo original como se já estivessem em RAS.
- **Anisotropia:** em todos os exames, o espaçamento no terceiro eixo é maior que no plano. Operações 3D devem considerar essa diferença. O espaçamento da grade não comprova a espessura nominal de reconstrução; essa informação de aquisição não está disponível.
- **Referências pulmonares:** as vinte máscaras são `uint16`, declaram unidade espacial `mm` e têm forma e affine numericamente iguais às respectivas tomografias (tolerância absoluta de `1e-4` para a affine). Isso confirma compatibilidade da grade numérica, não a correção anatômica das anotações nem resolve automaticamente a unidade ausente na imagem.
- **Medidas físicas:** a avaliação pode usar a geometria em milímetros da referência pulmonar, após validar a correspondência de grade com a previsão. Essa política está detalhada abaixo. O script de morfologia usa o raio em milímetros com a geometria da referência, sem reescrever os cabeçalhos das tomografias.
- **Comparação entre fontes:** a escala de trabalho 0–255 não padroniza kernel, protocolo de aquisição ou resolução física. Como o kernel não está disponível, não é possível estratificar os resultados por kernel com os metadados locais atuais.

### Verificação das unidades pelas máscaras

Conferência adicional em **20/10/2026**: comparadas as vinte tomografias com as três máscaras de cada caso (`Lung_Mask`, `Infection_Mask` e `Lung_and_Infection_Mask`), totalizando **60 pares**.

| Verificação | Resultado |
|---|---|
| Unidade espacial das 20 tomografias | `unknown` |
| Unidade espacial das 60 máscaras | `mm` |
| Forma da imagem e da máscara | Igual em todos os pares |
| Maior diferença absoluta entre affines | 0 |
| Maior diferença absoluta entre espaçamentos | 0 |
| Espaçamento do cabeçalho versus tamanho de voxel derivado da affine | Consistente em todos, com tolerância absoluta de `1e-6` |

**Conclusão para a avaliação:** há suporte consistente para usar a grade das máscaras, declarada em milímetros, como referência física das previsões produzidas na mesma grade das tomografias. A unidade é obtida da referência correspondente, não inferida apenas pela magnitude dos números. A concordância das três máscaras reforça a consistência do conjunto, mas elas podem compartilhar a mesma origem de metadados: não são três calibrações independentes e não comprovam, sozinhas, a exatidão física da aquisição original.

Na implementação da avaliação:

1. Carregar/canonicalizar previsão e referência de forma consistente e verificar forma, affine e ausência de reamostragem não registrada. Se a previsão for apenas um array, preservar os metadados da imagem da qual foi gerada.
2. Usar o espaçamento/affine da máscara pulmonar correspondente, cuja unidade é `mm`, para as distâncias físicas. Registrar no resultado `unidade_avaliacao=mm` e a origem dessa unidade: referência pulmonar alinhada.
3. Calcular Hausdorff nessa geometria e reportar a distância em milímetros **segundo os metadados da referência do conjunto**. Não usar distância em índices de voxel como se fosse distância em milímetros. Dice e Jaccard na mesma grade não dependem de unidade física.
4. Caso apareça um par com geometria incompatível ou referência sem unidade conhecida, interromper a avaliação física desse par e investigar. Não aplicar esta conclusão indiscriminadamente a novos dados.

Os cabeçalhos originais permanecem intactos e continuam sendo reportados como `unknown` nas tomografias. Não é necessário regravá-los para avaliar na geometria da referência. Esta conferência valida a consistência dos metadados; não substitui a inspeção anatômica do alinhamento.

Fontes: inspeção dos arquivos locais; [especificação dos campos NIfTI](https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields.html), que identifica `pixdim` como espaçamento e `xyzt_units` como suas unidades; [registro oficial do conjunto](https://zenodo.org/records/3757476), que disponibiliza as imagens e três conjuntos de máscaras, mas não fornece confirmação adicional das unidades na descrição pública consultada.

## Executar os experimentos

Execute os comandos abaixo a partir da raiz do projeto e com o ambiente ativado. Feche a janela para encerrar cada visualização. Todos os scripts usam `imagem_255` do pré-processamento e identificam a fonte automaticamente pelos prefixos `coronacases_` e `radiopaedia_` do nome do arquivo, sem distinguir maiúsculas/minúsculas. Para nomes diferentes, informe `--fonte Coronacases` ou `--fonte Radiopaedia`; uma fonte explícita tem prioridade. A fonte não é inferida pelas intensidades, pois isso pode aplicar a transformação errada.

Todos os cinco scripts escolhem automaticamente a fatia central (`número de fatias // 2`, com índices iniciando em zero). Não é necessário informar `--fatia`. Esse argumento permanece opcional para inspecionar outro corte. Em volumes com número par de fatias, é escolhida a segunda das duas centrais.

### Visualizar uma fatia

```bash
python visualizar_fatia.py data/raw/coronacases_001.nii.gz
```

Sem `--fatia`, o programa escolhe a fatia central. Para visualizar um Radiopaedia na escala disponível:

```bash
python visualizar_fatia.py data/raw/radiopaedia_10_85902_1.nii.gz
```

### Limiar fixo

```bash
python segmentacao_limiar_fixo.py data/raw/coronacases_001.nii.gz --limiar 144.5
```

O limiar agora é informado em 0–255. O padrão 144,5 corresponde a −400 HU na janela Coronacases padrão (centro −500, largura 1500). Para comparar, 127,5 e 161,5 correspondem a −500 e −300 HU nessa janela. Esses valores não são parâmetros validados para Radiopaedia. Limiares negativos, acima de 255 ou não finitos são rejeitados.

### Limiar automático de Otsu

```bash
python segmentacao_otsu.py data/raw/coronacases_001.nii.gz
```

Mostra somente a fatia e a máscara de Otsu, sem remoção de componentes ou morfologia. O limiar é calculado na imagem preparada em 0–255; não recebe `--limiar`. O recorte da janela Coronacases pode alterar o resultado em relação ao cálculo sobre HU originais.

### Componentes conexos

```bash
python segmentacao_componentes.py data/raw/coronacases_001.nii.gz
```

Mostra a fatia e a máscara de Otsu após remover componentes que tocam a borda. Não compara métodos e não recebe mais `--limiar`. Essa etapa ainda pode manter componentes internos indesejados.

### Refinamento morfológico

Mostra apenas a fatia e a máscara final lado a lado, após rastrear/remover a traqueia, remover componentes de borda, aplicar fechamento e preencher buracos.

```bash
python segmentacao_morfologia.py data/raw/coronacases_001.nii.gz --metodo otsu --raio 2
```

A semente agora é **automática por padrão**. O detector examina do topo para baixo os 40% superiores do volume (segundo a orientação RAS do cabeçalho), sem escolher uma semente abaixo da fatia exibida. Busca componentes de ar internos ao corpo, compactos e próximos da linha central, com área entre 30 mm² e `--area-max-traqueia` (400 mm² por padrão). Usa solidez mínima de 0,75 e razão mínima de 0,19 entre autovalores da distribuição espacial dos pixels. O corpo é estimado das intensidades da própria tomografia; os rótulos das máscaras não participam.

A escolha exige um único candidato com sobreposição, área e posição estáveis em **três fatias consecutivas**. Uma região central maior que o limite também impede aceitar arbitrariamente uma região pequena concorrente. Ausência ou ambiguidade interrompem a execução com uma mensagem para revisar o exame e informar a semente. Não há coordenadas específicas por paciente nem classificação anatômica garantida por esses critérios.

`--semente I J K` continua disponível e tem prioridade: substitua `I J K` por índices inteiros de um ponto **dentro do lúmen da traqueia em um corte superior**, observado no próprio exame. A fonte continua automática e a fatia exibida continua central por padrão. Os índices começam em zero e pertencem à imagem canonicalizada em RAS. Na exibição transposta usada pelos scripts, a coordenada horizontal é `I` e a vertical é `J`. `K` é a fatia da semente, não a fatia que será exibida. Escolha `K >= fatia exibida`; para observar outro corte, use `--fatia`. Essa relação pressupõe que a orientação anatômica declarada no cabeçalho esteja correta.

O rastreamento **2,5D** percorre `K, K-1, ...` até o corte de interesse. Usa a máscara e o centroide da fatia anterior para orientar operações exclusivamente no plano atual:

1. Prepara a fatia em 0–255 e calcula Otsu (padrão) ou limiar fixo.
2. Delimita uma ROI circular física em torno do centroide anterior (`--roi-mm 20`). Dentro dela, rotula componentes 8-conexos: selecionar os componentes alcançados pela semente/sobreposição equivale a crescimento de região binário. Na primeira fatia usa a semente; nas seguintes exige sobreposição com a máscara anterior e escolhe um ponto interno.
3. Rejeita componentes que toquem a borda da ROI ou da imagem. Isso impede aceitar como via aérea apenas um recorte artificial do pulmão. Verifica área máxima (400 mm²), aumento de área (até 2 vezes) e deslocamento do centroide (até 5 mm por fatia).
4. Se houver candidato incompatível, tenta reduzir o limiar em passos de 5, até `--ajuste-limiar-max 30` unidades da escala 0–255. Isso pode desfazer pontes de intensidade intermediária; não altera o limiar da segmentação pulmonar. Zero desliga essa adaptação. Sem continuidade, não pula fatias nem reinicia em outra região.
5. Aceita uma divisão em **dois ramos** quando ambos sobrepõem a região anterior: cada área deve ser pelo menos `max(3 mm², 15% da área anterior)` e a soma deve ficar entre a área anterior dividida e multiplicada por `--crescimento-max`. Os controles individuais também se aplicam. É uma hipótese geométrica, sem confirmação anatômica da carina nem confirmação temporal adicional da divisão.
6. Acompanha os dois ramos independentemente, com seus próprios centroides e critérios de parada. Permite somente uma geração de bifurcação; novas divisões encerram o ramo. Se dois rastros ocuparem pixels em comum, interrompe ambos por ambiguidade. Um ramo pode continuar quando o outro para.
7. Na fatia solicitada, reúne os ramos aceitos e subtrai essa região da máscara pulmonar. Não remove automaticamente os menores componentes internos, pois tamanho isolado não identifica a traqueia e pode excluir pulmão.

Os parâmetros são heurísticos e ainda não foram calibrados por fonte ou espaçamento entre fatias. `--deslocamento-max` é medido em mm por fatia, não por mm de avanço no eixo superior–inferior. O algoritmo não garante exclusão completa dos brônquios nem identificação correta se a semente estiver no componente errado.

Se todos os ramos pararem antes do alvo, nenhuma remoção de vias aéreas é feita nesse corte. Se apenas um chegar, a remoção pode ser parcial. Terminal e JSON registram as paradas. `--registro conversa_local/relatorios/traqueia_25d.json` guarda parâmetros, identificadores/pais dos ramos, centroides, sementes internas e limiares efetivamente usados. O script não salva um volume segmentado.

**Custo computacional:** não faz rotulação, morfologia ou crescimento 3D. Mantém máscaras 2D dos ramos ativos e histórico numérico. Entretanto, o leitor existente usa `nib.as_closest_canonical`, que pode materializar o volume na reorientação; portanto a leitura ainda não tem garantia de memória limitada a poucas fatias.

**Verificação anterior à implementação 2,5D (22/10/2026):** os vinte exames foram executados pela CLI com Otsu e parâmetros padrão, sem semente manual. Houve 17 detecções (10 Coronacases e 7 Radiopaedia), inspecionadas visualmente como pontos plausíveis na via aérea central superior, e 3 recusas:

| Exame | Motivo da recusa automática |
|---|---|
| `radiopaedia_7_85703_0` | Nenhum candidato persistente na região pesquisada. Os cortes de índices altos exibem anatomia basal; há suspeita de inconsistência entre ordem anatômica e cabeçalho. Revisar orientação antes de usar uma semente manual para rastreamento. |
| `radiopaedia_27_86410_0` | Região pequena concorrente com uma região central maior que o limite de área. |
| `radiopaedia_36_86526_0` | Região central maior que o limite torna a seleção automática ambígua pelo critério conservador. |

Essas 17 detecções não significam 17 remoções completas: em todas, o rastreamento parou antes da fatia central por seus critérios de continuidade, divisão, área ou deslocamento. Assim, não houve exclusão traqueal na fatia central nesses testes. A inspeção visual das sementes não substitui validação quantitativa e não confirma identificação da carina. Os 30 testes sintéticos/do pré-processamento passaram, incluindo prioridade manual, rejeição de ambiguidade e persistência em três cortes. A primeira versão da busca selecionou uma região inadequada em um Radiopaedia; a inspeção motivou a regra de recusa diante de uma alternativa central maior. Estes exames foram usados no desenvolvimento e não constituem um conjunto independente de avaliação.

**Verificação 2,5D em 22/10/2026:** a CLI foi executada nos vinte exames, com Otsu e padrões, sem semente manual. Manteve 17 sementes aceitas e as três recusas acima. Nenhum rastreamento chegou à fatia central. Em comparação ao algoritmo anterior, avançou mais em 7 exames, parou no mesmo corte em 9 e parou antes em 1. Houve 12 eventos aceitos com redução de limiar e 0 eventos de bifurcação aceitos. Esses números medem comportamento, não acurácia anatômica. Os 34 testes locais passaram, incluindo bifurcação, parada independente, fusão ambígua, vazamento na borda da ROI e ponte removida por ajuste de limiar. A solução ainda não resolve automaticamente todos os exames. Relatório detalhado local: `conversa_local/relatorios/rastreamento_25d_2026-10-22/RELATORIO.md` (ignorado pelo Git).

**Sequência na fatia exibida:** preparação 0–255 → limiarização → remoção dos componentes de borda → exclusão da traqueia rastreada → fechamento com raio em mm → preenchimento de buracos → exclusão novamente da região traqueal, para impedir sua reintrodução → exibição da fatia e da máscara final. Os rótulos da referência pulmonar não são usados para rastrear ou remover a traqueia; somente sua geometria é consultada.

`--raio` aceita um número decimal em **milímetros**, com padrão **2 mm** (não mais 2 pixels). Zero desliga somente o fechamento; o preenchimento de buracos permanece ativo.

O espaçamento é lido da máscara pulmonar de mesmo nome, procurada em `../masks/Lung_Mask/` a partir da pasta da tomografia. Para outra organização, use `--mascara caminho/mascara.nii.gz`. A referência precisa declarar `mm` e coincidir em forma e affine com a imagem após canonicalização. O script lê apenas sua geometria, sem usar rótulos para construir a previsão.

Para cada eixo, o raio equivalente em pixels é `raio_mm / spacing`. O elemento inclui os deslocamentos `(i, j)` que satisfazem `(i*sx)² + (j*sy)² <= raio_mm²`, preservando um disco físico mesmo com pixels retangulares. Um raio menor que ambos os espaçamentos produz apenas o pixel central e não altera a máscara no fechamento. O espaçamento entre fatias não entra nessa operação 2D. Referências ausentes, desalinhadas, sem unidade `mm` ou com geometria incompatível são rejeitadas.

 Para o método fixo, use `--metodo fixo --limiar 144.5`. Uma área maior após refinamento não implica necessariamente maior precisão.

Para executar qualquer um desses scripts em Radiopaedia, basta usar o caminho com o prefixo `radiopaedia_`, por exemplo:

```bash
python segmentacao_morfologia.py data/raw/radiopaedia_10_85902_1.nii.gz --metodo otsu --raio 2
```

`--centro` e `--largura` configuram a janela em HU somente para Coronacases. Radiopaedia mantém suas intensidades 0–255. A exibição padrão usa 0–255; no visualizador, `--vmin` e `--vmax` ajustam apenas o contraste dentro dessa faixa. Alterar a janela Coronacases também altera o significado de um limiar fixo em 0–255.

Cada exercício tem uma responsabilidade: visualização, limiar fixo (`segmentacao_limiar_fixo.py`), Otsu (`segmentacao_otsu.py`), remoção de borda e refinamento morfológico.

Os scripts dependem somente de `src.preprocessing` e das bibliotecas externas: não importam uns aos outros. A remoção de borda é uma operação curta implementada em cada exercício que precisa dela.

Os scripts exibem figuras; ainda não salvam automaticamente uma tabela de métricas ou máscaras finais.

## Organização

```text
.
├── visualizar_fatia.py
├── segmentacao_limiar_fixo.py
├── segmentacao_otsu.py
├── segmentacao_componentes.py
├── segmentacao_morfologia.py
├── src/
│   ├── __init__.py
│   └── preprocessing.py
├── data/
│   ├── README.md
│   └── manifest.csv
├── requirements.txt
├── README.md
├── .gitattributes
└── .gitignore
```

`src/preprocessing.py` concentra leitura, seleção de fatias, metadados, alinhamento e leitura das referências. Os scripts usam `preparar_fatia` e seu campo `imagem_255`: Coronacases recebe janela e conversão para 0–255; Radiopaedia preserva a escala disponível. `original` e `normalizada` (0–1) permanecem disponíveis. A faixa numérica comum não implica contraste equivalente entre fontes. Unidades desconhecidas não são convertidas para milímetros automaticamente.

## Verificações realizadas

O ambiente existente passou em `python -m pip check`. Na versão anterior, os scripts foram executados no exame `coronacases_001`, fatia 150, com renderização sem interface gráfica; aquela refatoração preservou seus resultados numéricos. A mudança atual de escala inclui recorte da janela antes de Otsu e pode alterar seus resultados. O pré-processamento foi exercitado também com dados sintéticos para escala NIfTI, índices, rótulos e alinhamento. Esses testes não representam avaliação do método nos vinte volumes nem confirmação de instalação em um ambiente novo.

