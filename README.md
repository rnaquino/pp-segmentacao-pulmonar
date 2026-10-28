# Segmentação pulmonar em CT — pipeline 2D e experimentos

> **Estado da entrega:** pipeline operacional de preparação e separação 2D; a reincorporação de regiões excluídas e a validação volumétrica completa pertencem à próxima etapa. **Não se trata de segmentação clinicamente validada.**

## Comece aqui

A interface para a próxima integrante é **`separacao_final.py`**. Ela aplica `pipeline_base.preparar_mascara_base` e depois `separacao_programacao_dinamica.separar_pulmoes`. Scripts experimentais permanecem na raiz por possuírem dependências cruzadas; a documentação fica em `docs/` e os testes em `tests/`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Exemplo de geração de máscaras (ajuste os caminhos para os arquivos disponíveis localmente):

```bash
python separacao_final.py data/raw/coronacases_001.nii.gz \
  --fatia 142 \
  --fonte Coronacases \
  --referencia data/masks/Lung_Mask/coronacases_001.nii.gz \
  --raio 3 \
  --saida resultados/coronacases_001_fatia_142.npz
```

O arquivo `.npz` contém `mascara_base`, `pulmao_esquerdo`, `pulmao_direito`, `mascara_combinada`, `status`, `metodo` e `raio_mm`. Os nomes de lados são **heurísticos**; confirme sua correspondência anatômica com as referências antes de relatar métricas por lado. A referência informada ao comando é usada para obter o espaçamento e **não** para determinar os pixels segmentados.

Para comparar os métodos experimentais, use:

```bash
python -m experimentos.comparar_metodos_separacao --help
python -m diagnosticos.diagnostico_exame_completo --help
python -m experimentos.inspecionar_metodos_separacao
```

Consulte [comandos e saídas](docs/execucao_e_entrega.md), [decisões metodológicas](docs/decisoes_metodologicas.md), [inventário e dependências](docs/inventario_arquivos.md) e [pré-processamento](docs/preprocessing.md).

### Parâmetros fixados nesta etapa

- Entrada: fatia axial 2D normalizada em 0–255 por `src/preprocessing.py`, com orientação RAS.
- Máscara candidata: Otsu por fatia, remoção de componentes de borda e retenção dos dois componentes principais e fragmentos próximos.
- Refinamento: fechamento com disco de **raio físico 3 mm**, seguido de preenchimento de buracos. Não confundir raio com diâmetro.
- Separação: programação dinâmica como escolha operacional. Erosão condicional permanece como dependência para localizar sementes, e o método de maior intensidade permanece para comparação.
- **Sem declaração de superioridade estatística** ou validação dos 20 volumes na versão entregue.

### Organização

```text
.
├── separacao_final.py            # interface da próxima etapa
├── pipeline_base.py              # Otsu, isolamento, morfologia
├── separacao_*.py                # método adotado e métodos comparativos
├── segmentacao_*.py              # componentes, morfologia e históricos
├── comparar_*.py, avaliar_*.py   # avaliação experimental
├── diagnostico_*.py, inspecionar_*.py, ...
├── src/preprocessing.py          # leitura e padronização NIfTI
├── docs/                         # inventário, decisões e instruções
├── tests/                        # testes sintéticos da interface
├── data/                         # dados locais (não empacotados)
└── resultados/                   # saídas locais (fora do Git)
```

### Limitações e compatibilidade

Esta entrega preserva os scripts e nomes de importação existentes. Mover scripts para pacotes sem antes refatorar os imports poderia quebrar a reprodução dos experimentos. Alguns comandos históricos descritos adiante representam fases anteriores da pesquisa e **não** substituem o pipeline operacional documentado acima.

---

## Registro histórico de desenvolvimento

O conteúdo a seguir foi mantido como registro técnico das etapas anteriores. Referências a trabalhos ainda pendentes e à antiga árvore de arquivos dizem respeito àquela fase; o estado de entrega vigente está no início deste documento.

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

Mostra apenas a fatia e a máscara final lado a lado, após remover componentes de borda, aplicar fechamento e preencher buracos.

```bash
python segmentacao_morfologia.py data/raw/coronacases_001.nii.gz --metodo otsu --raio 2
```

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

## Avaliação quantitativa — Dice e IoU

O projeto inclui o script `avaliar_segmentacao.py`,
responsável pela comparação entre máscaras produzidas
pelos algoritmos de segmentação e máscaras de referência
(ground truth).

As métricas implementadas são:

- Dice Similarity Coefficient (DSC): mede a sobreposição
  entre as regiões segmentadas.
- Intersection over Union (IoU/Jaccard): mede a razão
  entre a interseção e a união das regiões segmentadas.

Ambas variam de 0 a 1, sendo 1 a concordância perfeita.

O script aceita máscaras NIfTI (.nii e .nii.gz) e
NumPy (.npy), permitindo avaliar volumes inteiros
ou fatias específicas.

Exemplo de execução:

    python avaliar_segmentacao.py \
      caminho/mascara_predita.nii.gz \
      caminho/mascara_referencia.nii.gz

Para avaliar uma fatia específica, acrescente
`--fatia N`, sendo N o índice da fatia.

As máscaras devem possuir as mesmas dimensões,
orientação e alinhamento espacial. Para arquivos
NIfTI, a geometria é verificada automaticamente.

As métricas serão utilizadas para comparar
quantitativamente as segmentações obtidas pelos
diferentes métodos implementados no projeto.

### Comparação visual dos métodos de separação

Para comparar os três métodos nas fatias 129, 130, 133 e 134 do Coronacases 002:

    python -m experimentos.inspecionar_metodos_separacao data/raw/coronacases_002.nii.gz --referencia data/masks/Lung_Mask/coronacases_002.nii.gz --fatias 129 130 133 134 --raio 3 --saida resultados/inspecao_separacao_002.png

O parâmetro `--mostrar` exibe a figura na tela. Sem ele, a figura é apenas salva.

Para consultar os parâmetros disponíveis:

    python -m experimentos.inspecionar_metodos_separacao --help

