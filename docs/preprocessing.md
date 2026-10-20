# Pré-processamento compartilhado

O módulo `src/preprocessing.py` atende aos quatro requisitos do roteiro: extração 2D, janela pulmonar, normalização e organização padronizada dos dados.

| Requisito | Implementação |
|---|---|
| Extração de fatias 2D | `extrair_fatia`, após leitura/canonicalização |
| Janela de pulmão | `aplicar_janela_pulmao`: recorte por centro e largura em HU |
| Normalização | `normalizar`: recorte e transformação para 0–1 com faixa explícita |
| Estrutura padronizada | `preparar_fatia` e `preparar_paciente`: dicionário com campos fixos |

## Preparar uma fatia Coronacases

Na raiz do projeto, em um script ou sessão Python:

```python
from src.preprocessing import preparar_fatia

dados = preparar_fatia(
    "data/raw/coronacases_001.nii.gz",
    fonte="Coronacases",
    k=150,
    caminho_mascara="data/masks/Lung_Mask/coronacases_001.nii.gz",
    centro=-500,
    largura=1500,
)

print(dados["original"].shape)
print(dados["normalizada"].min(), dados["normalizada"].max())

# Para limiarização em HU, use sempre a original.
mascara_candidata = dados["original"] < -400

# A normalizada pode ser exibida com vmin=0 e vmax=1.
# A máscara pulmonar de referência serve para comparação/avaliação.
referencia = dados["mascara_pulmonar"]
```

Os limites da janela são `centro − largura/2` e `centro + largura/2`. Com os padrões, a faixa é −1250 a 250 HU. Valores abaixo/acima são recortados nesses limites; a normalização mapeia o limite inferior para 0 e o superior para 1. Os padrões seguem a recomendação dos autores do conjunto; os parâmetros continuam configuráveis para experimentação.

Essa é a normalização por faixa definida, não por mínimo e máximo de cada fatia. A mesma intensidade tem a mesma representação normalizada quando os parâmetros são iguais.

## Campos do registro padronizado

| Campo | Conteúdo |
|---|---|
| `patient_id`, `source` | Identificador e fonte |
| `image_path`, `lung_mask_path` | Caminhos de origem; máscara pode estar ausente |
| `indice_fatia` | Corte usado no terceiro eixo canonicalizado |
| `original` | Array 2D float32 na escala original |
| `janelada` | Array 2D recortado em HU; `None` nos Radiopaedia |
| `normalizada` | Array 2D float32 entre 0 e 1 |
| `imagem_255` | Array 2D float32 entre 0 e 255, usado pelos cinco scripts |
| `rotulos_referencia` | Array 2D int32 de classes; `None` sem referência |
| `mascara_pulmonar` | Referência booleana dos rótulos 1 e 2 por padrão; `None` sem referência |
| `metadados` | Forma do volume, espaçamento, orientação, unidades, affine e metadados da referência |
| `parametros` | Escala de entrada, janela, faixa de normalização e rótulos escolhidos |

Todos os arrays presentes têm a mesma forma 2D, sem redimensionamento, transposição ou reamostragem. A affine e o espaçamento permanecem associados ao volume original canonicalizado; use também `indice_fatia` para localizar o corte fisicamente.

Uma referência opcional não participa do cálculo da janela ou da normalização e não gera uma previsão. Pode haver uma fatia sem pulmão, resultando em referência booleana vazia.

## Reutilizar o catálogo

```python
import csv
from pathlib import Path
from src.preprocessing import preparar_paciente

raiz = Path.cwd()  # Execute a partir da raiz do projeto.
with (raiz / "data/manifest.csv").open(encoding="utf-8", newline="") as arquivo:
    registro = next(csv.DictReader(arquivo))

dados = preparar_paciente(registro, raiz, k=150)
```

Para processar os demais pacientes, percorra as linhas e trate um registro por vez. Não é necessário empilhar fatias de formas diferentes nem carregar todos os volumes juntos.

## Radiopaedia

```python
dados = preparar_fatia(
    "data/raw/radiopaedia_10_85902_1.nii.gz",
    fonte="Radiopaedia",
    limites_radiopaedia=(0, 255),
)
```

Essa ramificação não aplica janela em HU nem tenta recuperar HU. Preserva a fatia original e normaliza a escala explicitamente informada; `janelada` e a configuração de janela são `None`. Os limites 0–255 refletem a escala observada nas fatias inspecionadas deste conjunto e podem ser alterados conforme os dados. Valores fora da faixa geram aviso de recorte.

Os cinco scripts agora usam `imagem_255` também para Radiopaedia. Otsu e limiar fixo estão disponíveis; a comparação com limiar local/adaptativo continua pendente. A igualdade de faixa numérica não garante contraste equivalente entre as fontes.

## Compatibilidade e unidades

`aplicar_janela` mantém o comportamento anterior: janela mais normalização em uma chamada. Os cinco scripts usam `preparar_fatia` e o campo `imagem_255`. A fonte é detectada pelo prefixo do arquivo, com `--fonte` opcional para informá-la explicitamente, e `--limiar` recebe valores de 0 a 255. O padrão 144,5 corresponde a −400 HU somente na janela Coronacases padrão. Otsu passa a operar sobre a imagem preparada, podendo mudar com o recorte da janela. A saída `normalizada` e a API anterior continuam em 0–1. Nenhum script de segmentação importa outro.

Os cabeçalhos originais não são alterados. Unidade espacial desconhecida permanece `unknown`; se houver referência alinhada, a verificação emite um aviso. Não usar os números de espaçamento como milímetros sem confirmação.

## Verificar

```bash
python -m unittest discover -s tests -v
```

Os testes usam arquivos NIfTI sintéticos para conferir limites, não alteração da entrada, forma 2D, orientação, alinhamento das referências, classes, unidades e integração com o catálogo.

## Identificação automática da fonte

`preparar_fatia(caminho_imagem)` identifica a fonte pelos prefixos `coronacases_` e `radiopaedia_` do nome, sem distinguir maiúsculas/minúsculas, e seleciona a fatia central. Para nomes não reconhecidos, exige `fonte="Coronacases"` ou `fonte="Radiopaedia"`; um valor explícito tem prioridade. Não usa nome de diretório, tipo numérico ou intensidades para adivinhar a fonte. `preparar_intensidades` continua exigindo a fonte porque recebe apenas um array. `preparar_paciente` continua usando a fonte declarada no catálogo.
