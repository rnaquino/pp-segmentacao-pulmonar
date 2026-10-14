# Segmentação pulmonar por métodos clássicos

Projeto acadêmico de segmentação pulmonar em tomografia computadorizada, usando limiarização, componentes conexos e operações morfológicas. O desenvolvimento é feito em scripts Python locais; o notebook no Google Colab será preparado ao final.

## Estado atual

- Leitura de NIfTI e padronização da orientação com NiBabel.
- Visualização de uma fatia.
- Comparação de limiar fixo e Otsu em Coronacases.
- Remoção de componentes que tocam a borda da fatia.
- Experimentos com fechamento e preenchimento de buracos.
- Módulo compartilhado de pré-processamento.
- Inventário dos vinte volumes, com suas referências pulmonares.

Os scripts são experimentos em fatias 2D. Ainda faltam seleção anatômica robusta de componentes, separação dos lados, adaptação da segmentação aos Radiopaedia, processamento volumétrico, avaliação quantitativa e notebook final. As máscaras candidatas não são resultados clínicos validados.

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

## Executar os experimentos

Execute os comandos abaixo a partir da raiz do projeto e com o ambiente ativado. Feche a janela para encerrar cada visualização.

### Visualizar uma fatia

```bash
python visualizar_fatia.py data/raw/coronacases_001.nii.gz --fatia 150
```

Sem `--fatia`, o programa escolhe a fatia central. Para visualizar um Radiopaedia na escala disponível:

```bash
python visualizar_fatia.py data/raw/radiopaedia_10_85902_1.nii.gz --vmin 0 --vmax 255
```

### Limiar fixo

```bash
python segmentacao_inicial.py data/raw/coronacases_001.nii.gz --fatia 150 --limiar -400
```

Experimente −500, −400 e −300 na mesma fatia. Esses limiares em HU são destinados aos Coronacases.

### Otsu e componentes conexos

```bash
python segmentacao_componentes.py data/raw/coronacases_001.nii.gz --fatia 150 --limiar -400
```

O programa mostra a comparação entre limiar fixo e Otsu e remove regiões que tocam a borda. As cores dos componentes não representam classes anatômicas.

### Refinamento morfológico

```bash
python segmentacao_morfologia.py data/raw/coronacases_001.nii.gz --fatia 150 --metodo otsu --raio 2
```

`--raio` é um inteiro em pixels. Zero desliga somente o fechamento; o preenchimento de buracos permanece ativo. Para o método fixo, use `--metodo fixo --limiar -400`. Uma área maior após refinamento não implica necessariamente maior precisão.

Os scripts exibem figuras; ainda não salvam automaticamente uma tabela de métricas ou máscaras finais.

## Organização

```text
.
├── visualizar_fatia.py
├── segmentacao_inicial.py
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

`src/preprocessing.py` concentra leitura, seleção de fatias, metadados, alinhamento e leitura das referências. Janelamento é uma operação explícita; não é aplicado automaticamente à imagem usada para limiarização. Unidades desconhecidas não são convertidas para milímetros automaticamente.

## Verificações realizadas

O ambiente existente passou em `python -m pip check`. Os scripts foram executados no exame `coronacases_001`, fatia 150, com renderização sem interface gráfica; a refatoração preservou seus resultados numéricos. O pré-processamento foi exercitado também com dados sintéticos para escala NIfTI, índices, rótulos e alinhamento. Esses testes não representam avaliação do método nos vinte volumes nem confirmação de instalação em um ambiente novo.

## Primeiro envio ao GitHub

Crie no GitHub um repositório vazio, sem README, licença ou `.gitignore` gerados automaticamente, e copie sua URL. O projeto local está preparado para um primeiro commit na branch `main`.

Confira a seleção de arquivos antes de enviar:

```bash
git status
git diff --cached --stat
git commit -m "Organiza projeto de segmentação pulmonar"
```

Se o Git pedir identificação, configure seu nome e e-mail para este repositório:

```bash
git config user.name "SEU NOME"
git config user.email "SEU EMAIL"
```

Depois conecte o endereço real do repositório e envie. Substitua `URL_DO_REPOSITORIO` pela URL copiada:

```bash
git remote add origin URL_DO_REPOSITORIO
git push -u origin main
```

O envio exige autenticação no GitHub. Após novas alterações, selecione os arquivos pertinentes com `git add`, faça outro commit e use `git push`.
