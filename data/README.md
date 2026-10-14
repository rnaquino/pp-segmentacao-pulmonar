# Dados do projeto

Os exames e as máscaras não são incluídos no repositório Git. Obtenha o conjunto [COVID-19 CT Lung and Infection Segmentation Dataset no Zenodo](https://zenodo.org/records/3757476) e respeite a licença indicada no conjunto.

Extraia as vinte tomografias de `COVID-19-CT-Seg_20cases.zip` em `data/raw/`, mantendo os arquivos `.nii.gz` diretamente nessa pasta. Extraia os arquivos de `Lung_Mask.zip` em `data/masks/Lung_Mask/`. Não mantenha uma pasta intermediária adicional entre esses diretórios e os NIfTI.

As máscaras de infecção e as máscaras combinadas são opcionais nas etapas atuais. Se baixadas, coloque-as em `data/masks/Infection_Mask/` e `data/masks/Lung_and_Infection_Mask/`.

```text
data/
├── README.md
├── manifest.csv
├── raw/
│   ├── coronacases_001.nii.gz
│   └── ...
└── masks/
    ├── Lung_Mask/
    │   ├── coronacases_001.nii.gz
    │   └── ...
    ├── Infection_Mask/           # Opcional
    └── Lung_and_Infection_Mask/  # Opcional
```

## Inventário

`manifest.csv` contém uma linha por volume, com quatro colunas:

| Coluna | Significado |
|---|---|
| `patient_id` | Identificador do volume |
| `source` | Coronacases ou Radiopaedia |
| `image_path` | Caminho da tomografia |
| `lung_mask_path` | Caminho da referência pulmonar |

Os caminhos são relativos à raiz do projeto, não à pasta deste README. Há dez volumes de cada fonte. Os scripts atuais recebem um exame pela linha de comando; o processamento automático do inventário será implementado posteriormente.

Coronacases preserva unidades Hounsfield. Radiopaedia possui intensidades já transformadas; os limiares em HU não se aplicam diretamente a esses exames. A unidade espacial pode estar ausente nos cabeçalhos: confirme-a antes de calcular medidas físicas.
