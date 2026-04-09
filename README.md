# Laboratório 07 — Especialização de LLMs com LoRA e QLoRA

Pipeline completo de fine-tuning de um modelo de linguagem fundacional utilizando PEFT/LoRA e quantização QLoRA. O domínio escolhido para especialização é **futebol**, com um dataset sintético gerado via OpenAI API.

**Autor:** José Lucas Silva Souza

---

## Índice

1. [Visão Geral](#visão-geral)
2. [Requisitos de Hardware](#requisitos-de-hardware)
3. [Configuração do Ambiente](#configuração-do-ambiente)
4. [Alternativa: Rodar no Google Colab](#alternativa-rodar-no-google-colab)
5. [Passo 1 — Geração do Dataset Sintético](#passo-1--geração-do-dataset-sintético)
6. [Passo 2, 3 e 4 — Treinamento](#passo-2-3-e-4--treinamento)
7. [Estrutura do Projeto](#estrutura-do-projeto)
8. [Hiperparâmetros Obrigatórios](#hiperparâmetros-obrigatórios)
9. [Política de Uso de IA](#política-de-uso-de-ia)

---

## Visão Geral

Este projeto implementa um pipeline de fine-tuning eficiente para o modelo Llama 2 7B usando:

- **QLoRA**: quantização 4-bit (NF4) via `bitsandbytes` para viabilizar treinamento em hardware com memória limitada.
- **LoRA** (Low-Rank Adaptation): adaptadores de baixo posto via `peft`, evitando atualizar todos os parâmetros do modelo base.
- **SFTTrainer** da biblioteca `trl` para orquestrar o treinamento supervisionado com instruções.
- **Dataset sintético** gerado com a API da OpenAI, contendo 60 pares de pergunta e resposta sobre futebol.

---

## Requisitos de Hardware

| Recurso        | Mínimo       | Recomendado  |
| -------------- | ------------ | ------------ |
| GPU VRAM       | 12 GB        | 24 GB        |
| RAM            | 16 GB        | 32 GB        |
| Armazenamento  | 20 GB livres | 40 GB livres |

> A quantização 4-bit reduz significativamente o consumo de VRAM, mas o treinamento em CPU é extremamente lento e não recomendado para este projeto.

---

## Configuração do Ambiente

### 1. Clone o repositório

```bash
git clone https://github.com/lucasrbsouza/llm-lora-qlora-synthetic-finetuning-lab.git
cd llm-lora-qlora-synthetic-finetuning-lab
```

### 2. Crie e ative um ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Instale o PyTorch com suporte a CUDA

Antes de instalar o `requirements.txt`, instale o PyTorch compatível com sua versão de CUDA. Consulte [pytorch.org/get-started](https://pytorch.org/get-started/locally/) para o comando exato. Exemplo para CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Instale as dependências do projeto

```bash
pip install -r requirements.txt
```

### 5. Configure as variáveis de ambiente

```bash
cp .env.example .env
```

Abra o arquivo `.env` e preencha:

```env
OPENAI_API_KEY=sk-...          # Necessário apenas para gerar o dataset
HF_TOKEN=hf_...                # Token do Hugging Face (necessário para Llama 2)
BASE_MODEL=NousResearch/Llama-2-7b-hf
```

#### Sobre o token do Hugging Face

O modelo padrão (`NousResearch/Llama-2-7b-hf`) é um re-upload comunitário que não requer aprovação de licença. Se preferir usar o modelo oficial (`meta-llama/Llama-2-7b-hf`):

1. Acesse [huggingface.co/meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) e aceite os termos de uso.
2. Gere um token em [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Defina `BASE_MODEL=meta-llama/Llama-2-7b-hf` no `.env`.

---

## Alternativa: Rodar no Google Colab

Se preferível, o projeto pode ser rodado diretamente no Google Colab, sem necessidade de configurar ambiente local ou possuir uma GPU própria.

O notebook [`colab_training.ipynb`](colab_training.ipynb) contém todo o pipeline adaptado para o Colab.

### Passo a passo

1. Acesse [colab.research.google.com](https://colab.research.google.com)
2. Clique em `File > Upload notebook` e selecione o arquivo `colab_training.ipynb`
3. Ative a GPU: `Runtime > Change runtime type > GPU (T4)`
4. Na **célula 2**, preencha suas credenciais:

```python
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["HF_TOKEN"]       = "hf_..."
```

1. Execute as células em ordem com `Shift + Enter` ou clique em `Runtime > Run all`

### Sobre o dataset no Colab

- **Opção A:** rode a célula de geração para criar um novo dataset via OpenAI API
- **Opção B:** rode a célula alternativa que clona o repositório e usa o dataset já incluído em `data/` — sem gastar créditos da OpenAI

Ao final do treinamento, o notebook oferece uma célula para baixar o adaptador LoRA treinado como arquivo `.zip`.

---

## Passo 1 — Geração do Dataset Sintético

O script `generate_dataset.py` utiliza a API da OpenAI (GPT-3.5-turbo) para gerar 60 pares de pergunta e resposta sobre futebol, cobrindo seis categorias:

- Regras e regulamentos
- História e jogadores lendários
- Táticas e formações
- Competições e torneios
- Recordes e estatísticas
- Cultura e torcida

Os dados são embaralhados com semente fixa (`seed=42`) e divididos em 90% treino e 10% teste.

```bash
python generate_dataset.py
```

Saída esperada:

```text
Generating 60 football Q&A pairs...
Train: 54 pairs -> data/train.jsonl
Test:  6 pairs  -> data/test.jsonl
```

> O dataset já está incluído no repositório em `data/`. Este passo só é necessário se quiser regenerar os dados.

---

## Passo 2, 3 e 4 — Treinamento

O script `train.py` executa todo o pipeline de fine-tuning em sequência:

**Passo 2 — Quantização:** carrega o modelo base em 4-bit usando `BitsAndBytesConfig` com quantização NF4 e `compute_dtype=float16`.

**Passo 3 — LoRA:** aplica adaptadores LoRA com `r=64`, `lora_alpha=16`, e `lora_dropout=0.1` via `peft`.

**Passo 4 — Treinamento:** usa `SFTTrainer` com otimizador `paged_adamw_32bit`, scheduler cosine, e `warmup_ratio=0.03`.

```bash
python train.py
```

Ao final, o adaptador treinado é salvo em `outputs/football-lora/`.

---

## Estrutura do Projeto

```text
llm-lora-qlora-synthetic-finetuning-lab/
├── .env.example            # Template de variáveis de ambiente
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt        # Dependências do projeto
├── config.py               # Hiperparâmetros centralizados (dataclasses)
├── generate_dataset.py     # Geração do dataset sintético via OpenAI API
├── train.py                # Pipeline completo de fine-tuning
├── colab_training.ipynb    # Notebook para execução no Google Colab
├── data/
│   ├── train.jsonl         # 54 pares de treino (futebol)
│   └── test.jsonl          # 6 pares de teste (futebol)
└── outputs/
    └── football-lora/      # Adaptador LoRA salvo após o treinamento
```

---

## Hiperparâmetros Obrigatórios

| Componente   | Parâmetro                | Valor              |
| ------------ | ------------------------ | ------------------ |
| Quantização  | `bnb_4bit_quant_type`    | `nf4`              |
| Quantização  | `bnb_4bit_compute_dtype` | `float16`          |
| Quantização  | `load_in_4bit`           | `True`             |
| LoRA         | `r` (rank)               | `64`               |
| LoRA         | `lora_alpha`             | `16`               |
| LoRA         | `lora_dropout`           | `0.1`              |
| LoRA         | `task_type`              | `CAUSAL_LM`        |
| Treinamento  | `optim`                  | `paged_adamw_32bit`|
| Treinamento  | `lr_scheduler_type`      | `cosine`           |
| Treinamento  | `warmup_ratio`           | `0.03`             |

Todos os valores são definidos em [config.py](config.py) e consumidos pelos scripts sem repetição (princípio DRY).

---

## Política de Uso de IA

> Partes geradas/complementadas com IA, revisadas por **José Lucas Silva Souza**.
