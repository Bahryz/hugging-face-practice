# 🏷️ Zero-Shot Transaction Classifier

Classificador inteligente de transações financeiras usando **Zero-Shot Classification** com HuggingFace Transformers. O modelo classifica descrições de gastos em categorias sem necessidade de treinamento específico.

---

## 🎯 O Problema

Categorizar transações financeiras manualmente é tedioso e propenso a erros. Este projeto demonstra como um modelo multilingual de NLP pode classificar automaticamente transações escritas em **português** em categorias pré-definidas — sem nenhum dado de treino rotulado.

---

## 🧠 Como Funciona

```
Transação em texto livre (PT-BR)
        │
        ▼
┌────────────────────────────────┐
│  Zero-Shot Classification      │  ← HuggingFace Pipeline
│  Modelo: mDeBERTa-v3-base      │  ← Multilingual NLI
│  Hypothesis: "Esta despesa é   │
│  da categoria {}"               │
└────────────────────────────────┘
        │
        ▼
┌────────────────────────────────┐
│  Categorias Candidatas:         │
│  • alimentação                  │
│  • transporte e locomoção       │
│  • contas domésticas            │
│  • lazer e entretenimento       │
│  • compras de produtos e bens   │
└────────────────────────────────┘
        │
        ▼
  DataFrame com resultados
```

O classificador avalia cada transação contra todas as categorias candidatas e seleciona a de maior probabilidade.

---

## 🛠️ Tecnologias Utilizadas

| Tecnologia | Uso |
|---|---|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Linguagem principal |
| ![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black) | Transformers Pipeline |
| ![mDeBERTa](https://img.shields.io/badge/mDeBERTa--v3-NLI-blue?style=flat) | Modelo multilingual zero-shot |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | Exibição dos resultados |
| ![dotenv](https://img.shields.io/badge/python--dotenv-ECD53F?style=flat) | Gerenciamento de secrets |

---

## 🚀 Como Rodar

### 1. Clone o repositório
```bash
git clone https://github.com/Bahryz/hugging-face-practice.git
cd hugging-face-practice
```

### 2. Crie e ative um ambiente virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Configure o `.env`
Copie o arquivo `.env` e adicione seu token do HuggingFace (opcional — funciona sem token para modelos públicos):
```env
HF_TOKEN=hf_seu_token_aqui
```
> 💡 Obtenha seu token em [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 5. Execute
```bash
python teste.py
```

---

## 📊 Exemplo de Output

```
======================================================================
  CLASSIFICAÇÃO ZERO-SHOT DE TRANSAÇÕES FINANCEIRAS
======================================================================
                                                  Transação              Categoria Prevista
  Comprei um carro, um carro bem bonito pra te falar é uma RAM   compras de produtos e bens
  Fui ao parque e paguei uma entrada a roda gigante do parque     lazer e entretenimento
  Gastei 15 reais de uber pra levar minha mãe ao aeroporto       transporte e locomoção
  Paguei a conta de luz da casa, bem caro esse mês               contas domésticas
  Fui jantar no melhor sushi da cidade com minha esposa           alimentação
======================================================================
```

---

## 📁 Estrutura do Projeto

```
hugging-face-practice/
├── teste.py            # Script principal do classificador
├── .env                # Variáveis de ambiente (HF_TOKEN)
├── .gitignore          # Arquivos ignorados pelo Git
├── requirements.txt    # Dependências Python
└── README.md           # Este arquivo
```

---

## 👤 Autor

**Pedro Bahry** — AI Engineering Intern @ NTT DATA  
[![GitHub](https://img.shields.io/badge/GitHub-Bahryz-181717?style=flat&logo=github)](https://github.com/Bahryz)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Pedro%20Bahry-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/pedro-bahry-864b40305)
