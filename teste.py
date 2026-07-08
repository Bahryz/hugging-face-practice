"""
Classificador de transações financeiras utilizando inferência Zero-Shot.
Modelo: mDeBERTa-v3 ajustado para NLI (Natural Language Inference).
"""

import os
import pandas as pd
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Token opcional, útil para evitar rate limit de requisições de modelos no Hugging Face Hub
pipeline_kwargs = {}
if hf_token:
    pipeline_kwargs["token"] = hf_token

# O mDeBERTa-v3-base-mnli-xnli é o estado da arte para NLI multilíngue e responde muito bem em PT-BR
classificador = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    **pipeline_kwargs
)

transacoes = [
    "Comprei um carro, um carro bem bonito pra te falar é uma RAM",
    "Fui ao parque e paguei uma entrada a roda gigante do parque me di",
    "Gastei 15 reais de uber pra levar minha mãe ao aeroporto",
    "Paguei a conta de luz da casa, bem caro esse mês",
    "Fui jantar no melhor sushi da cidade com minha esposa"
]

categorias = [
    "alimentação",
    "transporte e locomoção",
    "contas domésticas",
    "lazer e entretenimento",
    "compras de produtos e bens"
]

categorias_previstas = []
scores_completos = []

for transacao in transacoes:
    # O hypothesis_template estrutura a frase para o modelo avaliar a premissa (NLI)
    resultado = classificador(
        transacao,
        candidate_labels=categorias,
        hypothesis_template="Esta despesa é da categoria {}"
    )

    # O pipeline já retorna as classes ordenadas pelo score de probabilidade (labels[0] é a de maior confiança)
    categorias_previstas.append(resultado["labels"][0])
    scores_completos.append(list(zip(resultado["labels"], resultado["scores"])))

df_resultado = pd.DataFrame({
    "Transação": transacoes,
    "Categoria Prevista": categorias_previstas
})

print("\n" + "=" * 70)
print("  CLASSIFICAÇÃO ZERO-SHOT DE TRANSAÇÕES FINANCEIRAS")
print("=" * 70)
print(df_resultado.to_string(index=False))
print("=" * 70)
