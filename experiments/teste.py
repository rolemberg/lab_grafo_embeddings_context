from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.diagnostic import answer_with_local_hf_model
from experiments.longmemeval_check import load_oracle, run_check, summarize

# carrega os dados
data = load_oracle("data/longmemeval_oracle.json")

# roda as 70 perguntas x 3 condições, com o Granite respondendo de verdade
results = run_check(data, answer_fn=answer_with_local_hf_model)

# mostra o resumo (F1 médio por condição)
print(summarize(results))

# salva o resultado bruto, pra eu poder analisar depois
results.to_csv("results/longmemeval_check.csv", index=False)