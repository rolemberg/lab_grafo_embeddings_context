"""
metrics/significance.py

Formaliza o que antes era calculado ad hoc, direto em cima do CSV,
toda vez que uma nova rodada do experimento comparativo saía: intervalo
de confiança de Wilson por condição/segmento, e teste de McNemar entre
duas condições (desenhado especificamente pra comparar `sessao_memoria`
vs `hibrido_sob_demanda`, mas funciona pra qualquer par).

Por que Wilson, não a fórmula clássica (p ± 1.96*erro_padrão): a
fórmula clássica quebra perto de 0% ou 100% de acerto (dá intervalo
[0,0] mesmo com poucas tentativas, que é overconfiante). Wilson lida
bem com esse caso -- ver `sem_retrieval`, que fica em 0% de acerto e
ainda assim tem intervalo > 0.

Por que McNemar, não um teste de duas proporções independentes: o
desenho do experimento comparativo é PAREADO -- cada (customer_id,
fact_type) passa pelas mesmas condições, não são amostras
independentes. McNemar usa só os pares discordantes (onde as duas
condições divergem), que é onde mora toda a informação sobre qual
método é melhor -- pares onde as duas acertam ou as duas erram não
dizem nada sobre isso.

Sem dependência de statsmodels -- só numpy/scipy, que já são
dependência do projeto (ver requirements.txt).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, binomtest


# ---------------------------------------------------------------------------
# WILSON SCORE INTERVAL
# ---------------------------------------------------------------------------

def wilson_interval(k: int, n: int, alpha: float = 0.05):
    """Intervalo de confiança de Wilson para uma proporção k/n.
    Retorna (lo, hi). n=0 -> (0.0, 1.0) (sem informação nenhuma)."""
    if n == 0:
        return 0.0, 1.0
    z = norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def accuracy_with_ci(results: pd.DataFrame, group_cols=("condition", "segment"), alpha: float = 0.05):
    """Tabela de acerto com intervalo de Wilson, agrupado por
    group_cols (default: condição x segmento, como no experimento
    comparativo)."""
    rows = []
    for keys, grp in results.groupby(list(group_cols)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n = len(grp)
        k = int(grp["correct"].sum())
        lo, hi = wilson_interval(k, n, alpha=alpha)
        rows.append({**dict(zip(group_cols, keys)), "n": n, "correct": k,
                     "accuracy": k / n if n else float("nan"),
                     "ci_lower": lo, "ci_upper": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MCNEMAR (exato, via teste binomial -- válido em qualquer N, não só
# quando N é grande; mais conservador que a aproximação qui-quadrado,
# mas não exige N mínimo pra ser correto)
# ---------------------------------------------------------------------------

def mcnemar_test(results: pd.DataFrame, condition_a: str, condition_b: str,
                  pair_cols=("customer_id", "fact_type"), segment: str | None = None):
    """Compara duas condições do experimento comparativo, usando o
    desenho pareado (mesmo customer_id/fact_type nas duas condições).

    Retorna dict com: n_pairs, n_a_only (a acertou, b errou), n_b_only
    (b acertou, a errou), n_concordant, p_value.
    """
    d = results if segment is None else results[results.segment == segment]

    a = d[d.condition == condition_a].set_index(list(pair_cols))["correct"]
    b = d[d.condition == condition_b].set_index(list(pair_cols))["correct"]
    paired = a.to_frame("a").join(b.to_frame("b"), how="inner")

    n_a_only = int(((paired.a) & (~paired.b)).sum())
    n_b_only = int(((~paired.a) & (paired.b)).sum())
    n_concordant = int(len(paired) - n_a_only - n_b_only)

    n_discordant = n_a_only + n_b_only
    if n_discordant == 0:
        p_value = 1.0  # nenhuma diferença observável -- não dá pra rejeitar igualdade
    else:
        # teste binomial exato: sob H0 (métodos equivalentes), cada par
        # discordante tem 50% de chance de favorecer qualquer um dos
        # lados -- testamos se o desequilíbrio observado é compatível
        # com isso.
        p_value = binomtest(min(n_a_only, n_b_only), n_discordant, 0.5,
                             alternative="two-sided").pvalue

    return {
        "condition_a": condition_a, "condition_b": condition_b,
        "segment": segment or "geral",
        "n_pairs": len(paired),
        "n_a_only": n_a_only, "n_b_only": n_b_only, "n_concordant": n_concordant,
        "p_value": p_value,
    }


def mcnemar_by_segment(results: pd.DataFrame, condition_a: str, condition_b: str,
                        pair_cols=("customer_id", "fact_type")):
    """Roda mcnemar_test geral + por segmento (leve/pesado), retorna
    DataFrame de 3 linhas -- é o que run_experiment.py imprime."""
    segments = [None] + sorted(results.segment.unique().tolist())
    rows = [mcnemar_test(results, condition_a, condition_b, pair_cols=pair_cols, segment=s)
            for s in segments]
    return pd.DataFrame(rows)