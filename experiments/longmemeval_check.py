"""
experiments/longmemeval_check.py

O QUE ESTE MÓDULO TESTA, ESPECIFICAMENTE: só a regra de decisão da
contribuição 1 (context clash por recência) -- "quando há evidência
conflitante sobre o mesmo fato em sessões diferentes, a sessão mais
recente vence" -- isolada, sem o resto do pipeline (grafo, PPR,
resolução de identidade).

POR QUE ISOLAR, EM VEZ DE RECONSTRUIR O PIPELINE INTEIRO: os dados do
LongMemEval são diálogo em texto livre, não campos estruturados
(intent/channel/entity_id). Rodar o pipeline completo exigiria uma
etapa de extração -- tipicamente via LLM -- que é exatamente o que
nossa Seção 2.5/2.6 usa para diferenciar o método dos comparáveis
(HippoRAG, G-Long, AtomMem, MemORAI, Mem0, Graphiti). Testar a REGRA DE
DECISÃO isoladamente, usando a própria estrutura de sessão/timestamp
que o benchmark já fornece, evita essa tensão -- não precisamos
extrair nada, só decidir qual sessão já fornecida é a mais recente.

DADOS: usa `longmemeval_oracle.json` (a versão já filtrada só com
sessões de evidência, sem as centenas de sessões de "ruído de
preenchimento" do benchmark completo) -- ver conversa/artigo para a
justificativa desse recorte.

RECORTE: categoria `knowledge-update` (78 perguntas no total), filtrada
para perguntas com evidência em MAIS DE UMA sessão (70 das 78) -- só
essas têm conflito de verdade pra resolver. As 8 restantes (evidência
numa sessão só) não testam a regra de recência, então ficam de fora.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

QUESTION_TYPE = "knowledge-update"


# ---------------------------------------------------------------------------
# CARREGAMENTO E FILTRO
# ---------------------------------------------------------------------------

def load_oracle(path: str):
    oracle_path = Path(path)
    if not oracle_path.is_absolute() and not oracle_path.exists():
        repo_root = Path(__file__).resolve().parent.parent
        candidate = repo_root / "data" / oracle_path.name
        if candidate.exists():
            oracle_path = candidate
    with open(oracle_path) as f:
        return json.load(f)


def _n_sessions_with_answer(q: dict) -> int:
    return sum(1 for sess in q["haystack_sessions"] if any(t.get("has_answer") for t in sess))


def select_conflict_questions(data: list, question_type: str = QUESTION_TYPE):
    """Só perguntas do tipo alvo, com evidência em >1 sessão -- ver
    docstring do módulo. No oracle atual: 70 de 78 para knowledge-update."""
    return [
        q for q in data
        if q["question_type"] == question_type and _n_sessions_with_answer(q) > 1
    ]


# ---------------------------------------------------------------------------
# CONSTRUÇÃO DE CONTEXTO -- as 3 condições
# ---------------------------------------------------------------------------

def _answer_bearing_text(session) -> str:
    """Concatena só os turnos marcados has_answer=True numa sessão."""
    return "\n".join(t["content"] for t in session if t.get("has_answer"))


def _sessions_by_recency(q: dict):
    """(indice, data, sessao) ordenado do mais recente para o mais
    antigo, só das sessões que têm turno de resposta."""
    idx_with_answer = [i for i, sess in enumerate(q["haystack_sessions"])
                        if any(t.get("has_answer") for t in sess)]
    ordered = sorted(idx_with_answer, key=lambda i: q["haystack_dates"][i], reverse=True)
    return [(i, q["haystack_dates"][i], q["haystack_sessions"][i]) for i in ordered]


def context_todas_sessoes(q: dict) -> str:
    """Baseline: todas as sessões de evidência concatenadas, sem
    nenhuma resolução de conflito -- equivalente em espírito ao
    `contexto_completo` do experimento comparativo principal."""
    parts = []
    for i, sess in enumerate(q["haystack_sessions"]):
        parts.append(f"[Sessão de {q['haystack_dates'][i]}]\n{_answer_bearing_text(sess)}")
    return "\n\n".join(parts)


def context_sessao_mais_recente(q: dict) -> str:
    """NOSSO MÉTODO: só o texto da sessão de evidência mais recente --
    a aplicação direta da política de clash por recência (contribuição
    1) fora do contexto do grafo."""
    ordered = _sessions_by_recency(q)
    idx, date, sess = ordered[0]
    return f"[Sessão de {date}]\n{_answer_bearing_text(sess)}"


def context_sessao_mais_antiga(q: dict) -> str:
    """Baseline de contraste: só a sessão MAIS ANTIGA -- existe pra
    mostrar que é a RECÊNCIA que importa, não só "uma sessão qualquer
    em vez de todas". Se esse baseline for tão bom quanto o nosso,
    recência não é o que está ajudando."""
    ordered = _sessions_by_recency(q)
    idx, date, sess = ordered[-1]
    return f"[Sessão de {date}]\n{_answer_bearing_text(sess)}"


CONTEXT_BUILDERS = {
    "todas_sessoes": context_todas_sessoes,
    "sessao_mais_recente": context_sessao_mais_recente,
    "sessao_mais_antiga": context_sessao_mais_antiga,
}


# ---------------------------------------------------------------------------
# SCORE -- F1 em nível de token (convenção do próprio LongMemEval/LoCoMo,
# não inventamos métrica nova aqui)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+")


def _tokenize(text) -> list:
    return _WORD_RE.findall(str(text).lower())


def token_f1(prediction, gold) -> float:
    pred_tokens = Counter(_tokenize(prediction))
    gold_tokens = Counter(_tokenize(gold))
    if not gold_tokens:
        return 0.0
    overlap = sum((pred_tokens & gold_tokens).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred_tokens.values())
    recall = overlap / sum(gold_tokens.values())
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# RESPONDEDOR DE SANITY-CHECK (sem LLM -- só valida a mecânica)
# ---------------------------------------------------------------------------

def answer_naive_echo(question: str, context: str) -> str:
    """Sanity-check sem LLM: devolve o contexto cru como "resposta".
    Não serve pra medir qualidade de resposta de verdade -- só confirma
    que o pipeline de dados/contexto/score roda sem quebrar. Pra
    números reais, use experiments.diagnostic.answer_with_local_hf_model
    na sua máquina (ver run_check abaixo, parâmetro answer_fn)."""
    return context


# ---------------------------------------------------------------------------
# EXPERIMENTO
# ---------------------------------------------------------------------------

def run_check(data: list, answer_fn=answer_naive_echo, question_type: str = QUESTION_TYPE):
    questions = select_conflict_questions(data, question_type=question_type)

    rows = []
    for q in questions:
        for condition, builder in CONTEXT_BUILDERS.items():
            context = builder(q)
            prediction = answer_fn(q["question"], context)
            f1 = token_f1(prediction, q["answer"])
            rows.append({
                "question_id": q["question_id"],
                "condition": condition,
                "question": q["question"],
                "gold_answer": q["answer"],
                "prediction": prediction,
                "f1": f1,
                "context_n_char": len(context),
            })
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame):
    return results.groupby("condition")[["f1", "context_n_char"]].mean().round(3)


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "longmemeval_oracle.json"
    data = load_oracle(path)

    questions = select_conflict_questions(data)
    print(f"Perguntas knowledge-update com conflito real: {len(questions)} de "
          f"{sum(1 for q in data if q['question_type'] == QUESTION_TYPE)}")

    results = run_check(data, answer_fn=answer_naive_echo)
    print("\n[SANITY-CHECK -- respondedor de eco, NÃO é qualidade de resposta real]")
    print(summarize(results))
    print("\nPara números reais, rode na sua máquina com:")
    print("  from experiments.diagnostic import answer_with_local_hf_model")
    print("  results = run_check(data, answer_fn=answer_with_local_hf_model)")