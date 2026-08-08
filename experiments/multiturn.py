"""
experiments/multiturn.py

Experimento MULTI-TURNO (seção 5.5 do artigo, contribuição 3): valida
o mecanismo de SessionMemory.add_turn() de verdade -- até agora ele só
tinha sido exercitado na demo manual (agent/session_memory.py::__main__),
nunca em experimento formal com métrica agregada sobre vários clientes.

CENÁRIO SIMULADO (3 momentos, por cliente):
  momento_0        -- sem nenhuma fala do cliente (só recência/frequência)
  turno_1_suporte  -- cliente menciona um problema/chamado de suporte
  turno_2_compra   -- cliente MUDA de assunto pra uma compra que fez

O QUE ISSO MEDE (duas coisas, não uma):

  1) RECALL ORIENTADO POR TÓPICO -- será que mencionar um assunto de
     fato aumenta a chance da entidade certa daquele assunto aparecer
     no contexto, comparado ao momento anterior (sem esse assunto
     mencionado)? Isso é o teste direto de que add_turn() funciona,
     não só existe.

  2) SUBSTITUIÇÃO, NÃO ACÚMULO -- o texto injetado deveria ficar do
     mesmo tamanho (top_k fixo) em todos os 3 momentos, nunca crescer.
     Se o tamanho do contexto crescer monotonicamente turno a turno,
     é sinal de que o texto está empilhando em vez de substituir --
     exatamente o failure mode (context distraction) que a arquitetura
     foi desenhada pra evitar (ver conversa/artigo, seção 4.6/6).

SELEÇÃO DE CLIENTES: só entram clientes com histórico nos DOIS tópicos
(pelo menos 1 evento support_ticket E 1 evento purchase, no log
conhecido via resolução de identidade) -- sem isso, a troca de assunto
no turno 2 não tem nada de verdade pra recuperar, e o experimento não
testaria nada.

NÃO USA LLM -- é um experimento de RECALL (a entidade certa está ou não
está no contexto retornado), não de geração de resposta. Roda igual
neste ambiente e na máquina do usuário, sem depender de Granite.
"""

from __future__ import annotations

import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.session_memory import SessionMemory
from agent.tool_contract import _entry_node, _known_channel_nodes
from config import (
    N_CUSTOMERS_MULTITURN, MULTITURN_TOP_K,
    MULTITURN_UTTERANCE_SUPORTE, MULTITURN_UTTERANCE_COMPRA, SEED,
)


# ---------------------------------------------------------------------------
# SELEÇÃO DE CLIENTES E GROUND TRUTH POR TÓPICO
# ---------------------------------------------------------------------------

def _most_recent_entity(log, known_nodes, event_type):
    """Entidade do evento mais recente de um tipo específico, dentro do
    log que a resolução de identidade consegue enxergar pra esse
    cliente. None se não houver nenhum evento desse tipo."""
    ev = log[log.local_customer_id.isin(known_nodes) & (log.event_type == event_type)]
    if len(ev) == 0:
        return None
    return ev.sort_values("ts", ascending=False).iloc[0].entity_id


def _select_multiturn_customers(log, facts, G, n=N_CUSTOMERS_MULTITURN, seed=SEED):
    """Clientes com pelo menos 1 evento support_ticket E 1 evento
    purchase no log conhecido -- só esses permitem testar troca de
    assunto de verdade (ver docstring do módulo)."""
    rng = random.Random(seed)
    candidates = []
    for global_id in facts:
        entry = _entry_node(global_id)
        known = _known_channel_nodes(G, entry)
        target_suporte = _most_recent_entity(log, known, "support_ticket")
        target_compra = _most_recent_entity(log, known, "purchase")
        if target_suporte is not None and target_compra is not None:
            candidates.append((global_id, target_suporte, target_compra))

    rng.shuffle(candidates)
    return candidates[:n]


# ---------------------------------------------------------------------------
# SIMULAÇÃO DA SESSÃO, 1 CLIENTE
# ---------------------------------------------------------------------------

def _simulate_session(customer_id, target_suporte, target_compra,
                       G, node_idx, embeddings, type_indexes, log, top_k=MULTITURN_TOP_K):
    session = SessionMemory(customer_id, G, node_idx, embeddings, type_indexes, log)
    rows = []

    def _record(stage):
        result = session.get_context(top_k=top_k)
        text = result["context_text"]
        rows.append({
            "customer_id": customer_id,
            "stage": stage,
            "turn_count": session.turn_count,
            "context_n_char": len(text),
            "suporte_presente": target_suporte in text,
            "compra_presente": target_compra in text,
        })

    _record("momento_0")

    session.add_turn(MULTITURN_UTTERANCE_SUPORTE)
    _record("turno_1_suporte")

    session.add_turn(MULTITURN_UTTERANCE_COMPRA)
    _record("turno_2_compra")

    return rows


# ---------------------------------------------------------------------------
# EXPERIMENTO COMPLETO
# ---------------------------------------------------------------------------

def run_multiturn_experiment(dataset, G, node_idx, embeddings, type_indexes,
                              n_customers=N_CUSTOMERS_MULTITURN, top_k=MULTITURN_TOP_K, seed=SEED):
    log = dataset["log"]
    facts = dataset["facts"]

    selected = _select_multiturn_customers(log, facts, G, n=n_customers, seed=seed)

    rows = []
    for customer_id, target_suporte, target_compra in selected:
        rows.extend(_simulate_session(
            customer_id, target_suporte, target_compra,
            G, node_idx, embeddings, type_indexes, log, top_k=top_k,
        ))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RESUMO
# ---------------------------------------------------------------------------

STAGE_ORDER = ["momento_0", "turno_1_suporte", "turno_2_compra"]


def summarize_recall(results: pd.DataFrame):
    """Taxa de presença da entidade-alvo de cada tópico, por estágio.
    O que se espera ver: suporte_presente sobe em turno_1_suporte (em
    relação a momento_0); compra_presente sobe em turno_2_compra."""
    tbl = results.groupby("stage")[["suporte_presente", "compra_presente"]].mean()
    return tbl.reindex(STAGE_ORDER)


def summarize_context_size(results: pd.DataFrame):
    """Tamanho médio do contexto por estágio -- deve ficar ESTÁVEL, não
    crescente, se a substituição (não acúmulo) estiver funcionando."""
    tbl = results.groupby("stage")["context_n_char"].agg(["mean", "std"])
    return tbl.reindex(STAGE_ORDER)


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data.synthetic_events import generate_dataset
    from graph.build_graph import build_graph
    from embeddings.structural import compute_structural_embeddings, build_type_indexes

    ds = generate_dataset()
    G, _ = build_graph(ds)
    nodes, node_idx, embeddings = compute_structural_embeddings(G)
    type_indexes = build_type_indexes(nodes, node_idx, embeddings)

    results = run_multiturn_experiment(ds, G, node_idx, embeddings, type_indexes)
    print(f"Total de clientes testados: {results.customer_id.nunique()}\n")
    print("Recall por tópico e estágio (esperado: sobe quando o tópico é mencionado):")
    print(summarize_recall(results).round(3))
    print("\nTamanho de contexto por estágio (esperado: estável, não crescente):")
    print(summarize_context_size(results).round(1))