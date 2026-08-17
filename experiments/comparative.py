"""
experiments/comparative.py

Experimento COMPARATIVO (seção 5.3 do artigo): compara 5 condições de
injeção de contexto, mantendo o LLM FIXO entre elas -- o único fator
que varia é COMO o contexto chega até o modelo. Isso isola o efeito do
método de retrieval da capacidade de tool-calling do modelo base (o
mesmo controle de confound discutido na seção 5 do esqueleto).

As 5 condições:
  1) sem_retrieval       -- nenhum contexto de cliente (baseline "cego")
  2) contexto_completo    -- despeja todo o histórico conhecido do cliente
  3) topk_estatico        -- top-k entidades por frequência simples (sem
                              PPR, sem propagação de grafo) -- baseline de
                              retrieval "raso"
  4) hibrido_sob_demanda  -- exatamente o que agent/tool_contract.py
                              devolveria (recall + PPR local), seed = 1
                              nó de entrada só, sem clash resolvido
  5) sessao_memoria       -- agent/session_memory.py em "momento 0" (sem
                              add_turn -- ver nota na função abaixo):
                              mesma ausência de foco que a condição 4,
                              mas com seed distribuído sobre N entidades
                              recentes e context clash resolvido por
                              recência. Isola as contribuições 1 e 3 do
                              artigo (política de clash; mecanismo de
                              memória de sessão) do efeito de foco/tópico,
                              que fica pro próximo experimento (multi-turno).

ESTRATIFICAÇÃO: clientes são amostrados em dois segmentos -- "leve"
(perto da mediana de engajamento) e "pesado" (cauda longa) -- porque a
vantagem do híbrido só deve aparecer nos clientes de cauda longa (ver
achado em agent/tool_contract.py: redução de contexto negativa pra
clientes leves, 99.3% pra um cliente pesado).

MESMA RESSALVA DE agent/tool_contract.py e experiments/diagnostic.py:
o respondedor default aqui (answer_naive_contains) NÃO é um LLM -- é
uma checagem de substring, só pra validar a mecânica das 4 condições
neste ambiente (sem GPU/modelo local). Pra medir o resultado real,
troque ANSWER_FN por answer_with_local_hf_model (importado de
experiments.diagnostic) e rode na sua máquina.
"""

from __future__ import annotations

import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tool_contract import _entry_node, _known_channel_nodes, buscar_contexto_cliente
from agent.session_memory import SessionMemory
from experiments.diagnostic import (
    _format_target_line,
    _format_event_line,
    answer_with_local_hf_model,  # re-exportado -- mesmo respondedor real usado no diagnóstico
)
from metrics.hallucination import classify_answer

# ---------------------------------------------------------------------------
# CONFIG DO EXPERIMENTO
# ---------------------------------------------------------------------------

from config import (
    SEED,
    CONDITIONS,
    FACT_TYPES,
    TOPK_STATIC_K,
    N_CUSTOMERS_PER_SEGMENT,
    HEAVY_SEGMENT_PERCENTILE,
    COMPARATIVE_FULL_CONTEXT_MAX_EVENTS,
    COMPARATIVE_CONTEXT_CHAR_BUDGET,
)


# ---------------------------------------------------------------------------
# CONDIÇÃO 1: SEM RETRIEVAL
# ---------------------------------------------------------------------------

def context_sem_retrieval(customer_id, **kwargs):
    return "Nenhuma informação adicional sobre este cliente está disponível."


# ---------------------------------------------------------------------------
# CONDIÇÃO 2: CONTEXTO COMPLETO (despejo bruto de tudo que o grafo conhece)
# ---------------------------------------------------------------------------

def context_completo(customer_id, G, log, **kwargs):
    entry = _entry_node(customer_id)
    known_nodes = _known_channel_nodes(G, entry)
    cust_events = log[log.local_customer_id.isin(known_nodes)]
    if len(cust_events) == 0:
        return f"Nenhum evento encontrado para o cliente {customer_id}."
    cust_events = cust_events.sort_values("ts", ascending=False).head(COMPARATIVE_FULL_CONTEXT_MAX_EVENTS)
    lines = [_format_event_line(row) for row in cust_events.itertuples()]
    header = f"Histórico completo de eventos do cliente {customer_id} ({len(lines)} eventos):"
    return header + "\n" + "\n".join(lines)


def _clip_context_text(context: str, max_chars: int = COMPARATIVE_CONTEXT_CHAR_BUDGET) -> str:
    if len(context) <= max_chars:
        return context
    half = max_chars // 2
    return context[:half] + "\n...[contexto truncado]...\n" + context[-half:]


# ---------------------------------------------------------------------------
# CONDIÇÃO 3: TOP-K ESTÁTICO (frequência simples, sem PPR/grafo)
# ---------------------------------------------------------------------------

def context_topk_estatico(customer_id, G, log, k=TOPK_STATIC_K, **kwargs):
    entry = _entry_node(customer_id)
    known_nodes = _known_channel_nodes(G, entry)
    cust_events = log[log.local_customer_id.isin(known_nodes)]
    if len(cust_events) == 0:
        return f"Nenhum evento encontrado para o cliente {customer_id}."

    top_entities = cust_events.entity_id.value_counts().head(k)
    lines = [f"## Contexto do cliente: {customer_id} (top-{k} por frequência)"]
    for ent, count in top_entities.items():
        ev = cust_events[cust_events.entity_id == ent].sort_values("ts", ascending=False)
        tipos = ", ".join(ev.event_type.value_counts().index[:3])
        lines.append(f"- {ent} ({count} interações) [{tipos}], último em ts={int(ev.ts.max())}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CONDIÇÃO 4: HÍBRIDO SOB DEMANDA (a tool de verdade, recall + PPR local)
# ---------------------------------------------------------------------------

def context_hibrido(customer_id, G, node_idx, embeddings, type_indexes, log,
                     k=TOPK_STATIC_K, **kwargs):
    result = buscar_contexto_cliente(customer_id, G, node_idx, embeddings, type_indexes,
                                      log, max_entidades=k)
    return result["context_text"]


# ---------------------------------------------------------------------------
# CONDIÇÃO 5: SESSÃO DE MEMÓRIA (hipocampo/conteúdo + clash por recência)
#
# CORREÇÃO DE UMA AFIRMAÇÃO ANTERIOR (deixada aqui de propósito, não
# apagada): este comentário dizia que a comparação com hibrido_sob_demanda
# "isolava exatamente as contribuições 1 e 3". Isso era impreciso -- os
# dois métodos diferem em VÁRIAS coisas ao mesmo tempo (nº de seeds,
# âncora, decaimento, E resolução de conflito), não é um ablation limpo.
# Corrigir um bug de recall (retrieval/hybrid.py) mudou o resultado
# dessa comparação de forma substancial, confirmando o confound. A
# condição 6 abaixo (sessao_memoria_sem_clash) é o ablation de verdade.
# ---------------------------------------------------------------------------

def context_sessao_memoria(customer_id, G, node_idx, embeddings, type_indexes, log,
                            k=TOPK_STATIC_K, **kwargs):
    session = SessionMemory(customer_id, G, node_idx, embeddings, type_indexes, log)
    result = session.get_context(top_k=k)
    return result["context_text"]


# ---------------------------------------------------------------------------
# CONDIÇÃO 6: MESMO SESSAO_MEMORIA, SEM RESOLUÇÃO DE CLASH -- ABLATION LIMPO
#
# Idêntica à condição 5 em tudo (mesmo seed evolutivo, mesma âncora,
# mesmo recall corrigido) -- a ÚNICA diferença é resolve_clash=False:
# lista eventos crus por entidade, sem priorizar o mais recente. Essa é
# a comparação que isola de fato a contribuição 1 (rho), sem o confound
# de nº de seeds que invalidou sessao_memoria vs hibrido_sob_demanda.
# ---------------------------------------------------------------------------

def context_sessao_memoria_sem_clash(customer_id, G, node_idx, embeddings, type_indexes, log,
                                      k=TOPK_STATIC_K, **kwargs):
    session = SessionMemory(customer_id, G, node_idx, embeddings, type_indexes, log, resolve_clash=False)
    result = session.get_context(top_k=k)
    return result["context_text"]


CONTEXT_BUILDERS = {
    "sem_retrieval": context_sem_retrieval,
    "contexto_completo": context_completo,
    "topk_estatico": context_topk_estatico,
    "hibrido_sob_demanda": context_hibrido,
    "sessao_memoria": context_sessao_memoria,
    "sessao_memoria_sem_clash": context_sessao_memoria_sem_clash,
}


# ---------------------------------------------------------------------------
# RESPONDEDOR DE SANITY-CHECK (mesmo espírito de experiments/diagnostic.py)
# ---------------------------------------------------------------------------

def answer_naive_contains(question, context, fact_type, expected_answer):
    """SANITY-CHECK APENAS -- não é um LLM. Só confirma se a resposta
    esperada aparece literalmente no texto do contexto -- valida se
    cada condição de fato contém (ou não) a informação necessária,
    não simula nenhum raciocínio real do modelo."""
    return expected_answer.strip().lower() if expected_answer.strip().lower() in context.lower() else ""


ANSWER_FN = None  # None = usa answer_naive_contains (sanity-check); troque por
                   # answer_with_local_hf_model para medir resultado real


# ---------------------------------------------------------------------------
# AMOSTRAGEM ESTRATIFICADA (leve vs pesado)
# ---------------------------------------------------------------------------

def _stratified_customers(log, facts, fact_type, n_per_segment,
                           heavy_percentile=HEAVY_SEGMENT_PERCENTILE, seed=SEED):
    rng = random.Random(seed)

    eligible = [
        gid for gid, f in facts.items()
        if (fact_type != "support_ticket_count" or f.support_ticket_count > 0)
    ]
    events_per_customer = log.global_customer_id_TRUTH.value_counts()
    eligible_counts = events_per_customer[events_per_customer.index.isin(eligible)]

    heavy_threshold = eligible_counts.quantile(heavy_percentile)
    heavy_pool = eligible_counts[eligible_counts >= heavy_threshold].index.tolist()

    median = eligible_counts.median()
    # "leve" = os mais próximos da mediana (não os mais raros/vazios, pra evitar
    # casos degenerados de 1 evento só)
    light_pool = (eligible_counts - median).abs().sort_values().index.tolist()

    heavy_sample = rng.sample(heavy_pool, k=min(n_per_segment, len(heavy_pool)))
    light_candidates = [c for c in light_pool if c not in heavy_sample]
    light_sample = light_candidates[:n_per_segment]

    return {"leve": light_sample, "pesado": heavy_sample}


# ---------------------------------------------------------------------------
# LOOP PRINCIPAL DO EXPERIMENTO
# ---------------------------------------------------------------------------

def run_comparative_experiment(dataset, G, node_idx, embeddings, type_indexes,
                                answer_fn=None, fact_types=FACT_TYPES,
                                n_customers_per_segment=N_CUSTOMERS_PER_SEGMENT,
                                conditions=CONDITIONS, seed=SEED):
    log = dataset["log"]
    facts = dataset["facts"]
    using_naive = answer_fn is None and ANSWER_FN is None
    answer_fn = answer_fn or ANSWER_FN

    rows = []
    for fact_type in fact_types:
        segments = _stratified_customers(log, facts, fact_type, n_customers_per_segment, seed=seed)

        for segment_label, customer_ids in segments.items():
            for customer_id in customer_ids:
                f = facts[customer_id]
                target_line, question, expected_answer = _format_target_line(fact_type, f)

                for condition in conditions:
                    builder = CONTEXT_BUILDERS[condition]
                    context = builder(
                        customer_id, G=G, node_idx=node_idx, embeddings=embeddings,
                        type_indexes=type_indexes, log=log,
                    )
                    context = _clip_context_text(context)

                    if using_naive:
                        model_answer = answer_naive_contains(question, context, fact_type, expected_answer)
                    else:
                        model_answer = answer_fn(question, context)

                    verdict_info = classify_answer(model_answer, expected_answer, fact_type)
                    correct = verdict_info["verdict"] == "correct"

                    rows.append({
                        "customer_id": customer_id,
                        "segment": segment_label,
                        "fact_type": fact_type,
                        "condition": condition,
                        "expected_answer": expected_answer,
                        "model_answer": model_answer,
                        "correct": correct,
                        "verdict": verdict_info["verdict"],
                        "context_n_char": len(context),
                        "context_text": context,  # texto completo -- sem isso, diagnosticar
                                                    # um caso específico exige reconstruir o
                                                    # contexto adivinhando parâmetros/seed, o
                                                    # que já causou análise incorreta antes.
                    })

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame):
    """Tabela pivô: acerto médio por (condição, segmento)."""
    return results.pivot_table(index="condition", columns="segment",
                                values="correct", aggfunc="mean")


def summarize_cost(results: pd.DataFrame):
    """Tabela pivô: tamanho médio de contexto (chars) por (condição, segmento) --
    o lado "custo" do trade-off qualidade x custo que sustenta a tese do artigo."""
    return results.pivot_table(index="condition", columns="segment",
                                values="context_n_char", aggfunc="mean")


def summarize_verdicts(results: pd.DataFrame):
    """Distribuição completa de veredito (correct/hallucination/abstention/
    malformed) por condição -- é aqui que aparece a diferença entre
    "sem_retrieval erra" e "sem_retrieval aluciona vs. sem_retrieval admite
    que não sabe", que uma métrica de acerto simples não revela."""
    return (results.groupby(["condition", "verdict"]).size()
            .unstack(fill_value=0)
            .apply(lambda row: row / row.sum(), axis=1))


# ---------------------------------------------------------------------------
# DEMO / SANITY CHECK
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data.synthetic_events import generate_dataset
    from graph.build_graph import build_graph
    from embeddings.structural import compute_structural_embeddings, build_type_indexes

    ds = generate_dataset()
    G, graph_stats = build_graph(ds)
    nodes, node_idx, embeddings = compute_structural_embeddings(G)
    type_indexes = build_type_indexes(nodes, node_idx, embeddings)

    print("=== Rodando harness com respondedor de SANITY-CHECK (não é LLM) ===")
    print("(confirma que cada condição contém -- ou não -- a informação necessária)\n")

    results = run_comparative_experiment(ds, G, node_idx, embeddings, type_indexes)
    print(f"Total de trials: {len(results)}\n")

    print("Acerto por condição x segmento:")
    print(summarize(results))

    print("\nCusto (chars médios de contexto) por condição x segmento:")
    print(summarize_cost(results).round(0))

    print("\nDistribuição de veredito por condição (correct/hallucination/abstention/malformed):")
    print(summarize_verdicts(results).round(2))

    print("\n" + "=" * 70)
    print("Para medir resultado REAL, na sua máquina (com o Granite em cache):")
    print("  from experiments.comparative import run_comparative_experiment, answer_with_local_hf_model")
    print("  results = run_comparative_experiment(ds, G, node_idx, embeddings, type_indexes,")
    print("                                        answer_fn=answer_with_local_hf_model)")
    print("=" * 70)