"""
metrics/retrieval_metrics.py

Métricas formais de qualidade de RETRIEVAL -- distintas das métricas de
resposta do agente (metrics/hallucination.py). Aqui avaliamos o ranking
de nós que o retrieval devolve, não a resposta final em linguagem
natural.

Formaliza dois tipos de avaliação que já vinham sendo calculados "na
unha" em outros módulos:

  1) CONTRA UM FATO ESPECÍFICO -- a entidade que responde a pergunta
     (ex: last_entity do ground truth) está no top-k devolvido?
     -> recall_at_k, precision_at_k, reciprocal_rank
     (explica o achado do comparative.py: hibrido/topk_estatico
     acertam só 80% pro segmento "pesado" -- é recall@k perdido, não
     erro do LLM)

  2) CONTRA O PPR COMPLETO COMO GABARITO DE RANKING -- o híbrido
     reproduz bem a ORDEM que o PPR no grafo inteiro produziria?
     -> ndcg_at_k, overlap_at_k
     (formaliza o overlap que retrieval/hybrid.py e experiments/cost.py
     calculavam com um `len(set_a & set_b)` cru)
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# 1) MÉTRICAS CONTRA UM CONJUNTO DE ITENS RELEVANTES (fato específico)
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: list, relevant: set, k: int | None = None) -> float:
    """Fração dos itens relevantes que aparecem nos top-k devolvidos.
    Com 1 item relevante só (caso comum aqui: 1 entidade-alvo), isso
    vira um hit-rate binário (0 ou 1)."""
    if not relevant:
        return float("nan")
    top = retrieved[:k] if k is not None else retrieved
    hits = len(set(top) & relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list, relevant: set, k: int | None = None) -> float:
    """Fração dos top-k devolvidos que são de fato relevantes."""
    top = retrieved[:k] if k is not None else retrieved
    if not top:
        return float("nan")
    hits = len(set(top) & relevant)
    return hits / len(top)


def reciprocal_rank(retrieved: list, relevant: set) -> float:
    """1/posição do primeiro item relevante encontrado (0 se nenhum).
    Componente de MRR quando agregado sobre várias consultas -- mede
    "quão perto do topo" o item certo apareceu, não só se apareceu."""
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def hit_rate_at_k(retrieved: list, relevant: set, k: int | None = None) -> bool:
    """Versão booleana de recall_at_k -- achou PELO MENOS UM item
    relevante no top-k? Útil quando `relevant` tem 1 item só (caso do
    ground truth de fato único, ex: last_entity)."""
    top = retrieved[:k] if k is not None else retrieved
    return len(set(top) & relevant) > 0


# ---------------------------------------------------------------------------
# 2) MÉTRICAS CONTRA UM RANKING DE REFERÊNCIA (ex: PPR completo)
# ---------------------------------------------------------------------------

def overlap_at_k(list_a: list, list_b: list, k: int | None = None) -> float:
    """Fração de sobreposição entre os top-k de duas listas -- métrica
    simples, já usada (sem nome formal) em retrieval/hybrid.py e
    experiments/cost.py. Normaliza pelo tamanho de list_a (o método
    sendo avaliado), não pela união."""
    top_a = set(list_a[:k] if k is not None else list_a)
    top_b = set(list_b[:k] if k is not None else list_b)
    if not top_a:
        return float("nan")
    return len(top_a & top_b) / len(top_a)


def ndcg_at_k(retrieved: list, reference_ranked: list, k: int) -> float:
    """NDCG@k do ranking `retrieved` usando `reference_ranked` (ex: a
    ordem do PPR no grafo inteiro) como gabarito de relevância graduada.

    Relevância de um item = posição inversa no ranking de referência
    (1º lugar = relevância máxima, decai linearmente até 0 fora do
    ranking de referência). Isso penaliza tanto "faltou um item bom"
    quanto "trouxe na ordem errada", ao contrário do overlap_at_k
    (que só vê conjunto, ignora ordem)."""
    n_ref = len(reference_ranked)
    if n_ref == 0 or k == 0:
        return float("nan")

    relevance = {item: (n_ref - i) / n_ref for i, item in enumerate(reference_ranked)}

    top_retrieved = retrieved[:k]
    dcg = sum(
        relevance.get(item, 0.0) / math.log2(i + 2)  # +2 pq i começa em 0
        for i, item in enumerate(top_retrieved)
    )

    ideal_relevances = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevances))

    return dcg / idcg if idcg > 0 else float("nan")


# ---------------------------------------------------------------------------
# CONVENIÊNCIA: AVALIAÇÃO COMBINADA
# ---------------------------------------------------------------------------

def evaluate_against_fact(retrieved: list, target_entity: str, k: int = 10) -> dict:
    """Avaliação conveniente pro caso de 1 fato específico (ex: a
    entidade last_entity do ground truth deve estar no top-k)."""
    relevant = {target_entity}
    return {
        "hit": hit_rate_at_k(retrieved, relevant, k),
        "recall_at_k": recall_at_k(retrieved, relevant, k),
        "reciprocal_rank": reciprocal_rank(retrieved, relevant),
        "rank_if_found": (retrieved.index(target_entity) + 1) if target_entity in retrieved else None,
    }


def evaluate_against_reference(retrieved: list, reference_ranked: list, k: int = 10) -> dict:
    """Avaliação conveniente pro caso de comparar contra um ranking de
    referência (ex: PPR no grafo inteiro)."""
    return {
        "overlap_at_k": overlap_at_k(retrieved, reference_ranked, k),
        "ndcg_at_k": ndcg_at_k(retrieved, reference_ranked, k),
    }


# ---------------------------------------------------------------------------
# DEMO / SANITY CHECK
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Testes unitários simples ===\n")

    retrieved = ["ent_5", "ent_22", "ent_9", "ent_3"]
    relevant = {"ent_22"}
    print(f"retrieved={retrieved}, relevant={relevant}")
    print(f"  recall_at_k(k=4)  = {recall_at_k(retrieved, relevant, 4):.3f}  (esperado 1.0)")
    print(f"  recall_at_k(k=1)  = {recall_at_k(retrieved, relevant, 1):.3f}  (esperado 0.0, não está no top-1)")
    print(f"  reciprocal_rank   = {reciprocal_rank(retrieved, relevant):.3f}  (esperado 0.5, posição 2)")
    print(f"  hit_rate_at_k(k=4)= {hit_rate_at_k(retrieved, relevant, 4)}  (esperado True)")

    reference = ["ent_5", "ent_22", "ent_9", "ent_3", "ent_7"]
    same_order = ["ent_5", "ent_22", "ent_9", "ent_3", "ent_7"]
    shuffled = ["ent_7", "ent_3", "ent_9", "ent_22", "ent_5"]  # ordem invertida
    missing_top = ["ent_3", "ent_7", "ent_9", "ent_22", "ent_99"]  # ent_5 (o melhor) nem aparece

    print(f"\nreference={reference}")
    print(f"  ndcg(same_order)  = {ndcg_at_k(same_order, reference, 5):.4f}  (esperado 1.0, ordem idêntica)")
    print(f"  ndcg(shuffled)    = {ndcg_at_k(shuffled, reference, 5):.4f}  (esperado bem < 1.0, mesmos itens, ordem ruim)")
    print(f"  ndcg(missing_top) = {ndcg_at_k(missing_top, reference, 5):.4f}  (pior ainda: o item mais relevante nem foi recuperado)")
    print(f"  overlap(shuffled, k=5) = {overlap_at_k(shuffled, reference, 5):.3f}  (esperado 1.0 -- overlap ignora ordem!)")

    # --- demo real, usando o pipeline completo ---
    print("\n\n=== Demo com o pipeline real (grafo + embeddings + retrieval) ===\n")
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data.synthetic_events import generate_dataset
    from graph.build_graph import build_graph
    from embeddings.structural import compute_structural_embeddings, build_type_indexes
    from retrieval.hybrid import hybrid_retrieve, full_graph_ppr

    ds = generate_dataset()
    G, _ = build_graph(ds)
    nodes, node_idx, embeddings = compute_structural_embeddings(G)
    type_indexes = build_type_indexes(nodes, node_idx, embeddings)

    facts = ds["facts"]
    events_per_customer = ds["log"].global_customer_id_TRUTH.value_counts()
    heaviest = events_per_customer.index[0]

    query = f"web_{heaviest.split('_')[-1]}"
    target_entity = facts[heaviest].last_entity

    ranked_hybrid, _, _ = hybrid_retrieve(query, G, node_idx, embeddings, type_indexes,
                                           k_recall=30, top_k=10, target_type="entity")
    ranked_full, _ = full_graph_ppr(G, query, top_k=30, target_type="entity")

    hybrid_ids = [n for n, _ in ranked_hybrid]
    full_ids = [n for n, _ in ranked_full]

    print(f"Cliente mais pesado: {heaviest} | alvo (last_entity): {target_entity}")
    print(f"\nAvaliação CONTRA O FATO (a entidade certa está no top-10 do híbrido?):")
    for k, v in evaluate_against_fact(hybrid_ids, target_entity, k=10).items():
        print(f"  {k}: {v}")

    print(f"\nAvaliação CONTRA O PPR COMPLETO (fidelidade de ranking, top-10):")
    for k, v in evaluate_against_reference(hybrid_ids, full_ids, k=10).items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")