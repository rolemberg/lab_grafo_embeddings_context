"""
retrieval/hybrid.py

Pipeline híbrido de retrieval (seção 4.3 do artigo):
  1) RECALL rápido: kNN sobre embedding estrutural (embeddings/structural.py)
     -> candidatos aproximados, baratos de calcular.
  2) RERANK fino: Personalized PageRank rodado só no SUBGRAFO INDUZIDO
     pelos candidatos + seus vizinhos diretos -- não no grafo inteiro.
     É isso que torna o método barato o suficiente pra rodar sob
     demanda, dentro de uma tool call do agente, em vez de precisar
     pré-computar tudo offline.

Este módulo também expõe full_graph_ppr(), a alternativa "ingênua" de
rodar PPR no grafo inteiro sem recall prévio -- usada só para a
comparação de custo (seção 5.4), não faz parte do pipeline de produção.
"""

from __future__ import annotations

import os
import sys
import time

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embeddings.structural import recall_candidates, node_type
from config import PPR_ALPHA, PPR_MAX_ITER, PPR_MAX_ITER_FALLBACK, MAX_NEIGHBORS_PER_CANDIDATE

# NOTA sobre MAX_NEIGHBORS_PER_CANDIDATE (valor em config.py, hoje 20): sem esse
# limite, nós-hub (entidades populares, tipos de evento) fazem o subgrafo
# "local" virar quase o grafo inteiro (medido: 92.9% dos nós sem esse limite).


# ---------------------------------------------------------------------------
# RERANK: PPR LOCAL NO SUBGRAFO INDUZIDO
# ---------------------------------------------------------------------------

def _top_weighted_neighbors(G, node, max_n):
    """Vizinhos de `node`, ordenados por peso de aresta decrescente,
    limitados a max_n. Evita que um nó-hub (ex: entidade muito popular)
    explique a vizinhança inteira do subgrafo local."""
    neighbors = G[node]  # dict {vizinho: {"weight": ..., ...}}
    if len(neighbors) <= max_n:
        return list(neighbors.keys())
    ranked = sorted(neighbors.items(), key=lambda kv: kv[1].get("weight", 0.0), reverse=True)
    return [n for n, _ in ranked[:max_n]]


def local_ppr_rerank(G, query_node, candidates, top_k=10,
                      max_neighbors_per_candidate=MAX_NEIGHBORS_PER_CANDIDATE):
    """Roda PPR só no subgrafo induzido por (candidatos + vizinhos
    diretos MAIS RELEVANTES de cada candidato + o próprio nó de
    consulta). Barato porque não toca o grafo inteiro -- o subgrafo
    tipicamente tem uma fração pequena dos nós totais, desde que a
    expansão de vizinhança seja limitada (ver _top_weighted_neighbors)."""
    if query_node not in G:
        return []

    local_nodes = set(candidates) | {query_node}
    for c in candidates:
        if c in G:
            local_nodes.update(_top_weighted_neighbors(G, c, max_neighbors_per_candidate))
    # garante que os vizinhos do próprio query também entram, mesmo
    # que nenhum candidato tenha puxado eles
    if query_node in G:
        local_nodes.update(_top_weighted_neighbors(G, query_node, max_neighbors_per_candidate))
    subG = G.subgraph(local_nodes)

    if query_node not in subG:
        return []

    personalization = {n: (1.0 if n == query_node else 0.0) for n in subG.nodes()}
    try:
        scores = nx.pagerank(subG, alpha=PPR_ALPHA, personalization=personalization,
                              weight="weight", max_iter=PPR_MAX_ITER)
    except nx.PowerIterationFailedConvergence:
        # subgrafos pequenos/estranhos às vezes não convergem no limite
        # padrão -- tenta de novo com mais margem antes de desistir
        scores = nx.pagerank(subG, alpha=PPR_ALPHA, personalization=personalization,
                              weight="weight", max_iter=PPR_MAX_ITER_FALLBACK, tol=1e-4)

    ranked = sorted(
        [(n, s) for n, s in scores.items() if n in candidates],
        key=lambda x: x[1], reverse=True,
    )
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# PIPELINE HÍBRIDO COMPLETO
# ---------------------------------------------------------------------------

DIRECT_RECENCY_TOPN = 10  # quantos vizinhos por recência direta somar ao recall por embedding


def _direct_recency_candidates(G, query_node, target_type="entity", n=DIRECT_RECENCY_TOPN):
    """Candidatos direto por PESO DE ARESTA (já ponderado por recência,
    Eq. 2), sem passar pelo espaço de embedding. Ver nota em
    hybrid_retrieve() -- isso é o fix pro gargalo de recall em
    perguntas tipo 'qual foi a última X', onde proximidade estrutural
    não é a mesma coisa que recência."""
    if query_node not in G:
        return []
    neighbors = [
        (nb, G[query_node][nb].get("weight", 0.0))
        for nb in G.neighbors(query_node)
        if node_type(nb) == target_type
    ]
    neighbors.sort(key=lambda x: x[1], reverse=True)
    return [nb for nb, _ in neighbors[:n]]


# ---------------------------------------------------------------------------
# RECALL + RERANK, QUERY ÚNICA
# ---------------------------------------------------------------------------

def hybrid_retrieve(query_node, G, node_idx, embeddings, type_indexes,
                     k_recall=30, top_k=10, target_type="entity"):
    """Recall (kNN) + rerank (PPR local). Retorna (ranked, t_recall_ms, t_rerank_ms)."""
    t0 = time.perf_counter()
    candidates = recall_candidates(query_node, node_idx, embeddings, type_indexes,
                                    target_type=target_type, k=k_recall)
    # UNIÃO com candidatos por recência direta (peso de aresta) -- fix pro
    # gargalo diagnosticado no experimento comparativo em escala grande:
    # recall via embedding busca proximidade ESTRUTURAL, não recência.
    # Uma entidade da interação mais recente do cliente pode não estar
    # "perto" dele no espaço de embedding (o embedding reflete padrão de
    # interação, não tempo) -- então perguntas do tipo "qual foi a
    # última X" perdiam a entidade certa antes mesmo do rerank rodar.
    # Como já calculamos peso por recência nas arestas (Eq. 2 do artigo),
    # isso é essencialmente de graça: sem embedding, sem custo extra
    # relevante, só ordena os vizinhos diretos já conhecidos.
    direct = _direct_recency_candidates(G, query_node, target_type=target_type, n=DIRECT_RECENCY_TOPN)
    candidates = list(dict.fromkeys(candidates + direct))  # une, preserva ordem, sem duplicar
    t1 = time.perf_counter()
    ranked = local_ppr_rerank(G, query_node, candidates, top_k=top_k)
    t2 = time.perf_counter()
    return ranked, (t1 - t0) * 1000, (t2 - t1) * 1000


# ---------------------------------------------------------------------------
# VARIANTE MULTI-SEED: recall + rerank a partir de VÁRIOS nós de query
# ponderados, não um só. Necessário para memória de sessão (memória
# curta), onde o vetor de personalização acumula massa de vários tópicos
# ao longo de uma ligação, em vez de uma única query fixa.
# Aditivo -- não altera hybrid_retrieve()/local_ppr_rerank() acima, que
# continuam servindo o caso de uma consulta única (ex: experiments/).
# ---------------------------------------------------------------------------

def recall_candidates_from_weights(seed_weights, node_idx, embeddings, type_indexes, G=None,
                                    target_type="entity", k_per_seed=20):
    """Recall unindo candidatos de CADA nó do vetor de seed (peso > 0),
    na ordem de maior peso primeiro. Não pondera o recall em si -- o
    peso de cada seed só importa depois, no rerank via PPR
    personalizado (local_ppr_rerank_from_weights).

    G opcional: se passado, une também candidatos por recência DIRETA
    (peso de aresta, sem embedding) a partir de cada nó do seed -- mesmo
    fix de hybrid_retrieve(), necessário aqui porque é esse caminho que
    agent/session_memory.py usa de fato (não o de query única)."""
    ordered_seeds = sorted(seed_weights.items(), key=lambda kv: kv[1], reverse=True)
    candidates, seen = [], set()
    for node, weight in ordered_seeds:
        if weight <= 0 or node not in node_idx:
            continue
        for c in recall_candidates(node, node_idx, embeddings, type_indexes,
                                    target_type=target_type, k=k_per_seed):
            if c not in seen:
                seen.add(c)
                candidates.append(c)
        if G is not None:
            for c in _direct_recency_candidates(G, node, target_type=target_type, n=DIRECT_RECENCY_TOPN):
                if c not in seen:
                    seen.add(c)
                    candidates.append(c)
    return candidates


def local_ppr_rerank_from_weights(G, seed_weights, candidates, top_k=10,
                                   max_neighbors_per_candidate=MAX_NEIGHBORS_PER_CANDIDATE):
    """Mesma mecânica de local_ppr_rerank(), mas o vetor de
    personalização vem de um DICT {nó: peso} já normalizado externamente
    (o estado acumulado da memória curta), em vez de massa 1.0 num único
    nó de query."""
    active_seeds = {n: w for n, w in seed_weights.items() if w > 0 and n in G}
    if not active_seeds:
        return []

    local_nodes = set(candidates) | set(active_seeds)
    for c in candidates:
        if c in G:
            local_nodes.update(_top_weighted_neighbors(G, c, max_neighbors_per_candidate))
    for s in active_seeds:
        local_nodes.update(_top_weighted_neighbors(G, s, max_neighbors_per_candidate))
    subG = G.subgraph(local_nodes)

    # personalização só sobre seeds que sobreviveram no subgrafo induzido;
    # renormaliza pra somar 1 (nx.pagerank não exige isso, mas deixa a
    # massa relativa entre seeds explícita e auditável)
    sub_seeds = {n: w for n, w in active_seeds.items() if n in subG}
    if not sub_seeds:
        return []
    total = sum(sub_seeds.values())
    personalization = {n: (sub_seeds.get(n, 0.0) / total) for n in subG.nodes()}

    try:
        scores = nx.pagerank(subG, alpha=PPR_ALPHA, personalization=personalization,
                              weight="weight", max_iter=PPR_MAX_ITER)
    except nx.PowerIterationFailedConvergence:
        scores = nx.pagerank(subG, alpha=PPR_ALPHA, personalization=personalization,
                              weight="weight", max_iter=PPR_MAX_ITER_FALLBACK, tol=1e-4)

    ranked = sorted(
        [(n, s) for n, s in scores.items() if n in candidates],
        key=lambda x: x[1], reverse=True,
    )
    return ranked[:top_k]


def hybrid_retrieve_from_weights(seed_weights, G, node_idx, embeddings, type_indexes,
                                  k_recall_per_seed=20, top_k=10, target_type="entity"):
    """Versão multi-seed de hybrid_retrieve(). Usada por
    agent/session_memory.py para servir consultas cujo foco evolui ao
    longo de uma sessão."""
    t0 = time.perf_counter()
    candidates = recall_candidates_from_weights(seed_weights, node_idx, embeddings, type_indexes, G=G,
                                                 target_type=target_type, k_per_seed=k_recall_per_seed)
    t1 = time.perf_counter()
    ranked = local_ppr_rerank_from_weights(G, seed_weights, candidates, top_k=top_k)
    t2 = time.perf_counter()
    return ranked, (t1 - t0) * 1000, (t2 - t1) * 1000


# ---------------------------------------------------------------------------
# BASELINE: PPR NO GRAFO INTEIRO (só para comparação de custo, 5.4)
# ---------------------------------------------------------------------------

def full_graph_ppr(G, query_node, top_k=10, target_type="entity"):
    """Alternativa sem recall prévio: PPR personalizado no grafo inteiro,
    depois filtra pelo tipo de nó desejado. Existe só para medir o
    speedup do híbrido -- não é o que o agente chamaria em produção."""
    t0 = time.perf_counter()
    personalization = {n: (1.0 if n == query_node else 0.0) for n in G.nodes()}
    scores = nx.pagerank(G, alpha=PPR_ALPHA, personalization=personalization, weight="weight")

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranked = [(n, s) for n, s in ranked if n != query_node and node_type(n) == target_type]
    ranked = ranked[:top_k]
    t1 = time.perf_counter()
    return ranked, (t1 - t0) * 1000


# ---------------------------------------------------------------------------
# DEMO / SANITY CHECK
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data.synthetic_events import generate_dataset
    from graph.build_graph import build_graph
    from embeddings.structural import compute_structural_embeddings, build_type_indexes

    ds = generate_dataset()
    G, graph_stats = build_graph(ds)
    print(f"Grafo: {graph_stats['n_nodes']} nós, {graph_stats['n_edges']} arestas\n")

    nodes, node_idx, embeddings = compute_structural_embeddings(G)
    type_indexes = build_type_indexes(nodes, node_idx, embeddings)

    query = "web_5"
    print(f"=== Consulta: {query} ===\n")

    ranked_hybrid, t_recall, t_rerank = hybrid_retrieve(
        query, G, node_idx, embeddings, type_indexes, target_type="entity",
    )
    print(f"[HÍBRIDO] recall: {t_recall:.2f}ms | rerank local: {t_rerank:.2f}ms | "
          f"total: {t_recall + t_rerank:.2f}ms")
    print("Top entidades relevantes (híbrido):")
    for n, s in ranked_hybrid:
        print(f"  {n:10s}  score={s:.5f}")

    print()
    ranked_full, t_full = full_graph_ppr(G, query, target_type="entity")
    print(f"[PPR NO GRAFO INTEIRO] total: {t_full:.2f}ms")
    print("Top entidades relevantes (PPR completo):")
    for n, s in ranked_full:
        print(f"  {n:10s}  score={s:.5f}")

    overlap = len(set(n for n, _ in ranked_hybrid) & set(n for n, _ in ranked_full))
    print(f"\nSobreposição top-10 híbrido vs PPR completo: {overlap}/10")
    t_hybrid_total = t_recall + t_rerank
    if t_hybrid_total > 0:
        print(f"Speedup do híbrido: {t_full / t_hybrid_total:.1f}x")