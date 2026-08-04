"""
Protótipo: pipeline híbrido para retrieval de contexto de cliente
a partir de uma base de eventos, usando:

  1) Embeddings de grafo -> recall rápido de candidatos (ANN)
  2) Personalized PageRank local -> rerank fino no subgrafo induzido

Objetivo: mostrar a MECÂNICA do híbrido, não ser produção-ready.
Em produção: trocar TruncatedSVD por node2vec/GraphSAGE, e o
NearestNeighbors por FAISS/HNSW.
"""

import time
import random
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1) GERA BASE DE EVENTOS SINTÉTICA
#    formato: cada linha = 1 evento (chave = customer_id, evento = event_type)
# ---------------------------------------------------------------------------

N_CUSTOMERS = 800
N_ENTITIES = 60          # produtos/páginas/campanhas
EVENT_TYPES = ["view", "add_to_cart", "purchase", "search", "support_ticket"]
N_EVENTS = 12000

def gen_event_log():
    rows = []
    # clientes têm "nichos" de interesse -> cria estrutura de comunidade real
    n_niches = 8
    niche_entities = {
        n: random.sample(range(N_ENTITIES), k=N_ENTITIES // n_niches * 2)
        for n in range(n_niches)
    }
    customer_niche = {c: random.randint(0, n_niches - 1) for c in range(N_CUSTOMERS)}

    t0 = 1_700_000_000
    for i in range(N_EVENTS):
        cust = random.randint(0, N_CUSTOMERS - 1)
        niche = customer_niche[cust]
        # 85% dos eventos do cliente ficam dentro do seu nicho (sinal real)
        if random.random() < 0.85:
            entity = random.choice(niche_entities[niche])
        else:
            entity = random.randint(0, N_ENTITIES - 1)
        event_type = random.choices(
            EVENT_TYPES, weights=[5, 3, 1, 3, 1]
        )[0]
        # sessão: agrupa eventos próximos no tempo do mesmo cliente
        session_id = f"{cust}_{i // 15}"
        ts = t0 + i * 30
        rows.append((f"cust_{cust}", event_type, f"ent_{entity}", session_id, ts))

    return pd.DataFrame(
        rows, columns=["customer_id", "event_type", "entity_id", "session_id", "ts"]
    )

log = gen_event_log()
print(f"Log de eventos: {len(log)} linhas, {log.customer_id.nunique()} clientes, "
      f"{log.entity_id.nunique()} entidades\n")

# ---------------------------------------------------------------------------
# 2) MODELAGEM EM GRAFO
#    nós: customer, entity, event_type
#    arestas: customer->entity (ponderada por frequência),
#             entity<->entity (co-ocorrência na mesma sessão)
# ---------------------------------------------------------------------------

G = nx.Graph()

# aresta cliente -> entidade, peso = quantas vezes o cliente interagiu com ela
cust_entity_w = log.groupby(["customer_id", "entity_id"]).size()
for (cust, ent), w in cust_entity_w.items():
    G.add_edge(cust, ent, weight=float(w))

# aresta entidade <-> entidade quando aparecem na mesma sessão
for session_id, grp in log.groupby("session_id"):
    ents = grp["entity_id"].unique()
    for i in range(len(ents)):
        for j in range(i + 1, len(ents)):
            if G.has_edge(ents[i], ents[j]):
                G[ents[i]][ents[j]]["weight"] += 0.5
            else:
                G.add_edge(ents[i], ents[j], weight=0.5)

print(f"Grafo: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas\n")

# ---------------------------------------------------------------------------
# 3) EMBEDDINGS ESTRUTURAIS (proxy rápido para node2vec/GraphSAGE)
#    aqui: espectral, via SVD truncado da matriz de adjacência ponderada.
#    Em produção -> node2vec (random walks + skip-gram) ou GraphSAGE (indutivo)
# ---------------------------------------------------------------------------

nodes = list(G.nodes())
node_idx = {n: i for i, n in enumerate(nodes)}
A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight")

EMB_DIM = 32
svd = TruncatedSVD(n_components=EMB_DIM, random_state=42)
embeddings = svd.fit_transform(A)  # shape (n_nodes, EMB_DIM)

# índice para busca rápida por similaridade (proxy de FAISS/HNSW)
nn_index = NearestNeighbors(n_neighbors=30, metric="cosine").fit(embeddings)

# ---------------------------------------------------------------------------
# 4) RECALL: candidatos rápidos via embeddings
# ---------------------------------------------------------------------------

def is_entity(n):
    return n.startswith("ent_")

# índice separado só com nós de entidade -> candidatos do mesmo "tipo" do que
# queremos entregar ao agente (ex: produtos/páginas relevantes pro cliente)
entity_nodes = [n for n in nodes if is_entity(n)]
entity_emb = embeddings[[node_idx[n] for n in entity_nodes]]
entity_nn_index = NearestNeighbors(n_neighbors=30, metric="cosine").fit(entity_emb)

def recall_candidates(query_node, k=30):
    qi = node_idx[query_node]
    dists, idxs = entity_nn_index.kneighbors(embeddings[qi:qi+1], n_neighbors=k)
    return [entity_nodes[i] for i in idxs[0]]

# ---------------------------------------------------------------------------
# 5) RERANK: Personalized PageRank rodado só no subgrafo local
#    (query + candidatos + vizinhos diretos) -> barato, não toca o grafo todo
# ---------------------------------------------------------------------------

def local_ppr_rerank(query_node, candidates, top_k=10):
    # subgrafo local: candidatos + seus vizinhos imediatos + o próprio query
    local_nodes = set(candidates) | {query_node}
    for c in candidates:
        local_nodes.update(G.neighbors(c))
    subG = G.subgraph(local_nodes)

    personalization = {n: (1.0 if n == query_node else 0.0) for n in subG.nodes()}
    scores = nx.pagerank(subG, alpha=0.85, personalization=personalization,
                          weight="weight", max_iter=100)

    ranked = sorted(
        [(n, s) for n, s in scores.items() if n in candidates],
        key=lambda x: x[1], reverse=True
    )
    return ranked[:top_k]

# ---------------------------------------------------------------------------
# 6) PIPELINE HÍBRIDO COMPLETO + comparação de latência
# ---------------------------------------------------------------------------

def hybrid_retrieve(query_customer, k_recall=30, top_k=10):
    t0 = time.perf_counter()
    candidates = recall_candidates(query_customer, k=k_recall)
    t1 = time.perf_counter()
    ranked = local_ppr_rerank(query_customer, candidates, top_k=top_k)
    t2 = time.perf_counter()
    return ranked, (t1 - t0) * 1000, (t2 - t1) * 1000

def full_graph_ppr(query_customer, top_k=10):
    t0 = time.perf_counter()
    personalization = {n: (1.0 if n == query_customer else 0.0) for n in G.nodes()}
    scores = nx.pagerank(G, alpha=0.85, personalization=personalization, weight="weight")
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranked = [(n, s) for n, s in ranked if n != query_customer][:top_k]
    t1 = time.perf_counter()
    return ranked, (t1 - t0) * 1000

# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------

query = "cust_5"
print(f"=== Consulta: {query} ===\n")

ranked_hybrid, t_recall, t_rerank = hybrid_retrieve(query)
print(f"[HÍBRIDO] recall: {t_recall:.2f}ms | rerank local: {t_rerank:.2f}ms | "
      f"total: {t_recall + t_rerank:.2f}ms")
print("Top nós relevantes (híbrido):")
for n, s in ranked_hybrid:
    print(f"  {n:15s}  score={s:.5f}")

print()
ranked_full, t_full = full_graph_ppr(query)
print(f"[PPR NO GRAFO INTEIRO] total: {t_full:.2f}ms")
print("Top nós relevantes (PPR completo):")
for n, s in ranked_full:
    print(f"  {n:15s}  score={s:.5f}")

overlap = len(set(n for n, _ in ranked_hybrid) & set(n for n, _ in ranked_full))
print(f"\nSobreposição top-10 híbrido vs PPR completo: {overlap}/10")
print(f"Speedup do híbrido: {t_full / (t_recall + t_rerank):.1f}x")

# ---------------------------------------------------------------------------
# 7) MONTAGEM DO CONTEXTO PARA O AGENTE
#    aqui é onde o grafo "vira" texto: pega os nós ranqueados e puxa de volta
#    os eventos originais do log que os explicam, formatando algo que um
#    agente de IA consegue efetivamente ler e usar.
# ---------------------------------------------------------------------------

def build_agent_context(customer_id, ranked_entities, max_events_per_entity=3):
    lines = [f"## Contexto do cliente: {customer_id}"]

    # resumo geral: contagem de eventos por tipo, direto do log
    cust_events = log[log.customer_id == customer_id]
    counts = cust_events.event_type.value_counts()
    resumo = ", ".join(f"{k}: {v}" for k, v in counts.items())
    lines.append(f"Resumo de atividade ({len(cust_events)} eventos totais): {resumo}")
    lines.append("")
    lines.append("### Entidades mais relevantes (ranqueadas por PageRank personalizado)")

    for ent, score in ranked_entities:
        # eventos do PRÓPRIO cliente relacionados a essa entidade
        ev = cust_events[cust_events.entity_id == ent].sort_values("ts", ascending=False)
        if len(ev) == 0:
            # entidade relevante na vizinhança do grafo, mas sem interação
            # direta do cliente -> ainda vale mencionar como "relacionada"
            lines.append(f"- {ent} (relevância={score:.3f}): relacionada por padrão "
                          f"de comportamento similar, sem interação direta registrada")
            continue
        tipos = ", ".join(ev.event_type.value_counts().index[:max_events_per_entity])
        lines.append(f"- {ent} (relevância={score:.3f}): {len(ev)} eventos "
                      f"[{tipos}], último em ts={int(ev.ts.max())}")

    return "\n".join(lines)


context = build_agent_context(query, ranked_hybrid)
print("\n" + "=" * 70)
print("CONTEXTO PRONTO PARA INJETAR NO PROMPT DO AGENTE:")
print("=" * 70)
print(context)