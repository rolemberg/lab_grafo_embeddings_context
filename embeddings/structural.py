"""
embeddings/structural.py

Transforma o grafo (graph/build_graph.py) em embeddings estruturais e um
índice de busca rápida por similaridade -- a etapa de RECALL do
pipeline híbrido (seção 4.2/4.3 do artigo).

Método: SVD truncado sobre a matriz de adjacência ponderada, como proxy
rápido para node2vec (random walks + skip-gram) ou GraphSAGE (indutivo).
Ver nota de produção no final do arquivo.

PONTO IMPORTANTE DE DESIGN: como a "unificação" entre canais não colapsa
nós (decisão tomada em graph/build_graph.py), o embedding espectral
propaga sinal entre "web_5", "app_5", "cc_anon_5" automaticamente
através das arestas de identidade -- não precisa de lógica extra aqui
para "juntar" o cliente entre canais. Uma aresta de identidade fraca
(probabilística, peso baixo) propaga menos sinal que uma forte (exata,
peso alto) -- a incerteza de identidade se traduz diretamente em
distância no espaço de embedding.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEED, EMB_DIM, N_NEIGHBORS_DEFAULT


# ---------------------------------------------------------------------------
# EMBEDDING ESTRUTURAL
# ---------------------------------------------------------------------------

def compute_structural_embeddings(G, emb_dim=EMB_DIM, seed=SEED):
    """Calcula embedding espectral (SVD truncado) para todo nó do grafo.

    Retorna:
      nodes:      lista de nós, na ordem usada pela matriz de embeddings
      node_idx:   {nó: índice na matriz}
      embeddings: array (n_nodes, emb_dim)
    """
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight")

    # SVD exige n_components < n_features; protege contra grafos pequenos
    n_components = min(emb_dim, A.shape[1] - 1) if A.shape[1] > 1 else 1

    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    embeddings = svd.fit_transform(A)

    return nodes, node_idx, embeddings


# ---------------------------------------------------------------------------
# ÍNDICES DE RECALL (por tipo de nó)
# ---------------------------------------------------------------------------

def node_type(n: str) -> str:
    """Versão pública de _node_type -- usada por outros módulos
    (ex: retrieval/hybrid.py) que precisam filtrar nós por tipo."""
    return _node_type(n)


def _node_type(n: str) -> str:
    """Classifica o nó pelo prefixo do id -- convenção usada em todo o
    pipeline (data/synthetic_events.py, graph/build_graph.py)."""
    if n.startswith("ent_"):
        return "entity"
    if n.startswith(("web_", "app_", "cc_anon_", "app_anon_")):
        return "customer_local"
    return "event_type"  # o que sobra: view, purchase, search, ...


def build_type_indexes(nodes, node_idx, embeddings, n_neighbors=N_NEIGHBORS_DEFAULT):
    """Constrói um índice kNN separado por tipo de nó -- essencial pra
    recall só retornar candidatos do tipo certo (ex: se o agente quer
    "entidades relevantes", não queremos que o kNN devolva outro cliente
    ou um tipo de evento).

    Retorna: {tipo: {"node_ids": [...], "index": NearestNeighbors, "emb": array}}
    """
    by_type = {}
    for n in nodes:
        by_type.setdefault(_node_type(n), []).append(n)

    indexes = {}
    for node_type, type_nodes in by_type.items():
        idxs = [node_idx[n] for n in type_nodes]
        type_emb = embeddings[idxs]
        k = min(n_neighbors, len(type_nodes))
        nn_index = NearestNeighbors(n_neighbors=k, metric="cosine").fit(type_emb)
        indexes[node_type] = {
            "node_ids": type_nodes,
            "index": nn_index,
            "emb": type_emb,
        }
    return indexes


def recall_candidates(query_node, node_idx, embeddings, type_indexes,
                       target_type="entity", k=30):
    """Dado um nó de consulta (ex: 'web_5'), devolve os k nós de
    `target_type` mais próximos no espaço de embedding -- candidatos
    rápidos, antes do rerank fino via PPR local."""
    if query_node not in node_idx:
        return []
    qi = node_idx[query_node]
    query_vec = embeddings[qi:qi + 1]

    bucket = type_indexes[target_type]
    k = min(k, len(bucket["node_ids"]))
    _, idxs = bucket["index"].kneighbors(query_vec, n_neighbors=k)
    return [bucket["node_ids"][i] for i in idxs[0]]


# ---------------------------------------------------------------------------
# DEMO / SANITY CHECK
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.synthetic_events import generate_dataset
    from graph.build_graph import build_graph

    ds = generate_dataset()
    G, graph_stats = build_graph(ds)
    print(f"Grafo: {graph_stats['n_nodes']} nós, {graph_stats['n_edges']} arestas")

    nodes, node_idx, embeddings = compute_structural_embeddings(G)
    print(f"Embeddings: {embeddings.shape}")

    type_indexes = build_type_indexes(nodes, node_idx, embeddings)
    print("Nós por tipo:")
    for t, bucket in type_indexes.items():
        print(f"  {t:15s}: {len(bucket['node_ids'])}")

    # --- sanity check 1: recall básico a partir de um cliente-local
    query = "web_5"
    candidates = recall_candidates(query, node_idx, embeddings, type_indexes,
                                    target_type="entity", k=10)
    print(f"\nTop-10 entidades recall'adas a partir de '{query}':")
    print(f"  {candidates}")

    # --- sanity check 2: a "unificação por aresta" está funcionando?
    # se web_5 e app_5 têm identidade exata, seus embeddings devem estar
    # próximos no espaço vetorial (mais próximos que web_5 vs um cliente
    # aleatório sem ligação nenhuma).
    identities = ds["identities"]
    ident5 = identities["cust_5"]
    if "app" in ident5.channel_ids and ident5.has_exact_key.get("app"):
        web_vec = embeddings[node_idx["web_5"]]
        app_vec = embeddings[node_idx[ident5.channel_ids["app"]]]
        random_vec = embeddings[node_idx["web_100"]] if "web_100" in node_idx else None

        def cos_sim(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        sim_same = cos_sim(web_vec, app_vec)
        print(f"\nSanity check -- similaridade cosseno:")
        print(f"  web_5 vs app_5 (mesma identidade, chave exata): {sim_same:.4f}")
        if random_vec is not None:
            sim_random = cos_sim(web_vec, random_vec)
            print(f"  web_5 vs web_100 (cliente diferente):          {sim_random:.4f}")