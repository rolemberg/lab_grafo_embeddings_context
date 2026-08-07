"""
graph/build_graph.py

Constrói o grafo G a partir do log de eventos multi-canal gerado por
data/synthetic_events.py.

DECISÃO DE DESIGN: a "unificação" entre canais NÃO colapsa nós antes de
montar o grafo. Em vez disso, cada (cliente, canal) é um nó separado, e
a ligação entre canais é representada como uma ARESTA DE IDENTIDADE:
  - exata: peso fixo alto (chave compartilhada, ex: loyalty_id)
  - probabilística: peso = score de similaridade (comportamental +
    textual), só adicionada acima de um limiar
Isso deixa a incerteza de identidade explícita no grafo, em vez de
forçar uma decisão binária de "é a mesma pessoa ou não" antes do
retrieval -- o PPR local naturalmente propaga menos relevância através
de uma aresta de identidade fraca do que de uma forte.

Tipos de nó:
  - customer local: "web_5", "app_anon_12", "cc_anon_7", ...
  - entity: "ent_12", ...
  - event_type: "view", "purchase", ...

Tipos de aresta:
  - customer_local -> entity            (peso = freq. de interação)
  - entity <-> entity                   (co-ocorrência de sessão)
  - customer_local -> event_type        (peso = freq.)
  - customer_local <-> customer_local   (identidade entre canais)

IMPORTANTE: a resolução de identidade probabilística aqui NUNCA olha
`global_customer_id_TRUTH` -- essa coluna é gabarito de avaliação. O
pipeline só enxerga `local_customer_id`, `channel` e o texto ruidoso
associado (`noisy_text`), simulando o que um sistema real teria
disponível antes de qualquer resolução.

NOTA DE PRODUÇÃO: este grafo é montado em memória (networkx) a cada
execução -- adequado para o experimento (grafo pequeno, reprodutibilidade
sem infra externa). Em produção, o mesmo modelo de nós/arestas mapearia
diretamente para um banco de grafos com PPR nativo (ex: Neo4j + GDS,
que já implementa Personalized PageRank sobre subgrafo induzido -- a
mesma ideia da seção 4.3, só que nativa do banco), permitindo
persistência incremental do grafo à medida que novos eventos chegam,
em vez de reconstrução do zero.
"""

from __future__ import annotations

import difflib
import os
import sys
from collections import defaultdict

import networkx as nx

# permite rodar tanto via `python -m graph.build_graph` quanto direto
# (`python graph/build_graph.py`) -- ambos precisam achar embeddings.semantic
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embeddings.semantic import fit_text_vectorizer, similarity as semantic_similarity

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

import numpy as np

from config import (
    IDENTITY_EXACT_WEIGHT, IDENTITY_PROB_THRESHOLD, IDENTITY_TOPK_CANDIDATES,
    W_ENTITY_OVERLAP, W_TEXT_SIM, REFERENCE_CHANNEL, EDGE_RECENCY_HALF_LIFE_SECONDS,
)

# NOTA sobre IDENTITY_PROB_THRESHOLD (valor em config.py, hoje 0.40):
# recalibrado para a escala do score TF-IDF (embeddings/semantic.py substituiu
# o difflib). Após introduzir a distribuição de engajamento em cauda longa em
# data/synthetic_events.py, a precisão nesse ponto caiu de 61.1% -> 19.7%.
# Isso NÃO é regressão a "consertar" subindo o limiar -- é o efeito esperado
# de esparsidade: a maioria dos clientes agora tem poucos eventos (mediana
# ~6), então o sinal de overlap comportamental usado no blocking fica fraco
# demais pra desambiguar identidade com confiança. Material real para a
# seção 7 (limitações): a qualidade da resolução de identidade depende
# diretamente de quanto histórico cada fragmento de canal acumulou --
# clientes de cauda longa são estruturalmente mais difíceis de resolver,
# não só um problema de calibração de limiar.


# ---------------------------------------------------------------------------
# ARESTAS DE INTERAÇÃO (cliente-local -> entidade, entidade <-> entidade)
# ---------------------------------------------------------------------------

def _recency_weighted_group_sum(log, group_cols, half_life_seconds=EDGE_RECENCY_HALF_LIFE_SECONDS):
    """Soma por grupo, mas cada linha contribui exp(-idade * ln2 / meia_vida)
    em vez de 1.0. "Idade" é relativa ao ts mais recente do log inteiro.

    half_life_seconds=None -> comportamento antigo (equivalente a .size()),
    útil pra rodar a ablation "com vs. sem recência" sem duplicar código.
    """
    if half_life_seconds is None:
        return log.groupby(group_cols).size().astype(float)

    ref_ts = log["ts"].max()
    age = ref_ts - log["ts"]
    decay_rate = np.log(2) / half_life_seconds
    contrib = np.exp(-decay_rate * age)
    return log.assign(_contrib=contrib).groupby(group_cols)["_contrib"].sum()


def add_interaction_edges(G, log, half_life_seconds=EDGE_RECENCY_HALF_LIFE_SECONDS):
    """customer_local -> entity, peso = soma ponderada por recência de
    quantas vezes esse cliente-local interagiu com essa entidade (ver
    _recency_weighted_group_sum -- half_life_seconds=None recupera a
    contagem bruta antiga)."""
    w = _recency_weighted_group_sum(log, ["local_customer_id", "entity_id"], half_life_seconds)
    for (cust, ent), weight in w.items():
        G.add_edge(cust, ent, weight=float(weight), edge_type="interaction")


def add_session_cooccurrence_edges(G, log):
    """entity <-> entity quando aparecem na mesma sessão (sessão já é
    escopada por canal no gerador, então isso não vaza sinal entre
    canais -- só reforça estrutura de nicho dentro de uma sessão real)."""
    for session_id, grp in log.groupby("session_id"):
        ents = grp["entity_id"].unique()
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                a, b = ents[i], ents[j]
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 0.5
                else:
                    G.add_edge(a, b, weight=0.5, edge_type="session_cooccurrence")


def add_event_type_edges(G, log, half_life_seconds=EDGE_RECENCY_HALF_LIFE_SECONDS):
    """customer_local -> event_type, peso = soma ponderada por recência.
    Permite que o PPR capture "esse cliente ANDA abrindo muito chamado de
    suporte" (peso puxado pro recente) em vez de "esse cliente abriu
    muito chamado alguma vez" (contagem histórica pura) -- sinal
    estrutural, não só via entidade."""
    w = _recency_weighted_group_sum(log, ["local_customer_id", "event_type"], half_life_seconds)
    for (cust, et), weight in w.items():
        G.add_edge(cust, et, weight=float(weight), edge_type="event_type")


# ---------------------------------------------------------------------------
# ARESTAS DE IDENTIDADE -- EXATA
# ---------------------------------------------------------------------------

def add_exact_identity_edges(G, identities):
    """Conecta nós customer_local entre canais quando ambos têm chave
    exata (ex: loyalty_id compartilhado entre web e app)."""
    n_edges = 0
    for ident in identities.values():
        exact_channels = [ch for ch, has_key in ident.has_exact_key.items() if has_key]
        for i in range(len(exact_channels)):
            for j in range(i + 1, len(exact_channels)):
                a = ident.channel_ids[exact_channels[i]]
                b = ident.channel_ids[exact_channels[j]]
                G.add_edge(a, b, weight=IDENTITY_EXACT_WEIGHT,
                           edge_type="identity_exact")
                n_edges += 1
    return n_edges


# ---------------------------------------------------------------------------
# ARESTAS DE IDENTIDADE -- PROBABILÍSTICA
# ---------------------------------------------------------------------------

def _entity_sets_by_local_customer(log):
    sets = defaultdict(set)
    for local_id, ent in zip(log.local_customer_id, log.entity_id):
        sets[local_id].add(ent)
    return sets


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _text_similarity(vectorizer, a: str, b: str) -> float:
    """Similaridade semântica via TF-IDF de n-gramas de caractere
    (embeddings/semantic.py). Produção: embedding treinado (ex:
    sentence-transformers) sobre nome/telefone/endereço concatenados."""
    return semantic_similarity(vectorizer, a, b)


def resolve_and_add_probabilistic_identity_edges(
    G, identities, log,
    reference_channel=REFERENCE_CHANNEL,
    threshold=IDENTITY_PROB_THRESHOLD,
    top_k=IDENTITY_TOPK_CANDIDATES,
):
    """Para cada nó customer_local SEM chave exata, gera candidatos via
    blocking (overlap de entidades com clientes do canal de referência),
    depois refina com similaridade textual, e cria aresta de identidade
    probabilística para o melhor candidato acima do limiar.

    Retorna um DataFrame de auditoria (local_id, canal, candidato
    escolhido, score, acertou?) -- o "acertou?" só existe aqui para
    avaliação offline, nunca é usado para decidir a aresta.
    """
    entity_sets = _entity_sets_by_local_customer(log)

    # ajusta o vetorizador semântico UMA VEZ sobre todo o corpus de texto
    # ruidoso disponível (todos os canais, todos os clientes) -- é assim
    # que TF-IDF deve ser usado, nunca refeito por par
    all_texts = [t for ident in identities.values() for t in ident.noisy_text.values()]
    vectorizer = fit_text_vectorizer(all_texts)

    # pool de referência: todos os customer_local do canal-âncora
    reference_pool = []
    reference_text = {}
    reference_truth = {}
    for gid, ident in identities.items():
        if reference_channel in ident.channel_ids:
            local_id = ident.channel_ids[reference_channel]
            reference_pool.append(local_id)
            reference_text[local_id] = ident.noisy_text[reference_channel]
            reference_truth[local_id] = gid

    audit_rows = []
    n_edges = 0

    for gid, ident in identities.items():
        for channel, has_key in ident.has_exact_key.items():
            if has_key or channel == reference_channel:
                continue  # já tem aresta exata, ou é o próprio canal-âncora

            local_id = ident.channel_ids[channel]
            local_entities = entity_sets.get(local_id, set())
            local_text = ident.noisy_text[channel]

            # --- blocking: rankeia candidatos por overlap comportamental
            scored = []
            for ref_id in reference_pool:
                ref_entities = entity_sets.get(ref_id, set())
                overlap = _jaccard(local_entities, ref_entities)
                if overlap > 0:
                    scored.append((ref_id, overlap))
            scored.sort(key=lambda x: x[1], reverse=True)
            candidates = scored[:top_k]

            # --- refina com similaridade textual (TF-IDF) só nos top-K
            best_id, best_score = None, 0.0
            for ref_id, overlap in candidates:
                text_sim = _text_similarity(vectorizer, local_text, reference_text[ref_id])
                combined = W_ENTITY_OVERLAP * overlap + W_TEXT_SIM * text_sim
                if combined > best_score:
                    best_id, best_score = ref_id, combined

            correct = best_id is not None and reference_truth.get(best_id) == gid
            audit_rows.append((local_id, channel, best_id, best_score, correct))

            if best_id is not None and best_score >= threshold:
                G.add_edge(local_id, best_id, weight=best_score,
                           edge_type="identity_probabilistic")
                n_edges += 1

    import pandas as pd
    audit = pd.DataFrame(audit_rows, columns=[
        "local_id", "channel", "matched_reference", "score", "correct_match",
    ])
    return audit, n_edges


# ---------------------------------------------------------------------------
# API DE ALTO NÍVEL
# ---------------------------------------------------------------------------

def build_graph(dataset):
    """dataset = saída de data.synthetic_events.generate_dataset()"""
    log = dataset["log"]
    identities = dataset["identities"]

    G = nx.Graph()

    add_interaction_edges(G, log)
    add_session_cooccurrence_edges(G, log)
    add_event_type_edges(G, log)
    n_exact = add_exact_identity_edges(G, identities)
    audit, n_prob = resolve_and_add_probabilistic_identity_edges(G, identities, log)

    stats = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_identity_exact_edges": n_exact,
        "n_identity_probabilistic_edges": n_prob,
        "identity_resolution_audit": audit,
    }
    return G, stats


# ---------------------------------------------------------------------------
# DEMO / SANITY CHECK
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data.synthetic_events import generate_dataset

    ds = generate_dataset()
    G, stats = build_graph(ds)

    print(f"Grafo: {stats['n_nodes']} nós, {stats['n_edges']} arestas")
    print(f"  arestas de identidade exata:          {stats['n_identity_exact_edges']}")
    print(f"  arestas de identidade probabilística:  {stats['n_identity_probabilistic_edges']}")

    audit = stats["identity_resolution_audit"]
    attempted = audit[audit.matched_reference.notna()]
    linked = attempted[attempted.score >= IDENTITY_PROB_THRESHOLD]
    print(f"\n=== Auditoria de resolução de identidade probabilística ===")
    print(f"  Total de vínculos sem chave exata: {len(audit)}")
    print(f"  Candidato encontrado (score>0):    {len(attempted)}")
    print(f"  Acima do limiar ({IDENTITY_PROB_THRESHOLD}) -> aresta criada: {len(linked)}")
    if len(linked) > 0:
        precision = linked.correct_match.mean()
        print(f"  Precisão entre os que viraram aresta: {precision:.1%}")
    if len(attempted) > 0:
        recall_at_threshold = len(linked[linked.correct_match]) / len(attempted[attempted.correct_match]) \
            if attempted.correct_match.sum() > 0 else float("nan")
        print(f"  Recall (dos que tinham match correto possível): {recall_at_threshold:.1%}")

    print("\nExemplos de decisões (5 primeiras):")
    print(audit.head(5).to_string(index=False))