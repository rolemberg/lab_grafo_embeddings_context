"""
experiments/cost.py

Experimento de CUSTO (seção 5.4 do artigo): varredura de escala pra
encontrar o ponto de CROSSOVER onde o retrieval híbrido (recall + PPR
local) realmente compensa vs. rodar PPR personalizado no grafo inteiro.

MOTIVAÇÃO DIRETA: em retrieval/hybrid.py, medimos speedup de só 1.6x
num grafo de ~1940 nós -- pequeno demais pra o overhead da construção
do subgrafo valer a pena. A pergunta real não é "o híbrido é mais
rápido?", é "A PARTIR DE QUE ESCALA o híbrido compensa?" -- essa é a
curva que este módulo produz.

TAMBÉM MEDIMOS um segundo gargalo, descoberto ao construir isto: a
resolução de identidade probabilística (graph/build_graph.py) compara
cada cliente sem chave exata contra TODOS os clientes do canal de
referência -- O(n²) em número de clientes. Isso pode se tornar o
verdadeiro teto de escala do pipeline, antes mesmo do PPR. Por isso
medimos o tempo de CONSTRUÇÃO DO GRAFO separado do tempo de RETRIEVAL.
"""

from __future__ import annotations

import os
import random
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_events import generate_dataset
from graph.build_graph import build_graph
from embeddings.structural import compute_structural_embeddings, build_type_indexes
from retrieval.hybrid import hybrid_retrieve, full_graph_ppr

# ---------------------------------------------------------------------------
# CONFIG DA VARREDURA
# ---------------------------------------------------------------------------

from config import SEED, SCALE_POINTS_N_CUSTOMERS, EVENTS_PER_CUSTOMER_RATIO, N_QUERIES_PER_SCALE, N_REPETITIONS_PER_SCALE_POINT


# ---------------------------------------------------------------------------
# UMA MEDIÇÃO POR PONTO DE ESCALA
# ---------------------------------------------------------------------------

def _measure_scale_point(n_customers, n_queries=N_QUERIES_PER_SCALE, seed=SEED, emb_dim=None):
    from config import EMB_DIM
    emb_dim = emb_dim or EMB_DIM
    n_events = n_customers * EVENTS_PER_CUSTOMER_RATIO

    t0 = time.perf_counter()
    ds = generate_dataset(n_customers=n_customers, n_events=n_events, seed=seed)
    t1 = time.perf_counter()
    t_data_gen = t1 - t0

    t0 = time.perf_counter()
    G, graph_stats = build_graph(ds)
    t1 = time.perf_counter()
    t_build_graph = t1 - t0

    t0 = time.perf_counter()
    nodes, node_idx, embeddings = compute_structural_embeddings(G, emb_dim=emb_dim)
    type_indexes = build_type_indexes(nodes, node_idx, embeddings)
    t1 = time.perf_counter()
    t_embeddings = t1 - t0

    # amostra clientes-web existentes como consultas
    web_nodes = [n for n in nodes if n.startswith("web_")]
    rng = random.Random(seed)
    queries = rng.sample(web_nodes, k=min(n_queries, len(web_nodes)))

    hybrid_times, full_times, overlaps = [], [], []
    for q in queries:
        ranked_hybrid, t_recall, t_rerank = hybrid_retrieve(
            q, G, node_idx, embeddings, type_indexes, target_type="entity",
        )
        ranked_full, t_full = full_graph_ppr(G, q, target_type="entity")

        hybrid_times.append(t_recall + t_rerank)
        full_times.append(t_full)

        top_hybrid = set(n for n, _ in ranked_hybrid)
        top_full = set(n for n, _ in ranked_full)
        denom = max(len(top_hybrid), 1)
        overlaps.append(len(top_hybrid & top_full) / denom)

    avg_hybrid = sum(hybrid_times) / len(hybrid_times)
    avg_full = sum(full_times) / len(full_times)

    return {
        "n_customers": n_customers,
        "n_nodes": graph_stats["n_nodes"],
        "n_edges": graph_stats["n_edges"],
        "t_data_gen_s": t_data_gen,
        "t_build_graph_s": t_build_graph,
        "t_embeddings_s": t_embeddings,
        "avg_t_hybrid_ms": avg_hybrid,
        "avg_t_full_ppr_ms": avg_full,
        "speedup": avg_full / avg_hybrid if avg_hybrid > 0 else float("nan"),
        "avg_overlap_top10": sum(overlaps) / len(overlaps),
    }


# ---------------------------------------------------------------------------
# REPETIÇÃO + AGREGAÇÃO POR PONTO DE ESCALA
#
# UMA medição só de latência é ruído de carga de máquina disfarçado de
# resultado -- vimos isso na prática: o mesmo n_customers=600, mesmo
# SEED, deu speedup=2.23x numa execução e 0.98x na seguinte. Por isso
# cada ponto de escala roda N_REPETITIONS_PER_SCALE_POINT vezes (variando
# o seed em cada repetição -- reprodutível, mas não a mesma amostra de
# consultas toda vez) e reporta a MEDIANA, não uma execução isolada.
# ---------------------------------------------------------------------------

def _measure_scale_point_repeated(n_customers, n_queries=N_QUERIES_PER_SCALE,
                                   seed=SEED, n_repeats=N_REPETITIONS_PER_SCALE_POINT, emb_dim=None):
    reps = [
        _measure_scale_point(n_customers, n_queries=n_queries, seed=seed + rep, emb_dim=emb_dim)
        for rep in range(n_repeats)
    ]
    reps_df = pd.DataFrame(reps)

    agg = {
        "n_customers": n_customers,
        "n_nodes": reps_df["n_nodes"].median(),
        "n_edges": reps_df["n_edges"].median(),
        "t_data_gen_s": reps_df["t_data_gen_s"].median(),
        "t_build_graph_s": reps_df["t_build_graph_s"].median(),
        "t_embeddings_s": reps_df["t_embeddings_s"].median(),
        "avg_t_hybrid_ms": reps_df["avg_t_hybrid_ms"].median(),
        "avg_t_full_ppr_ms": reps_df["avg_t_full_ppr_ms"].median(),
        "speedup": reps_df["speedup"].median(),
        "speedup_std": reps_df["speedup"].std(),
        "speedup_min": reps_df["speedup"].min(),
        "speedup_max": reps_df["speedup"].max(),
        "avg_overlap_top10": reps_df["avg_overlap_top10"].median(),
        "n_repeats": n_repeats,
    }
    return agg


# ---------------------------------------------------------------------------
# VARREDURA COMPLETA
# ---------------------------------------------------------------------------

def run_cost_experiment(scale_points=SCALE_POINTS_N_CUSTOMERS, n_queries=N_QUERIES_PER_SCALE,
                         seed=SEED, n_repeats=N_REPETITIONS_PER_SCALE_POINT, emb_dim=None):
    rows = []
    for n_customers in scale_points:
        print(f"[cost.py] medindo escala n_customers={n_customers} ({n_repeats} repetições, "
              f"emb_dim={emb_dim or 'config.EMB_DIM'}) ...")
        row = _measure_scale_point_repeated(n_customers, n_queries=n_queries, seed=seed,
                                             n_repeats=n_repeats, emb_dim=emb_dim)
        rows.append(row)
        print(f"  -> n_nodes={row['n_nodes']:.0f} | build_graph={row['t_build_graph_s']:.2f}s | "
              f"speedup mediana={row['speedup']:.2f}x (min={row['speedup_min']:.2f}x, "
              f"max={row['speedup_max']:.2f}x, std={row['speedup_std']:.2f}) | "
              f"overlap={row['avg_overlap_top10']:.1%}")
    return pd.DataFrame(rows)


def find_crossover(results: pd.DataFrame):
    """Primeiro ponto de escala em que speedup > 1.0 -- onde o híbrido
    passa a compensar em relação ao PPR no grafo inteiro."""
    above_one = results[results.speedup > 1.0]
    if len(above_one) == 0:
        return None
    return above_one.iloc[0]


# ---------------------------------------------------------------------------
# DEMO / EXECUÇÃO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_cost_experiment()

    print("\n" + "=" * 90)
    print("RESULTADO DA VARREDURA DE ESCALA")
    print("=" * 90)
    cols = ["n_customers", "n_nodes", "n_edges", "t_build_graph_s",
            "avg_t_hybrid_ms", "avg_t_full_ppr_ms", "speedup", "speedup_std", "avg_overlap_top10"]
    print(results[cols].to_string(index=False))
    print(f"\n(speedup = mediana de {int(results.n_repeats.iloc[0])} repetições por ponto de escala; "
          f"speedup_std = desvio padrão entre repetições -- quanto maior, menos confiável o ponto)")

    crossover = find_crossover(results)
    print("\n" + "=" * 90)
    if crossover is not None:
        print(f"CROSSOVER: híbrido passa a compensar a partir de "
              f"~{int(crossover.n_customers)} clientes ({int(crossover.n_nodes)} nós), "
              f"speedup={crossover.speedup:.2f}x")
    else:
        print("Nenhum ponto testado teve speedup > 1.0 -- o híbrido não compensou "
              "em NENHUMA das escalas testadas. Considere testar escalas maiores "
              "(editar SCALE_POINTS_N_CUSTOMERS) antes de descartar o método.")

    # aviso sobre o outro gargalo (construção do grafo / resolução de identidade)
    print("\nCusto de CONSTRUÇÃO DO GRAFO (inclui resolução de identidade O(n²)):")
    print(results[["n_customers", "t_data_gen_s", "t_build_graph_s", "t_embeddings_s"]].to_string(index=False))

    # --- gráfico (salvo como arquivo, não plt.show -- ambiente sem display) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        axes[0].errorbar(results.n_nodes, results.speedup,
                          yerr=[results.speedup - results.speedup_min, results.speedup_max - results.speedup],
                          marker="o", capsize=3)
        axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=1)
        axes[0].set_xlabel("nº de nós no grafo")
        axes[0].set_ylabel("speedup mediana (min-max)")
        axes[0].set_title("Speedup do híbrido vs. escala do grafo")

        axes[1].plot(results.n_nodes, results.t_build_graph_s, marker="o", label="construção do grafo")
        axes[1].plot(results.n_nodes, results.t_data_gen_s, marker="s", label="geração de dados")
        axes[1].set_xlabel("nº de nós no grafo")
        axes[1].set_ylabel("tempo (s)")
        axes[1].set_title("Custo de construção (inclui resolução de identidade)")
        axes[1].legend()

        fig.tight_layout()
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost_scaling.png")
        fig.savefig(out_path, dpi=120)
        print(f"\nGráfico salvo em: {out_path}")
    except ImportError:
        print("\n(matplotlib não disponível -- pulei a geração do gráfico)")