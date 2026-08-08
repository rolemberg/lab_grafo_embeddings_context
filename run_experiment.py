"""
run_experiment.py

Entrypoint único: orquestra o pipeline inteiro de ponta a ponta --
geração de dados -> construção do grafo -> embeddings -> os 4
experimentos (diagnóstico, comparativo, custo).

Por padrão roda com o respondedor de SANITY-CHECK (não é LLM) nos
experimentos que precisam de um "agente" -- é assim que este ambiente
consegue rodar, sem GPU/modelo local. Para medir resultado REAL, troque
answer_fn por experiments.diagnostic.answer_with_local_hf_model e rode
na sua máquina (onde o Granite já está em cache) -- ver bloco
`if __name__ == "__main__"` no final deste arquivo.

O experimento de CUSTO (5.4) não depende de LLM nenhum -- roda
igualmente aqui e na sua máquina, com o mesmo resultado.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

from data.synthetic_events import generate_dataset
from graph.build_graph import build_graph
from embeddings.structural import compute_structural_embeddings, build_type_indexes

from experiments.diagnostic import run_diagnostic_experiment, summarize as summarize_diagnostic
from experiments.comparative import (
    run_comparative_experiment,
    summarize as summarize_comparative,
    summarize_cost as summarize_comparative_cost,
    summarize_verdicts,
)
from experiments.cost import run_cost_experiment, find_crossover
from experiments.multiturn import (
    run_multiturn_experiment,
    summarize_recall as summarize_multiturn_recall,
    summarize_context_size as summarize_multiturn_context_size,
)
from metrics.significance import accuracy_with_ci, mcnemar_by_segment

import config


# ---------------------------------------------------------------------------
# MONTAGEM DO PIPELINE (data -> graph -> embeddings)
# ---------------------------------------------------------------------------

def build_pipeline(n_customers=None, n_events=None, seed=None, verbose=True):
    """Roda data/synthetic_events.py -> graph/build_graph.py ->
    embeddings/structural.py em sequência, medindo tempo de cada etapa.
    Retorna um dict com tudo que os experimentos precisam."""
    seed = seed if seed is not None else config.SEED
    n_customers = n_customers or config.N_CUSTOMERS
    n_events = n_events or config.N_EVENTS

    t0 = time.perf_counter()
    ds = generate_dataset(n_customers=n_customers, n_events=n_events, seed=seed)
    t1 = time.perf_counter()

    G, graph_stats = build_graph(ds)
    t2 = time.perf_counter()

    nodes, node_idx, embeddings = compute_structural_embeddings(G)
    type_indexes = build_type_indexes(nodes, node_idx, embeddings)
    t3 = time.perf_counter()

    if verbose:
        print("=" * 78)
        print("PIPELINE: dados -> grafo -> embeddings")
        print("=" * 78)
        print(f"  geração de dados:    {t1 - t0:6.2f}s | "
              f"{len(ds['log'])} eventos, {ds['log'].global_customer_id_TRUTH.nunique()} clientes ativos")
        print(f"  construção do grafo: {t2 - t1:6.2f}s | "
              f"{graph_stats['n_nodes']} nós, {graph_stats['n_edges']} arestas "
              f"({graph_stats['n_identity_exact_edges']} identidade exata, "
              f"{graph_stats['n_identity_probabilistic_edges']} probabilística)")
        print(f"  embeddings + índices: {t3 - t2:6.2f}s")
        print()

    return {
        "dataset": ds, "G": G, "graph_stats": graph_stats,
        "nodes": nodes, "node_idx": node_idx, "embeddings": embeddings,
        "type_indexes": type_indexes,
    }


# ---------------------------------------------------------------------------
# PERSISTÊNCIA DE RESULTADOS -- cada execução salva um CSV com timestamp,
# pra nunca mais perder um resultado (inclusive rodadas com --granite,
# que custam tempo real pra reproduzir).
# ---------------------------------------------------------------------------

def _save_results(results: dict, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths = []
    for name, df in results.items():
        path = os.path.join(out_dir, f"{name}_{stamp}.csv")
        df.to_csv(path, index=False)
        saved_paths.append(path)
    return saved_paths


# ---------------------------------------------------------------------------
# ORQUESTRAÇÃO DOS 3 EXPERIMENTOS
# ---------------------------------------------------------------------------

def run_all(answer_fn=None, run_diagnostic=True, run_comparative=True, run_cost=True, run_multiturn=True,
            n_customers=None, n_events=None, seed=None, cost_scale_points=None):
    using_naive = answer_fn is None
    if using_naive:
        print("!! Nenhum answer_fn informado -- rodando com respondedor de SANITY-CHECK "
              "(não é LLM). Para resultado real, veja o rodapé deste arquivo. !!\n")

    pipeline = build_pipeline(n_customers=n_customers, n_events=n_events, seed=seed)
    ds, G = pipeline["dataset"], pipeline["G"]
    node_idx, embeddings, type_indexes = pipeline["node_idx"], pipeline["embeddings"], pipeline["type_indexes"]

    results = {}

    if run_diagnostic:
        print("=" * 78)
        print("EXPERIMENTO 1/4 -- DIAGNÓSTICO (seção 5.2: ruído x posição)")
        print("=" * 78)
        diag = run_diagnostic_experiment(ds, answer_fn=answer_fn)
        print(f"Total de trials: {len(diag)} | acerto geral: {diag.correct.mean():.1%}\n")
        print("Acerto por (ruído x posição):")
        print(summarize_diagnostic(diag))
        print()
        results["diagnostic"] = diag

    if run_comparative:
        print("=" * 78)
        print("EXPERIMENTO 2/4 -- COMPARATIVO (seção 5.3: 5 condições)")
        print("=" * 78)
        comp = run_comparative_experiment(ds, G, node_idx, embeddings, type_indexes, answer_fn=answer_fn)
        print(f"Total de trials: {len(comp)}\n")
        print("Acerto por condição x segmento:")
        print(summarize_comparative(comp))
        print("\nCusto (chars médios) por condição x segmento:")
        print(summarize_comparative_cost(comp).round(0))
        print("\nDistribuição de veredito por condição:")
        print(summarize_verdicts(comp).round(2))
        print()
        results["comparative"] = comp

        print("Intervalo de confiança (Wilson, 95%) por condição x segmento:")
        acc_ci = accuracy_with_ci(comp)
        print(acc_ci.round(3).to_string(index=False))
        print()

        print("Teste de McNemar: sessao_memoria vs hibrido_sob_demanda "
              "(pareado por customer_id x fact_type):")
        mcnemar_tbl = mcnemar_by_segment(comp, "sessao_memoria", "hibrido_sob_demanda")
        print(mcnemar_tbl.round(4).to_string(index=False))
        print("(p_value < 0.05 -> a diferença não é explicável por acaso de amostra)")
        print()
        results["comparative_ci"] = acc_ci
        results["comparative_mcnemar"] = mcnemar_tbl

    if run_cost:
        print("=" * 78)
        print("EXPERIMENTO 3/4 -- CUSTO (seção 5.4: varredura de escala, sem LLM)")
        print("=" * 78)
        cost = run_cost_experiment(scale_points=cost_scale_points or config.SCALE_POINTS_N_CUSTOMERS)
        crossover = find_crossover(cost)
        print()
        if crossover is not None:
            print(f"CROSSOVER: híbrido compensa a partir de ~{int(crossover.n_customers)} "
                  f"clientes ({int(crossover.n_nodes)} nós), speedup={crossover.speedup:.2f}x")
        results["cost"] = cost

    if run_multiturn:
        print("=" * 78)
        print("EXPERIMENTO 4/4 -- MULTI-TURNO (seção 5.5: memória de sessão, sem LLM)")
        print("=" * 78)
        mt = run_multiturn_experiment(ds, G, node_idx, embeddings, type_indexes)
        print(f"Total de clientes testados: {mt.customer_id.nunique()}\n")
        print("Recall por tópico e estágio (esperado: sobe quando o tópico é mencionado):")
        print(summarize_multiturn_recall(mt).round(3))
        print("\nTamanho de contexto por estágio (esperado: estável, não crescente):")
        print(summarize_multiturn_context_size(mt).round(1))
        print()
        results["multiturn"] = mt

    if results:
        saved_paths = _save_results(results)
        print("Resultados salvos em:")
        for p in saved_paths:
            print(f"  {p}")
        print()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roda o pipeline completo de experimentos.")
    parser.add_argument("--skip-diagnostic", action="store_true")
    parser.add_argument("--skip-comparative", action="store_true")
    parser.add_argument("--skip-cost", action="store_true")
    parser.add_argument("--skip-multiturn", action="store_true")
    parser.add_argument("--granite", action="store_true",
                         help="Usa o Granite local (ibm-granite/granite-4.1-8b) em vez do "
                              "respondedor de sanity-check. Só funciona na sua máquina, com "
                              "o modelo em cache -- não roda neste ambiente de execução.")
    args = parser.parse_args()

    answer_fn = None
    if args.granite:
        from experiments.diagnostic import answer_with_local_hf_model
        answer_fn = answer_with_local_hf_model

    results = run_all(
        answer_fn=answer_fn,
        run_diagnostic=not args.skip_diagnostic,
        run_comparative=not args.skip_comparative,
        run_cost=not args.skip_cost,
        run_multiturn=not args.skip_multiturn,
    )

    # =========================================================================
    # COMO RODAR COM UM LLM REAL (na sua máquina, com o Granite em cache):
    #
    #   python run_experiment.py --granite
    #
    # Ou dentro de um script/notebook:
    #
    #   from run_experiment import run_all
    #   from experiments.diagnostic import answer_with_local_hf_model
    #   results = run_all(answer_fn=answer_with_local_hf_model)
    #
    # Pra pular etapas (ex: só rodar o comparativo):
    #
    #   python run_experiment.py --granite --skip-diagnostic --skip-cost
    # =========================================================================