"""
agent/tool_contract.py

Define a TOOL que um agente de LLM chamaria via tool-calling para
buscar contexto de cliente sob demanda (seção 4.4 do artigo) -- em vez
de injetar todo o histórico do cliente no prompt de sistema.

Esta é a peça que conecta a infraestrutura de retrieval (graph/,
embeddings/, retrieval/) a um agente de fato: expõe uma função com
assinatura de tool-calling padrão (nome, descrição, schema JSON de
parâmetros) e um dispatcher que recebe uma tool_call no formato usado
por APIs de LLM (compatível com o formato Anthropic /v1/messages) e
devolve texto pronto pra ser injetado como tool_result.

DECISÃO DE DESIGN IMPORTANTE: o resumo cross-canal devolvido pela tool
NÃO usa a identidade verdadeira (data/synthetic_events.py::identities,
que é gabarito de avaliação). Em vez disso, a tool percorre as ARESTAS
DE IDENTIDADE que o próprio grafo descobriu (graph/build_graph.py --
exatas + probabilísticas) a partir do nó de entrada do cliente. Isso
significa que um erro de resolução de identidade se propaga
diretamente pra qualidade do contexto entregue ao agente -- o mesmo
efeito seria observado em produção.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.hybrid import hybrid_retrieve
from config import REFERENCE_CHANNEL, MAX_EVENTS_PER_ENTITY

IDENTITY_EDGE_TYPES = {"identity_exact", "identity_probabilistic"}  # vocabulário
                       # compartilhado com graph/build_graph.py (não é hiperparâmetro)


# ---------------------------------------------------------------------------
# SCHEMA DA TOOL (formato compatível com tool-calling de LLM)
# ---------------------------------------------------------------------------

TOOL_SCHEMA = {
    "name": "buscar_contexto_cliente",
    "description": (
        "Busca contexto relevante sobre um cliente específico: entidades "
        "relevantes (produtos/páginas/campanhas relacionadas ao seu "
        "comportamento), histórico de interação e resumo de atividade. "
        "Use esta ferramenta sempre que precisar de informação específica "
        "sobre um cliente que não está disponível na conversa atual -- não "
        "assuma ou invente informação sobre o cliente sem consultá-la."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "description": "ID canônico do cliente, ex: 'cust_5'.",
            },
            "max_entidades": {
                "type": "integer",
                "description": "Número máximo de entidades relevantes a retornar.",
                "default": 10,
            },
        },
        "required": ["customer_id"],
    },
}


# ---------------------------------------------------------------------------
# RESOLUÇÃO: customer_id -> nó de entrada no grafo
# ---------------------------------------------------------------------------

def _entry_node(customer_id: str, reference_channel: str = REFERENCE_CHANNEL) -> str:
    """Convenção usada em todo o pipeline: o canal de referência (web)
    sempre existe para todo cliente sintético, então o nó de entrada é
    previsível. Em produção, essa resolução seria feita por um serviço
    de identidade externo (ex: procurar pelo customer_id de negócio em
    um índice), não por convenção de nome."""
    idx = customer_id.split("_")[-1]
    return f"{reference_channel}_{idx}"


def _known_channel_nodes(G, entry_node: str) -> set:
    """Percorre SÓ arestas de identidade (exata + probabilística) a
    partir do nó de entrada, coletando os nós customer_local que o
    GRAFO acredita pertencerem ao mesmo cliente. Não usa nenhuma
    informação de gabarito -- só o que já está codificado no grafo."""
    if entry_node not in G:
        return set()

    known = {entry_node}
    frontier = [entry_node]
    while frontier:
        current = frontier.pop()
        for neighbor, edge_data in G[current].items():
            if edge_data.get("edge_type") in IDENTITY_EDGE_TYPES and neighbor not in known:
                known.add(neighbor)
                frontier.append(neighbor)
    return known


# ---------------------------------------------------------------------------
# MONTAGEM DO TEXTO DE CONTEXTO (o que de fato vira tool_result)
# ---------------------------------------------------------------------------

def _build_context_text(customer_id, known_nodes, log, ranked_entities):
    """Monta o texto final -- é aqui que o grafo "vira" texto legível
    pelo agente. Usa só os eventos dos nós customer_local que o grafo
    already resolveu como pertencentes a esse cliente (known_nodes)."""
    lines = [f"## Contexto do cliente: {customer_id}"]

    cust_events = log[log.local_customer_id.isin(known_nodes)]
    n_channels = cust_events.channel.nunique() if len(cust_events) else 0
    lines.append(
        f"Canais com atividade encontrada: {n_channels} "
        f"({', '.join(sorted(cust_events.channel.unique())) if n_channels else 'nenhum'})"
    )

    if len(cust_events) == 0:
        lines.append("Nenhum evento encontrado para este cliente.")
        return "\n".join(lines)

    counts = cust_events.event_type.value_counts()
    resumo = ", ".join(f"{k}: {v}" for k, v in counts.items())
    lines.append(f"Resumo de atividade ({len(cust_events)} eventos totais): {resumo}")
    lines.append("")
    lines.append("### Entidades mais relevantes (ranqueadas por PageRank personalizado)")

    for ent, score in ranked_entities:
        ev = cust_events[cust_events.entity_id == ent].sort_values("ts", ascending=False)
        if len(ev) == 0:
            lines.append(
                f"- {ent} (relevância={score:.3f}): relacionada por padrão de "
                f"comportamento similar, sem interação direta registrada"
            )
            continue
        tipos = ", ".join(ev.event_type.value_counts().index[:MAX_EVENTS_PER_ENTITY])
        lines.append(
            f"- {ent} (relevância={score:.3f}): {len(ev)} eventos [{tipos}], "
            f"último em ts={int(ev.ts.max())}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# API DA TOOL
# ---------------------------------------------------------------------------

def buscar_contexto_cliente(customer_id, G, node_idx, embeddings, type_indexes, log,
                             max_entidades=10, k_recall=30):
    """Implementação da tool. Retorna um dict com o texto pronto pra
    injetar como tool_result, mais metadados úteis pra instrumentar os
    experimentos depois (tamanho do contexto, tempo gasto, quantos
    canais foram encontrados)."""
    entry = _entry_node(customer_id)
    known_nodes = _known_channel_nodes(G, entry)

    ranked_entities, t_recall, t_rerank = hybrid_retrieve(
        entry, G, node_idx, embeddings, type_indexes,
        k_recall=k_recall, top_k=max_entidades, target_type="entity",
    )

    context_text = _build_context_text(customer_id, known_nodes, log, ranked_entities)

    return {
        "context_text": context_text,
        "n_channels_resolved": len(known_nodes),
        "n_char": len(context_text),
        "t_recall_ms": t_recall,
        "t_rerank_ms": t_rerank,
    }


# ---------------------------------------------------------------------------
# DISPATCHER (formato tool_call de API de LLM -> tool_result)
# ---------------------------------------------------------------------------

def call_tool(tool_call: dict, pipeline_state: dict) -> dict:
    """Recebe uma tool_call no formato {"name": ..., "input": {...}}
    (compatível com o bloco `tool_use` da API Anthropic) e devolve um
    bloco de tool_result. `pipeline_state` empacota tudo que a tool
    precisa (G, node_idx, embeddings, type_indexes, log) -- montado
    uma vez por sessão do agente, não a cada chamada."""
    if tool_call["name"] != TOOL_SCHEMA["name"]:
        return {"type": "tool_result", "is_error": True,
                "content": f"Tool desconhecida: {tool_call['name']}"}

    args = tool_call.get("input", {})
    customer_id = args.get("customer_id")
    if not customer_id:
        return {"type": "tool_result", "is_error": True,
                "content": "Parâmetro obrigatório ausente: customer_id"}

    max_entidades = args.get("max_entidades", 10)

    result = buscar_contexto_cliente(
        customer_id,
        pipeline_state["G"], pipeline_state["node_idx"],
        pipeline_state["embeddings"], pipeline_state["type_indexes"],
        pipeline_state["log"],
        max_entidades=max_entidades,
    )
    return {"type": "tool_result", "is_error": False, "content": result["context_text"]}


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

    pipeline_state = {
        "G": G, "node_idx": node_idx, "embeddings": embeddings,
        "type_indexes": type_indexes, "log": ds["log"],
    }

    print("=== Chamando a tool diretamente ===\n")
    result = buscar_contexto_cliente("cust_5", G, node_idx, embeddings, type_indexes, ds["log"])
    print(result["context_text"])
    print(f"\n[metadados] canais resolvidos={result['n_channels_resolved']} | "
          f"tamanho={result['n_char']} chars | "
          f"recall={result['t_recall_ms']:.2f}ms | rerank={result['t_rerank_ms']:.2f}ms")

    print("\n\n=== Chamando via dispatcher (formato tool_use de API) ===\n")
    tool_call = {"name": "buscar_contexto_cliente", "input": {"customer_id": "cust_5", "max_entidades": 5}}
    tool_result = call_tool(tool_call, pipeline_state)
    print(f"is_error={tool_result['is_error']}")
    print(tool_result["content"])

    # --- comparação com "jogar tudo no prompt" (motivação do artigo) ---
    # IMPORTANTE: essa comparação só faz sentido em função do volume de
    # histórico do cliente -- por isso comparamos um cliente leve (perto
    # da mediana) com o mais pesado da cauda longa, em vez de um único
    # cliente fixo, que poderia cair em qualquer ponto da distribuição.
    print("\n\n=== Comparação: contexto sob demanda vs. despejar tudo (ground truth) ===")
    identities = ds["identities"]
    events_per_customer = ds["log"].global_customer_id_TRUTH.value_counts()

    def compare_for(customer_id, label):
        result = buscar_contexto_cliente(customer_id, G, node_idx, embeddings, type_indexes, ds["log"])
        true_channel_nodes = set(identities[customer_id].channel_ids.values())
        full_dump = ds["log"][ds["log"].local_customer_id.isin(true_channel_nodes)]
        full_dump_text = full_dump.to_csv(index=False)
        reducao = 1 - result["n_char"] / len(full_dump_text) if len(full_dump_text) else float("nan")
        print(f"\n[{label}] {customer_id} -- {events_per_customer.get(customer_id, 0)} eventos totais")
        print(f"  Contexto sob demanda (tool):  {result['n_char']:7d} caracteres")
        print(f"  Despejo bruto de tudo (CSV):  {len(full_dump_text):7d} caracteres")
        print(f"  Redução: {reducao:.1%}")

    median_customer = events_per_customer.index[len(events_per_customer) // 2]
    heaviest_customer = events_per_customer.index[0]
    compare_for(median_customer, "cliente mediano")
    compare_for(heaviest_customer, "cliente mais engajado (cauda longa)")