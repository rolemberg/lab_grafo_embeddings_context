"""
agent/session_memory.py
 
Camada de MEMÓRIA CURTA que envolve o retrieval híbrido existente
(retrieval/hybrid.py) e a tool de contexto (agent/tool_contract.py) numa
sessão de atendimento com estado -- fechando três lacunas entre a
arquitetura conceitual (grafo por cliente + hipocampo/conteúdo) e o
pipeline single-shot original.
 
1) MEMÓRIA CURTA / SEED EVOLUTIVO
   SessionMemory mantém um vetor de personalização (seed_weights) que
   acumula massa ao longo dos turnos de uma ligação -- não é mais um PPR
   de tiro único por chamada de tool. O TEXTO injetado é sempre
   REGENERADO do zero a cada chamada de get_context() e SUBSTITUI o
   anterior -- nunca empilha -- para não recriar o problema de context
   distraction que motivou toda a arquitetura.
 
2) CATEGORIA / DOMÍNIO PARA COLD-START
   Quando o cliente introduz um tópico sem histórico direto (ex: nunca
   abriu support_ticket antes), o seed cai para o nó de event_type
   correspondente -- um hub já presente no grafo, compartilhado por
   todos os clientes (graph/build_graph.py::add_event_type_edges) --
   em vez de não ter nenhum nó equivalente pra ativar.
   TOPIC_TO_EVENT_TYPE abaixo é um placeholder determinístico por
   palavra-chave; em produção, isso é a saída de um classificador de
   intenção (ou, no domínio de telco discutido no artigo, os campos já
   estruturados de intent/journey_details -- sem precisar de NLP).
 
3) CONTEXT CLASH (prioridade por recência)
   Quando o PPR ativa múltiplos eventos referentes à MESMA entidade, o
   texto final apresenta só o mais recente como fato vigente,
   comprimindo os demais numa nota de contagem -- em vez de listar
   estados possivelmente conflitantes lado a lado.
"""

from __future__ import annotations

import os 
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field

from agent.tool_contract import _entry_node, _known_channel_nodes
from retrieval.hybrid import hybrid_retrieve_from_weights
from config import MAX_EVENTS_PER_ENTITY, EDGE_RECENCY_HALF_LIFE_SECONDS


# ---------------------------------------------------------------------------
# CONFIG DESTE MÓDULO
# ---------------------------------------------------------------------------
 
SHORT_TERM_DECAY = 0.8      # fator aplicado ao seed acumulado a cada novo turno
                             # ("esquecimento de curto prazo" dentro da própria ligação).
                             # Calibrado empiricamente via experiments/multiturn.py: com
                             # 0.6 (valor original), uma entidade corretamente identificada
                             # no momento 0 decaía rápido demais (0.6²=0.36 do peso
                             # original após 2 turnos) e perdia a disputa por massa de
                             # personalização contra nós de tópico recém-mencionados
                             # (peso cheio) -- 9 clientes pioravam contra só 2 que
                             # melhoravam. Com 0.8 (0.8²=0.64), a proporção caiu pra 4
                             # contra 1 -- ainda não neutro, mas mais que reduzido à
                             # metade. Ver achado registrado no artigo, seção 6/7.
INITIAL_SEED_TOP_N = 5       # quantas entidades recentes formam o seed do momento 0
TOPIC_MASS = 1.0             # massa injetada por tópico novo mencionado pelo cliente
ANCHOR_MASS = 1.0            # massa reforçada no nó do próprio cliente a cada turno --
                              # sem isso, quando o seed cai num nó de categoria (hub
                              # compartilhado por todos os clientes), o PPR perde de
                              # vista "de quem" é a busca e dilui o sinal específico
                              # do cliente. Achado confirmado por
                              # experiments/multiturn.py: sem ancoragem, recall da
                              # entidade certa CAI depois do cliente mencionar o
                              # assunto, em vez de subir.
#
# TENTATIVA TESTADA E REVERTIDA: ancorar também o "pick" nº1 do momento 0
# (mesma lógica de piso, massa menor) piorou o resultado (5 pioram/1
# melhora, contra 4/1 sem a mudança) -- a hipótese era que esse pick já
# era "confirmado", mas na prática ele não é necessariamente a entidade
# relevante pro TÓPICO específico sendo perguntado, só a mais saliente
# em geral; protegê-lo compete com o tópico genuíno em vez de coincidir
# com ele. Registrado como tentativa negativa, não implementado.
 
# placeholder determinístico -- ver nota (2) no docstring do módulo
TOPIC_TO_EVENT_TYPE = {
    "compra": "purchase", "comprei": "purchase", "comprou": "purchase",
    "carrinho": "add_to_cart",
    "suporte": "support_ticket", "chamado": "support_ticket",
    "problema": "support_ticket", "reclamacao": "support_ticket",
    "busca": "search", "pesquisa": "search", "procurando": "search",
    "visualizei": "view", "vi": "view",
}
_ENTITY_PATTERN = re.compile(r"\bent_\d+\b")

# ---------------------------------------------------------------------------
# ESTADO DA SESSÃO
# ---------------------------------------------------------------------------

@dataclass

class SessionMemory:
    customer_id: str
    G: object
    node_idx: dict
    embeddings: object
    type_indexes: dict
    log: object
    resolve_clash: bool = True  # ver _build_context_text -- False = ablation
                                  # da contribuição 1 (lista eventos crus, sem
                                  # priorizar o mais recente), pra isolar seu
                                  # efeito de tudo mais que SessionMemory faz
                                  # (seed evolutivo, âncora, recall corrigido)
 
    entry_node: str = field(init=False)
    known_nodes: set = field(init=False)
    seed_weights: dict = field(default_factory=dict, init=False)
    turn_count: int = field(default=0, init=False)

    def __post_init__(self):
        self.entry_node = _entry_node(self.customer_id)
        self.known_nodes = _known_channel_nodes(self.G, self.entry_node)
        self._seed_from_recency()

    # -----------------------------------------------------------------
    # MOMENTO 0 -- seed inicial sem nenhuma fala do cliente, só a partir
    # do próprio histórico (recência/frequência), sem salto associativo
    # nenhum ainda.
    # -----------------------------------------------------------------
    def _seed_from_recency(self, top_n: int = INITIAL_SEED_TOP_N):
        cust_events = self.log[self.log.local_customer_id.isin(self.known_nodes)]
        if len(cust_events) == 0:
            return
        # peso por RECÊNCIA de verdade (não frequência bruta) -- mesma
        # fórmula de decaimento usada nas arestas do grafo
        # (graph/build_graph.py::_recency_weighted_group_sum), pra que uma
        # entidade com 1 evento bem recente não perca pra outra com muitos
        # eventos antigos. Bug real encontrado rodando com LLM: fact_type
        # "last_entity" (pergunta sobre a interação mais recente) falhava
        # em 6/6 casos porque o seed antigo (value_counts().head(top_n))
        # descartava a entidade certa quando ela não era também a mais
        # frequente.
        ref_ts = cust_events["ts"].max()
        age = ref_ts - cust_events["ts"]
        decay_rate = np.log(2) / EDGE_RECENCY_HALF_LIFE_SECONDS
        contrib = np.exp(-decay_rate * age)
        scores = cust_events.assign(_contrib=contrib).groupby("entity_id")["_contrib"].sum()
        top_entities = scores.sort_values(ascending=False).head(top_n)
        total = top_entities.sum()
        for ent, score in top_entities.items():
            if ent in self.node_idx:
                self.seed_weights[ent] = self.seed_weights.get(ent, 0.0) + score / total

    # -----------------------------------------------------------------
    # NOVO TURNO -- decai o que já estava acumulado, então injeta massa
    # nova a partir do que o cliente disse neste turno.
    # -----------------------------------------------------------------
    def add_turn(self, utterance: str, weight: float = TOPIC_MASS):
        self.turn_count += 1
        for node in list(self.seed_weights):
            self.seed_weights[node] *= SHORT_TERM_DECAY

        # ÂNCORA: reforça o nó do próprio cliente a cada turno, ANTES de
        # somar massa em qualquer categoria. Sem isso, um nó de categoria
        # compartilhado (ex: "support_ticket") domina o vetor de
        # personalização e o PPR passa a responder "o que é comum entre
        # TODOS os clientes desse tópico", não "o que é específico DESSE
        # cliente" -- ver nota em ANCHOR_MASS acima.
        #
        # Usa MAX, não soma: reforçar com += faz a âncora CRESCER a cada
        # turno (decai 0.6, soma 1.0 de novo -> tende a 2.5 depois de
        # poucos turnos), o que sufoca o próprio tópico recém-mencionado
        # -- mesma classe de problema (diluição), só que causada pelo
        # fix em vez de pelo hub. Com max(), a âncora fica establizada em
        # ANCHOR_MASS, nunca menos (o piso) e nunca crescendo sem limite.
        if self.entry_node in self.node_idx:
            current = self.seed_weights.get(self.entry_node, 0.0)
            self.seed_weights[self.entry_node] = max(current, ANCHOR_MASS)

        text = utterance.lower()
        matched_any = False

        # 1) menção direta a uma entidade conhecida (ex: "ent_12")
        for ent in _ENTITY_PATTERN.findall(text):
            if ent in self.node_idx:
                self.seed_weights[ent] = self.seed_weights.get(ent, 0.0) + weight
                matched_any = True

        # 2) categoria/domínio por palavra-chave -> fallback pro nó de
        #    event_type (cold-start: funciona mesmo sem histórico direto
        #    desse cliente nesse tópico). DEDUPLICADO por event_type: a
        #    frase pode bater em várias palavras-chave do mesmo tópico
        #    ("problema", "chamado" e "suporte" todas mapeiam pra
        #    support_ticket) -- sem isso, o mesmo tópico levava 3x a
        #    massa só por sorte de vocabulário, sufocando a âncora acima.
        matched_event_types = {
            event_type for keyword, event_type in TOPIC_TO_EVENT_TYPE.items()
            if keyword in text and event_type in self.node_idx
        }
        for event_type in matched_event_types:
            self.seed_weights[event_type] = self.seed_weights.get(event_type, 0.0) + weight
            matched_any = True

        return matched_any

     # -----------------------------------------------------------------
    # RETRIEVAL -- roda sobre o vetor de seed ACUMULADO até agora.
    # -----------------------------------------------------------------
    def get_context(self, top_k: int = 10, k_recall_per_seed: int = 20) -> dict:
        if not self.seed_weights:
            return {
                "context_text": f"## Contexto do cliente: {self.customer_id}\n"
                                 f"Nenhum sinal de foco ainda -- sem histórico e sem tópico mencionado.",
                "ranked_entities": [], "t_recall_ms": 0.0, "t_rerank_ms": 0.0,
            }
 
        ranked, t_recall, t_rerank = hybrid_retrieve_from_weights(
            self.seed_weights, self.G, self.node_idx, self.embeddings, self.type_indexes,
            k_recall_per_seed=k_recall_per_seed, top_k=top_k, target_type="entity",
        )
        context_text = self._build_context_text(ranked)
        return {
            "context_text": context_text, "ranked_entities": ranked,
            "t_recall_ms": t_recall, "t_rerank_ms": t_rerank,
            "seed_weights_snapshot": dict(self.seed_weights),
        }

     # -----------------------------------------------------------------
    # MONTAGEM DO TEXTO -- com resolução de context clash por recência.
    # -----------------------------------------------------------------
    def _build_context_text(self, ranked_entities) -> str:
        lines = [f"## Contexto do cliente: {self.customer_id} (turno {self.turn_count})"]
 
        cust_events = self.log[self.log.local_customer_id.isin(self.known_nodes)]
        if len(cust_events) == 0:
            lines.append("Nenhum evento encontrado para este cliente.")
            return "\n".join(lines)

        # RESUMO AGREGADO -- faltava aqui (existe em tool_contract.py, não
        # tinha sido portado). Sem isso, perguntas sobre CONTAGEM TOTAL por
        # tipo de evento (ex: fact_type="support_ticket_count") não têm de
        # onde tirar a resposta em lugar nenhum do texto -- bug real
        # encontrado rodando com LLM: 2/2 casos desse fact_type falharam
        # por esse motivo exato.
        counts = cust_events.event_type.value_counts()
        resumo = ", ".join(f"{k}: {v}" for k, v in counts.items())
        lines.append(f"Resumo de atividade ({len(cust_events)} eventos totais): {resumo}")
        lines.append("")

        for ent, score in ranked_entities:
            ev = cust_events[cust_events.entity_id == ent].sort_values("ts", ascending=False)
            if len(ev) == 0:
                lines.append(
                    f"- {ent} (relevância={score:.3f}): relacionada por padrão de "
                    f"comportamento similar, sem interação direta registrada"
                )
                continue
 
            # CLASH: pode haver mais de um event_type diferente pra mesma
            # entidade (ex: view e depois purchase) -- o mais recente vira
            # o "estado vigente"; o resto vira nota de contagem, não fica
            # listado como se fossem fatos igualmente válidos.
            # CLASH: pode haver mais de um event_type diferente pra mesma
            # entidade (ex: view e depois purchase). Ramo A
            # (resolve_clash=True, comportamento normal): o mais recente
            # vira o "estado vigente", o resto vira nota de contagem --
            # esta é a Contribuição 1 do artigo. Ramo B
            # (resolve_clash=False, ablation): lista TODOS os eventos
            # crus, sem priorizar nenhum -- exatamente o que um sistema
            # de retrieval ingênuo faria, mas com o MESMO seed evolutivo,
            # âncora e recall corrigido do resto do SessionMemory. É essa
            # comparação (True vs False, tudo mais igual) que isola o
            # efeito de rho sem o confound de número de seeds que
            # invalidou a comparação sessao_memoria vs hibrido_sob_demanda.
            if self.resolve_clash:
                latest = ev.iloc[0]
                older_count = len(ev) - 1
                # NÃO usar um número cru aqui (ex: "mais 12 eventos
                # anteriores") -- achado real, rodando com LLM em escala:
                # quando a pergunta é sobre CONTAGEM AGREGADA
                # (support_ticket_count), o texto acaba com vários números
                # "mais N eventos anteriores" (um por entidade do top-k)
                # boiando perto do número certo (o do Resumo de atividade),
                # e o modelo se confunde e devolve resposta vazia
                # (malformed) em vez de arriscar qual número é o certo.
                # Sem número aqui, ambíguo, mas por outro motivo bom: só o
                # Resumo de atividade tem número de contagem no texto
                # inteiro.
                older_note = " (havia também interações anteriores registradas)" if older_count else ""
                lines.append(
                    f"- {ent} (relevância={score:.3f}): estado mais recente = "
                    f"{latest.event_type} em ts={int(latest.ts)}{older_note}"
                )
            else:
                eventos = "; ".join(f"{row.event_type} em ts={int(row.ts)}" for row in ev.itertuples())
                lines.append(f"- {ent} (relevância={score:.3f}): {eventos}")
 
        return "\n".join(lines)



# ---------------------------------------------------------------------------
# DEMO -- reproduz o cenário de teste discutido (fatura -> produto),
# adaptado ao domínio sintético deste repositório (support_ticket -> purchase)
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    from data.synthetic_events import generate_dataset
    from graph.build_graph import build_graph
    from embeddings.structural import compute_structural_embeddings, build_type_indexes
 
    ds = generate_dataset()
    G, graph_stats = build_graph(ds)
    nodes, node_idx, embeddings = compute_structural_embeddings(G)
    type_indexes = build_type_indexes(nodes, node_idx, embeddings)
    log = ds["log"]
 
    # acha um cliente com pelo menos 1 support_ticket e 1 purchase, pra
    # que o cenário de teste (assunto muda de categoria no meio da
    # ligação) tenha sinal real pra propagar
    facts = ds["facts"]
    candidates = [gid for gid, f in facts.items() if f.support_ticket_count > 0]
    customer_id = candidates[0] if candidates else "cust_5"
 
    print(f"=== Sessão de teste: {customer_id} ===\n")
    session = SessionMemory(customer_id, G, node_idx, embeddings, type_indexes, log)
 
    print("--- Momento 0 (sem fala do cliente, só recência/frequência) ---")
    r0 = session.get_context(top_k=5)
    print(r0["context_text"])
    print(f"[seed_weights] {r0['seed_weights_snapshot']}\n")
 
    print("--- Momento 1 (cliente confirma: 'quero falar sobre um chamado de suporte') ---")
    session.add_turn("quero falar sobre um chamado de suporte")
    r1 = session.get_context(top_k=5)
    print(r1["context_text"])
    print(f"[seed_weights] {r1['seed_weights_snapshot']}\n")
 
    print("--- Momento 2 (assunto muda: cliente menciona uma compra) ---")
    session.add_turn("na verdade é sobre uma compra que eu fiz")
    r2 = session.get_context(top_k=5)
    print(r2["context_text"])
    print(f"[seed_weights] {r2['seed_weights_snapshot']}\n")
 
    print("=" * 70)
    print("Verificação: o texto de cada momento SUBSTITUI o anterior "
          "(não empilha) -- confirmar visualmente que os 3 blocos acima "
          "são independentes, cada um do tamanho de top_k=5, não crescente.")