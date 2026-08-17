"""
config.py

Configuração centralizada: seeds, dimensões e hiperparâmetros usados em
todo o pipeline (data/, graph/, embeddings/, retrieval/, agent/,
experiments/). Antes deste módulo, cada arquivo carregava suas
próprias constantes soltas -- risco real de dessincronização (ex:
FACT_TYPES estava duplicado em experiments/diagnostic.py e
experiments/comparative.py; bastava editar um dos dois pra os
experimentos pararem de ser comparáveis entre si).

Os módulos importam daqui; parâmetros pontuais continuam podendo ser
sobrescritos via argumento de função quando fizer sentido (ex:
generate_dataset(n_customers=...) para testes de escala em
experiments/cost.py), mas o DEFAULT sempre vem deste arquivo.
"""

import os


# ---------------------------------------------------------------------------
# CONFIGURAÇÃO DO MODELO
# ---------------------------------------------------------------------------

_DEFAULT_LOCAL_LLM_MODEL_PATH = (
    "~/.cache/huggingface/hub/models--ibm-granite--granite-4.1-8b"
)

# ── Modelo LLM local (HuggingFace) ────────────────────────────
# Caminho completo onde o modelo foi baixado
# O HuggingFace salva em: models--<org>--<model>/snapshots/<hash>/
# Prioridade de resolução:
# 1) variável de ambiente LLM_MODEL_PATH (se definida)
# 2) caminho local padrão, apenas se existir neste host
# 3) string vazia (força uso de LLM_MODEL_ID)
_llm_model_path_env = os.getenv("LLM_MODEL_PATH", "")
if _llm_model_path_env is not None:
    LLM_MODEL_PATH: str = _llm_model_path_env
elif os.path.isdir(os.path.expanduser(_DEFAULT_LOCAL_LLM_MODEL_PATH)):
    LLM_MODEL_PATH: str = os.path.expanduser(_DEFAULT_LOCAL_LLM_MODEL_PATH)
else:
    LLM_MODEL_PATH: str = ""
# Se quiser usar o model ID direto (baixa automaticamente se não tiver):
LLM_MODEL_ID: str = os.getenv("LLM_MODEL_ID", "ibm-granite/granite-4.1-8b")

# ── Modelo Embedding ───────────────────────────────────────────
# Granite Embeddings também pode ser local ou via HuggingFace ID
EMBEDDING_MODEL_ID: str = os.getenv(
    "EMBEDDING_MODEL_ID", "ibm-granite/granite-embedding-278m-multilingual"
)

# Guard rails de inferência para reduzir OOM no MPS durante testes longos
LLM_MAX_INPUT_TOKENS = 1024
LLM_MAX_NEW_TOKENS = 40
LLM_CONTEXT_CHAR_BUDGET = 12000

# Limites de contexto do experimento comparativo
COMPARATIVE_FULL_CONTEXT_MAX_EVENTS = 120
COMPARATIVE_CONTEXT_CHAR_BUDGET = 12000


# ---------------------------------------------------------------------------
# REPRODUTIBILIDADE
# ---------------------------------------------------------------------------

SEED = 42


# ---------------------------------------------------------------------------
# data/synthetic_events.py
# ---------------------------------------------------------------------------

N_CUSTOMERS = 10000
N_ENTITIES = 60
N_NICHES = 8
EVENT_TYPES = ["view", "add_to_cart", "purchase", "search", "support_ticket"]
EVENT_TYPE_WEIGHTS = [5, 3, 1, 3, 1]
CHANNELS = ["web", "app", "call_center"]

FRAC_WITH_LOYALTY_KEY = 0.6      # fração de clientes com chave exata web<->app
FRAC_WITH_CALL_CENTER = 0.35     # fração de clientes que também usa call_center

N_EVENTS = 120000

CUSTOMER_ENGAGEMENT_ALPHA = 1.2  # shape do Pareto p/ distribuição de engajamento
                                  # em cauda longa -- menor = cauda mais pesada


# ---------------------------------------------------------------------------
# graph/build_graph.py
# ---------------------------------------------------------------------------

REFERENCE_CHANNEL = "web"        # canal-âncora: sempre existe p/ todo cliente sintético

# Decaimento por recência nas arestas cliente->entidade e cliente->event_type.
# Antes: peso = contagem bruta de ocorrências (log.groupby(...).size()).
# Agora: cada ocorrência contribui exp(-idade / EDGE_RECENCY_HALF_LIFE_SECONDS
# * ln(2)), então uma interação de ontem pesa mais que uma de 6 meses atrás
# com a MESMA contagem total. "Idade" é relativa ao evento mais recente do
# log inteiro (não "agora" do relógio real), pra ficar reprodutível com
# dado sintético que não tem timestamp real.
# None desliga o decaimento e volta ao comportamento antigo (contagem pura)
# -- útil pra comparar os dois modos no artigo (seção de ablation).
EDGE_RECENCY_HALF_LIFE_SECONDS = 30 * 24 * 3600   # meia-vida de 30 dias

IDENTITY_EXACT_WEIGHT = 5.0      # peso fixo p/ aresta de identidade exata (chave compartilhada)
IDENTITY_PROB_THRESHOLD = 0.40   # score mínimo p/ criar aresta de identidade probabilística
                                  # (recalibrado para a escala do score TF-IDF + distribuição
                                  # de engajamento em cauda longa -- ver nota em graph/build_graph.py)
IDENTITY_TOPK_CANDIDATES = 5     # blocking: só avalia top-K candidatos por overlap comportamental
W_ENTITY_OVERLAP = 0.5           # peso do sinal comportamental no score combinado de identidade
W_TEXT_SIM = 0.5                 # peso do sinal textual (TF-IDF) no score combinado


# ---------------------------------------------------------------------------
# embeddings/structural.py
# ---------------------------------------------------------------------------

# EMB_DIM=32 foi calibrado e validado só até ~2 mil nós (N_CUSTOMERS~1000
# default). Achado real, rodando com ~20 mil nós (N_CUSTOMERS~10000): overlap
# do recall híbrido contra PPR completo caiu de ~65% para ~40%, e a
# significância do experimento comparativo (McNemar) foi junto -- SVD
# truncado com dimensão fixa perde capacidade discriminativa conforme o
# grafo cresce. Testado empiricamente: EMB_DIM=128 recupera overlap
# (~68%, contra ~55% em 32) num grafo de ~10 mil nós, com custo de tempo de
# embedding ainda desprezível (<1s). Use auto_emb_dim() abaixo para escalar
# automaticamente em vez de fixar um valor só, ou ajuste manualmente se
# rodar em escala muito diferente da testada.
EMB_DIM = 128
N_NEIGHBORS_DEFAULT = 30


def auto_emb_dim(n_nodes: int, min_dim: int = 32, max_dim: int = 256) -> int:
    """Sugestão de EMB_DIM proporcional ao tamanho do grafo -- regra
    empírica, não teórica: dobra a dimensão a cada ordem de grandeza de
    nós, calibrada nos pontos observados (2 mil nós -> 32 já era
    suficiente; 20 mil nós -> precisou de ~128 pra recuperar overlap).
    Não substitui validação -- é um ponto de partida melhor que a
    constante fixa."""
    import math
    scale = max(1, n_nodes // 2000)
    dim = min_dim * (2 ** int(math.log2(scale + 1)))
    return max(min_dim, min(max_dim, dim))


# ---------------------------------------------------------------------------
# embeddings/semantic.py
# ---------------------------------------------------------------------------

CHAR_NGRAM_RANGE = (2, 3)        # n-gramas de caractere p/ TF-IDF de resolução de identidade


# ---------------------------------------------------------------------------
# retrieval/hybrid.py
# ---------------------------------------------------------------------------

PPR_ALPHA = 0.85
PPR_MAX_ITER = 100
PPR_MAX_ITER_FALLBACK = 500      # se o subgrafo não convergir no limite normal
MAX_NEIGHBORS_PER_CANDIDATE = 20 # limita expansão de vizinhança -- evita nós-hub
                                  # explodindo o subgrafo "local" (ver nota em retrieval/hybrid.py)


# ---------------------------------------------------------------------------
# agent/tool_contract.py
# ---------------------------------------------------------------------------

MAX_EVENTS_PER_ENTITY = 3        # quantos tipos de evento mostrar por entidade no contexto formatado


# ---------------------------------------------------------------------------
# experiments/diagnostic.py e experiments/comparative.py (compartilhado)
# ---------------------------------------------------------------------------

FACT_TYPES = ["last_entity", "support_ticket_count"]


# ---------------------------------------------------------------------------
# experiments/diagnostic.py
# ---------------------------------------------------------------------------

NOISE_LEVELS = [0.0, 0.5, 0.9]               # fração do contexto que é ruído
POSITIONS = ["inicio", "meio", "fim"]                # onde o alvo fica embutido
TOTAL_CONTEXT_LINES = 12                # tamanho do contexto quando noise_pct=1.0
N_CUSTOMERS_PER_CONDITION = 2           # trials por combinação (ruído x posição x fato)


# ---------------------------------------------------------------------------
# experiments/comparative.py
# ---------------------------------------------------------------------------

CONDITIONS = ["sem_retrieval", "contexto_completo", "topk_estatico", "hibrido_sob_demanda",
              "sessao_memoria", "sessao_memoria_sem_clash"]
TOPK_STATIC_K = 3                # tamanho do top-k estático (mesmo k do híbrido -- comparação justa)
N_CUSTOMERS_PER_SEGMENT = 30     # por segmento (leve/pesado) e por tipo de fato -- 30 dá
                                   # ~110 pares avaliáveis no teste de McNemar (sessão vs.
                                   # híbrido), suficiente pra significância estatística limpa
                                   # (ver achado: com N=5 o efeito aparece mas fica marginal,
                                   # p=0.0625; com N~30 deu p=0.0009)
HEAVY_SEGMENT_PERCENTILE = 0.95   # cliente "pesado" = acima deste percentil de engajamento


# ---------------------------------------------------------------------------
# experiments/cost.py
# ---------------------------------------------------------------------------

SCALE_POINTS_N_CUSTOMERS = [100, 300, 600, 1000, 1500]
EVENTS_PER_CUSTOMER_RATIO = 15    # mesma densidade do dataset default (N_EVENTS / N_CUSTOMERS)
N_QUERIES_PER_SCALE = 8           # consultas amostradas por ponto de escala
N_REPETITIONS_PER_SCALE_POINT = 5  # repetições por ponto de escala -- reportar mediana,
                                    # não uma medição só (latência tem ruído de carga da
                                    # máquina entre execuções -- ver achado no artigo, seção 6)


# ---------------------------------------------------------------------------
# experiments/multiturn.py
#
# Experimento MULTI-TURNO (contribuição 3 do artigo): exercita
# SessionMemory.add_turn() de verdade, ao longo de uma sessão simulada
# de 3 momentos (0=sem fala, 1=cliente menciona suporte, 2=cliente muda
# de assunto pra compra) -- não existia nenhum experimento formal
# fazendo isso antes, só a demo manual em agent/session_memory.py.
# ---------------------------------------------------------------------------

N_CUSTOMERS_MULTITURN = 30   # clientes com histórico nos dois tópicos (suporte E compra)
MULTITURN_TOP_K = 6          # top-k de entidades retornado em cada get_context()

# utterances placeholder -- mesmo espírito de TOPIC_TO_EVENT_TYPE em
# session_memory.py (palavra-chave determinística, não classificador de
# intenção real -- ver nota lá)
MULTITURN_UTTERANCE_SUPORTE = "tenho um problema, quero abrir um chamado de suporte"
MULTITURN_UTTERANCE_COMPRA = "na verdade é sobre uma compra que eu fiz recentemente"