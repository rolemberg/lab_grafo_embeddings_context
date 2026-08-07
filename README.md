# Lab: Grafo + Embeddings + Contexto para Retrieval

Projeto de laboratorio para estudar retrieval com embeddings em grafo, combinando sinal estrutural e semantico para montar contexto de cliente sob demanda, dentro de uma arquitetura de memoria de agente inspirada em Complementary Learning Systems (memoria curta + memoria longa episodica/hipocampo-conteudo).

O pipeline compara 4 formas de injecao de contexto, mede custo de inferencia e estima risco de alucinacao com checagem automatica contra ground truth sintetico.

## Objetivo

Responder, com reproducibilidade, perguntas como:

- Quando o retrieval hibrido (recall vetorial + PPR local) melhora qualidade vs baselines simples?
- Em que escala o custo do hibrido passa a compensar o PPR no grafo inteiro?
- Como ruido e posicao da informacao relevante degradam resposta do agente (lost-in-the-middle)?
- Como uma memoria de sessao com seed evolutivo por turno se comporta ao longo de uma conversa multi-turno?
- Como resolver contradicoes entre eventos ativados para a mesma entidade (context clash)?

## Arquitetura conceitual

A memoria do agente se divide em duas camadas:

- **Memoria curta**: buffer efemero da conversa/ligacao atual. Nao tem grafo proprio -- alimenta um vetor de personalizacao (seed) que evolui por turno (decaimento de curto prazo + massa nova a cada topico mencionado).
- **Memoria longa** (persistente, por cliente): subdividida em **hipocampo** (indice associativo esparso -- o grafo `G`, que nao guarda conteudo, so aponta pra ele) e **conteudo** (os episodios/eventos reais, consultados apenas para os nos que o hipocampo ativa via PPR).

Quando um no ativado tem mais de um evento associado, aplica-se uma politica de **context clash por recencia**: o evento mais recente vira o fato apresentado, os demais viram uma nota de contagem -- evita listar estados possivelmente contraditorios lado a lado.

## Arquitetura do pipeline (implementacao)

1. Geracao de dados sinteticos multi-canal em `data/synthetic_events.py` (dominio generico) ou `data/synthetic_events_telco.py` (dominio de atendimento de fatura -- mesma mecanica, vocabulario de intents diferente, drop-in)
2. Construcao de grafo heterogeneo em `graph/build_graph.py` -- arestas `interaction` e `event_type` agora ponderadas por **decaimento de recencia** (meia-vida configuravel), nao mais contagem bruta
3. Embeddings estruturais (SVD) e indices por tipo em `embeddings/structural.py`
4. Retrieval hibrido (kNN recall + PPR local rerank) em `retrieval/hybrid.py` -- versao original de query unica **+ variante multi-seed** (`hybrid_retrieve_from_weights`) para seeds acumulados ao longo de uma sessao
5. Tool de contexto para agente, consulta unica, em `agent/tool_contract.py`
6. **Memoria de sessao** com estado entre turnos em `agent/session_memory.py` -- fecha memoria curta, fallback de categoria para cold-start, e context clash por recencia
7. Avaliacao em `experiments/diagnostic.py`, `experiments/comparative.py`, `experiments/cost.py`

## Estrutura do repositorio

```
.
├── agent/
│   ├── tool_contract.py         # buscar_contexto_cliente(foco) -> str, consulta unica
│   └── session_memory.py        # SessionMemory: memoria curta + hipocampo/conteudo + clash
├── data/
│   ├── synthetic_events.py      # gerador generico (e-commerce/suporte)
│   └── synthetic_events_telco.py  # gerador de dominio telco (fatura), mesma API
├── embeddings/
│   ├── semantic.py
│   └── structural.py
├── experiments/
│   ├── comparative.py
│   ├── cost.py
│   └── diagnostic.py
├── graph/
│   └── build_graph.py           # arestas com decaimento por recencia (EDGE_RECENCY_HALF_LIFE_SECONDS)
├── metrics/
│   ├── hallucination.py         # veredito de 4 vias: CORRECT/HALLUCINATION/ABSTENTION/MALFORMED
│   └── retrieval_metrics.py
├── retrieval/
│   └── hybrid.py                # + hybrid_retrieve_from_weights (variante multi-seed)
├── config.py
└── run_experiment.py
```

**Status de integracao**: `agent/session_memory.py`, `data/synthetic_events_telco.py` e o decaimento por recencia em `graph/build_graph.py` foram desenvolvidos e testados isoladamente (ver secao "Memoria de sessao" abaixo), mas **ainda nao estao conectados a `run_experiment.py`** -- os 3 experimentos formais (diagnostic/comparative/cost) continuam rodando contra o pipeline de consulta unica (`tool_contract.py`), nao contra `session_memory.py`. Integrar isso e um proximo passo, nao um resultado ja medido.

## Requisitos

Recomendado:

- Python 3.10+
- `pip` ou `uv`

Dependencias principais do pipeline:

- pandas
- numpy
- networkx
- scikit-learn
- matplotlib (opcional, para grafico de custo)

Dependencias opcionais para rodar com LLM local (Granite):

- torch
- transformers

## Instalacao rapida

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas numpy networkx scikit-learn matplotlib
```

Para execucao com LLM local:

```bash
pip install torch transformers
```

O caminho do modelo local e resolvido via variavel de ambiente `LLM_MODEL_PATH`, com fallback para `~/.cache/huggingface/hub/...` (ver `config.py`) -- nao ha caminho de usuario hardcoded.

## Como rodar

Rodar pipeline completo com respondedor de sanity-check (sem LLM real):

```bash
python run_experiment.py
```

Rodar com modelo local Granite (na sua maquina, com cache local configurado):

```bash
python run_experiment.py --granite
```

Rodar apenas o experimento comparativo:

```bash
python run_experiment.py --skip-diagnostic --skip-cost
```

Rodar a demo de memoria de sessao (multi-turno, dominio generico), isolada dos experimentos formais:

```bash
python -m agent.session_memory
```

Rodar o gerador de dominio telco isoladamente (sanity check de distribuicao de intents):

```bash
python -m data.synthetic_events_telco
```

Para usar o dominio telco no lugar do generico em qualquer script, troque o import:

```python
# from data.synthetic_events import generate_dataset
from data.synthetic_events_telco import generate_dataset
```

## Memoria de sessao (`agent/session_memory.py`)

Classe `SessionMemory`, com estado por atendimento/ligacao:

- **Momento 0** (sem fala do cliente): seed inicial = top-N entidades mais recentes/frequentes do proprio cliente, normalizado.
- **A cada turno** (`add_turn(utterance)`): decai o seed acumulado (`SHORT_TERM_DECAY`), depois injeta massa nova nos nos que o texto do turno ativa -- por mencao direta a entidade (`ent_\d+`) ou por categoria via `TOPIC_TO_EVENT_TYPE` (placeholder de palavra-chave; em producao seria a saida de um classificador de intencao, ou os proprios campos estruturados no dominio telco).
- **`get_context()`**: roda `hybrid_retrieve_from_weights` sobre o seed acumulado, aplica clash por recencia, e retorna um bloco de texto que **substitui** o anterior a cada chamada -- nunca acumula, evitando recriar context distraction dentro da propria sessao.

Achado registrado, ainda sem fix: quando o seed cai para um no de categoria generico (hub compartilhado por muitos clientes) sem ancoragem forte nos nos conhecidos do cliente especifico, o sinal do cliente pode ser **diluido** em vez de reforcado -- ver commits/discussao para reproducao.

## Experimentos

### 1) Diagnostico (secao 5.2)

Arquivo: `experiments/diagnostic.py`

- Varia `% de ruido` no contexto
- Varia posicao do fato-alvo (`inicio`, `meio`, `fim`)
- Mede acerto/alucinacao por condicao -- este e o experimento de lost-in-the-middle

### 2) Comparativo (secao 5.3)

Arquivo: `experiments/comparative.py`

Compara 4 condicoes:

1. `sem_retrieval`
2. `contexto_completo`
3. `topk_estatico`
4. `hibrido_sob_demanda`

Tambem estratifica por segmento de cliente:

- `leve`
- `pesado`

### 3) Custo/escala (secao 5.4)

Arquivo: `experiments/cost.py`

- Varre pontos de escala (`n_customers`)
- Compara latencia de retrieval hibrido vs PPR em grafo inteiro
- Estima ponto de crossover (quando speedup > 1.0) -- **achado**: em escalas pequenas o hibrido pode ser mais lento que o PPR completo (overhead de indução de subgrafo domina); o crossover nao e assumido, e buscado.

## Parametros de configuracao

Arquivo central: `config.py`

Principais grupos:

- Reprodutibilidade (`SEED`)
- Tamanho de dataset (`N_CUSTOMERS`, `N_EVENTS`)
- Grafo e identidade (`IDENTITY_*`, pesos)
- **Decaimento de recencia** (`EDGE_RECENCY_HALF_LIFE_SECONDS`, padrao 30 dias; `None` recupera contagem bruta, util para ablation)
- Embeddings (`EMB_DIM`, `N_NEIGHBORS_DEFAULT`)
- Retrieval/PPR (`PPR_ALPHA`, `MAX_NEIGHBORS_PER_CANDIDATE`)
- Escala de custo (`SCALE_POINTS_N_CUSTOMERS`)
- Modelo local (`LLM_MODEL_PATH`, `LLM_MODEL_ID`)

## Interpretacao de saidas

Ao rodar `run_experiment.py`, observe:

- Tabela de acerto por condicao/segmento
- Tamanho medio de contexto por condicao (proxy de custo)
- Distribuicao de veredito (`correct`, `hallucination`, `abstention`, `malformed`)
- Crossover do experimento de custo

Regra pratica:

- Melhor metodo tende a combinar maior acerto com menor contexto medio
- Ganho de custo deve ser analisado junto de overlap/top-k e qualidade final

## Importante sobre validade dos resultados

- Sem `--granite`, os experimentos de resposta usam um respondedor de sanity-check (nao e LLM).
- Para resultado real de comportamento de modelo, use `--granite` em ambiente local com modelo disponivel.
- O experimento de custo (latencia de retrieval) nao depende de LLM e e comparavel entre ambientes.
- A resolucao de identidade probabilistica tem precisao baixa (~13%) com recall alto (~87%) no regime de cauda longa do gerador atual -- efeito esperado de esparsidade de historico por cliente-canal, nao um bug de calibracao de limiar (ver nota em `graph/build_graph.py`).
- O recall via embedding pode nao recuperar um no relevante topologicamente (visto empiricamente) -- o metodo hibrido tem teto de qualidade definido pela etapa de recall, nao so pelo rerank via PPR.

## Nota sobre submissao double-blind

Se este repositorio for referenciado como material suplementar de submissao com revisao cega, revisar antes:

- Nenhum caminho de arquivo, nome de variavel ou comentario deve identificar o(s) autor(es) (`config.py` ja foi auditado e corrigido nessa frente).
- O nome do repositorio e a URL do GitHub, se citados no PDF, quebram o anonimato -- nao incluir link direto durante o periodo de revisao.

## Proximos passos sugeridos

- Integrar `agent/session_memory.py` aos 3 experimentos formais (hoje eles rodam so contra `tool_contract.py`, consulta unica)
- Corrigir a diluicao de hub em cold-start (ancorar expansao de categoria nos `known_nodes` do cliente)
- Corrigir o recall-miss do embedding estrutural (uniao com vizinhos diretos no grafo, nao so vizinhos por similaridade)
- Adicionar camada estatica de "produto que o cliente possui", distinta da jornada temporal
- Adicionar `requirements.txt` ou `pyproject.toml`
- Persistir resultados em CSV/Parquet por execucao
- Versionar seeds e configuracoes por rodada experimental
- Incluir benchmark com mais de um modelo LLM
- Avaliar contra benchmarks padrao da categoria (LoCoMo, LongMemEval), hoje nao usados