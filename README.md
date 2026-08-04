# Lab: Grafo + Embeddings + Contexto para Retrieval

Projeto de laboratorio para estudar retrieval com embeddings em grafo, combinando sinal estrutural e semantico para montar contexto de cliente sob demanda.

O pipeline compara 4 formas de injecao de contexto, mede custo de inferencia e estima risco de alucinacao com checagem automatica contra ground truth sintetico.

## Objetivo

Responder, com reproducibilidade, perguntas como:

- Quando o retrieval hibrido (recall vetorial + PPR local) melhora qualidade vs baselines simples?
- Em que escala o custo do hibrido passa a compensar o PPR no grafo inteiro?
- Como ruido e posicao da informacao relevante degradam resposta do agente?

## Arquitetura do Pipeline

1. Geracao de dados sinteticos multi-canal em `data/synthetic_events.py`
2. Construcao de grafo heterogeneo em `graph/build_graph.py`
3. Embeddings estruturais (SVD) e indices por tipo em `embeddings/structural.py`
4. Retrieval hibrido (kNN recall + PPR local rerank) em `retrieval/hybrid.py`
5. Tool de contexto para agente em `agent/tool_contract.py`
6. Avaliacao em `experiments/diagnostic.py`, `experiments/comparative.py`, `experiments/cost.py`

## Estrutura do Repositorio

```
.
├── agent/
│   └── tool_contract.py
├── data/
│   └── synthetic_events.py
├── embeddings/
│   ├── semantic.py
│   └── structural.py
├── experiments/
│   ├── comparative.py
│   ├── cost.py
│   └── diagnostic.py
├── graph/
│   └── build_graph.py
├── metrics/
│   ├── hallucination.py
│   └── retrieval_metrics.py
├── retrieval/
│   └── hybrid.py
├── config.py
└── run_experiment.py
```

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

## Instalacao Rapida

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

## Como Rodar

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

## Experimentos

### 1) Diagnostico (secao 5.2)

Arquivo: `experiments/diagnostic.py`

- Varia `% de ruido` no contexto
- Varia posicao do fato-alvo (`inicio`, `meio`, `fim`)
- Mede acerto/alucinacao por condicao

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
- Estima ponto de crossover (quando speedup > 1.0)

## Parametros de Configuracao

Arquivo central: `config.py`

Principais grupos:

- Reprodutibilidade (`SEED`)
- Tamanho de dataset (`N_CUSTOMERS`, `N_EVENTS`)
- Grafo e identidade (`IDENTITY_*`, pesos)
- Embeddings (`EMB_DIM`, `N_NEIGHBORS_DEFAULT`)
- Retrieval/PPR (`PPR_ALPHA`, `MAX_NEIGHBORS_PER_CANDIDATE`)
- Escala de custo (`SCALE_POINTS_N_CUSTOMERS`)
- Modelo local (`LLM_MODEL_PATH`, `LLM_MODEL_ID`)

## Interpretacao de Saidas

Ao rodar `run_experiment.py`, observe:

- Tabela de acerto por condicao/segmento
- Tamanho medio de contexto por condicao (proxy de custo)
- Distribuicao de veredito (`correct`, `hallucination`, `abstention`, `malformed`)
- Crossover do experimento de custo

Regra pratica:

- Melhor metodo tende a combinar maior acerto com menor contexto medio
- Ganho de custo deve ser analisado junto de overlap/top-k e qualidade final

## Importante sobre Validade dos Resultados

- Sem `--granite`, os experimentos de resposta usam um respondedor de sanity-check (nao e LLM).
- Para resultado real de comportamento de modelo, use `--granite` em ambiente local com modelo disponivel.
- O experimento de custo (latencia de retrieval) nao depende de LLM e e comparavel entre ambientes.

## Proximos Passos Sugeridos

- Adicionar `requirements.txt` ou `pyproject.toml`
- Persistir resultados em CSV/Parquet por execucao
- Versionar seeds e configuracoes por rodada experimental
- Incluir benchmark com mais de um modelo LLM

