context_degradation_experiment/
├── data/
│   └── synthetic_events.py      # gerador multi-canal + identidade + ground truth de fatos
├── graph/
│   └── build_graph.py           # constrói G a partir do log de eventos
├── embeddings/
│   ├── structural.py            # SVD (proxy node2vec/GraphSAGE)
│   └── semantic.py              # embedding de texto p/ recall + resolução de identidade
├── retrieval/
│   └── hybrid.py                # recall kNN + PPR local rerank (o que já existe, refatorado)
├── agent/
│   └── tool_contract.py         # buscar_contexto_cliente(foco: str) -> str
├── experiments/
│   ├── diagnostic.py            # 5.2 — curva de degradação (ruído x posição)
│   ├── comparative.py           # 5.3 — 4 condições
│   └── cost.py                  # 5.4 — latência híbrido vs PPR completo (já existe, migra pra cá)
├── metrics/
│   ├── hallucination.py         # verificador automático contra ground truth
│   └── retrieval_metrics.py     # NDCG, recall@k, overlap
├── config.py                    # seeds, dimensões, hiperparâmetros centralizados
└── run_experiment.py            # entrypoint que orquestra tudo