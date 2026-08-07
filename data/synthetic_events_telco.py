"""
data/synthetic_events_telco.py

Variante de domínio de data/synthetic_events.py -- MESMA mecânica de
geração (fragmentação de identidade entre canais, cauda longa de
engajamento via Pareto, nichos/perfis, ground truth de fatos), só
trocando o VOCABULÁRIO por intenções de atendimento de fatura, inspirado
na estrutura da base real mostrada pelo usuário (colunas interaction_id,
intent/journey_details, channel, id_document, direction, status).

IMPORTANTE: nenhum dado real foi copiado -- nenhum CPF, nenhum
interaction_id, nenhum timestamp, nenhuma contagem da base original.
Só a IDEIA estrutural (que intenções existem, e que elas têm frequência
bem desbalanceada -- poucas intenções dominam o volume, contestação e
bloqueio são raras) foi reaproveitada como inspiração para os pesos
sintéticos abaixo.

API idêntica a data/synthetic_events.py (generate_dataset() com as
mesmas chaves de retorno) -- drop-in replacement: graph/build_graph.py,
embeddings/, retrieval/ e agent/ não precisam de nenhuma alteração para
consumir esta variante, já que não dependem do vocabulário específico
de EVENT_TYPES/ENTITY, só da FORMA do log (local_customer_id, channel,
global_customer_id_TRUTH, event_type, entity_id, session_id, ts).

DIFERENÇA DE VOCABULÁRIO (a única coisa que muda de fato):
  - "entity" (genérico, ex-produto/página) -> aqui representa uma LINHA
    ou PRODUTO específico do cliente (ex: uma linha móvel, um plano de
    fibra) -- o que o intent de fatura está se referindo.
  - "event_type" (genérico, ex-view/purchase) -> aqui é a INTENÇÃO de
    atendimento (central_faturas, pagar_fatura, ...), no mesmo papel
    estrutural: nó compartilhado entre todos os clientes, usado em
    agent/session_memory.py como fallback de categoria para cold-start.
  - CustomerFacts.support_ticket_count/entities é mantido com esse NOME
    por compatibilidade com metrics/ e experiments/ (que já esperam
    esses campos), mas agora conta eventos de
    "esclarecimento_contestacao" -- o intent mais próximo, em espírito,
    de "abriu um chamado por algo ter dado errado".
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from config import (
    SEED, N_CUSTOMERS, N_ENTITIES, N_NICHES, CHANNELS,
    FRAC_WITH_LOYALTY_KEY, FRAC_WITH_CALL_CENTER, N_EVENTS,
    CUSTOMER_ENGAGEMENT_ALPHA,
)

# ---------------------------------------------------------------------------
# VOCABULÁRIO DE DOMÍNIO -- a parte que de fato muda em relação ao genérico
# ---------------------------------------------------------------------------

# intenções inspiradas na base real (nomes de intent/journey_details
# observados), com pesos sintéticos que reproduzem a FORMA da
# distribuição real (2ª via e pagamento dominam; contestação e bloqueio
# são raros) -- não os números reais.
INTENT_TYPES = [
    "central_faturas",
    "segunda_via_fatura",
    "detalhe_fatura",
    "pagar_fatura",
    "esclarecimento_contestacao",
    "bloqueio_franquia",
]
INTENT_WEIGHTS = [5, 5, 4, 5, 1, 1]

# intent usado como "categoria de risco" para o ground truth de fatos
# (papel equivalente a support_ticket no domínio genérico)
_RISK_INTENT = "esclarecimento_contestacao"


# ---------------------------------------------------------------------------
# ESTRUTURAS DE GROUND TRUTH (idêntico a synthetic_events.py)
# ---------------------------------------------------------------------------

@dataclass
class ChannelIdentity:
    """Mapeia um customer_id "canônico" (global) para seus IDs por canal."""
    global_id: str
    channel_ids: dict = field(default_factory=dict)
    has_exact_key: dict = field(default_factory=dict)
    noisy_text: dict = field(default_factory=dict)


@dataclass
class CustomerFacts:
    """Ground truth de fatos por cliente -- usado como resposta certa
    nos experimentos de diagnóstico/comparativo. Nomes de campo mantidos
    genéricos por compatibilidade com metrics/ e experiments/."""
    global_id: str
    last_entity: str | None
    last_entity_channel: str | None
    last_entity_ts: int | None
    primary_niche: int
    primary_niche_share: float
    support_ticket_count: int          # ver nota no topo do arquivo
    support_ticket_entities: list


# ---------------------------------------------------------------------------
# GERAÇÃO DE IDENTIDADE MULTI-CANAL (idêntico a synthetic_events.py)
# ---------------------------------------------------------------------------

def _random_name():
    first = "".join(random.choices(string.ascii_lowercase, k=6)).capitalize()
    last = "".join(random.choices(string.ascii_lowercase, k=8)).capitalize()
    return f"{first} {last}"


def _noisy_variant(text, p_typo=0.15):
    chars = list(text)
    for i in range(len(chars)):
        if chars[i].isalpha() and random.random() < p_typo:
            chars[i] = random.choice(string.ascii_lowercase)
    return "".join(chars)


def gen_identities(n_customers=N_CUSTOMERS):
    identities = {}
    for i in range(n_customers):
        gid = f"cust_{i}"
        name = _random_name()
        ident = ChannelIdentity(global_id=gid)

        ident.channel_ids["web"] = f"web_{i}"
        ident.has_exact_key["web"] = True
        ident.noisy_text["web"] = name

        has_loyalty = random.random() < FRAC_WITH_LOYALTY_KEY
        if has_loyalty:
            ident.channel_ids["app"] = f"app_{i}"
            ident.has_exact_key["app"] = True
            ident.noisy_text["app"] = name
        else:
            ident.channel_ids["app"] = f"app_anon_{i}"
            ident.has_exact_key["app"] = False
            ident.noisy_text["app"] = _noisy_variant(name)

        if random.random() < FRAC_WITH_CALL_CENTER:
            ident.channel_ids["call_center"] = f"cc_anon_{i}"
            ident.has_exact_key["call_center"] = False
            ident.noisy_text["call_center"] = _noisy_variant(name, p_typo=0.25)

        identities[gid] = ident

    return identities


# ---------------------------------------------------------------------------
# GERAÇÃO DO LOG DE EVENTOS -- aqui entra o vocabulário de fatura
# ---------------------------------------------------------------------------

def _customer_engagement_weights(n_customers, alpha=CUSTOMER_ENGAGEMENT_ALPHA):
    """Cauda longa de engajamento -- mesma justificativa de
    synthetic_events.py: sem isso, o histórico médio por cliente fica
    raso demais pro problema de degradação de contexto aparecer."""
    raw = 1.0 + np.random.pareto(alpha, size=n_customers)
    probs = raw / raw.sum()
    return probs


def gen_event_log(identities, n_events=N_EVENTS, n_entities=N_ENTITIES,
                   n_niches=N_NICHES):
    """Gera o log de eventos multi-canal, no vocabulário de fatura.

    `entity_id` aqui representa uma linha/produto específico do cliente
    (ex: "ent_12" = uma das linhas móveis dele) -- o intent de fatura
    sempre se refere a UM produto/linha por evento, assim como no Excel
    real cada interação tinha um id_document e um journey_details.
    """
    niche_entities = {
        n: random.sample(range(n_entities), k=max(2, n_entities // n_niches * 2))
        for n in range(n_niches)
    }
    global_ids = list(identities.keys())
    customer_niche = {gid: random.randint(0, n_niches - 1) for gid in global_ids}

    engagement_probs = _customer_engagement_weights(len(global_ids))
    gid_sequence = np.random.choice(global_ids, size=n_events, p=engagement_probs)

    rows = []
    t0 = 1_700_000_000
    for i in range(n_events):
        gid = gid_sequence[i]
        ident = identities[gid]

        available_channels = list(ident.channel_ids.keys())
        channel = random.choice(available_channels)
        local_id = ident.channel_ids[channel]

        niche = customer_niche[gid]
        if random.random() < 0.85:
            entity = random.choice(niche_entities[niche])
        else:
            entity = random.randint(0, n_entities - 1)

        intent = random.choices(INTENT_TYPES, weights=INTENT_WEIGHTS)[0]
        session_id = f"{local_id}_{i // 15}"
        ts = t0 + i * 30

        rows.append((
            local_id, channel, gid, intent, f"ent_{entity}",
            session_id, ts,
        ))

    log = pd.DataFrame(rows, columns=[
        "local_customer_id", "channel", "global_customer_id_TRUTH",
        "event_type", "entity_id", "session_id", "ts",
    ])
    return log, customer_niche


# ---------------------------------------------------------------------------
# GROUND TRUTH DE FATOS DO CLIENTE
# ---------------------------------------------------------------------------

def build_customer_facts(log, customer_niche):
    facts = {}
    for gid, grp in log.groupby("global_customer_id_TRUTH"):
        grp_sorted = grp.sort_values("ts", ascending=False)
        last = grp_sorted.iloc[0]

        niche_counts = grp["entity_id"].value_counts()
        total = niche_counts.sum()
        top_entity_share = float(niche_counts.iloc[0] / total) if total else 0.0

        risk = grp[grp.event_type == _RISK_INTENT]

        facts[gid] = CustomerFacts(
            global_id=gid,
            last_entity=last.entity_id,
            last_entity_channel=last.channel,
            last_entity_ts=int(last.ts),
            primary_niche=customer_niche[gid],
            primary_niche_share=top_entity_share,
            support_ticket_count=int(len(risk)),
            support_ticket_entities=risk.entity_id.unique().tolist(),
        )
    return facts


# ---------------------------------------------------------------------------
# API DE ALTO NÍVEL (mesma assinatura de synthetic_events.py)
# ---------------------------------------------------------------------------

def generate_dataset(n_customers=N_CUSTOMERS, n_events=N_EVENTS, seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

    identities = gen_identities(n_customers)
    log, customer_niche = gen_event_log(identities, n_events=n_events)
    facts = build_customer_facts(log, customer_niche)

    return {
        "identities": identities,
        "log": log,
        "customer_niche": customer_niche,
        "facts": facts,
    }


# ---------------------------------------------------------------------------
# DEMO / SANITY CHECK
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds = generate_dataset()
    log = ds["log"]
    facts = ds["facts"]

    print(f"Log: {len(log)} eventos, {log.global_customer_id_TRUTH.nunique()} clientes globais")
    print(f"\nDistribuição de intenções (deve lembrar a forma da base real -- "
          f"2ª via e pagamento dominando, contestação/bloqueio raros):")
    print(log.event_type.value_counts())

    events_per_customer = log.global_customer_id_TRUTH.value_counts()
    print(f"\nEventos por cliente -- mediana: {events_per_customer.median():.0f}, "
          f"máximo: {events_per_customer.max()}")

    example_gid = "cust_5"
    f = facts[example_gid]
    print(f"\n=== Ground truth de fatos: {example_gid} ===")
    print(f"  Última linha/produto acessado: {f.last_entity} "
          f"(canal={f.last_entity_channel}, ts={f.last_entity_ts})")
    print(f"  Perfil primário: {f.primary_niche} (share do produto top={f.primary_niche_share:.2f})")
    print(f"  Contestações/esclarecimentos: {f.support_ticket_count} -> produtos: {f.support_ticket_entities}")