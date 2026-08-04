"""
data/synthetic_events.py

Gerador de base de eventos sintética MULTI-CANAL, com:
  1) Identidade fragmentada entre canais (algumas com chave exata,
     outras exigindo resolução probabilística) -> ground truth de
     identidade, usada para avaliar a etapa 4.1 do método.
  2) Ground truth de FATOS DO CLIENTE, usada nos experimentos 5.2/5.3
     como "resposta certa" que o agente deve produzir.

Este módulo é a base de todo o resto do pipeline (grafo, embeddings,
retrieval, experimentos) -- ele não depende de nenhum outro módulo do
projeto.

Domínio simulado:
  - 3 canais: "web", "app", "call_center"
  - web e app compartilham loyalty_id para uma fração dos clientes
    (chave exata de identidade)
  - call_center nunca tem chave exata -> precisa ser resolvido via
    similaridade de texto (nome/telefone parcial simulados)
  - clientes têm "nichos" de interesse -> estrutura de comunidade real
    no grafo resultante
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
    SEED, N_CUSTOMERS, N_ENTITIES, N_NICHES, EVENT_TYPES, EVENT_TYPE_WEIGHTS,
    CHANNELS, FRAC_WITH_LOYALTY_KEY, FRAC_WITH_CALL_CENTER, N_EVENTS,
    CUSTOMER_ENGAGEMENT_ALPHA,
)


# ---------------------------------------------------------------------------
# ESTRUTURAS DE GROUND TRUTH
# ---------------------------------------------------------------------------

@dataclass
class ChannelIdentity:
    """Mapeia um customer_id "canônico" (global) para seus IDs por canal."""
    global_id: str
    channel_ids: dict = field(default_factory=dict)   # {channel: channel_local_id}
    has_exact_key: dict = field(default_factory=dict) # {channel: bool}
    noisy_text: dict = field(default_factory=dict)     # {channel: str} p/ resolução probabilística


@dataclass
class CustomerFacts:
    """Ground truth de fatos do cliente -- usado como resposta certa nos
    experimentos de diagnóstico (5.2) e comparativo (5.3)."""
    global_id: str
    last_entity: str | None
    last_entity_channel: str | None
    last_entity_ts: int | None
    primary_niche: int
    primary_niche_share: float
    support_ticket_count: int
    support_ticket_entities: list


# ---------------------------------------------------------------------------
# GERAÇÃO DE IDENTIDADE MULTI-CANAL
# ---------------------------------------------------------------------------

def _random_name():
    first = "".join(random.choices(string.ascii_lowercase, k=6)).capitalize()
    last = "".join(random.choices(string.ascii_lowercase, k=8)).capitalize()
    return f"{first} {last}"


def _noisy_variant(text, p_typo=0.15):
    """Introduz ruído leve no texto p/ simular variação entre canais
    (ex: nome digitado diferente no call center)."""
    chars = list(text)
    for i in range(len(chars)):
        if chars[i].isalpha() and random.random() < p_typo:
            chars[i] = random.choice(string.ascii_lowercase)
    return "".join(chars)


def gen_identities(n_customers=N_CUSTOMERS):
    """Gera a estrutura de identidade fragmentada por canal.

    web: sempre presente, id = cust_{i}_web
    app: presente para todos; compartilha loyalty_id com web para uma
         fração dos clientes (chave exata) -- o resto tem app_id
         independente, sem chave, precisa resolução por similaridade.
    call_center: presente só para uma fração; nunca tem chave exata.
    """
    identities = {}
    for i in range(n_customers):
        gid = f"cust_{i}"
        name = _random_name()
        ident = ChannelIdentity(global_id=gid)

        # web: sempre tem, e é a "raiz" -- id local == id global por convenção
        ident.channel_ids["web"] = f"web_{i}"
        ident.has_exact_key["web"] = True
        ident.noisy_text["web"] = name

        # app: chave exata (loyalty_id) só para uma fração
        has_loyalty = random.random() < FRAC_WITH_LOYALTY_KEY
        if has_loyalty:
            ident.channel_ids["app"] = f"app_{i}"  # mesma raiz -> chave exata
            ident.has_exact_key["app"] = True
            ident.noisy_text["app"] = name
        else:
            ident.channel_ids["app"] = f"app_anon_{i}"
            ident.has_exact_key["app"] = False
            ident.noisy_text["app"] = _noisy_variant(name)

        # call_center: nunca tem chave exata; só uma fração interage por lá
        if random.random() < FRAC_WITH_CALL_CENTER:
            ident.channel_ids["call_center"] = f"cc_anon_{i}"
            ident.has_exact_key["call_center"] = False
            ident.noisy_text["call_center"] = _noisy_variant(name, p_typo=0.25)

        identities[gid] = ident

    return identities


# ---------------------------------------------------------------------------
# GERAÇÃO DO LOG DE EVENTOS
# ---------------------------------------------------------------------------

def _customer_engagement_weights(n_customers, alpha=CUSTOMER_ENGAGEMENT_ALPHA):
    """Gera um peso de engajamento por cliente via distribuição de Pareto,
    simulando a realidade omnichannel: uma minoria de clientes gera a
    maior parte do volume de eventos (long tail), a maioria tem
    histórico curto. Sem isso, o histórico médio por cliente fica raso
    demais para o problema de degradação de contexto (seção 5.2) sequer
    aparecer -- ver discussão em agent/tool_contract.py."""
    raw = 1.0 + np.random.pareto(alpha, size=n_customers)
    probs = raw / raw.sum()
    return probs


def gen_event_log(identities, n_events=N_EVENTS, n_entities=N_ENTITIES,
                   n_niches=N_NICHES):
    """Gera o log de eventos multi-canal.

    Cada linha carrega tanto o ID local do canal (o que um sistema real
    veria antes de qualquer resolução de identidade) quanto o global_id
    (ground truth, usado só para avaliação -- não deve vazar para o
    pipeline de retrieval).

    A escolha de QUAL cliente gera cada evento não é uniforme -- segue
    uma distribuição de Pareto (_customer_engagement_weights), criando
    uma cauda longa realista de engajamento.
    """
    niche_entities = {
        n: random.sample(range(n_entities), k=max(2, n_entities // n_niches * 2))
        for n in range(n_niches)
    }
    global_ids = list(identities.keys())
    customer_niche = {gid: random.randint(0, n_niches - 1) for gid in global_ids}

    # sorteia TODOS os clientes-donos-de-evento de uma vez, de forma
    # vetorizada, respeitando o peso de engajamento de cada um
    engagement_probs = _customer_engagement_weights(len(global_ids))
    gid_sequence = np.random.choice(global_ids, size=n_events, p=engagement_probs)

    rows = []
    t0 = 1_700_000_000
    for i in range(n_events):
        gid = gid_sequence[i]
        ident = identities[gid]

        # escolhe canal dentre os disponíveis para esse cliente
        available_channels = list(ident.channel_ids.keys())
        channel = random.choice(available_channels)
        local_id = ident.channel_ids[channel]

        niche = customer_niche[gid]
        if random.random() < 0.85:
            entity = random.choice(niche_entities[niche])
        else:
            entity = random.randint(0, n_entities - 1)

        event_type = random.choices(EVENT_TYPES, weights=EVENT_TYPE_WEIGHTS)[0]
        session_id = f"{local_id}_{i // 15}"
        ts = t0 + i * 30

        rows.append((
            local_id, channel, gid, event_type, f"ent_{entity}",
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

def build_customer_facts(log, customer_niche, niche_entities=None):
    """Deriva o ground truth de fatos por cliente global, direto do log
    (usando global_customer_id_TRUTH -- ou seja, com identidade já
    resolvida). Isso é o "gabarito" contra o qual o verificador de
    alucinação (metrics/hallucination.py) vai comparar as respostas do
    agente."""
    facts = {}
    for gid, grp in log.groupby("global_customer_id_TRUTH"):
        grp_sorted = grp.sort_values("ts", ascending=False)
        last = grp_sorted.iloc[0]

        niche_counts = grp["entity_id"].value_counts()
        total = niche_counts.sum()
        # nicho primário = canal de entidade mais frequente, como proxy
        top_entity_share = float(niche_counts.iloc[0] / total) if total else 0.0

        tickets = grp[grp.event_type == "support_ticket"]

        facts[gid] = CustomerFacts(
            global_id=gid,
            last_entity=last.entity_id,
            last_entity_channel=last.channel,
            last_entity_ts=int(last.ts),
            primary_niche=customer_niche[gid],
            primary_niche_share=top_entity_share,
            support_ticket_count=int(len(tickets)),
            support_ticket_entities=tickets.entity_id.unique().tolist(),
        )
    return facts


# ---------------------------------------------------------------------------
# API DE ALTO NÍVEL
# ---------------------------------------------------------------------------

def generate_dataset(n_customers=N_CUSTOMERS, n_events=N_EVENTS, seed=SEED):
    """Ponto de entrada único: gera identidades, log de eventos e ground
    truth de fatos, de forma reprodutível."""
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
    identities = ds["identities"]

    print(f"Log: {len(log)} eventos, {log.global_customer_id_TRUTH.nunique()} clientes globais")
    print(f"Canais: {log.channel.value_counts().to_dict()}")

    events_per_customer = log.global_customer_id_TRUTH.value_counts()
    top1pct_n = max(1, len(events_per_customer) // 100)
    top1pct_share = events_per_customer.iloc[:top1pct_n].sum() / events_per_customer.sum()
    print(f"\nDistribuição de eventos por cliente:")
    print(f"  mínimo:  {events_per_customer.min()}")
    print(f"  mediana: {events_per_customer.median():.0f}")
    print(f"  média:   {events_per_customer.mean():.1f}")
    print(f"  máximo:  {events_per_customer.max()}")
    print(f"  top 1% dos clientes concentram {top1pct_share:.1%} dos eventos")

    n_no_key = sum(
        1 for ident in identities.values()
        for ch, has_key in ident.has_exact_key.items() if not has_key
    )
    print(f"Vínculos canal->cliente SEM chave exata (exigem resolução): {n_no_key}")

    example_gid = "cust_5"
    print(f"\n=== Ground truth de fatos: {example_gid} ===")
    f = facts[example_gid]
    print(f"  Última entidade: {f.last_entity} (canal={f.last_entity_channel}, ts={f.last_entity_ts})")
    print(f"  Nicho primário: {f.primary_niche} (share da entidade top={f.primary_niche_share:.2f})")
    print(f"  Chamados de suporte: {f.support_ticket_count} -> entidades: {f.support_ticket_entities}")

    print(f"\n=== Identidade fragmentada: {example_gid} ===")
    ident = identities[example_gid]
    for ch, local_id in ident.channel_ids.items():
        print(f"  {ch:12s} local_id={local_id:20s} chave_exata={ident.has_exact_key[ch]}")