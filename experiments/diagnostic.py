"""
experiments/diagnostic.py

Experimento de DIAGNÓSTICO (seção 5.2 do artigo): mede como o
acerto/alucinação do agente degrada em função de:
  (a) % de RUÍDO no contexto (distratores irrelevantes misturados com
      a informação que responde a pergunta)
  (b) POSIÇÃO da informação relevante dentro do contexto (início, meio, fim)

Clássico setup "needle in a haystack" / lost-in-the-middle, aplicado ao
domínio de contexto de cliente (em vez de texto genérico).

====================================================================
IMPORTANTE SOBRE EXECUÇÃO REAL -- LEIA ANTES DE INTERPRETAR RESULTADOS
====================================================================
Este harness roda de ponta a ponta neste ambiente com um respondedor
de SANITY-CHECK (answer_naive_keyword_search) que NÃO é um LLM -- é um
parser determinístico que só confirma que o alvo está de fato embutido
e extraível no contexto construído. Ele é, por construção, ~100% de
acerto em qualquer condição de ruído/posição -- serve pra validar a
MECÂNICA do experimento, não pra medir degradação nenhuma.

Para medir degradação de verdade, troque ANSWER_FN por
answer_with_local_hf_model(...), que usa o modelo local
"ibm-granite/granite-4.1-8b" via transformers/AutoModelForCausalLM.
Essa função só roda NA SUA MÁQUINA (onde o modelo já está em cache em
~/.cache/huggingface/hub/models--ibm-granite--granite-4.1-8b) -- não
neste ambiente de execução, que não tem GPU, não tem acesso a esse
cache local, e não tem rede liberada para huggingface.co.
"""

from __future__ import annotations

import os
import random
import re
import sys
from functools import lru_cache

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.hallucination import classify_answer, check_answer  # noqa: F401 -- check_answer mantido por compat

# ---------------------------------------------------------------------------
# CONFIG DO EXPERIMENTO
# ---------------------------------------------------------------------------

from config import (
    SEED,
    NOISE_LEVELS,
    POSITIONS,
    TOTAL_CONTEXT_LINES,
    FACT_TYPES,
    N_CUSTOMERS_PER_CONDITION,
    LLM_MODEL_PATH,
    LLM_MODEL_ID,
    LLM_MAX_INPUT_TOKENS,
    LLM_MAX_NEW_TOKENS,
    LLM_CONTEXT_CHAR_BUDGET,
)


_FORCE_CPU_AFTER_MPS_OOM = False


# ---------------------------------------------------------------------------
# FORMATAÇÃO DE LINHAS DE EVENTO (contexto e ruído usam o mesmo formato --
# ruído tem que ser plausível, não obviamente descartável)
# ---------------------------------------------------------------------------

def _format_event_line(row) -> str:
    return (f"- Evento: interação com {row.entity_id} via {row.event_type} "
            f"em ts={int(row.ts)} (canal={row.channel})")


def _format_target_line(fact_type: str, facts) -> tuple[str, str, str]:
    """Constrói a frase-alvo (a informação que a pergunta busca), a
    pergunta em si, e a resposta esperada, a partir do ground truth
    (data/synthetic_events.py::CustomerFacts)."""
    if fact_type == "last_entity":
        target_line = (f"- Evento: última interação registrada foi com "
                        f"{facts.last_entity} em ts={facts.last_entity_ts} "
                        f"(canal={facts.last_entity_channel})")
        question = "Qual foi a última entidade (produto/página) com que este cliente interagiu?"
        expected_answer = facts.last_entity

    elif fact_type == "support_ticket_count":
        target_line = (f"- Evento: cliente abriu {facts.support_ticket_count} "
                        f"chamado(s) de suporte no total")
        question = "Quantos chamados de suporte este cliente abriu no total?"
        expected_answer = str(facts.support_ticket_count)

    else:
        raise ValueError(f"fact_type desconhecido: {fact_type}")

    return target_line, question, expected_answer


# ---------------------------------------------------------------------------
# POOL DE RUÍDO (eventos de OUTROS clientes, mesmo formato do alvo)
# ---------------------------------------------------------------------------

def _build_noise_pool(log, exclude_local_ids: set, pool_size=2000, seed=None):
    """Amostra linhas de evento de clientes DIFERENTES do alvo (exclui
    os local_customer_id verdadeiros do cliente sendo testado, pra não
    vazar sinal relevante disfarçado de ruído)."""
    candidates = log[~log.local_customer_id.isin(exclude_local_ids)]
    sample = candidates.sample(n=min(pool_size, len(candidates)), random_state=seed)
    return [_format_event_line(row) for row in sample.itertuples()]


# ---------------------------------------------------------------------------
# CONSTRUÇÃO DO CONTEXTO (needle in a haystack)
# ---------------------------------------------------------------------------

def build_context(target_line, noise_pool, noise_pct, position,
                   total_lines=TOTAL_CONTEXT_LINES, customer_id="cliente"):
    """Monta o texto de contexto: `target_line` embutido entre
    `n_noise` linhas de ruído, na posição pedida (início/meio/fim)."""
    n_noise = int(round(noise_pct * (total_lines - 1)))
    noise_lines = random.sample(noise_pool, k=min(n_noise, len(noise_pool)))

    lines = list(noise_lines)
    if position == "inicio":
        idx = 0
    elif position == "fim":
        idx = len(lines)
    else:  # meio
        idx = len(lines) // 2
    lines.insert(idx, target_line)

    header = f"Segue o histórico de eventos do cliente {customer_id}:"
    return header + "\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# RESPONDEDORES (interface plugável -- ANSWER_FN é o que o experimento chama)
# ---------------------------------------------------------------------------

def answer_naive_keyword_search(question, context, fact_type):
    """SANITY-CHECK APENAS -- não é um LLM. Faz parsing determinístico
    procurando a linha com o formato específico da frase-alvo (ver
    _format_target_line) e extrai o valor. Serve só pra confirmar que
    o alvo está embutido e extraível no contexto construído -- é, por
    construção, praticamente imune a ruído/posição, então NUNCA deve
    ser interpretado como medida de degradação real."""
    if fact_type == "last_entity":
        m = re.search(r"última interação registrada foi com (ent_\d+)", context)
        return m.group(1) if m else ""
    elif fact_type == "support_ticket_count":
        m = re.search(r"cliente abriu (\d+) chamado", context)
        return m.group(1) if m else ""
    return ""


def _resolve_model_source(model_name_or_path: str | None) -> str:
    """Resolve um caminho de modelo utilizável pelo transformers.

    Se `model_name_or_path` apontar para a raiz do cache do HF
    (models--... com subpasta snapshots/), retorna um snapshot concreto.
    """
    candidate = model_name_or_path or LLM_MODEL_PATH or LLM_MODEL_ID

    if not isinstance(candidate, str):
        return LLM_MODEL_ID

    if not os.path.isdir(candidate):
        return candidate

    # Caso comum: já aponta para a pasta correta (com tokenizer/config).
    if os.path.exists(os.path.join(candidate, "tokenizer.json")):
        return candidate

    snapshots_dir = os.path.join(candidate, "snapshots")
    refs_main = os.path.join(candidate, "refs", "main")

    if not os.path.isdir(snapshots_dir):
        return candidate

    # Preferir o snapshot referenciado por refs/main, quando existir.
    if os.path.exists(refs_main):
        with open(refs_main, "r", encoding="utf-8") as f:
            snap = f.read().strip()
        resolved = os.path.join(snapshots_dir, snap)
        if snap and os.path.isdir(resolved):
            return resolved

    # Fallback: primeiro snapshot disponível (ordenado para estabilidade).
    snapshots = sorted(
        d for d in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, d))
    )
    if snapshots:
        return os.path.join(snapshots_dir, snapshots[0])

    return candidate


@lru_cache(maxsize=4)
def _load_local_hf_stack(model_source: str, device: str):
    """Carrega tokenizer+modelo uma única vez por (fonte, device)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForCausalLM.from_pretrained(model_source, torch_dtype="auto")
    model.to(device)
    model.eval()
    return tokenizer, model


def answer_with_local_hf_model(question, context,
                                model_name_or_path="ibm-granite/granite-4.1-8b",
                                device=None, max_new_tokens=None,
                                max_input_tokens=None, context_char_budget=None):
    """Respondedor REAL via modelo local (transformers). SÓ RODA NA SUA
    MÁQUINA -- requer `pip install transformers torch` e o modelo já em
    cache local. Não executa neste ambiente (sem GPU, sem acesso ao
    cache do usuário, sem rede para huggingface.co).

    Uso (na sua máquina):
        from experiments.diagnostic import answer_with_local_hf_model
        ANSWER_FN = answer_with_local_hf_model
    """
    import torch

    global _FORCE_CPU_AFTER_MPS_OOM

    max_new_tokens = max_new_tokens or LLM_MAX_NEW_TOKENS
    max_input_tokens = max_input_tokens or LLM_MAX_INPUT_TOKENS
    context_char_budget = context_char_budget or LLM_CONTEXT_CHAR_BUDGET

    if device is None:
        if _FORCE_CPU_AFTER_MPS_OOM:
            device = "cpu"
        elif torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    model_source = _resolve_model_source(model_name_or_path)
    tokenizer, model = _load_local_hf_stack(model_source, device)

    # Contextos muito grandes no comparativo explodem memória no MPS.
    if len(context) > context_char_budget:
        half = context_char_budget // 2
        context = context[:half] + "\n...[contexto truncado]...\n" + context[-half:]

    prompt = (f"Contexto:\n{context}\n\n"
              f"Pergunta: {question}\n"
              f"Responda apenas com o valor solicitado, sem explicação.\nResposta:")

    # BUG REAL encontrado rodando em escala (clientes "pesados", contexto
    # grande): tokenizer(..., truncation=True) trunca do lado DIREITO por
    # padrão -- corta o FIM do prompt, que é exatamente onde estão a
    # pergunta e "Resposta:". Isso produzia resposta vazia (malformed)
    # consistentemente em clientes com histórico grande o bastante pra
    # estourar max_input_tokens, disfarçado de "o modelo não sabe
    # responder" quando na verdade ele nunca viu a pergunta. Fix: truncar
    # do lado ESQUERDO -- se precisar cortar, corta contexto antigo, não
    # a pergunta.
    tokenizer.truncation_side = "left"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(device)

    with torch.inference_mode():
        try:
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        except RuntimeError as exc:
            msg = str(exc).lower()
            # Fallback automático: MPS pode estourar no comparativo em contextos longos.
            if device == "mps" and "out of memory" in msg:
                _FORCE_CPU_AFTER_MPS_OOM = True
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()

                cpu_tokenizer, cpu_model = _load_local_hf_stack(model_source, "cpu")
                cpu_tokenizer.truncation_side = "left"  # mesmo fix -- ver nota acima
                cpu_inputs = cpu_tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_input_tokens,
                ).to("cpu")
                output = cpu_model.generate(**cpu_inputs, max_new_tokens=max_new_tokens, do_sample=False)
                generated = output[0][cpu_inputs["input_ids"].shape[1]:]
                return cpu_tokenizer.decode(generated, skip_special_tokens=True).strip()
            raise

    generated = output[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # ACHADO NÃO TOTALMENTE EXPLICADO: em ~6% dos casos do segmento pesado
    # (fact_type support_ticket_count), a decodificação gulosa
    # (do_sample=False) produzia resposta vazia -- confirmado NÃO ser
    # truncamento de prompt (contexto bem abaixo do limite de tokens
    # nesses casos específicos). Causa raiz não identificada sem rodar o
    # modelo de verdade (não disponível no ambiente onde este código foi
    # escrito). Mitigação pragmática, não diagnóstico definitivo: se a
    # geração gulosa vier vazia, tenta 1x com amostragem -- documentar
    # isso explicitamente na Seção 7 se a taxa de retry for não-trivial
    # (adicione logging/contagem se for medir isso para o artigo).
    if not answer:
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=True, temperature=0.3, top_p=0.9)
        generated = output[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return answer


ANSWER_FN = answer_with_local_hf_model  # <-- troque aqui para medir degradação real


# ---------------------------------------------------------------------------
# CHECAGEM DE RESPOSTA (placeholder -- será formalizado em metrics/hallucination.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LOOP PRINCIPAL DO EXPERIMENTO
# ---------------------------------------------------------------------------

def run_diagnostic_experiment(dataset, answer_fn=None,
                               noise_levels=NOISE_LEVELS, positions=POSITIONS,
                               fact_types=FACT_TYPES,
                               n_customers_per_condition=N_CUSTOMERS_PER_CONDITION,
                               seed=SEED):
    answer_fn = answer_fn or ANSWER_FN
    random.seed(seed)

    log = dataset["log"]
    facts = dataset["facts"]
    identities = dataset["identities"]

    rows = []
    for fact_type in fact_types:
        # elegibilidade: só entram clientes onde o fato é "interessante"
        # (ex: pra contagem de chamados, exige >0 -- senão a pergunta é trivial)
        eligible = [
            gid for gid, f in facts.items()
            if (fact_type != "support_ticket_count" or f.support_ticket_count > 0)
        ]
        if len(eligible) == 0:
            continue
        trial_customers = random.sample(eligible, k=min(n_customers_per_condition, len(eligible)))

        for customer_id in trial_customers:
            f = facts[customer_id]
            target_line, question, expected_answer = _format_target_line(fact_type, f)

            true_local_ids = set(identities[customer_id].channel_ids.values())
            noise_pool = _build_noise_pool(log, exclude_local_ids=true_local_ids, seed=seed)

            for noise_pct in noise_levels:
                for position in positions:
                    if noise_pct == 0.0 and position != positions[0]:
                        continue  # sem ruído, posição não é definida -- evita trial duplicado

                    context = build_context(target_line, noise_pool, noise_pct, position,
                                             customer_id=customer_id)
                    model_answer = answer_fn(question, context, fact_type) \
                        if answer_fn is answer_naive_keyword_search \
                        else answer_fn(question, context)
                    verdict_info = classify_answer(model_answer, expected_answer, fact_type)
                    correct = verdict_info["verdict"] == "correct"

                    rows.append({
                        "customer_id": customer_id,
                        "fact_type": fact_type,
                        "noise_pct": noise_pct,
                        "position": position,
                        "expected_answer": expected_answer,
                        "model_answer": model_answer,
                        "correct": correct,
                        "verdict": verdict_info["verdict"],
                        "context_n_char": len(context),
                    })

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame):
    """Tabela pivô: acerto médio por (noise_pct, position)."""
    return results.pivot_table(index="noise_pct", columns="position",
                                values="correct", aggfunc="mean")


# ---------------------------------------------------------------------------
# DEMO / SANITY CHECK
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data.synthetic_events import generate_dataset

    ds = generate_dataset()

    print("=== Rodando harness com respondedor de SANITY-CHECK (não é LLM) ===")
    print("(confirma que o alvo é extraível do contexto -- não mede degradação real)\n")

    results = run_diagnostic_experiment(ds)
    print(f"Total de trials: {len(results)}")
    print(f"Acerto geral: {results.correct.mean():.1%}\n")

    print("Acerto por (ruído x posição):")
    print(summarize(results))

    print("\nAcerto por tipo de fato:")
    print(results.groupby("fact_type").correct.mean())

    print("\nExemplo de trial (linha crua):")
    print(results.iloc[0].to_dict())

    print("\n" + "=" * 70)
    print("Para medir degradação REAL, na sua máquina (com o Granite em cache):")
    print("  from experiments.diagnostic import run_diagnostic_experiment, answer_with_local_hf_model")
    print("  results = run_diagnostic_experiment(ds, answer_fn=answer_with_local_hf_model)")
    print("=" * 70)