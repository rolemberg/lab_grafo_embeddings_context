"""
metrics/hallucination.py

Verificador formal de resposta contra ground truth -- substitui os
placeholders simplificados usados em experiments/diagnostic.py
(check_answer) e experiments/comparative.py (answer_naive_contains).

POR QUE ISSO MERECE UM MÓDULO PRÓPRIO, NÃO SÓ UM BOOLEANO:
a pergunta que sustenta a seção 5 do artigo não é só "o modelo
acertou?", é "COMO ele errou, quando errou?". Um modelo que responde
"não tenho informação suficiente" quando o contexto não traz o dado
certo está se comportando de forma fundamentalmente diferente de um
que inventa uma entidade plausível com confiança. O primeiro é
incerteza honesta; o segundo é alucinação de fato -- e é exatamente
esse segundo caso que motiva a preocupação com degradação de contexto
no artigo. Colapsar os dois em um único "correct=False" (como os
experimentos faziam até agora) esconde a distinção.

Classificação de 4 vias (Verdict):
  CORRECT      -- resposta bate com o gabarito (com normalização/tolerância)
  HALLUCINATION -- resposta é específica, bem formada, e ERRADA (o caso perigoso)
  ABSTENTION   -- modelo explicitamente diz que não sabe / não tem info
  MALFORMED    -- resposta vazia ou não contém nada parseável no formato esperado
"""

from __future__ import annotations

import re
import unicodedata

CORRECT = "correct"
HALLUCINATION = "hallucination"
ABSTENTION = "abstention"
MALFORMED = "malformed"

# ---------------------------------------------------------------------------
# NORMALIZAÇÃO DE TEXTO
# ---------------------------------------------------------------------------

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _normalize(s: str) -> str:
    s = _strip_accents(s.lower().strip())
    s = re.sub(r"[^\w\s]", " ", s)  # tira pontuação
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ---------------------------------------------------------------------------
# DETECÇÃO DE ABSTENÇÃO
# ---------------------------------------------------------------------------

_ABSTENTION_PATTERNS = [
    "nao sei", "nao tenho informacao", "nao ha informacao",
    "nao e possivel determinar", "sem dados suficientes", "nao consta",
    "desconhecido", "nao encontrei", "informacao nao disponivel",
    "nao foi possivel", "nao ha dados", "i don't know", "i do not have",
    "insufficient information", "no information available",
]


def _is_abstention(normalized_text: str) -> bool:
    return any(pattern in normalized_text for pattern in _ABSTENTION_PATTERNS)


# ---------------------------------------------------------------------------
# EXTRAÇÃO DE VALORES TIPADOS DA RESPOSTA
# ---------------------------------------------------------------------------

_NUMBER_WORDS_PT = {
    "zero": 0, "um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3,
    "quatro": 4, "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9,
    "dez": 10, "onze": 11, "doze": 12, "treze": 13, "catorze": 14,
    "quatorze": 14, "quinze": 15, "dezesseis": 16, "dezessete": 17,
    "dezoito": 18, "dezenove": 19, "vinte": 20,
}


def _extract_number(normalized_text: str) -> int | None:
    """Tenta extrair um número da resposta -- primeiro dígitos, depois
    numerais escritos por extenso (só cobre 0-20; suficiente pro range
    de chamados de suporte deste experimento)."""
    m = re.search(r"\b(\d+)\b", normalized_text)
    if m:
        return int(m.group(1))
    for word, value in _NUMBER_WORDS_PT.items():
        if re.search(rf"\b{word}\b", normalized_text):
            return value
    return None


def _extract_entity_id(text: str) -> str | None:
    """Extrai um id de entidade no formato ent_N -- não normaliza pra
    minúsculas antes por precisão, mas aceita variação de caixa."""
    m = re.search(r"ent_(\d+)", text, flags=re.IGNORECASE)
    return f"ent_{m.group(1)}" if m else None


# ---------------------------------------------------------------------------
# CLASSIFICAÇÃO PRINCIPAL
# ---------------------------------------------------------------------------

def classify_answer(model_answer: str, expected_answer: str, fact_type: str) -> dict:
    """Classifica a resposta do modelo em CORRECT / HALLUCINATION /
    ABSTENTION / MALFORMED, com detalhes extras pra depuração.

    Retorna um dict: {"verdict": ..., "extracted": ..., "detail": ...}
    """
    if model_answer is None or not model_answer.strip():
        return {"verdict": MALFORMED, "extracted": None, "detail": "resposta vazia"}

    normalized = _normalize(model_answer)

    if fact_type == "last_entity":
        extracted = _extract_entity_id(model_answer)
        if extracted is None:
            if _is_abstention(normalized):
                return {"verdict": ABSTENTION, "extracted": None, "detail": "sem entidade, padrão de abstenção"}
            return {"verdict": MALFORMED, "extracted": None, "detail": "sem entidade parseável na resposta"}

        expected_norm = expected_answer.strip().lower()
        if extracted.lower() == expected_norm:
            return {"verdict": CORRECT, "extracted": extracted, "detail": None}
        return {"verdict": HALLUCINATION, "extracted": extracted,
                "detail": f"afirmou {extracted}, esperado {expected_answer}"}

    elif fact_type == "support_ticket_count":
        extracted = _extract_number(normalized)
        if extracted is None:
            if _is_abstention(normalized):
                return {"verdict": ABSTENTION, "extracted": None, "detail": "sem número, padrão de abstenção"}
            return {"verdict": MALFORMED, "extracted": None, "detail": "sem número parseável na resposta"}

        try:
            expected_n = int(expected_answer.strip())
        except ValueError:
            expected_n = None

        if expected_n is not None and extracted == expected_n:
            return {"verdict": CORRECT, "extracted": extracted, "detail": None}
        delta = (extracted - expected_n) if expected_n is not None else None
        return {"verdict": HALLUCINATION, "extracted": extracted,
                "detail": f"afirmou {extracted}, esperado {expected_n} (delta={delta})"}

    else:
        # fallback genérico: substring normalizada, sem categorização fina
        if _is_abstention(normalized):
            return {"verdict": ABSTENTION, "extracted": None, "detail": "padrão de abstenção"}
        expected_norm = _normalize(expected_answer)
        if expected_norm in normalized:
            return {"verdict": CORRECT, "extracted": model_answer.strip(), "detail": None}
        return {"verdict": HALLUCINATION, "extracted": model_answer.strip(),
                "detail": "resposta não contém o valor esperado"}


# ---------------------------------------------------------------------------
# API DE COMPATIBILIDADE (usada por experiments/diagnostic.py e comparative.py)
# ---------------------------------------------------------------------------

def check_answer(model_answer: str, expected_answer: str, fact_type: str = "generic") -> bool:
    """Versão booleana -- mantém compatibilidade com o que os
    experimentos já usavam, mas agora é um wrapper de classify_answer.
    Prefira usar classify_answer diretamente para reter a distinção
    entre alucinação/abstenção/malformado."""
    return classify_answer(model_answer, expected_answer, fact_type)["verdict"] == CORRECT


# ---------------------------------------------------------------------------
# DEMO / SANITY CHECK
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cases = [
        # (model_answer, expected_answer, fact_type, esperado)
        ("ent_22", "ent_22", "last_entity", CORRECT),
        ("A entidade foi ent_22.", "ent_22", "last_entity", CORRECT),
        ("Foi ent_9, tenho certeza.", "ent_22", "last_entity", HALLUCINATION),
        ("Não tenho informação suficiente sobre isso.", "ent_22", "last_entity", ABSTENTION),
        ("", "ent_22", "last_entity", MALFORMED),
        ("O cliente parece satisfeito.", "ent_22", "last_entity", MALFORMED),

        ("3", "3", "support_ticket_count", CORRECT),
        ("O cliente abriu três chamados.", "3", "support_ticket_count", CORRECT),
        ("5", "3", "support_ticket_count", HALLUCINATION),
        ("Não há dados suficientes para responder.", "3", "support_ticket_count", ABSTENTION),
        ("Não sei quantos.", "0", "support_ticket_count", ABSTENTION),
    ]

    print(f"{'resposta do modelo':45s} {'esperado':10s} {'fact_type':22s} {'veredito':14s} {'ok?'}")
    n_correct_predictions = 0
    for model_answer, expected_answer, fact_type, expected_verdict in cases:
        result = classify_answer(model_answer, expected_answer, fact_type)
        ok = "OK" if result["verdict"] == expected_verdict else "  <-- DIVERGE"
        if result["verdict"] == expected_verdict:
            n_correct_predictions += 1
        print(f"{model_answer[:43]:45s} {expected_answer:10s} {fact_type:22s} "
              f"{result['verdict']:14s} {ok}")

    print(f"\n{n_correct_predictions}/{len(cases)} casos de teste bateram com o veredito esperado.")