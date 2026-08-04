"""
embeddings/semantic.py

Embedding de TEXTO (nome/telefone ruidoso) usado na resolução de
identidade probabilística (seção 4.1 do artigo) -- substitui o proxy
ingênuo baseado em difflib.SequenceMatcher usado inicialmente em
graph/build_graph.py.

Método: TF-IDF sobre n-gramas de CARACTERE (não de palavra) + cosseno.
N-gramas de caractere são a escolha certa aqui porque o "documento" é
um nome curto e ruidoso (ex: "Joao Silva" vs "Joao Silv4") -- n-gramas
de palavra tratariam isso como tokens completamente diferentes; de
caractere capturam a proximidade real mesmo com erro de digitação.

Isso é uma técnica clássica e genuinamente usada em sistemas de
resolução de identidade / entity matching em produção (ex: como parte
de um pipeline de blocking + matching) -- não é um proxy de brinquedo
como o difflib. Em produção com mais contexto disponível (endereço,
telefone estruturado, e-mail), a evolução natural seria um embedding
treinado (ex: sentence-transformers) sobre texto concatenado, mas
TF-IDF de char n-gram já é uma baseline forte e interpretável.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHAR_NGRAM_RANGE


# ---------------------------------------------------------------------------
# API PRINCIPAL
# ---------------------------------------------------------------------------

def fit_text_vectorizer(texts: list[str]) -> TfidfVectorizer:
    """Ajusta um TF-IDF de n-gramas de caractere sobre um corpus de
    textos curtos (nomes/telefones ruidosos). Deve ser ajustado UMA VEZ
    sobre todo o corpus disponível (todos os canais), não por par."""
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",  # n-gramas de caractere respeitando fronteiras de palavra
        ngram_range=CHAR_NGRAM_RANGE,
        min_df=1,
    )
    vectorizer.fit(texts)
    return vectorizer


def embed_texts(vectorizer: TfidfVectorizer, texts: list[str]):
    """Retorna matriz esparsa TF-IDF (n_texts, n_features)."""
    return vectorizer.transform(texts)


def similarity(vectorizer: TfidfVectorizer, text_a: str, text_b: str) -> float:
    """Similaridade de cosseno entre dois textos individuais. Uso pontual
    (par a par) -- para muitas comparações, prefira embed_texts +
    cosine_similarity em lote (mais rápido, evita refazer o transform)."""
    vecs = vectorizer.transform([text_a, text_b])
    return float(cosine_similarity(vecs[0], vecs[1])[0, 0])


def top_k_similar(vectorizer: TfidfVectorizer, query_text: str,
                   candidate_texts: list[str], k: int = 5):
    """Dado um texto de consulta e uma lista de candidatos, devolve os
    top-k (índice, score) mais similares por cosseno. Usado após o
    blocking comportamental em build_graph.py -- já reduz o universo de
    candidatos antes de chegar aqui."""
    query_vec = vectorizer.transform([query_text])
    cand_vecs = vectorizer.transform(candidate_texts)
    sims = cosine_similarity(query_vec, cand_vecs)[0]
    order = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in order]


# ---------------------------------------------------------------------------
# DEMO / COMPARAÇÃO COM O PROXY ANTERIOR (difflib)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import difflib

    # pares de teste: (nome original, variante ruidosa, deveria bater?)
    pairs = [
        ("Joao Silva", "Joao Silvz", True),      # 1 typo
        ("Joao Silva", "Jofo Silvz", True),      # 2 typos
        ("Joao Silva", "Maria Souza", False),    # pessoa diferente
        ("Ana Pereira", "Ana Pereura", True),    # 1 typo
        ("Ana Pereira", "Ana Ferreira", False),  # sobrenome parecido, pessoa diferente
        ("Carlos Mendes", "Carlos Mendez", True),
    ]

    corpus = [p[0] for p in pairs] + [p[1] for p in pairs]
    vectorizer = fit_text_vectorizer(corpus)

    print(f"{'nome_a':15s} {'nome_b':15s} {'esperado':9s} {'tfidf_cos':10s} {'difflib':10s}")
    for a, b, expected in pairs:
        tfidf_score = similarity(vectorizer, a, b)
        difflib_score = difflib.SequenceMatcher(None, a, b).ratio()
        flag = "" if (tfidf_score > 0.5) == expected else "  <-- diverge do esperado"
        print(f"{a:15s} {b:15s} {str(expected):9s} {tfidf_score:.4f}    {difflib_score:.4f}{flag}")

    print("\n=== top_k_similar: buscando 'Joao Silva' entre candidatos ===")
    candidates = ["Joao Silvz", "Maria Souza", "Joana Silva", "Jofo Silvz", "Pedro Costa"]
    results = top_k_similar(vectorizer, "Joao Silva", candidates, k=3)
    for idx, score in results:
        print(f"  {candidates[idx]:15s} score={score:.4f}")