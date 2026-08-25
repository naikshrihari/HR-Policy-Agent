"""Port of TOPIC_CLASSIFICATION_SCRIPT (+ Spanish).

A weighted unigram/bigram vocabulary model that classifies the turn's topic from the
user query (primary) with damped assistance from the agent answer (secondary).  The
English and Spanish routes share this engine and differ only in their vocabulary and a
handful of tuning constants, supplied via :mod:`topic_vocab`.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List

from . import topic_vocab

# ---- CONFIG (identical across both routes) --------------------------------
MIN_SCORE = 2
ACK_MAX_TOKENS = 4
FALLBACK_HR_TOPIC = "POLICY_INQUIRY"
FALLBACK_TOPIC = "OFF_TOPIC"
ANSWER_WEIGHT = 0.5
ANSWER_ONLY_MIN = 3
ANSWER_ASSISTED = ["POLICY_INQUIRY"]
ANSWER_MAX_CHARS = 1500

_BASE_STOPWORDS_EN = [
    "a", "an", "the", "and", "or", "but", "to", "of", "in", "on", "at", "for", "with",
    "about", "is", "are", "am", "was", "were", "be", "been", "being", "does", "did",
    "can", "could", "will", "would", "should", "have", "has", "had", "keep", "keeps",
    "this", "that", "these", "those", "it", "its", "there", "get", "gets", "i", "im",
    "we", "us", "so", "just", "please", "also", "some", "any", "if", "as",
]
_BASE_STOPWORDS_ES = [
    "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "u", "de", "del",
    "en", "al", "que", "es", "son", "esta", "estan", "estoy", "fue", "ser", "hay",
    "por", "con", "se", "le", "les", "lo", "ya", "si", "pero", "tambien", "muy",
]

AUDIENCE = {
    "user", "users", "member", "members", "employee", "employees", "worker", "workers",
    "team", "tm", "tms", "staff", "empleado", "empleados", "miembro", "miembros",
    "usuario", "usuarios", "trabajador", "trabajadores", "personal",
}

PERSON_NOUNS = [
    "coworker", "coworkers", "worker", "manager", "supervisor", "boss", "lead", "he",
    "she", "guy", "guys", "companero", "companera", "companeros", "jefe", "jefa",
    "gerente", "colega", "ella", "alguien", "supervisora",
]
BEHAVIOR_WORDS = [
    "inappropriate", "creepy", "unwanted", "touching", "staring", "following",
    "harassing", "inapropiado", "inapropiada", "inapropiados", "inapropiadas",
]
LEAVE_NOUNS = ["pl", "pto", "vacation", "vacations", "vacaciones", "sick", "days",
               "dias", "hours", "horas", "leave"]
PAY_NOUNS = ["paycheck", "paychecks", "pay", "check", "checks", "cheque", "pago",
             "wages", "salary", "salario", "sueldo", "quincena", "paga", "nomina"]
ERROR_NOUNS = ["correction", "corrections", "wrong", "incorrect", "error", "errors",
               "mistake", "missing", "short", "shorted", "correccion", "equivocado",
               "equivocada", "incorrecto", "incorrecta", "erroneo", "erronea", "falta",
               "faltan", "faltante", "faltantes"]

TOPIC_ORDER = [
    "CRISIS", "HARASSMENT_REPORT", "UNION_INQUIRY", "DISPUTE", "COMPLAINT",
    "PERSONAL_DATA", "POLICY_INQUIRY", "META_REQUEST", "CLARIFICATION", "ACKNOWLEDGMENT",
]
_VOCAB_KEY = {
    "CRISIS": "V_CRISIS", "HARASSMENT_REPORT": "V_HARASSMENT", "UNION_INQUIRY": "V_UNION",
    "DISPUTE": "V_DISPUTE", "COMPLAINT": "V_COMPLAINT", "PERSONAL_DATA": "V_PERSONAL_DATA",
    "POLICY_INQUIRY": "V_POLICY", "META_REQUEST": "V_META", "CLARIFICATION": "V_CLARIFICATION",
    "ACKNOWLEDGMENT": "V_ACK",
}


def _norm(s: str) -> str:
    s = str(s).lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _word_hits(vocab_word: str, token: str) -> bool:
    if vocab_word.endswith("*"):
        return token.startswith(vocab_word[:-1])
    return token == vocab_word


def _uni_hit(vocab_word: str, toks: List[str]) -> bool:
    return any(_word_hits(vocab_word, t) for t in toks)


def _bi_hit(vocab_bigram: str, toks: List[str]) -> bool:
    # JS `[w1, w2] = bigram.split(" ")` takes the first two words, ignoring any extra.
    parts = vocab_bigram.split(" ")
    w1, w2 = parts[0], parts[1]
    for i in range(len(toks) - 1):
        if _word_hits(w1, toks[i]) and _word_hits(w2, toks[i + 1]):
            return True
    return False


def _score_vocab(vocab: Dict[str, List[str]], toks: List[str]) -> Dict[str, Any]:
    score = 0
    hits: List[str] = []
    for w in vocab.get("uni", []):
        if _uni_hit(w, toks):
            score += 1
            hits.append(w)
    for w in vocab.get("uniS", []):
        if _uni_hit(w, toks):
            score += 2
            hits.append(w)
    for b in vocab.get("bi", []):
        if _bi_hit(b, toks):
            score += 3
            hits.append(b)
    return {"score": score, "hits": hits}


def _answer_to_text(v: Any, depth: int = 0) -> str:
    if v is None or depth > 3:
        return ""
    if isinstance(v, str):
        t = v.strip()
        if t.startswith("{") or t.startswith("["):
            import json
            try:
                return _answer_to_text(json.loads(t), depth + 1)
            except json.JSONDecodeError:
                return t
        return t
    if isinstance(v, list):
        return " ".join(_answer_to_text(x, depth + 1) for x in v)
    if isinstance(v, dict):
        return " ".join(_answer_to_text(x, depth + 1) for x in v.values())
    return str(v)


def classify_topic(raw_input: str, raw_answer: Any, language: str = "EN") -> Dict[str, str]:
    """Return ``{"topic_matched": "<TOPIC>"}`` for the given query + answer."""
    cfg = topic_vocab.CONFIG["ES" if str(language).upper() == "ES" else "EN"]
    vocab_all = cfg["vocab"]

    stop = set(_BASE_STOPWORDS_EN) | set(_BASE_STOPWORDS_ES) | set(cfg["stopwords_extra_es"])
    workforce = set(cfg["workforce"])
    audience_workforce_order = cfg["audience_workforce_order"]

    def tokenize(s: str) -> List[str]:
        return [t for t in _norm(s).split(" ") if t and t not in stop]

    raw_answer_text = _answer_to_text(raw_answer)[:ANSWER_MAX_CHARS]

    tokens_all = tokenize(raw_input)

    def drop_workforce_scope(toks: List[str]) -> List[str]:
        out: List[str] = []
        i = 0
        while i < len(toks):
            t = toks[i]
            nx = toks[i + 1] if i + 1 < len(toks) else None
            if t == "non" and nx in ("rep", "represented"):
                i += 2
                continue
            if t in workforce:
                i += 2 if (nx and nx in AUDIENCE) else 1
                continue
            if t == "union" and nx and nx in AUDIENCE:
                i += 2
                continue
            if t in AUDIENCE and nx == "sindicato":
                i += 2
                continue
            if audience_workforce_order and t in AUDIENCE and nx and nx in workforce:
                i += 2
                continue
            out.append(t)
            i += 1
        return out

    tokens_scoped = drop_workforce_scope(tokens_all)
    tokens_answer = drop_workforce_scope(tokenize(raw_answer_text))

    def has_tok(lst: List[str]) -> bool:
        return any(t in lst for t in tokens_scoped)

    norm_input = _norm(raw_input)
    balance_bonus = 3 if (re.search(cfg["balance_regex"], norm_input) and has_tok(LEAVE_NOUNS)) else 0
    pay_error_bonus = 3 if (has_tok(PAY_NOUNS) and has_tok(ERROR_NOUNS)) else 0
    harass_bonus = 3 if (has_tok(PERSON_NOUNS) and has_tok(BEHAVIOR_WORDS)) else 0
    has_first_person = bool(re.search(cfg["first_person_regex"], norm_input))

    def toks_for(name: str) -> List[str]:
        return tokens_all if name == "UNION_INQUIRY" else tokens_scoped

    def gate_ok(name: str) -> bool:
        if name == "PERSONAL_DATA":
            return has_first_person
        if name == "ACKNOWLEDGMENT":
            return len(tokens_scoped) <= ACK_MAX_TOKENS
        return True

    def bonus_for(name: str) -> int:
        if name == "HARASSMENT_REPORT":
            return harass_bonus
        if name == "DISPUTE":
            return pay_error_bonus
        if name == "PERSONAL_DATA":
            return balance_bonus
        return 0

    if not tokens_all:
        return {"topic_matched": FALLBACK_TOPIC}

    scores: Dict[str, Dict[str, Any]] = {}
    winner = None

    # PASS 1 — query-driven, first match wins.
    for name in TOPIC_ORDER:
        vocab = vocab_all[_VOCAB_KEY[name]]
        gok = gate_ok(name)
        q = _score_vocab(vocab, toks_for(name)) if gok else {"score": 0, "hits": []}
        b = bonus_for(name) if gok else 0
        if b > 0:
            q["score"] += b
        assisted = gok and len(tokens_answer) > 0 and name in ANSWER_ASSISTED
        a = _score_vocab(vocab, tokens_answer) if assisted else {"score": 0, "hits": []}
        a_w = ANSWER_WEIGHT * a["score"]
        scores[name] = {"query": q["score"], "answer": a["score"], "blended": q["score"] + a_w}
        eligible = gok and (
            q["score"] >= MIN_SCORE
            or (assisted and q["score"] >= 1 and q["score"] + a_w >= MIN_SCORE)
        )
        if winner is None and eligible:
            winner = name
    if winner:
        return {"topic_matched": winner}

    # PASS 2 — answer-only rescue for assisted topics.
    for name in ANSWER_ASSISTED:
        if len(tokens_answer) > 0 and (ANSWER_WEIGHT * scores.get(name, {}).get("answer", 0)) >= ANSWER_ONLY_MIN:
            return {"topic_matched": name}

    hr = _score_vocab(vocab_all["V_HR_ISH"], tokens_scoped)
    if hr["score"] >= 1:
        return {"topic_matched": FALLBACK_HR_TOPIC}
    return {"topic_matched": FALLBACK_TOPIC}
