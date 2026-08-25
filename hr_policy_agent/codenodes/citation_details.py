"""Port of GET_THE_CITATION_DETAILS (English + Spanish) — v7.7.

Renders card-style HTML "Sources" citations for the HR Policy Agent.  The two
original nodes share identical logic and differ only in user-visible LABELS, the
Spanish gap markers, and which RAG node codes they read; that variation is captured
in the ``LABELS`` / ``GAP_MARKERS`` tables and selected by ``language``.

Pipeline: gap-marker suppression -> gather RAG citations/supporting chunks ->
dedupe -> answer-coverage scoring (0.6*unigram + 0.4*bigram, normalized by
ANSWER_COV_FULL) -> relative gate + absolute floor + phrase-evidence gate ->
top-N selection -> excerpt extraction (sentence anchor / sliding window / snippet
degrade) -> HTML cards.  The production DEBUG_SCORES switch was False, so the
diagnostic table is not reproduced here.
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from typing import Any, Dict, List, Optional

# ---------------- Tunables (verbatim from v7.7) ----------------
USE_RELATIVE_GATE = True
REL_GATE = 0.60
ABS_FLOOR = 0.55
MIN_RELEVANCE = 0.80
MAX_CITATIONS = 2
FALLBACK_TOP_N = 2
FALLBACK_MIN_RELEVANCE = 0.05
ANSWER_COV_FULL = 0.4
MIN_PHRASE_EVIDENCE = 0.15
MIN_PHRASE_BIGRAMS = 8
SKIP_BIGRAMS = True
PHRASE_GATE_IN_FALLBACK = False
SNIPPET_LEN = 220
UPPER_RATIO = 0.6
UPPER_RUN_MIN = 4
OVERLAP_PREFIX_LEN = 250
SENT_MIN_SCORE = 0.30
SENT_NEIGHBOR_REL = 0.75
SENT_MAX_COUNT = 1
SENT_MIN_LEN = 25
SENT_MAX_CHARS = 180
WINDOW_MIN_SCORE = 0.20
SUPPRESS_LOW_SCORE_CARDS = False
HEADING_MIN_MATCH = 0.5
REQUIRE_ANSWER = True
UNIGRAM_W = 0.6
BIGRAM_W = 0.4

GAP_MARKERS = [
    "topic isn't covered", "topic is not covered", "topic not available",
    "topic isn't available", "topic is not available",
    "there is no specific information regarding",
    "No existe información específica con respecto a",
    "No hay información específica sobre",
    "policy does not specify",
    "no está cubierto", "no está disponible",
]

LABELS = {
    "EN": {"source": "Source", "sources": "Sources", "section": "Section",
           "show_full": "Show full excerpt"},
    "ES": {"source": "Fuente", "sources": "Fuentes", "section": "Sección",
           "show_full": "Mostrar extracto completo"},
}

_STOPWORDS = set((
    "a an and are as at be but by can could do does for from has have how i if in is it its "
    "may me my of on or our so than that the their them then there these they this to us was "
    "we were what when where which who will with would you your "
    "al como con cual cuando de del donde el ella ellos en es esta este esto la las lo los "
    "mas me mi mis no nos o para pero por que quien se si sin son su sus tu tus un una unas "
    "unos y ya "
    "company companys team member members station casinos casino policy policies "
    "employee employees employment work resources human hr shall must please refer "
    "details information following applicable eligible "
    "empresa equipo miembro miembros politica politicas empleado empleados trabajo "
    "without within other others another before after could would should also such "
    "among upon into onto per via any all more most only some may might made make "
    "makes both each than then still well being been because between during under "
    "over about above below according kindly additionally regarding respect further "
    "including includes included provide provides provided ensure ensures obtain "
    "obtained subject action taking take used using "
    "inside outside around across behind beyond toward towards against near"
).split())

_MONTH_NAMES = ["", "january", "february", "march", "april", "may2", "june", "july",
                "august", "september", "october", "november", "december"]


def _expand_dates(s: Any) -> str:
    def repl(m: "re.Match[str]") -> str:
        mi = int(m.group(1))
        name = _MONTH_NAMES[mi] if 1 <= mi <= 12 else ""
        return f" {name} {m.group(2)} {m.group(3)} "
    return re.sub(r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b", repl, str("" if s is None else s))


def _tokenize(s: Any) -> List[str]:
    t = _expand_dates(s).lower()
    t = unicodedata.normalize("NFD", t)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    out = []
    for w in t.split():
        if not w or w in _STOPWORDS:
            continue
        if re.fullmatch(r"\d+", w) or len(w) >= 3:
            out.append(w)
    return out


def _uniq(arr: List[str]) -> List[str]:
    seen, out = set(), []
    for w in arr:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _count_matches(terms: List[str], chunk_text: str) -> int:
    chunk_tokens = _tokenize(chunk_text)
    if not chunk_tokens or not terms:
        return 0
    matched = 0
    for qt in terms:
        for ct in chunk_tokens:
            if ct == qt or (len(qt) >= 5 and len(ct) >= 5 and ct[:5] == qt[:5]):
                matched += 1
                break
    return matched


def _build_bigrams(text: str) -> Dict[str, bool]:
    toks = _tokenize(text)
    s: Dict[str, bool] = {}
    for i in range(len(toks) - 1):
        s[toks[i] + "_" + toks[i + 1]] = True
    if SKIP_BIGRAMS:
        for i in range(len(toks) - 2):
            s[toks[i] + "_" + toks[i + 2]] = True
    return s


def _bigram_coverage(answer_bigram_list: List[str], chunk_bigram_set: Dict[str, bool]) -> float:
    if not answer_bigram_list:
        return 0.0
    hit = sum(1 for b in answer_bigram_list if chunk_bigram_set.get(b))
    return hit / len(answer_bigram_list)


def _normalize_for_gap(s: Any) -> str:
    t = str("" if s is None else s).upper().replace("‘", "'").replace("’", "'")
    t = t.replace("“", '"').replace("”", '"')
    t = unicodedata.normalize("NFD", t)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", t)


def _strip_framing(s: Any) -> str:
    t = str("" if s is None else s)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"according to[^:.\n]{0,120}[:.]", " ", t, flags=re.I)
    t = re.sub(r"seg[uú]n[^:.\n]{0,120}[:.]", " ", t, flags=re.I)
    t = re.sub(r"kindly refer to[^.\n]{0,120}\.", " ", t, flags=re.I)
    t = re.sub(r"consulte (?:a|con)[^.\n]{0,120}\.", " ", t, flags=re.I)
    return t.strip()


def _get_chunk_text(c: Dict[str, Any]) -> str:
    if not isinstance(c, dict):
        return ""
    if isinstance(c.get("citedText"), str):
        return c["citedText"]
    if isinstance(c.get("textChunk"), str):
        return c["textChunk"]
    return ""


def _get_doc_title(c: Dict[str, Any]) -> str:
    crit = (c or {}).get("documentIdentificationCriteria") or {}
    candidates = [crit.get("documentTitle"), c.get("documentTitle"), crit.get("documentName"),
                  c.get("documentName"), c.get("title"), c.get("fileName"), c.get("filename"),
                  crit.get("fileName"), c.get("sourceDocument"), c.get("docTitle")]
    title = ""
    for v in candidates:
        if isinstance(v, str) and len(v.strip()) > 1:
            title = v.strip()
            break
    if not title:
        return "Document"
    title = re.sub(r"\.(pdf|docx?|pptx?|xlsx?|txt|html?)$", "", title, flags=re.I)
    title = re.sub(r"[_]+", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def _clean_text(raw: Any) -> str:
    t = str("" if raw is None else raw)
    t = t.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    kept = []
    for line in t.split("\n"):
        if re.match(r"^\s*#{1,6}\s+\S", line) and len(line.strip()) <= 80:
            continue
        tokens = line.strip().split()
        if len(tokens) < 4:
            kept.append(line)
            continue
        upper = sum(1 for w in tokens if len(w) > 2 and any(c.isupper() for c in w) and w == w.upper())
        if (upper / len(tokens)) <= UPPER_RATIO:
            kept.append(line)
    t = " ".join(kept)
    fixes = {"dierent": "different", "oers": "offers", "sta": "staff", "eective": "effective"}
    t = re.sub(r"\b(dierent|oers|sta|eective)\b", lambda m: fixes[m.group(0).lower()], t, flags=re.I)
    run_pattern = r"\b(?:[A-Z0-9]{2,}\s+){" + str(UPPER_RUN_MIN - 1) + r",}[A-Z0-9]{2,}\b"
    t = re.sub(run_pattern, " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _esc(s: Any) -> str:
    return html.escape(str("" if s is None else s), quote=False)


def _split_sentences(t: Any) -> List[str]:
    s = str("" if t is None else t)
    s = re.sub(r"\s[-–—•▪◦‣]\s+(?=\S)", "\x01", s)
    s = re.sub(r"([.!?;])\s+(?=[A-Z0-9À-Þ¿¡“\"(•])", "\\1\x01", s)
    parts = []
    for seg in s.split("\x01"):
        seg = seg.strip()
        seg = re.sub(r"^[-–—•▪◦‣]\s*", "", seg)
        if seg:
            parts.append(seg)
    return parts


class _Variant:
    __slots__ = ("text", "terms", "bigram_list", "bigram_set")

    def __init__(self, text: str):
        self.text = text
        self.terms = _uniq(_tokenize(text))
        self.bigram_set = _build_bigrams(text)
        self.bigram_list = _uniq(list(self.bigram_set.keys()))


def _sentence_support(sentence: str, answer_text: str, answer_bigram_set: Dict[str, bool]) -> float:
    terms = _uniq(_tokenize(sentence))
    if len(terms) < 3:
        return 0.0
    uni = _count_matches(terms, answer_text) / len(terms)
    bigs = list(_build_bigrams(sentence).keys())
    hit = sum(1 for b in bigs if answer_bigram_set.get(b))
    big = (hit / len(bigs)) if bigs else 0.0
    return 0.5 * uni + 0.5 * big


def _sentence_support_best(sentence: str, variants: List[_Variant]) -> float:
    return max((_sentence_support(sentence, v.text, v.bigram_set) for v in variants), default=0.0)


def _best_window(text: Any, variants: List[_Variant], elide_before: bool, elide_after: bool) -> Dict[str, Any]:
    words = [w for w in str("" if text is None else text).split() if w]
    best = None
    start = 0
    while start < len(words):
        parts, length, end = [], 0, start
        while end < len(words) and (length + len(words[end]) + (1 if length else 0)) <= SENT_MAX_CHARS:
            length += len(words[end]) + (1 if length else 0)
            parts.append(words[end])
            end += 1
        if not parts:
            parts.append(words[start])
            end = start + 1
        win_text = " ".join(parts)
        sc = _sentence_support_best(win_text, variants)
        if best is None or sc > best["sc"]:
            best = {"text": win_text, "sc": sc, "s": start, "e": end}
        if end >= len(words):
            break
        start += max(1, (end - start) // 2)
    if not best:
        return {"text": str(text)[:SENT_MAX_CHARS], "score": 0.0}
    w = best["text"]
    if best["s"] > 0 or elide_before:
        w = "… " + w
    if best["e"] < len(words) or elide_after:
        w = w + " …"
    return {"text": w, "score": best["sc"]}


def _extract_support(chunk_text: str, variants: List[_Variant]) -> Optional[Dict[str, Any]]:
    segments = _split_sentences(chunk_text)
    scored = []
    for i, s in enumerate(segments):
        if len(s) < SENT_MIN_LEN:
            continue
        scored.append({"s": s, "i": i, "sc": _sentence_support_best(s, variants)})
    if not scored:
        return None
    anchor = max(scored, key=lambda x: x["sc"])
    if anchor["sc"] < SENT_MIN_SCORE:
        return None
    by_seg = {x["i"]: x for x in scored}
    win = [anchor]
    total = len(anchor["s"])
    lo = hi = anchor["i"]
    while len(win) < SENT_MAX_COUNT:
        prev = by_seg.get(lo - 1)
        nxt = by_seg.get(hi + 1)
        prev_ok = prev and prev["sc"] >= anchor["sc"] * SENT_NEIGHBOR_REL and (total + len(prev["s"])) <= SENT_MAX_CHARS
        next_ok = nxt and nxt["sc"] >= anchor["sc"] * SENT_NEIGHBOR_REL and (total + len(nxt["s"])) <= SENT_MAX_CHARS
        if not prev_ok and not next_ok:
            break
        if next_ok and (not prev_ok or nxt["sc"] >= prev["sc"]):
            win.append(nxt)
            total += len(nxt["s"])
            hi = nxt["i"]
        else:
            win.insert(0, prev)
            total += len(prev["s"])
            lo = prev["i"]
    joined = " ".join(x["s"] for x in win)
    if len(joined) > SENT_MAX_CHARS:
        joined = _best_window(joined, variants, lo > 0, hi < len(segments) - 1)["text"]
    else:
        if lo > 0:
            joined = "… " + joined
        if hi < len(segments) - 1:
            joined = joined + " …"
    if len(joined) >= len(chunk_text) - 40:
        return None
    return {"text": joined, "segments": segments}


def _with_breaks(t: str) -> str:
    e = _esc(t)
    e = re.sub(r"([.:;])\s+(?=[A-Z0-9“\"•])", r"\1<br>", e)
    e = re.sub(r"\s*([•▪◦‣])\s*", r"<br>\1 ", e)
    e = re.sub(r"(<br>\s*)+", "<br>", e)
    return re.sub(r"^<br>", "", e)


def _make_snippet(t: str) -> Dict[str, Any]:
    if len(t) <= SNIPPET_LEN:
        return {"snippet": _esc(t), "truncated": False}
    cut = t[:SNIPPET_LEN]
    last_space = cut.rfind(" ")
    if last_space > SNIPPET_LEN * 0.6:
        cut = cut[:last_space]
    return {"snippet": _esc(cut) + "&hellip;", "truncated": True}


def _rag_answer_text(rag_outputs: List[Dict[str, Any]]) -> str:
    combined = ""
    for out in rag_outputs:
        if isinstance(out, dict) and isinstance(out.get("value"), str) and len(out["value"].strip()) > 10:
            combined += " " + out["value"]
    return _strip_framing(combined)


def get_citation_details(agent_response: str, agent_response_topic: str,
                         rag_outputs: List[Dict[str, Any]], query_text: str,
                         language: str = "EN") -> str:
    """Port of GET_THE_CITATION_DETAILS.  Returns the HTML "Sources" block ("" if none)."""
    lang = "ES" if str(language).upper() == "ES" else "EN"
    labels = LABELS[lang]
    rag_outputs = [o for o in (rag_outputs or []) if isinstance(o, dict)]

    normalized_response = _normalize_for_gap(agent_response_topic)
    for m in GAP_MARKERS:
        nm = _normalize_for_gap(m).strip()
        if nm and nm in normalized_response:
            return ""  # gap marker -> suppress citations

    # Gather candidate chunks (citations first, then supporting chunks).
    raw_chunks = []
    for out in rag_outputs:
        cits = out.get("citations") if isinstance(out.get("citations"), list) else []
        chunks = out.get("supportingChunks") if isinstance(out.get("supportingChunks"), list) else []
        for c in cits:
            raw_chunks.append((c, True))
        for c in chunks:
            raw_chunks.append((c, False))
    if not raw_chunks:
        return ""

    # Enrich + dedupe (identical + same-doc prefix overlap).
    seen, seen_prefix, enriched = set(), set(), []
    for pos, (c, is_cit) in enumerate(raw_chunks):
        raw_text = _get_chunk_text(c)
        text = _clean_text(raw_text)
        if not text or len(text) < 20:
            continue
        norm = re.sub(r"[^a-z0-9]", "", text.lower())
        if norm in seen:
            continue
        seen.add(norm)
        prefix_fp = _get_doc_title(c) + "|" + norm[:OVERLAP_PREFIX_LEN]
        if prefix_fp in seen_prefix:
            continue
        seen_prefix.add(prefix_fp)
        match_text = re.sub(r"\bcol\d+\b", " ", text + " " + _get_doc_title(c), flags=re.I)
        enriched.append({"cite": c, "text": text, "rawText": raw_text, "matchText": match_text,
                         "isCitation": is_cit, "origPos": pos, "selected": False})

    def make_packs(txts: List[str]) -> List[_Variant]:
        return [v for v in (_Variant(t) for t in txts) if v.terms]

    def selection_texts() -> List[str]:
        arr = []
        final_plain = _strip_framing(agent_response)
        if len(final_plain) > 20:
            arr.append(final_plain)
        rag_plain = _rag_answer_text(rag_outputs)
        if len(rag_plain) > 20 and rag_plain not in arr:
            arr.append(rag_plain)
        return arr

    def excerpt_texts() -> List[str]:
        arr = []
        topic_plain = _strip_framing(agent_response_topic)
        if len(topic_plain) > 20:
            arr.append(topic_plain)
        if not arr:
            return selection_texts()
        rag_plain = _rag_answer_text(rag_outputs)
        if len(rag_plain) > 20 and rag_plain not in arr:
            arr.append(rag_plain)
        return arr

    query_terms = _uniq(_tokenize(query_text))
    selection_variants = make_packs(selection_texts())
    excerpt_variants = make_packs(excerpt_texts()) or selection_variants
    has_answer = len(selection_variants) > 0
    has_excerpt_answer = len(excerpt_variants) > 0

    if REQUIRE_ANSWER and not has_answer:
        return ""

    for e in enriched:
        e["matched"] = _count_matches(query_terms, e["matchText"])
        if has_answer:
            best_raw = best_uni = best_big = 0.0
            best_var = 0
            chunk_bigrams = _build_bigrams(e["matchText"])
            for vi, v in enumerate(selection_variants):
                uni_cov = _count_matches(v.terms, e["matchText"]) / len(v.terms)
                big_cov = _bigram_coverage(v.bigram_list, chunk_bigrams)
                combined = UNIGRAM_W * uni_cov + BIGRAM_W * big_cov
                raw = combined / ANSWER_COV_FULL
                if raw > best_raw:
                    best_raw, best_var, best_uni, best_big = raw, vi, uni_cov, big_cov
            e["rawScore"] = best_raw
            e["variantIdx"] = best_var
            winner = selection_variants[best_var]
            e["phraseOk"] = (len(winner.bigram_list) < MIN_PHRASE_BIGRAMS) or (best_big >= MIN_PHRASE_EVIDENCE)
        else:
            e["rawScore"] = (e["matched"] / len(query_terms)) if query_terms else 0.0
            e["phraseOk"] = True

    top_raw = max((e["rawScore"] for e in enriched), default=0.0)

    def primary_ok(e: Dict[str, Any]) -> bool:
        if not e["phraseOk"]:
            return False
        if not has_answer and e["matched"] < 1:
            return False
        if USE_RELATIVE_GATE:
            return e["rawScore"] >= ABS_FLOOR and e["rawScore"] >= top_raw * REL_GATE
        return e["rawScore"] >= MIN_RELEVANCE

    def by_score(e: Dict[str, Any]):
        return (-e["rawScore"], e["origPos"])

    candidates = sorted([e for e in enriched if primary_ok(e)], key=by_score)[:MAX_CITATIONS]

    if not candidates:
        candidates = sorted(
            [e for e in enriched if e["isCitation"] and e["rawScore"] >= FALLBACK_MIN_RELEVANCE
             and (e["phraseOk"] if PHRASE_GATE_IN_FALLBACK else True)],
            key=by_score)[:MAX_CITATIONS]
    if not candidates:
        candidates = sorted(
            [e for e in enriched if e["rawScore"] >= FALLBACK_MIN_RELEVANCE
             and (e["phraseOk"] if PHRASE_GATE_IN_FALLBACK else True)
             and (has_answer or e["matched"] >= 1)],
            key=by_score)[:min(FALLBACK_TOP_N, MAX_CITATIONS)]

    if not candidates:
        return ""

    card_style = "border:1px solid #d9d9d6;border-radius:8px;padding:10px 12px;margin:6px 0;"
    badge_style = "display:inline-block;background:#e6f1fb;color:#0c447c;font-size:12px;font-weight:600;min-width:20px;height:20px;line-height:20px;border-radius:50%;text-align:center;margin-right:8px;"
    title_style = "font-size:13px;font-weight:600;vertical-align:middle;"
    section_style = "font-size:12px;color:#8a8984;font-style:italic;margin:6px 0 0 28px;"
    snippet_style = "font-size:13px;color:#5f5e5a;margin:8px 0 0 28px;line-height:1.5;"
    full_style = "font-size:13px;color:#5f5e5a;margin:6px 0 0 28px;line-height:1.6;"

    seen_visible = set()
    blocks = []
    card_no = 0
    for e in candidates:
        title = _get_doc_title(e["cite"])
        support = _extract_support(e["text"], excerpt_variants) if has_excerpt_answer else None

        if support:
            visible_html = _esc(support["text"])
            show_full = True
            full_html = "<br>".join(_esc(seg) for seg in support["segments"])
        elif has_excerpt_answer:
            win = _best_window(e["text"], excerpt_variants, False, False)
            if win["score"] < WINDOW_MIN_SCORE and selection_variants:
                win2 = _best_window(e["text"], selection_variants, False, False)
                if win2["score"] > win["score"]:
                    win = win2
            if win["score"] < WINDOW_MIN_SCORE:
                if SUPPRESS_LOW_SCORE_CARDS:
                    continue
                snip = _make_snippet(e["text"])
                visible_html = snip["snippet"]
                show_full = snip["truncated"]
                full_html = _with_breaks(e["text"])
            else:
                visible_html = _esc(win["text"])
                show_full = len(win["text"]) < len(e["text"]) - 10
                full_html = _with_breaks(e["text"])
        else:
            snip = _make_snippet(e["text"])
            visible_html = snip["snippet"]
            show_full = snip["truncated"]
            full_html = _with_breaks(e["text"])

        vis_fp = title + "|" + re.sub(r"[^a-z0-9]", "", visible_html.lower())[:200]
        if vis_fp in seen_visible:
            continue
        seen_visible.add(vis_fp)

        card_no += 1
        excerpt_html = f'<div style="{snippet_style}">&ldquo;{visible_html}&rdquo;</div>'
        if show_full:
            excerpt_html += (
                f'<details style="margin-left:28px;"><summary style="font-size:12px;color:#185fa5;cursor:pointer;">'
                f'{labels["show_full"]}</summary>'
                f'<div style="{full_style}">{full_html}</div></details>'
            )
        blocks.append(
            f'<div style="{card_style}"><span style="{badge_style}">{card_no}</span>'
            f'<span style="{title_style}">{_esc(title)}</span>{excerpt_html}</div>'
        )

    if not blocks:
        return ""
    label = labels["source"] if len(blocks) == 1 else f'{labels["sources"]} ({len(blocks)})'
    return ("<br><br><details open><summary><b>" + label + "</b></summary>"
            + "".join(blocks) + "</details><hr>")
