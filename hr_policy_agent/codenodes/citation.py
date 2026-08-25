"""Ports of the citation-selection code nodes.

* RETURN_CITATION_SCRIPT (+ Spanish) — the citation selection engine (R1–R7/V1–V2)
* RETURN_AGENT_RESPONSE  (+ Spanish) — pick the final agent response by priority
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

MIN_RUN = 5
_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*(?:/\d+)?|[a-z]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _longest_common_run(a_tokens: List[str], c_tokens: List[str]) -> int:
    """Longest common *consecutive* run of tokens (DP, rolling array). R4a."""
    n = len(c_tokens)
    prev = [0] * (n + 1)
    run = 0
    for at in a_tokens:
        curr = [0] * (n + 1)
        for k in range(1, n + 1):
            if at == c_tokens[k - 1]:
                curr[k] = prev[k - 1] + 1
                if curr[k] > run:
                    run = curr[k]
        prev = curr
    return run


def _facts_ok(a_tokens: List[str], c_tokens: List[str]) -> bool:
    """R4b — numbers appearing in the same word context must be identical."""
    for ci, ctok in enumerate(c_tokens):
        if not ctok[:1].isdigit():
            continue
        c_prev = c_tokens[ci - 1] if ci > 0 else "^"
        c_next = c_tokens[ci + 1] if ci < len(c_tokens) - 1 else "$"
        context_in_answer = False
        same_number = False
        for ai, atok in enumerate(a_tokens):
            if not atok[:1].isdigit():
                continue
            a_prev = a_tokens[ai - 1] if ai > 0 else "^"
            a_next = a_tokens[ai + 1] if ai < len(a_tokens) - 1 else "$"
            if a_prev == c_prev and a_next == c_next:
                context_in_answer = True
                if atok == ctok:
                    same_number = True
        if context_in_answer and not same_number:
            return False
    return True


def _coerce_citations(cit: Any) -> List[Dict[str, Any]]:
    if cit is None:
        return []
    if isinstance(cit, str):
        s = cit.strip()
        if s == "" or s.lower() in ("undefined", "null"):
            return []
        try:
            cit = json.loads(s)
        except json.JSONDecodeError:
            return []
    if isinstance(cit, list):
        return [c for c in cit if isinstance(c, dict)]
    if isinstance(cit, dict):
        return [cit]
    return []


def return_citation_script(rag_outputs: List[Optional[Dict[str, Any]]]) -> str:
    """Port of RETURN_CITATION_SCRIPT.

    ``rag_outputs`` is the list of RAG node outputs that may have run (each shaped
    ``{"value": str, "citations": [...]}``).  Only one branch runs per query, but we
    accept the full list to mirror the original guarded lookup.
    Returns a JSON string ``{"Document_Title": [...], "Citation_Details": [...]}``.
    """
    default = json.dumps({"Document_Title": [], "Citation_Details": []})

    citation_inputs: List[Any] = []
    answer_inputs: List[Any] = []
    for out in rag_outputs:
        if out and isinstance(out, dict):
            citation_inputs.append(out.get("citations"))
            answer_inputs.append(out.get("value"))

    # Merge non-blank answer text.
    answer_text = ""
    for av in answer_inputs:
        as_ = "" if av is None else str(av).strip()
        if as_ and as_.lower() not in ("undefined", "null"):
            answer_text = as_ if answer_text == "" else answer_text + "\n" + as_

    # Build candidates: citedText + documentTitle only.
    cand_texts: List[str] = []
    cand_titles: List[Optional[str]] = []
    for cit in citation_inputs:
        for entry in _coerce_citations(cit):
            txt = entry.get("citedText")
            if txt is None or str(txt).strip() == "":
                continue
            ttl = None
            dic = entry.get("documentIdentificationCriteria")
            if isinstance(dic, dict):
                dt = dic.get("documentTitle")
                if dt is not None and str(dt).strip() != "":
                    ttl = str(dt)
            cand_texts.append(str(txt))
            cand_titles.append(ttl)

    if answer_text == "" or not cand_texts:
        return default

    a_tokens = _tokenize(answer_text)
    if not a_tokens:
        return default

    best_run = 0
    best_idx = -1
    for c, ctext in enumerate(cand_texts):
        c_tokens = _tokenize(ctext)
        if not c_tokens:
            continue
        run = _longest_common_run(a_tokens, c_tokens)
        if run < MIN_RUN:
            continue
        if not _facts_ok(a_tokens, c_tokens):
            continue
        if run > best_run:  # R6/R7 — longest run, earliest on ties
            best_run = run
            best_idx = c

    if best_idx >= 0:
        title = candtitle if (candtitle := cand_titles[best_idx]) is not None else "Untitled Document"
        return json.dumps({
            "Document_Title": [title],
            "Citation_Details": [cand_texts[best_idx]],
        })
    return default


def _unwrap_result(out: Any) -> Any:
    """Unwrap the {result, console, timedOut} envelope and parse JSON strings."""
    if isinstance(out, dict) and "result" in out:
        out = out["result"]
    if isinstance(out, str):
        try:
            out = json.loads(out)
        except json.JSONDecodeError:
            pass
    return out


def return_agent_response(
    script_output: Any,
    citation_only_output: Any,
    answer_agent_output: Any,
) -> str:
    """Port of RETURN_AGENT_RESPONSE — choose the final response by priority.

    P1: RETURN_CITATION_SCRIPT's Citation_Details[0] (deterministic selection)
    P2: GET_THE_RELEVANT_CITATION_ONLY's Citation_Details[0] (LLM fallback)
    P3: ANSWER_AGENT_'s value (final fallback)
    """
    agent_response = ""
    script_citation_length = 0

    out = _unwrap_result(script_output)
    if isinstance(out, dict) and isinstance(out.get("Citation_Details"), list):
        cd = out["Citation_Details"]
        script_citation_length = len(cd)
        if script_citation_length > 0 and str(cd[0]).strip() != "":
            agent_response = str(cd[0])

    if script_citation_length == 0 and not agent_response:
        out = citation_only_output
        if isinstance(out, dict) and isinstance(out.get("value"), str):
            out = out["value"]
        if isinstance(out, str):
            try:
                out = json.loads(out)
            except json.JSONDecodeError:
                pass
        if (
            isinstance(out, dict)
            and isinstance(out.get("Citation_Details"), list)
            and len(out["Citation_Details"]) > 0
            and str(out["Citation_Details"][0]).strip() != ""
        ):
            agent_response = str(out["Citation_Details"][0]).split(", documentIdentificationCriteria=")[0]

    if not agent_response:
        out = answer_agent_output
        if isinstance(out, dict) and isinstance(out.get("value"), str):
            out = out["value"]
        if isinstance(out, str):
            agent_response = out

    return agent_response
