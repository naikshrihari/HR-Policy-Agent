"""Resolve Oracle-style ``{{$context...}}`` template expressions.

The workflow's LLM prompts and node inputs interpolate values with expressions like::

    {{$context.$nodes.INPUT_USER_QUERY.$output.searchQuery}}
    {{$context.$nodes.INTENT_ROUTE_LLM.$output.tmType.toUpperCase()}}
    {{$context.$workflow.$conversationId}}
    {{$context.$system.$inputMessage}}

This module implements just enough of that mini expression language to substitute
values into prompt strings.  Boolean *conditions* and *switch* expressions from the
original graph are NOT evaluated here — those are ported to explicit Python in
``graph.py`` for clarity and safety.
"""

from __future__ import annotations

import re
from typing import Any, Dict

_EXPR_RE = re.compile(r"\{\{\s*(.*?)\s*\}\}", re.DOTALL)

# A single path segment: a .name accessor, a [index] accessor, or a ().method call.
_SEGMENT_RE = re.compile(r"\.([A-Za-z_$][\w$]*)|\[(\d+)\]|\.([A-Za-z_$][\w$]*)\(\)")


class _Missing:
    """Sentinel that stringifies to '' but keeps chaining safe."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"


MISSING = _Missing()


def _apply_method(value: Any, method: str) -> Any:
    if method == "toUpperCase":
        return str(value).upper()
    if method == "toLowerCase":
        return str(value).lower()
    if method == "trim":
        return str(value).strip()
    if method == "length":
        try:
            return len(value)
        except TypeError:
            return 0
    return value


def _walk(root: Any, expr: str) -> Any:
    """Walk a dotted/indexed expression against ``root``."""
    value = root
    # Tokenize into .name / [idx] / .method()
    pos = 0
    token_re = re.compile(r"\.([A-Za-z_$][\w$]*)\(\)|\.([A-Za-z_$][\w$]*)|\[(\d+)\]")
    for m in token_re.finditer(expr):
        method_call, name, index = m.group(1), m.group(2), m.group(3)
        if value is MISSING or value is None:
            return MISSING
        if method_call is not None:
            value = _apply_method(value, method_call)
        elif name is not None:
            key = name.lstrip("$")  # $output -> output
            if isinstance(value, dict):
                value = value.get(key, MISSING)
            else:
                value = getattr(value, key, MISSING)
        elif index is not None:
            idx = int(index)
            try:
                value = value[idx]
            except (IndexError, KeyError, TypeError):
                value = MISSING
    return value


def resolve_expression(expr: str, context: Dict[str, Any]) -> Any:
    """Resolve one expression (without the surrounding braces)."""
    expr = expr.strip()
    if not expr.startswith("$context"):
        return expr
    body = expr[len("$context"):]

    # Determine the root: $nodes.<CODE>, $workflow.$X, $system.$X
    if body.startswith(".$nodes"):
        rest = body[len(".$nodes"):]
        m = re.match(r"\.([A-Za-z_$][\w$]*)", rest)
        if not m:
            return MISSING
        code = m.group(1)
        node_output = (context.get("nodes") or {}).get(code, MISSING)
        remainder = rest[m.end():]
        # remainder starts with .$output...
        remainder = remainder.replace(".$output", "", 1)
        return _walk(node_output, remainder)

    if body.startswith(".$workflow"):
        rest = body[len(".$workflow"):]
        key = rest.lstrip(".$")
        wf = context.get("workflow", {})
        return wf.get(key, MISSING)

    if body.startswith(".$system"):
        rest = body[len(".$system"):]
        key = rest.lstrip(".$")
        sysd = context.get("system", {})
        return sysd.get(key, MISSING)

    return MISSING


def render(template: str, context: Dict[str, Any]) -> str:
    """Substitute every ``{{...}}`` expression in ``template``."""

    def _sub(match: "re.Match[str]") -> str:
        value = resolve_expression(match.group(1), context)
        if value is MISSING or value is None:
            return ""
        return str(value)

    return _EXPR_RE.sub(_sub, template)


def build_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the resolver context from graph state."""
    from datetime import datetime

    now = datetime.now()
    return {
        "nodes": state.get("nodes", {}),
        "workflow": {
            "conversationId": state.get("conversation_id", ""),
            "traceId": state.get("trace_id", ""),
            "inputMessage": state.get("input_message", ""),
        },
        "system": {
            "inputMessage": state.get("input_message", ""),
            "currentDate": now.strftime("%Y-%m-%d"),
            "currentDateTime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "chatHistory": state.get("chat_history", ""),
        },
    }
