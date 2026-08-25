"""Runner for the LLM nodes.

Loads the verbatim prompt for a node from ``prompts/<CODE>.txt``, resolves the
``{{$context...}}`` placeholders against graph state, then either calls the configured
chat model or (for the offline ``fake`` provider) the deterministic heuristic.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict

from .config import Settings
from .heuristics import fake_llm
from .llm import get_chat_model, parse_json
from .templating import build_context, render

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")

# Nodes whose output is a JSON object (structured output specification in the workflow).
STRUCTURED_NODES = {
    "INTENT_ROUTE_LLM",
    "INPUT_USER_QUERY",
    "GET_THE_RELEVANT_CITATION_ONLY",
    "GET_THE_RELEVANT_CITATION_ONLY_SPANISH",
}


@lru_cache(maxsize=None)
def load_prompt(code: str) -> str:
    path = os.path.join(_PROMPT_DIR, code + ".txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


@lru_cache(maxsize=None)
def _llm_config() -> Dict[str, Any]:
    path = os.path.join(_PROMPT_DIR, "_llm_config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_llm_node(code: str, state: Dict[str, Any], settings: Settings) -> Any:
    """Execute LLM node ``code`` and return its ``$output`` value."""
    if settings.llm_provider.lower() == "fake":
        return fake_llm(code, state)

    prompt_text = load_prompt(code)
    rendered = render(prompt_text, build_context(state))

    cfg = _llm_config().get(code, {})
    temperature = cfg.get("temperature")
    max_tokens = cfg.get("max_tokens")
    model = get_chat_model(settings, temperature=temperature, max_tokens=max_tokens)

    from langchain_core.messages import HumanMessage

    response = model.invoke([HumanMessage(content=rendered)])
    content = response.content if hasattr(response, "content") else str(response)

    if code in STRUCTURED_NODES:
        return parse_json(content)
    return content
