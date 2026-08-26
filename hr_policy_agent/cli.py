"""Command-line entrypoint for the HR Policy Agent.

Usage:
    python -m hr_policy_agent.cli "How many PL days do I accrue?"
    python -m hr_policy_agent.cli            # interactive REPL
    python -m hr_policy_agent.cli --html ... # keep the raw HTML response

The agent's response is HTML (it mirrors the Oracle chat-widget message). For terminal
use the CLI renders it as clean plain text by default; pass --html for the raw markup.
"""

from __future__ import annotations

import argparse
import html
import re
import sys

from .agent import HRPolicyAgent
from .config import get_settings


def html_to_text(s: str) -> str:
    """Render the HTML response as readable plain text for the terminal."""
    if not s:
        return ""
    # Collapsible source cards -> a simple "Sources:" section.
    s = re.sub(r"<summary[^>]*>\s*(?:<b>)?\s*Show full excerpt\s*(?:</b>)?\s*</summary>", "", s, flags=re.I)
    s = re.sub(r"<summary[^>]*>\s*<b>\s*(Sources?[^<]*)</b>\s*</summary>", r"\n\1:\n", s, flags=re.I)
    s = re.sub(r"<summary[^>]*>(.*?)</summary>", r"\n\1:\n", s, flags=re.I | re.S)
    # Line breaks and block boundaries.
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</(div|p|details|hr)>", "\n", s, flags=re.I)
    s = re.sub(r"<hr\s*/?>", "\n", s, flags=re.I)
    # Drop every remaining (inline) tag as a space so adjacent spans don't fuse,
    # then unescape entities.
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    # Tidy whitespace.
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Station Casinos HR Policy Agent")
    parser.add_argument("message", nargs="*", help="The question to ask (omit for interactive mode)")
    parser.add_argument("--person-number", help="Team Member person number (overrides the mock default)")
    parser.add_argument("--html", action="store_true", help="Print the raw HTML response instead of plain text")
    parser.add_argument("--quiet", action="store_true", help="Suppress the startup banner")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: deterministic routing + one compact LLM call (much faster on local models)")
    parser.add_argument("--timings", action="store_true", help="Print per-LLM-node timings to stderr")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.fast:
        settings.fast_mode = True
    if args.timings:
        settings.timing = True
    if not args.quiet:
        model = f" ({settings.llm_model})" if settings.llm_provider != "fake" else ""
        print(f"[HR Policy Agent] LLM provider: {settings.llm_provider}{model} | "
              f"fast={settings.fast_mode} | mock HCM={settings.use_mock_hcm} RAG={settings.use_mock_rag}")
        _ollama_preflight(settings)
        print()
    agent = HRPolicyAgent(settings)

    render = (lambda x: x) if args.html else html_to_text

    def answer_with_progress(msg: str, cid=None):
        if settings.llm_provider != "fake" and not args.quiet:
            print("Thinking… (the first call also loads the model into memory)", flush=True)
        return agent.run(msg, person_number=args.person_number, conversation_id=cid)

    if args.message:
        state = answer_with_progress(" ".join(args.message))
        print(render(state.get("final_response", "")))
        return 0

    if not args.quiet:
        print("Interactive mode. Type a question, or 'quit' to exit.\n")
    conversation_id = None
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if msg.lower() in ("quit", "exit"):
            break
        if not msg:
            continue
        state = answer_with_progress(msg, conversation_id)
        conversation_id = state.get("conversation_id")
        print("\nagent>", render(state.get("final_response", "")), "\n")
    return 0


def _ollama_preflight(settings) -> None:
    """When using Ollama, warn early if the server is down or the model isn't pulled,
    instead of hanging silently for minutes."""
    if settings.llm_provider.lower() != "ollama":
        return
    import json
    import urllib.request

    url = settings.ollama_base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.load(resp)
    except Exception as exc:  # noqa: BLE001 - best-effort diagnostic
        print(f"  [warn] Can't reach Ollama at {settings.ollama_base_url} ({exc}).")
        print("         Start it with 'ollama serve' (or check HRPA_OLLAMA_BASE_URL).")
        return
    models = [m.get("name", "") for m in data.get("models", [])]
    want = settings.llm_model
    base = want.split(":")[0]
    if not any(m == want or m.split(":")[0] == base for m in models):
        print(f"  [warn] Model '{want}' is not pulled in Ollama — the first call will "
              "download it (can be slow / look frozen).")
        print(f"         Pull it now with:  ollama pull {want}")
        if models:
            print(f"         Installed models: {', '.join(models)}")


if __name__ == "__main__":
    sys.exit(main())
