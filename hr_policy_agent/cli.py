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
    agent = HRPolicyAgent(settings)
    if not args.quiet:
        print(f"[HR Policy Agent] LLM provider: {settings.llm_provider} | "
              f"fast={settings.fast_mode} | mock HCM={settings.use_mock_hcm} RAG={settings.use_mock_rag}\n")

    render = (lambda x: x) if args.html else html_to_text

    if args.message:
        print(render(agent.answer(" ".join(args.message), person_number=args.person_number)))
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
        state = agent.run(msg, person_number=args.person_number, conversation_id=conversation_id)
        conversation_id = state.get("conversation_id")
        print("\nagent>", render(state.get("final_response", "")), "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
