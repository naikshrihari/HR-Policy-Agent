"""Command-line entrypoint for the HR Policy Agent.

Usage:
    python -m hr_policy_agent.cli "How many PL days do I accrue?"
    python -m hr_policy_agent.cli            # interactive REPL
"""

from __future__ import annotations

import argparse
import sys

from .agent import HRPolicyAgent
from .config import get_settings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Station Casinos HR Policy Agent")
    parser.add_argument("message", nargs="*", help="The question to ask (omit for interactive mode)")
    parser.add_argument("--person-number", help="Team Member person number (overrides the mock default)")
    args = parser.parse_args(argv)

    settings = get_settings()
    agent = HRPolicyAgent(settings)
    print(f"[HR Policy Agent] LLM provider: {settings.llm_provider} | "
          f"mock HCM={settings.use_mock_hcm} RAG={settings.use_mock_rag}\n")

    if args.message:
        print(agent.answer(" ".join(args.message), person_number=args.person_number))
        return 0

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
        print("\nagent>", state.get("final_response", ""), "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
