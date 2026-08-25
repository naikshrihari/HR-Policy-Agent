# HR Policy Agent (LangChain + LangGraph)

A Python port of the Oracle Fusion AI Agent Studio workflow
**`HR_POLICY_WORKFLOW_AGENT_V35`** — the *Station Casinos HCM Policy Agent* — rebuilt
as a [LangGraph](https://langchain-ai.github.io/langgraph/) state machine with
[LangChain](https://python.langchain.com/) chat models.

The original is a single "data pipeline" agent (86 nodes) that answers HR policy
questions grounded in the correct Team Member handbook. This port reproduces the same
control flow, prompts, and deterministic business logic in Python.

## What it does

For each Team Member question it:

1. **Identifies the user** — Oracle User Session → HCM `getWorkerDetails` REST call →
   classifies the Team Member as **Tavern**, **Represented**, or **Non Represented**.
2. **Understands the query** — normalizes the question and routes on **intent**
   (`GREETING` / `POLICY` / `REDIRECT`) and **language** (`EN` / `ES`).
3. **Retrieves** the answer from the correct handbook via a RAG document tool.
4. **Generates** the grounded answer, then post-processes it:
   - picks the best answer (falls back when a topic isn't covered),
   - builds an **HR-system referral** (ADP / Benefits Connect / Absence Resources /
     HCM / Ask your Manager / Human Resources),
   - selects and renders **citation cards**,
   - classifies the **topic** and flags **HR routing** (crisis / harassment / complaint
     / union inquiry),
   - logs the turn to the chat store.
5. Optionally collects **thumbs-up/down feedback** (see `feedback.py`).

Everything runs **offline by default** using deterministic stubs, so you can try it with
no API keys or endpoints.

## Quick start

```bash
pip install -e .            # or: pip install -r requirements.txt

# One-shot
python -m hr_policy_agent.cli "How many PL days do I accrue?"

# Interactive
python -m hr_policy_agent.cli
```

```python
from hr_policy_agent import HRPolicyAgent

agent = HRPolicyAgent()
print(agent.answer("¿Cuándo es el día de pago?"))          # Spanish route
print(agent.answer("Where can I see my pay stub?"))         # → ADP referral
```

## Using a real LLM / real backends

Configuration is entirely environment-driven (see `hr_policy_agent/config.py`):

| Variable | Purpose | Default |
|---|---|---|
| `HRPA_LLM_PROVIDER` | `fake` (offline), `openai`, `anthropic`, `ollama` | `fake` |
| `HRPA_LLM_MODEL` | model id for the chosen provider | `gpt-4o-mini` |
| `HRPA_LLM_API_KEY` / `HRPA_LLM_BASE_URL` | provider credentials / endpoint | – |
| `HRPA_USE_MOCK_HCM` / `HRPA_HCM_BASE_URL` | HCM `getWorkerDetails` REST | mock |
| `HRPA_USE_MOCK_RAG` / `HRPA_RAG_CORPUS_DIR` | handbook retrieval corpus | mock |
| `HRPA_USE_MOCK_CHAT_STORE` / `HRPA_CHAT_STORE_URL` | analytics/feedback logging | mock |
| `HRPA_ENABLE_HUMAN_FEEDBACK` | enable the feedback loop | `false` |

With `HRPA_LLM_PROVIDER` set to a real provider, the verbatim prompts in
`hr_policy_agent/prompts/` are rendered and sent to the model. To run real RAG, drop
handbook text/markdown files under `HRPA_RAG_CORPUS_DIR/<tool-code>/` and
`pip install '.[rag]'`.

## How the Oracle workflow maps to this project

| Oracle node type | Where it lives here |
|---|---|
| `START` / `END`, node graph & routing | `hr_policy_agent/graph.py` (LangGraph `StateGraph`) |
| `LLM` nodes (18) | prompts in `hr_policy_agent/prompts/*.txt`, run by `llm_nodes.py` |
| `CODE` nodes (18, JavaScript) | ported to `hr_policy_agent/codenodes/` |
| `RAG_DOCUMENT_TOOL` nodes (5) | `hr_policy_agent/services/rag.py` |
| `EXTERNAL_REST` (HCM, chat store) | `hr_policy_agent/services/hcm.py`, `services/chat_store.py` |
| `TOOL` (user session) | `hr_policy_agent/services/session.py` |
| `HUMAN` feedback nodes | `hr_policy_agent/feedback.py` |
| `CONDITION` / `SWITCH` | Python routers in `graph.py` |
| `{{$context.$nodes...}}` templating | `hr_policy_agent/templating.py` |

### Ported CODE nodes (`hr_policy_agent/codenodes/`)

| Module | Original node(s) |
|---|---|
| `person.py` | `ENCRYPT_PERSON_NUMBER`, `RETRIEVE_PERSON_DETAILS_SCRIPT` |
| `transforms.py` | `COMBINE_USER_QUERY…`, `GET_THE_BEST_ANSWER`, `HR_ROUTING_CLASSIFICATION` (EN/ES) |
| `citation.py` | `RETURN_CITATION_SCRIPT`, `RETURN_AGENT_RESPONSE` (EN/ES) |
| `topic_classifier.py` + `topic_vocab.py` | `TOPIC_CLASSIFICATION_SCRIPT` (EN/ES) |
| `system_mapping.py` | `FETCH_SYSTEM_MAPPING_SCRIPT` (EN/ES) |
| `citation_details.py` | `GET_THE_CITATION_DETAILS` v7.7 (EN/ES) |

## Notes on fidelity

- The original ships an English and a near-identical Spanish copy of every node. Because
  the two are semantically identical, this port carries `language` in state and runs one
  **language-aware** pipeline, selecting Spanish prompts, messages, and RAG tools when the
  Intent Router detects `ES`. Observable behavior matches the original flow (including the
  Spanish-Tavern "English only" redirect).
- Node outputs are stored under `state["nodes"][CODE]`, mirroring the Oracle
  `$context.$nodes.<CODE>.$output` model; CODE-node results keep the `{"result": ...}`
  envelope so the template references resolve unchanged.
- `GET_THE_CITATION_DETAILS`' production `DEBUG_SCORES` switch was off, so the diagnostic
  table is not reproduced; all scoring/gating/excerpt logic is.
- The multi-turn human feedback loop is factored into `feedback.py` rather than the
  single-turn Q&A graph.

## Tests

```bash
pip install pytest && python -m pytest -q
```

Covers the ported code nodes (classification, topic scoring, system mapping, citation
selection) and the end-to-end graph on the offline provider.
