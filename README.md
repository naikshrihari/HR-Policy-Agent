# HR Policy Agent (LangChain + LangGraph)

A Python port of the Oracle Fusion AI Agent Studio workflow
**`HR_POLICY_WORKFLOW_AGENT_V35`** the *Station Casinos HCM Policy Agent* rebuilt
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
`hr_policy_agent/prompts/` are rendered and sent to the model.

### Real RAG over your handbooks

Drop handbook files — **`.txt`, `.md`, `.pdf`, or `.docx`** — into a per-tool sub-folder
of the corpus directory (the folder names are the Oracle tool codes):

```
data/corpus/
  TAVERN_ENGLISH_DOCUMENT_TOOL_WORKFLOW_V3/      # Tavern handbook (EN)
  REP_ENGLISH_DOCUMENT_TOOL_WORKFLOW_V3/         # Represented handbook (EN)
  NON_REP_ENGLISH_DOCUMENT_TOOL_WORKFLOW_V3/     # Non-Represented handbook (EN)
  REP_SPANISH_DOCUMENT_TOOL_WORKFLOW/            # Represented handbook (ES)
  NON_REP_SPANISH_DOCUMENT_TOOL_WORKFLOW_V3/     # Non-Represented handbook (ES)
```

There are two retrieval backends (choose with `HRPA_RAG_BACKEND`):

#### 1. TF-IDF — no embeddings, no vector DB (default, simplest)

```bash
pip install '.[rag]' '.[docs]'     # scikit-learn + pypdf/docx2txt (all torch-free)
export HRPA_USE_MOCK_RAG=false      # HRPA_RAG_BACKEND defaults to "tfidf"
export HRPA_RAG_CORPUS_DIR=data/corpus
python -m hr_policy_agent.cli "How much bereavement leave do I get?"
```

Documents are loaded, chunked, and indexed in memory at startup. Great to get going.

#### 2. Embeddings + Chroma vector database (semantic search, persistent)

**Step 1 — embed your PDFs/DOCX into the vector DB** (run once, and whenever docs change):

```bash
pip install '.[vectordb]'                 # langchain-chroma + pypdf/docx2txt + openai embeddings

# pick an embedding model — OpenAI is torch-free and best on Windows:
export HRPA_EMBEDDING_PROVIDER=openai
export HRPA_EMBEDDING_MODEL=text-embedding-3-small
export HRPA_LLM_API_KEY=sk-...            # your OpenAI key (used for embeddings)

python -m scripts.ingest                  # reads data/corpus/<tool>/…  →  data/chroma/
```

The ingest script reads every `.txt/.md/.pdf/.docx` under each tool folder, splits it into
~1400-char chunks, embeds each chunk, and stores it in a **Chroma collection named after
the handbook**. Options: `--corpus-dir`, `--chroma-dir`, `--reset` (rebuild from scratch).

**Step 2 — serve using the vector DB:**

```bash
export HRPA_USE_MOCK_RAG=false
export HRPA_RAG_BACKEND=chroma
export HRPA_EMBEDDING_PROVIDER=openai      # must match what you ingested with
export HRPA_LLM_API_KEY=sk-...
python -m hr_policy_agent.cli "How much bereavement leave do I get?"
```

At query time each handbook's Chroma collection is loaded and queried by embedding
similarity; the top chunks feed the answer + citation cards. If a collection is empty
(you haven't ingested it yet) that route falls back to the mock passage.

**Embedding provider options** (`HRPA_EMBEDDING_PROVIDER`):

| Value | Model | Notes |
|---|---|---|
| `openai` | `text-embedding-3-small` | **Recommended.** Torch-free, needs an API key. |
| `huggingface` | `all-MiniLM-L6-v2` | Local/offline, but pulls in `torch` (`pip install '.[rag-semantic]'`). |
| `fake` | deterministic hash | No deps — for offline demos/tests only, not real semantics. |

> **Windows tip:** the `openai` embedding provider and the TF-IDF backend are both
> torch-free, so they avoid the `WinError 206` long-path failure that `torch` triggers.
> Only `huggingface` embeddings / `.[rag-semantic]` need torch — run those from a short
> path such as `C:\dev\HR-Policy-Agent` with long paths enabled.

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
