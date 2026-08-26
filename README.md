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

## Web UI (for end users)

A browser chat page is included — the friendliest way to use the agent:

```bash
pip install '.[web]'
python -m hr_policy_agent.web        # then open http://localhost:8000
#   options: --host 0.0.0.0 --port 8080   (share on your network)
```

It reads the same `.env` configuration as the CLI (LLM provider, RAG backend, etc.),
renders the answer with its citation cards, keeps conversation context, and shows the
active provider/model in the header. Endpoints: `GET /` (page), `POST /api/ask`
(`{message, conversation_id?}` → `{response, language, conversation_id}`), `GET /api/config`.

```python
from hr_policy_agent import HRPolicyAgent

agent = HRPolicyAgent()
print(agent.answer("¿Cuándo es el día de pago?"))          # Spanish route
print(agent.answer("Where can I see my pay stub?"))         # → ADP referral
```

## Using a real LLM / real backends

> **The startup banner tells you what's active**, e.g.
> `LLM provider: ollama | fast=True | mock HCM=True RAG=False`. If it says
> `provider: fake` your answers are echoed retrieved text (not written), and if it says
> `RAG=True` a **mock retriever returns the same canned passage for every question** —
> set `HRPA_LLM_PROVIDER` and `HRPA_USE_MOCK_RAG=false` to fix both.

**Settings persist via a `.env` file** — copy `.env.example` to `.env` in the project
root and edit it once, instead of re-exporting environment variables in every terminal
(PowerShell `$env:` vars don't survive a new window). Real environment variables still
override `.env`. A minimal working `.env` for local Ollama + your own docs:

```ini
HRPA_LLM_PROVIDER=ollama
HRPA_LLM_MODEL=llama3.2:3b
HRPA_FAST_MODE=true
HRPA_USE_MOCK_RAG=false
HRPA_RAG_CORPUS_DIR=data/corpus
```

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

Retrieval quality notes:
* Documents are split **section-aware** — each handbook heading (e.g. "Voting Leave")
  starts a new chunk, so a topic is retrieved as a unit rather than blended with its
  neighbours. Tune with `HRPA_RAG_CHUNK_SIZE` (default 900) / `HRPA_RAG_CHUNK_OVERLAP`.
* TF-IDF is **keyword** search: a short query like "can I take voting leave?" works, but
  paraphrases that share no words with the handbook can miss. For robust semantic
  matching (understands intent, not just words), use the Chroma + embeddings backend
  below.

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

| Value | Default model | Notes |
|---|---|---|
| `ollama` | `nomic-embed-text` | **Local & private**, torch-free, no API key — needs a running Ollama server. |
| `openai` | `text-embedding-3-small` | Torch-free, high quality, needs an API key. |
| `huggingface` | `all-MiniLM-L6-v2` | Local/offline, but pulls in `torch` (`pip install '.[rag-semantic]'`). |
| `fake` | deterministic hash | No deps — for offline demos/tests only, not real semantics. |

#### Using Ollama for embeddings (fully local)

[Install Ollama](https://ollama.com), pull an embedding model, then ingest and serve with
the `ollama` provider — nothing leaves your machine and no API key is needed:

```bash
ollama pull nomic-embed-text            # or: mxbai-embed-large

pip install '.[vectordb-ollama]'
export HRPA_EMBEDDING_PROVIDER=ollama
export HRPA_EMBEDDING_MODEL=nomic-embed-text          # optional; this is the default
export HRPA_OLLAMA_BASE_URL=http://localhost:11434    # optional; this is the default

python -m scripts.ingest                # embed data/corpus/... into data/chroma/

export HRPA_USE_MOCK_RAG=false
export HRPA_RAG_BACKEND=chroma
python -m hr_policy_agent.cli "How much bereavement leave do I get?"
```

You can also run the **chat model** locally on the same server:
`HRPA_LLM_PROVIDER=ollama` with `HRPA_LLM_MODEL=llama3.1` (or any model you've pulled).
The embedding provider used at query time must match the one used during ingest.

> **Windows tip:** the `openai` embedding provider and the TF-IDF backend are both
> torch-free, so they avoid the `WinError 206` long-path failure that `torch` triggers.
> Only `huggingface` embeddings / `.[rag-semantic]` need torch — run those from a short
> path such as `C:\dev\HR-Policy-Agent` with long paths enabled.

## Performance (local models / Ollama)

A full policy turn makes **5–6 sequential LLM calls**, and several ported Oracle prompts
are large (the intent router is ~21 KB, the answer prompt ~22 KB). On a local CPU model
that adds up to minutes per question. Two levers help a lot:

**1. Fast mode** — deterministic routing + a single compact answer call:

```bash
python -m hr_policy_agent.cli --fast "What's the policy on removal of an unruly guest?"
# or: export HRPA_FAST_MODE=true
```

Fast mode routes intent/language and reformulates the query with deterministic logic (no
LLM), skips the answer-polishing pass, and gives the one real answer call a short prompt.
Measured on a policy question: **6 calls / ~69 KB of prompts → 1 call / ~0.5 KB.** Routing
is slightly less nuanced, but for handbook Q&A it's usually indistinguishable — and far
faster.

**2. Use a small model and see where time goes:**

```bash
ollama pull llama3.2:3b            # a 3B model is far faster than 8B on CPU
export HRPA_LLM_MODEL=llama3.2:3b
python -m hr_policy_agent.cli --fast --timings "How much bereavement leave do I get?"
# [timing] ANSWER_AGENT_: 3.4s (prompt 478 chars)
```

Other knobs: `HRPA_LLM_MAX_TOKENS` bounds answer length (Ollama `num_predict`);
`HRPA_OLLAMA_NUM_CTX` (default 8192) is the context window. The first call after starting
Ollama also pays a one-time model-load cost.

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
