# AI Sales Assistant for SMB Banking

A production-grade conversational AI that helps bank managers sell SMB products (current accounts, acquiring, credit) via Telegram or REST API. The agent qualifies the client, retrieves grounded product information from a vector database, calculates tariffs on the fly, and creates CRM leads — all within a single multi-turn dialogue. Every response is automatically scored by an LLM judge to measure quality without human labelling.

---

## Problem & Goal

SMB sales calls follow a predictable qualification funnel: collect business type and monthly turnover, map them to the right tariff, handle objections with accurate numbers, and convert to a lead. Human managers are inconsistent; generic chatbots hallucinate pricing. This project shows how prompt engineering, RAG, and structured function calling can automate that funnel reliably — and how to *measure* reliability with an automated evaluation pipeline.

---

## Architecture

```
User (Telegram / REST)
        │
        ▼
  FastAPI  (main.py)
        │
        ▼
LangGraph ReAct Agent  (agent.py)
        ├── RAG Tool        ──► ChromaDB  (53 chunks, all-MiniLM-L6-v2)
        ├── Tariff Tool     ──► tariffs.json  (rule-based calculator)
        └── Lead Tool       ──► leads.db  (SQLite CRM)
        │
        ▼
ConversationBufferMemory  (window = 10 turns, client profile in session state)
        │
        ▼
LLM Judge  (evaluator.py)  ──► logs/evaluation_log.jsonl
```

**Request flow:**
1. User message arrives via Telegram webhook or `POST /chat`
2. LangGraph agent prepends system prompt + session history, then runs a ReAct loop
3. Agent calls tools as needed (RAG for facts, Tariff for pricing, Lead for conversion)
4. Response is returned; an async background task scores it with the LLM judge
5. Scores are appended to `evaluation_log.jsonl` for offline analysis

---

## Tech Stack

| Component | Technology | Version |
|---|---|---|
| API framework | FastAPI + Uvicorn | 0.115.5 / 0.32.1 |
| Agent framework | LangGraph (ReAct) | ≥ 1.0.0 |
| LLM | Groq — `llama3-groq-70b-8192-tool-use-preview` | langchain-groq 1.1.2 |
| Embeddings | `all-MiniLM-L6-v2` (local, CPU) | sentence-transformers 5.5.1 |
| Vector DB | ChromaDB | 1.5.9 |
| Validation | Pydantic v2 | 2.10.3 |
| Telegram | python-telegram-bot | 21.7 |
| Testing | pytest + pytest-asyncio | 8.3.4 / 0.24.0 |
| Python | CPython | 3.11+ |

---

## Prompt Engineering

### System Prompt (`app/prompts/system_prompt.md`)

The agent plays **Alex**, a senior bank manager with 7 years of SMB experience. The prompt enforces a three-stage sales funnel:

1. **Qualification** — ask at most 2 questions per turn (business form, sector, monthly turnover, desired products). No lists, natural dialogue only.
2. **Proposal** — only after business type and turnover are known. Every number must come from a tool call (`search_knowledge_base` or `calculate_tariff`). Hard rule: *no invented figures*.
3. **Closing** — once the client signals readiness, collect name + phone and call `create_lead`.

Key guardrails explicitly in the prompt:
- "Never state rates, amounts, or deadlines without calling a tool first."
- "Offer only products that exist in the knowledge base."
- No banking jargon (the prompt maps terms: "annuity" → "equal monthly payments").

### RAG Prompt (`app/prompts/rag_prompt.md`)

Used by the RAG tool after retrieving the top-3 chunks (cosine similarity, 500-token chunks, 50-token overlap). The prompt instructs the synthesis LLM to:
- Answer *strictly* from the provided context — no additions.
- Quote exact numbers as they appear in the source.
- If the context doesn't contain an answer, return a fixed phrase directing the user to a live manager — preventing silent hallucination.
- Keep the response to 2–4 sentences with no meta-phrases like "Based on the context…".

### LLM Judge (`app/prompts/judge_prompt.md`)

Every agent response is scored asynchronously on three metrics (0–10 integer each):

| Metric | Measures |
|---|---|
| `relevance` | Does the response address what was asked? |
| `groundedness` | Are all stated facts traceable to real bank products? |
| `sales_effectiveness` | Does the response move the dialogue toward a lead / account opening? |

The judge also produces a `reasoning` field (2–3 sentences), making failures debuggable without manual inspection. Scores are written to `evaluation_log.jsonl` with timestamp and session ID, enabling offline aggregation (e.g., average scores per prompt version, per product category).

The judge uses `with_structured_output(EvaluationResult)` — Pydantic enforces score bounds [0, 10] at the framework level, so malformed judge responses raise a `ValidationError` rather than silently polluting the log.

---

## Key Design Decisions

### LangGraph instead of AgentExecutor

`AgentExecutor` (LangChain classic) accumulates tool outputs in a single growing string, which causes context overflow on long conversations and makes tool-call debugging opaque. LangGraph represents the agent loop as an explicit state machine with typed `messages` — each node is observable, the graph can be interrupted and resumed, and the message history stays structured. This also makes it straightforward to inject session history as typed `HumanMessage`/`AIMessage` objects without mixing formats.

### Prompt versioning in `.md` files

All prompts live in `app/prompts/*.md` and are loaded at runtime via `_load_prompt(name)`. This means:
- Prompt changes are tracked by git diff like any source file.
- A/B testing different prompt versions requires no code changes — swap the file, restart, re-run evals.
- The LLM judge evaluates the *deployed* prompt, not a hardcoded string buried in Python.

### API key rotation on rate limit

Groq free-tier keys have per-minute token limits. `GroqKeyManager` (`app/key_manager.py`) holds a pool of keys (`GROQ_API_KEY_1/2/3`) and cycles through them when it detects a 429 response. Both the main agent and the LLM judge share the same key manager. On rotation, the agent rebuilds its LangGraph and tools with the new key (the embedding model and ChromaDB vectorstore are reused — they load once at startup). If all keys are exhausted, the user receives a user-friendly retry message instead of a stack trace.

---

## Results

| Metric | Value |
|---|---|
| Unit tests | 109 passing, 2 skipped (integration, require live API key) |
| Test coverage | RAG tool, tariff calculator, lead tool, evaluator, memory module, key manager |
| Avg Relevance | 6.6 / 10 |
| Avg Groundedness | 8.3 / 10 |
| Avg Sales Effectiveness | 5.9 / 10 |
| Avg Overall | 6.9 / 10 |
| Avg turn latency | ~6.5 s (Groq free tier) |
| API key resilience | Key rotation triggered during eval run (key 1 → key 2); zero user-visible errors |
| Full sales funnel | qualification → RAG retrieval → tariff calculation → lead creation |

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd sales-assistant
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — add at least one Groq key (GROQ_API_KEY_1)
# Optionally add TELEGRAM_BOT_TOKEN for Telegram mode

# 3. Ingest knowledge base into ChromaDB (one-time)
python scripts/ingest.py

# 4. Run the API
uvicorn app.main:app --reload

# 5. Chat via REST
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "message": "Хочу открыть счёт для ИП"}'

# 6. Run tests
pytest tests/ -v

# 7. Run prompt evaluations (requires GROQ_API_KEY)
python scripts/run_evals.py
```

---

## Project Structure

```
sales-assistant/
├── app/
│   ├── agent.py          # LangGraph ReAct agent; key rotation retry loop
│   ├── evaluator.py      # LLM judge; async scoring; JSONL logging
│   ├── key_manager.py    # GroqKeyManager; AllKeysExhaustedError; _is_rate_limit
│   ├── main.py           # FastAPI app; Telegram webhook handler
│   ├── memory.py         # SessionMemory; ClientProfile; MemoryManager
│   ├── data/
│   │   ├── tariffs.json          # Structured tariff rules (RKO, acquiring, credit)
│   │   └── knowledge_base/       # Source documents for RAG
│   │       ├── rko.md
│   │       ├── acquiring.md
│   │       └── credit.md
│   ├── prompts/
│   │   ├── system_prompt.md      # Agent persona, dialogue strategy, guardrails
│   │   ├── rag_prompt.md         # RAG synthesis; anti-hallucination rules
│   │   └── judge_prompt.md       # Evaluation rubric; per-metric score anchors
│   └── tools/
│       ├── rag_tool.py           # ChromaDB retrieval + synthesis via rag_prompt
│       ├── tariff_tool.py        # Rule-based tariff calculator from tariffs.json
│       └── lead_tool.py          # SQLite CRM; UUID lead IDs
├── scripts/
│   ├── ingest.py         # Chunk knowledge_base/*.md → ChromaDB (run once)
│   └── run_evals.py      # Batch prompt evaluation; prints aggregate scores
├── tests/
│   ├── test_key_manager.py   # GroqKeyManager unit tests; rate-limit retry mocks
│   ├── test_tools.py         # Tariff calculator (boundary tests); lead tool (SQLite)
│   ├── test_prompts.py       # Evaluator unit tests; memory module; prompt file checks
│   └── test_rag.py           # RAG tool unit tests
├── logs/
│   └── evaluation_log.jsonl  # Append-only; one JSON object per agent turn
├── .env.example          # All required env vars with descriptions
├── requirements.txt
└── pytest.ini
```
