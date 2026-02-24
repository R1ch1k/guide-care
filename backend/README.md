# GuideCare Backend

FastAPI backend with LangGraph orchestration, NICE guideline graph traversal engine, and real-time WebSocket chat.

## Stack

- **FastAPI** — async REST + WebSocket
- **PostgreSQL** — patient and conversation storage (SQLAlchemy + asyncpg)
- **LangGraph** — state machine orchestration for the clinical pipeline
- **OpenAI API** — variable extraction, clarification questions, guideline selection fallback
- **Guideline Engine** — pure-Python BFS traversal of NICE decision trees (no LLM needed)

## Setup

### With Docker (recommended)

```bash
# From the repo root
cp backend/.env.example backend/.env
# Edit backend/.env — set OPENAI_API_KEY

docker-compose up --build
```

Backend runs on http://localhost:8000. API docs at http://localhost:8000/docs.

### Without Docker

```bash
cd backend

# Create virtualenv
python -m venv .venv && source .venv/bin/activate

# Install deps
pip install -r requirements.txt

# Set environment
cp .env.example .env
# Edit .env — set DATABASE_URL pointing to your PostgreSQL, set OPENAI_API_KEY

# Run
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Requires a running PostgreSQL instance.

## Pipeline

The LangGraph state machine runs this flow for each user message:

```
load_patient → triage → clarify → select_guideline → extract_variables → walk_graph → format_output
```

1. **load_patient** — fetch patient record from DB
2. **triage** — keyword heuristics (or external API) to assess urgency
3. **clarify** — if variables are missing, LLM generates targeted clarification questions
4. **select_guideline** — keyword mapping to one of 10 NICE guidelines, LLM fallback
5. **extract_variables** — LLM extracts clinical variables from conversation + regex post-processing
6. **walk_graph** — BFS traversal of the guideline decision tree using extracted variables
7. **format_output** — template-based recommendation from action nodes, LLM fallback

## Guideline Engine

The core of the system. Located in `src/app/guideline_engine.py`.

Loads 10 NICE guideline JSON files from `data/guidelines/` and their evaluators from `data/evaluators/`. Traverses condition/action node graphs by evaluating:
- Simple variable checks (boolean, string equality)
- Numeric comparisons (age, temperature, IOP)
- Blood pressure ranges (systolic/diastolic)
- Compound conditions (AND/OR)
- Treatment type matching

Returns: reached action nodes, full decision path, and any missing variables.

## API Reference

### REST

| Method | Path | Description |
|--------|------|-------------|
| GET | /patients | List patients |
| GET | /patients/{id} | Patient details |
| GET | /patients/{id}/context | Patient context for LLM |
| POST | /patients | Create patient |
| POST | /conversations | Start conversation |
| GET | /conversations/{id} | Conversation history |

### WebSocket

`/ws/chat/{patient_id}` — send/receive JSON messages:

```json
{"role": "user", "content": "patient has a sore throat, 38.5C fever", "meta": {}}
```

Response events stream back as the pipeline progresses (triage, clarification, recommendation).

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| DATABASE_URL | Yes | — | `postgresql+asyncpg://user:pass@host:5432/db` |
| OPENAI_API_KEY | Yes | — | OpenAI API key |
| OPENAI_MODEL | No | gpt-4o | Model for LLM calls |
| CORS_ORIGINS | No | * | Allowed origins (comma-separated) |
| LANGGRAPH_API_URL | No | — | External LangGraph endpoint |
| LANGGRAPH_API_KEY | No | — | External LangGraph key |
| LANGGRAPH_WORKFLOW_ID | No | — | External workflow ID |

## Testing

```bash
# Install dev deps
pip install -r requirements-dev.txt

# Option A: existing PostgreSQL
export TEST_DATABASE_URL="postgresql+asyncpg://test:test@localhost:5432/guidecare_test"
pytest -q tests

# Option B: Testcontainers (requires Docker)
pip install testcontainers
pytest -q tests
```

## File Structure

```
backend/
├── data/
│   ├── guidelines/         # 10 NICE guideline decision tree JSONs
│   └── evaluators/         # 10 evaluator logic JSONs
├── src/app/
│   ├── main.py             # FastAPI app entry point
│   ├── core/config.py      # Pydantic settings
│   ├── db/
│   │   ├── models.py       # Patient, Conversation, Message models
│   │   └── session.py      # Async session factory
│   ├── guideline_engine.py # Graph traversal + extraction helpers (837 lines)
│   ├── llm.py              # Async OpenAI wrapper
│   ├── orchestration/
│   │   ├── graph.py        # LangGraph StateGraph definition
│   │   ├── nodes.py        # Node functions wired to deps
│   │   ├── state.py        # ConversationState TypedDict
│   │   ├── deps.py         # Real implementations of all pipeline steps
│   │   └── runner.py       # process_user_turn() entry point
│   ├── routes/
│   │   ├── patients.py
│   │   └── conversations.py
│   └── ws_manager.py       # WebSocket ConnectionManager
├── Dockerfile
├── requirements.txt
└── .env.example
```
