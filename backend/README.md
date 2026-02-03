GuideCare backend (FastAPI)
===========================

Overview
- FastAPI backend providing:
  - PostgreSQL storage for patients + conversations
  - REST endpoints:
    - GET /patients
    - GET /patients/{id}
    - GET /patients/{id}/context
    - POST /patients
    - POST /conversations
    - GET /conversations/{id}
  - WebSocket endpoint: /ws/chat/{patient_id} for real-time chat
  - LangGraph integration placeholder (async HTTP call)

Quick start (using docker-compose)
1. Copy `.env.example` to `.env` and update values (or set env vars).
2. From repo root run:
   docker-compose up --build
3. Backend will be available at http://localhost:8000
   - OpenDocs: http://localhost:8000/docs

Environment
- DATABASE_URL: SQLAlchemy async connection string, e.g. `postgresql+asyncpg://user:pass@postgres:5432/db`
- LANGGRAPH_API_URL: Base URL for LangGraph (optional)
- LANGGRAPH_API_KEY: API key for LangGraph (optional)
- LANGGRAPH_WORKFLOW_ID: workflow ID to call (optional)

Notes
- On startup the server will create tables and seed sample patients if DB empty.
- The WebSocket endpoint expects JSON messages: {"role":"user","content":"...","meta":{...}}
- To integrate with LangGraph, set LANGGRAPH_API_URL, LANGGRAPH_API_KEY and LANGGRAPH_WORKFLOW_ID.

Testing
- Install dev dependencies:
  ```bash
  pip install -r backend/requirements-dev.txt
  ```

- Option A — use an existing Postgres (set `TEST_DATABASE_URL`):
  ```bash
  export TEST_DATABASE_URL="postgresql+asyncpg://test:test@localhost:5432/guidecare_test"
  # On PowerShell (Windows):
  # $env:TEST_DATABASE_URL = "postgresql+asyncpg://test:test@localhost:5432/guidecare_test"
  TEST_DATABASE_URL="$TEST_DATABASE_URL" python backend/scripts/init_db.py
  pytest -q backend/tests
  ```

- Option B — use Testcontainers (no local DB required):
  ```bash
  pip install testcontainers
  pytest -q backend/tests
  ```
  Tests will automatically start a temporary Postgres container (requires Docker engine accessible to your user).

Next steps & improvements
- Add migrations (alembic) for production
- Use proper auth & CORS policies
- Add persistent background queue for LangGraph retry / error handling
- Add tests and CI
