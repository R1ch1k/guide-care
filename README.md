# GuideCare — NHS NICE Guideline Clinical Decision Support

A full-stack clinical decision support system that traverses NICE (National Institute for Health and Care Excellence) guideline decision trees to provide evidence-based recommendations. Combines a real graph-traversal engine with LLM-powered variable extraction and clarification.

## Architecture

```
Frontend (Next.js)  <--WebSocket-->  Backend (FastAPI + LangGraph)
                                         |
                                    Guideline Engine
                                    (graph traversal)
                                         |
                                    LLM (OpenAI API or local gpt-oss-20b)
```

**Frontend** — Next.js chat UI with WebSocket real-time messaging, patient selector, data import modal, and guideline-based recommendations display.

**Backend** — FastAPI with async PostgreSQL (SQLAlchemy + asyncpg), LangGraph state machine orchestration, and WebSocket endpoint for per-patient conversations.

**Guideline Engine** — Pure-Python BFS traversal of NICE guideline decision trees. Evaluates condition nodes (numeric comparisons, BP ranges, AND/OR logic) and returns reached action nodes with the full decision path.

**LLM Layer** — Used for triage (urgency + guideline selection), variable extraction from conversation, and clarification question generation. Supports OpenAI API or a local model.

## NICE Guidelines Covered

| ID | Guideline | Topic |
|----|-----------|-------|
| NG84 | Sore throat (acute) | Antibiotic prescribing |
| NG91 | Otitis media (acute) | Ear infection management |
| NG112 | UTI (lower) | Urinary tract infection |
| NG133 | Hypertension in pregnancy | Pre-eclampsia screening |
| NG136 | Hypertension in adults | BP diagnosis and management |
| NG184 | Bite wounds | Animal/human bite management |
| NG222 | Depression in adults | Treatment pathways |
| NG232 | Head injury | Assessment and early management |
| NG81 (Glaucoma) | Chronic open-angle glaucoma | IOP-based treatment |
| NG81 (Hypertension) | Ocular hypertension | Risk-based treatment |

Each guideline is stored as two JSON files:
- `backend/data/guidelines/<id>.json` — decision tree (condition/action nodes + edges)
- `backend/data/evaluators/<id>_eval.json` — condition evaluation logic per node

## Pipeline Flow

Each user message goes through a LangGraph state machine:

1. **Load Patient** — fetch patient record from the database
2. **Triage** — LLM determines urgency (green/amber/red) and selects the best-matching NICE guideline
3. **Select Guideline** — load the guideline JSON decision tree
4. **Clarify** — if required clinical variables are missing, generate a clarification question
5. **Extract Variables** — LLM extracts structured clinical variables from the conversation
6. **Walk Graph** — BFS traversal of the guideline decision tree using extracted variables
7. **Format Recommendation** — produce a structured recommendation with the decision pathway

On completion, a `Diagnosis` record is automatically persisted to the database.

## LLM Modes: API vs Local Model

### OpenAI API (default, recommended for demos)

Uses `gpt-4o` via the OpenAI API. Set your key in the backend `.env`:

```bash
OPENAI_API_KEY=sk-proj-your-key-here
OPENAI_MODEL=gpt-4o
```

Pros: No GPU required, fast, high accuracy.
Cons: Requires API key, costs per token, data leaves your machine.

### Local model: gpt-oss-20b (for research / offline use)

A 21B-parameter MoE model (3.6B active) that runs on a single A100 GPU. Used in the Colab test suite notebooks (`gpt_oss_20b_final_test_suite.ipynb` and `gpt_api_final_test_suite.ipynb`).

To switch the test notebook to local model:
1. Open the notebook on Google Colab with an A100 runtime
2. In the configuration cell, set `USE_API = False`
3. The notebook will download and load the model in BF16

The backend currently uses the OpenAI API only. To use a local model in the backend, you would replace `app/llm.py` with a local inference wrapper.

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Node.js 18+
- An OpenAI API key (for the backend LLM features)

### 1. Clone and configure

```bash
git clone https://github.com/R1ch1k/guide-care.git
cd guide-care
git checkout sqlite-database
```

Create the backend environment file:

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` and set your `OPENAI_API_KEY`.

### 2. Start the backend

```bash
docker-compose up --build
```

This starts:
- PostgreSQL on port 5432
- FastAPI backend on port 8000 (API docs at http://localhost:8000/docs)

On first startup, the backend creates tables and seeds sample patients.

**Running without Docker** (development):

```bash
cd backend
pip install -r requirements.txt
PYTHONPATH=src uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Start the frontend

```bash
npm install
npm run dev
```

Open http://localhost:3000 in your browser.

### 4. Use the application

1. Select a patient from the sidebar (or import your own via the **Connect** button)
2. Describe symptoms in the chat (e.g. "patient has a sore throat, 38.5C fever, no cough")
3. The system will:
   - Triage the symptoms for urgency (green/amber/red)
   - Select the appropriate NICE guideline
   - Extract clinical variables from the conversation
   - Ask clarification questions if variables are missing
   - Traverse the guideline decision tree
   - Return the evidence-based recommendation with the decision path
4. Completed diagnoses are automatically saved and can be exported via the API

## Importing Patient Data

Click the **Connect** button in the patient info panel to open the data import modal.

### CSV Upload

Upload a `.csv` file with the following columns:

| nhs_number | first_name | last_name | date_of_birth | gender | conditions | medications | allergies |
|------------|------------|-----------|---------------|--------|------------|-------------|-----------|
| 123-456-7890 | Jane | Smith | 1985-03-15 | Female | Asthma, Anxiety | Salbutamol | Penicillin |

- `conditions` and `allergies` can be comma-separated strings or JSON arrays
- `medications` can be a JSON array of `{"name": "...", "dose": "..."}` objects or comma-separated names
- A sample CSV template is available for download in the modal

### Excel Upload

Upload a `.xlsx` file with the same column headers in the first row.

## Backend API

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/patients` | List all patients |
| GET | `/patients/{id}` | Get patient details |
| GET | `/patients/{id}/context` | Get patient context for LLM |
| POST | `/patients` | Create a patient |
| POST | `/patients/import` | Import patients from CSV/Excel file |
| POST | `/conversations` | Start a new conversation |
| GET | `/conversations/{id}` | Get conversation history |
| GET | `/diagnoses` | List all completed diagnoses |
| GET | `/diagnoses/{id}` | Get a single diagnosis |
| GET | `/diagnoses/export?format=json\|csv` | Export all diagnoses |

### WebSocket

Connect to `/ws/chat/{patient_id}` for real-time chat. Send JSON:

```json
{"role": "user", "content": "patient has a sore throat", "meta": {}}
```

The server streams events back including triage results, clarification questions, and final recommendations.

## Project Structure

```
guide-care/
├── app/                        # Next.js frontend pages
├── components/
│   ├── ChatPanel.tsx           # Main chat interface with WebSocket
│   ├── PatientInfoPanel.tsx    # Patient details sidebar
│   ├── ConnectDataModal.tsx    # Data import modal (CSV/Excel upload)
│   ├── SampleInputModal.tsx    # Sample inputs with multi-guideline examples
│   ├── DecisionCard.tsx        # Decision pathway visualization
│   ├── AddPatientModal.tsx     # Manual patient creation form
│   └── ui/                     # Shared UI components (shadcn)
├── lib/                        # Frontend utilities and types
├── backend/
│   ├── data/
│   │   ├── guidelines/         # 10 NICE guideline JSON decision trees
│   │   └── evaluators/         # 10 evaluator JSON files
│   ├── src/
│   │   ├── main.py             # FastAPI app entry point
│   │   └── app/
│   │       ├── core/config.py  # Settings (DB, OpenAI, CORS)
│   │       ├── db/
│   │       │   ├── models.py   # Patient, Conversation, Diagnosis models
│   │       │   └── session.py  # Async SQLAlchemy session
│   │       ├── api/
│   │       │   ├── patients.py     # Patient CRUD + CSV/Excel import
│   │       │   ├── conversations.py # Conversation endpoints
│   │       │   └── diagnoses.py    # Diagnosis list, detail, export
│   │       ├── guideline_engine.py  # Graph traversal + recommendation formatting
│   │       ├── llm.py              # Async OpenAI wrapper
│   │       ├── orchestration/
│   │       │   ├── graph.py    # LangGraph state machine definition
│   │       │   ├── nodes.py    # LangGraph node functions
│   │       │   ├── state.py    # ConversationState TypedDict
│   │       │   ├── deps.py     # Triage, clarify, extract, traverse, format
│   │       │   └── runner.py   # process_user_turn() entry point
│   │       ├── schemas.py      # Pydantic request/response models
│   │       ├── crud.py         # Database CRUD operations
│   │       ├── seed.py         # Sample patient data seeder
│   │       └── ws_manager.py   # WebSocket manager + diagnosis auto-persist
│   ├── tests/                  # Backend test suite
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
├── docker-compose.yml
└── package.json
```

## Test Notebooks

Two Colab notebooks validate the full pipeline independently of the backend:

- **gpt_oss_20b_final_test_suite.ipynb** — runs on local gpt-oss-20b model (A100 GPU)
- **gpt_api_final_test_suite.ipynb** — runs on OpenAI API (gpt-4o)

Both test: variable extraction (30 cases), multi-turn clarification (10 cases), recommendation formatting (10 cases), and error handling (10 cases).

Latest API results: 10/10 pipeline success, 85% extraction accuracy, 96.2% overall score.

## Environment Variables

### Backend (`backend/.env`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| DATABASE_URL | Yes | — | PostgreSQL async connection string |
| OPENAI_API_KEY | Yes | — | OpenAI API key for LLM features |
| OPENAI_MODEL | No | gpt-4o | OpenAI model to use |
| CORS_ORIGINS | No | * | Comma-separated allowed origins |

### Frontend (`.env.local`)

| Variable | Required | Description |
|----------|----------|-------------|
| NEXT_PUBLIC_BACKEND_URL | No | Backend URL (defaults to http://localhost:8000) |

## Safety Notice

This tool is for healthcare professionals only and does not replace clinical judgment. Always consider individual patient context, contraindications, and local protocols. All recommendations cite the source NICE guideline.
