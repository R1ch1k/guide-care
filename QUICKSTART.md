# Quick Start

## 1. Backend

```bash
# Copy env and set your OpenAI key
cp backend/.env.example backend/.env
# Edit backend/.env → OPENAI_API_KEY=sk-proj-...

# Start PostgreSQL + backend
docker-compose up --build
```

Backend: http://localhost:8000
API docs: http://localhost:8000/docs

## 2. Frontend

```bash
npm install
npm run dev
```

Frontend: http://localhost:3000

## 3. Try it

1. Open http://localhost:3000
2. Select a patient
3. Type: "patient has a sore throat, temperature 38.5, no cough"
4. The system selects NICE NG84, extracts variables, traverses the decision tree, and returns a recommendation

## Switching LLM Mode (Test Notebooks)

The Colab test notebooks support both OpenAI API and a local model:

```python
# In the notebook configuration cell:
USE_API = True   # OpenAI API (gpt-4o) — works anywhere
USE_API = False  # Local gpt-oss-20b — requires A100 GPU on Colab
```

The backend always uses the OpenAI API (configured via `backend/.env`).
