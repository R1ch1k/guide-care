import httpx
from typing import Any, Callable, Dict, Optional, TypedDict
from uuid import UUID

from sqlalchemy import select

from app.core.config import settings
from app.db.models import Patient
from app.db.session import AsyncSessionLocal


class OrchestrationDeps(TypedDict):
    fetch_patient: Callable[..., Any]
    triage_agent: Callable[..., Any]
    gpt_clarifier: Callable[..., Any]
    select_guideline: Callable[..., Any]
    extract_variables_20b: Callable[..., Any]
    walk_guideline_graph: Callable[..., Any]
    format_output_20b: Callable[..., Any]


async def fetch_patient(patient_id: str) -> Dict[str, Any]:
    """Load a patient record from DB as a plain dict."""
    pid = UUID(patient_id)
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Patient).where(Patient.id == pid))
        p = result.scalars().first()
        if not p:
            return {"id": patient_id}

        return {
            "id": str(p.id),
            "nhs_number": p.nhs_number,
            "first_name": p.first_name,
            "last_name": p.last_name,
            "age": p.age,
            "gender": p.gender,
            "conditions": p.conditions or [],
            "medications": p.medications or [],
            "allergies": p.allergies or [],
            "recent_vitals": p.recent_vitals or {},
            "clinical_notes": p.clinical_notes or [],
        }


async def triage_agent(symptoms: str, history: list, patient_record: dict) -> Dict[str, Any]:
    """Call teammate triage API if configured; otherwise fallback."""
    if settings.TRIAGE_API_URL:
        async with httpx.AsyncClient(timeout=settings.AI_TIMEOUT_SECONDS) as client:
            r = await client.post(
                settings.TRIAGE_API_URL.rstrip("/") + "/triage",
                json={"symptoms": symptoms, "history": history, "patient": patient_record},
            )
            r.raise_for_status()
            return r.json()

    # Fallback heuristics
    s = (symptoms or "").lower()
    urgent_words = ["chest pain", "shortness of breath", "collapse", "seizure", "stroke"]
    if any(w in s for w in urgent_words):
        return {"urgency": "high", "assessment": "Potentially urgent symptoms detected"}
    return {"urgency": "moderate", "assessment": "Initial non-urgent triage (fallback)"}


async def gpt_clarifier(
    symptoms: str,
    history: list,
    patient_record: dict,
    triage: dict,
    answers: dict,
) -> Dict[str, Any]:
    """
    Replace with GPT-4 call later.
    For now, deterministic fallback that asks up to 2 questions for headache/dizziness.
    """
    if answers:
        return {"done": True, "questions": []}

    s = (symptoms or "").lower()
    if "headache" in s and ("dizzy" in s or "dizziness" in s):
        return {
            "done": False,
            "questions": [
                "Have you lost consciousness?",
                "Any vomiting or blurred vision?",
            ],
        }

    return {"done": True, "questions": []}


async def select_guideline(symptoms: str, triage: dict, answers: dict, patient_record: dict) -> str:
    s = (symptoms or "").lower()

    if "headache" in s:
        return "NG232"
    if "blood pressure" in s or "hypertension" in s:
        return "NG136"
    if "depression" in s:
        return "NG222"
    return "NG204"


async def extract_variables_20b(guideline: str, history: list, patient: dict, clarifications: dict) -> Dict[str, Any]:
    """Call local model API if configured, else fallback extraction."""
    if settings.LOCAL_20B_API_URL:
        async with httpx.AsyncClient(timeout=settings.AI_TIMEOUT_SECONDS) as client:
            r = await client.post(
                settings.LOCAL_20B_API_URL.rstrip("/") + "/extract",
                json={
                    "guideline": guideline,
                    "history": history,
                    "patient": patient,
                    "clarifications": clarifications,
                },
            )
            r.raise_for_status()
            return r.json()

    # Fallback extraction
    out: Dict[str, Any] = {}
    for q, a in (clarifications or {}).items():
        ql = q.lower()
        al = str(a).lower()
        if "lost consciousness" in ql:
            out["loss_consciousness"] = al.startswith("n") is False
        if "blurred vision" in ql:
            out["blurred_vision"] = "yes" in al or "blurred" in al
        if "vomiting" in ql:
            out["vomiting"] = "yes" in al or "vomit" in al
    return out


async def walk_guideline_graph(
    guideline: str,
    variables: dict,
    current_node: Optional[str],
    pathway: list,
) -> Dict[str, Any]:
    """
    Placeholder decision walker. Replace with your real NICE graph walk.
    Returns terminal immediately for now.
    """
    next_path = list(pathway or [])
    next_node = current_node or "start"
    next_path.append(next_node)

    return {
        "current_node": next_node,
        "pathway_walked": next_path,
        "terminal": True,
    }


async def format_output_20b(
    guideline: str,
    triage: dict,
    variables: dict,
    pathway: list,
    patient: dict,
) -> Dict[str, Any]:
    """Call local model formatter if configured; fallback text otherwise."""
    if settings.LOCAL_20B_API_URL:
        async with httpx.AsyncClient(timeout=settings.AI_TIMEOUT_SECONDS) as client:
            r = await client.post(
                settings.LOCAL_20B_API_URL.rstrip("/") + "/format",
                json={
                    "guideline": guideline,
                    "triage": triage,
                    "variables": variables,
                    "pathway": pathway,
                    "patient": patient,
                },
            )
            r.raise_for_status()
            return r.json()

    name = patient.get("first_name", "there")
    return {
        "final_recommendation": f"Based on NICE guideline {guideline}, {name}, please seek medical advice if symptoms worsen or persist.",
        "citation": guideline,
    }


def build_orchestration_deps() -> OrchestrationDeps:
    return {
        "fetch_patient": fetch_patient,
        "triage_agent": triage_agent,
        "gpt_clarifier": gpt_clarifier,
        "select_guideline": select_guideline,
        "extract_variables_20b": extract_variables_20b,
        "walk_guideline_graph": walk_guideline_graph,
        "format_output_20b": format_output_20b,
    }
