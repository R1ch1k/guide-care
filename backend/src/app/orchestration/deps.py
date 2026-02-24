"""
Orchestration dependencies — real implementations using the guideline engine
and OpenAI API.

Each callable is injected into the LangGraph nodes via build_orchestration_deps().
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional, TypedDict
from uuid import UUID

from sqlalchemy import select

from app.core.config import settings
from app.db.models import Patient
from app.db.session import AsyncSessionLocal
from app.guideline_engine import (
    VAR_DESCRIPTIONS,
    build_patient_record,
    extract_best_question,
    extract_json_from_text,
    fix_variable_extraction,
    fix_variable_extraction_v2,
    format_recommendation_template,
    get_guideline,
    get_missing_variables_for_next_step,
    load_all_guidelines,
    traverse_guideline_graph,
)
from app.llm import generate

logger = logging.getLogger(__name__)


class OrchestrationDeps(TypedDict):
    fetch_patient: Callable[..., Any]
    triage_agent: Callable[..., Any]
    gpt_clarifier: Callable[..., Any]
    select_guideline: Callable[..., Any]
    extract_variables_20b: Callable[..., Any]
    walk_guideline_graph: Callable[..., Any]
    format_output_20b: Callable[..., Any]


# ===================================================================
# 1. Fetch patient from DB
# ===================================================================


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


# ===================================================================
# 2. Triage agent — keyword heuristics + optional external API
# ===================================================================


async def triage_agent(
    symptoms: str, history: list, patient_record: dict
) -> Dict[str, Any]:
    """Triage based on symptoms. Uses external API if configured, else heuristics."""
    if settings.TRIAGE_API_URL:
        import httpx

        async with httpx.AsyncClient(timeout=settings.AI_TIMEOUT_SECONDS) as client:
            r = await client.post(
                settings.TRIAGE_API_URL.rstrip("/") + "/triage",
                json={
                    "symptoms": symptoms,
                    "history": history,
                    "patient": patient_record,
                },
            )
            r.raise_for_status()
            return r.json()

    # Heuristic fallback
    s = (symptoms or "").lower()
    urgent_words = [
        "chest pain",
        "shortness of breath",
        "collapse",
        "seizure",
        "stroke",
        "unconscious",
        "severe bleeding",
        "heart attack",
    ]
    if any(w in s for w in urgent_words):
        return {"urgency": "high", "assessment": "Potentially urgent symptoms detected"}
    return {"urgency": "moderate", "assessment": "Initial non-urgent triage (heuristic)"}


# ===================================================================
# 3. GPT Clarifier — LLM-driven clarification using guideline engine
# ===================================================================


async def gpt_clarifier(
    symptoms: str,
    history: list,
    patient_record: dict,
    triage: dict,
    answers: dict,
) -> Dict[str, Any]:
    """Generate clarification questions driven by the guideline evaluator.

    Uses the guideline engine to find missing variables, then asks the LLM
    to generate a natural-language question targeting the next missing variable.
    """
    # If we already have answers and no guideline selected yet, skip clarification
    if answers:
        return {"done": True, "questions": []}

    # Try to determine guideline from symptoms for early clarification
    s = (symptoms or "").lower()
    guideline_id = _guess_guideline(s)

    if not guideline_id:
        return {"done": True, "questions": []}

    g_data = get_guideline(guideline_id)
    if not g_data:
        return {"done": True, "questions": []}

    # Build known variables from patient record
    known_vars: Dict[str, Any] = {}
    if patient_record.get("age"):
        known_vars["age"] = patient_record["age"]
    if patient_record.get("gender"):
        known_vars["gender"] = patient_record["gender"]

    # Check what's missing
    nodes = g_data["guideline"]["nodes"]
    edges = g_data["guideline"]["edges"]
    evaluator = g_data["merged_evaluator"]

    missing = get_missing_variables_for_next_step(nodes, edges, evaluator, known_vars)

    if not missing:
        return {"done": True, "questions": []}

    # Generate questions for up to 2 missing variables
    questions: List[str] = []
    for target_var in missing[:2]:
        try:
            prompt = f"""You are a medical assistant helping a doctor gather information.

GUIDELINE: NICE {guideline_id}
PATIENT SYMPTOMS: {symptoms}

We need to determine: {target_var}

Generate ONE specific clarification question to ask the doctor. Be concise and professional."""

            raw = await generate(prompt, max_tokens=150, temperature=0.1)
            question = extract_best_question(raw)
            questions.append(question)
        except Exception as e:
            logger.warning("Failed to generate clarification question: %s", e)
            questions.append(f"Could you provide information about {target_var}?")

    return {"done": False, "questions": questions}


# ===================================================================
# 4. Select guideline — keyword mapping + optional LLM
# ===================================================================


def _guess_guideline(symptoms_lower: str) -> Optional[str]:
    """Map symptom keywords to NICE guideline IDs."""
    mapping = [
        (["sore throat", "throat pain", "tonsil"], "NG84"),
        (["head injury", "hit head", "head trauma", "fell", "concussion"], "NG232"),
        (["blood pressure", "hypertension", "bp reading", "bp "], "NG136"),
        (["depression", "antidepressant", "low mood"], "NG222"),
        (["uti", "urinary tract", "urinary infection", "dysuria"], "NG112"),
        (["pregnant", "pregnancy", "gestational", "pre-eclampsia"], "NG133"),
        (["bite", "cat bite", "dog bite", "animal bite"], "NG184"),
        (["ear pain", "otitis", "ear infection"], "NG91"),
        (["glaucoma", "iop", "intraocular"], "NG81_GLAUCOMA"),
        (["ocular hypertension", "raised iop", "eye pressure"], "NG81_HYPERTENSION"),
    ]
    for keywords, gid in mapping:
        if any(kw in symptoms_lower for kw in keywords):
            return gid
    return None


async def select_guideline_fn(
    symptoms: str, triage: dict, answers: dict, patient_record: dict
) -> str:
    """Select the most appropriate NICE guideline for the patient's symptoms."""
    s = (symptoms or "").lower()

    # Try keyword mapping first
    gid = _guess_guideline(s)
    if gid:
        return gid

    # Fallback: ask LLM to pick from available guidelines
    if settings.OPENAI_API_KEY:
        try:
            data = load_all_guidelines()
            available = ", ".join(sorted(data.keys()))
            prompt = f"""Given a patient with these symptoms: {symptoms}

Available NICE guidelines: {available}

Which single guideline ID is most appropriate? Reply with ONLY the guideline ID."""

            raw = await generate(prompt, max_tokens=20, temperature=0.0)
            candidate = raw.strip().upper().replace(" ", "_")
            if candidate in data:
                return candidate
        except Exception as e:
            logger.warning("LLM guideline selection failed: %s", e)

    return "NG204"  # fallback


# ===================================================================
# 5. Extract variables — LLM + regex helpers
# ===================================================================


async def extract_variables_20b(
    guideline: str, history: list, patient: dict, clarifications: dict
) -> Dict[str, Any]:
    """Extract clinical variables from conversation using LLM + regex helpers.

    Matches the notebook pipeline approach:
    1. Build extraction prompt with patient record and variable descriptions
    2. Call LLM for JSON extraction
    3. Apply regex-based fix_variable_extraction helpers
    """
    # Build scenario from conversation history
    scenario_parts = []
    for msg in history or []:
        if isinstance(msg, dict) and msg.get("role") == "user":
            scenario_parts.append(msg.get("content", ""))
    scenario = " ".join(scenario_parts)

    # Add patient context
    if patient.get("age"):
        scenario += f" {patient['age']} year old"
    if patient.get("gender"):
        scenario += f" {patient['gender']}"
    if patient.get("conditions"):
        scenario += f" with {', '.join(patient['conditions'])}"

    # Get required variables for this guideline
    g_data = get_guideline(guideline)
    if not g_data:
        return {}

    all_vars = list(
        set(
            v
            for spec in g_data["merged_evaluator"].values()
            if isinstance(spec, dict)
            for v in _extract_var_names(spec)
        )
    )[:15]  # Limit to keep prompt manageable

    if not all_vars:
        return {}

    patient_record_section = build_patient_record(scenario)
    var_list = [VAR_DESCRIPTIONS.get(v, f"{v} (extract this value)") for v in all_vars]

    prompt = f"""You are extracting clinical variables from a patient conversation.

PATIENT RECORD:
{patient_record_section}

CLINICAL SCENARIO:
{scenario}

Extract these variables in JSON format:
{chr(10).join(['- ' + v for v in var_list])}

Output ONLY valid JSON with snake_case keys. Use exact key names without descriptions.

JSON:
"""

    try:
        raw = await generate(prompt, max_tokens=300, temperature=0.0)
        extracted = extract_json_from_text(raw)
    except Exception as e:
        logger.warning("LLM extraction failed: %s", e)
        extracted = {}

    # Apply regex-based fixes
    extracted = fix_variable_extraction(extracted, scenario)
    extracted = fix_variable_extraction_v2(extracted, scenario)

    # Merge in clarification answers
    for q, a in (clarifications or {}).items():
        q_lower = q.lower()
        for var_name in all_vars:
            var_words = var_name.lower().replace("_", " ").split()
            if any(w in q_lower for w in var_words if len(w) > 3):
                if var_name not in extracted:
                    extracted[var_name] = a
                break

    return extracted


def _extract_var_names(spec: dict) -> List[str]:
    """Recursively extract variable names from a condition spec."""
    names = []
    if "variable" in spec:
        names.append(spec["variable"])
    if "and" in spec and isinstance(spec["and"], list):
        for sub in spec["and"]:
            if isinstance(sub, dict):
                names.extend(_extract_var_names(sub))
    if "conditions" in spec and isinstance(spec["conditions"], list):
        for sub in spec["conditions"]:
            if isinstance(sub, dict):
                names.extend(_extract_var_names(sub))
    return names


# ===================================================================
# 6. Walk guideline graph — real decision tree traversal
# ===================================================================


async def walk_guideline_graph_fn(
    guideline: str,
    variables: dict,
    current_node: Optional[str],
    pathway: list,
) -> Dict[str, Any]:
    """Walk the NICE guideline decision tree using the real graph engine.

    Returns traversal result with reached actions, path, and missing variables.
    """
    g_data = get_guideline(guideline)
    if not g_data:
        logger.warning("Guideline %s not found, returning terminal", guideline)
        return {
            "current_node": current_node or "start",
            "pathway_walked": list(pathway or []),
            "terminal": True,
        }

    nodes = g_data["guideline"]["nodes"]
    edges = g_data["guideline"]["edges"]
    evaluator = g_data["merged_evaluator"]

    result = traverse_guideline_graph(nodes, edges, evaluator, variables)

    walked = [f"{p[0]}({p[2]})" for p in result["path"]]
    last_node = result["path"][-1][0] if result["path"] else (current_node or "start")

    # Terminal if we reached action nodes and have no missing variables
    terminal = bool(result["reached_actions"]) and not result["missing_variables"]

    return {
        "current_node": last_node,
        "pathway_walked": walked,
        "terminal": terminal,
        "reached_actions": result["reached_actions"],
        "missing_variables": result["missing_variables"],
    }


# ===================================================================
# 7. Format output — template-based (no LLM needed)
# ===================================================================


async def format_output_20b(
    guideline: str,
    triage: dict,
    variables: dict,
    pathway: list,
    patient: dict,
) -> Dict[str, Any]:
    """Format final recommendation using template-based formatting.

    Uses action nodes from guideline graph traversal. Falls back to LLM
    if no actions are available in the traversal result.
    """
    g_data = get_guideline(guideline)

    # Get action nodes from a fresh traversal
    actions = []
    if g_data:
        nodes = g_data["guideline"]["nodes"]
        edges = g_data["guideline"]["edges"]
        evaluator = g_data["merged_evaluator"]
        result = traverse_guideline_graph(nodes, edges, evaluator, variables)
        actions = result["reached_actions"]

    if actions:
        # Build scenario string from patient data
        scenario_parts = []
        if patient.get("age"):
            scenario_parts.append(f"{patient['age']} year old")
        if patient.get("gender"):
            scenario_parts.append(patient["gender"])
        if patient.get("conditions"):
            scenario_parts.append(f"with {', '.join(patient['conditions'])}")
        scenario = " ".join(scenario_parts) if scenario_parts else ""

        recommendation = format_recommendation_template(
            guideline, scenario, actions, variables
        )
        return {
            "final_recommendation": recommendation,
            "citation": guideline,
        }

    # Fallback: LLM-based formatting
    if settings.OPENAI_API_KEY:
        try:
            name = patient.get("first_name", "the patient")
            prompt = f"""Based on NICE guideline {guideline}, provide a concise clinical
recommendation for {name}. Known variables: {json.dumps(variables)}.
Keep it under 3 sentences and professional."""

            rec = await generate(prompt, max_tokens=200, temperature=0.0)
            return {"final_recommendation": rec, "citation": guideline}
        except Exception as e:
            logger.warning("LLM formatting failed: %s", e)

    # Final fallback
    name = patient.get("first_name", "there")
    return {
        "final_recommendation": (
            f"Based on NICE guideline {guideline}, {name}, "
            "please seek medical advice if symptoms worsen or persist."
        ),
        "citation": guideline,
    }


# ===================================================================
# Build all dependencies
# ===================================================================


def build_orchestration_deps() -> OrchestrationDeps:
    """Assemble all orchestration callables."""
    # Pre-load guidelines at startup
    load_all_guidelines()

    return {
        "fetch_patient": fetch_patient,
        "triage_agent": triage_agent,
        "gpt_clarifier": gpt_clarifier,
        "select_guideline": select_guideline_fn,
        "extract_variables_20b": extract_variables_20b,
        "walk_guideline_graph": walk_guideline_graph_fn,
        "format_output_20b": format_output_20b,
    }
