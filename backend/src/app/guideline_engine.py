"""
NICE Guideline Graph Traversal Engine

Loads guideline decision trees and evaluator logic from JSON files,
traverses them based on patient variables, and provides helper functions
for variable extraction, clarification question parsing, and
recommendation formatting.

Ported from the gpt_oss_20b_final_test_suite.ipynb notebook.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data directory (relative to this file)
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_GUIDELINES_DIR = _DATA_DIR / "guidelines"
_EVALUATORS_DIR = _DATA_DIR / "evaluators"

# Filename -> guideline ID mapping
_FILENAME_TO_ID = {
    "ng84": "NG84",
    "ng91": "NG91",
    "ng112": "NG112",
    "ng133": "NG133",
    "ng136": "NG136",
    "ng184": "NG184",
    "ng222": "NG222",
    "ng232": "NG232",
    "ng81_chronic_glaucoma": "NG81_GLAUCOMA",
    "ng81_ocular_hypertension": "NG81_HYPERTENSION",
}

# Reverse mapping: guideline ID -> filename stem
_ID_TO_FILENAME = {v: k for k, v in _FILENAME_TO_ID.items()}


# ===================================================================
# Guideline data loading
# ===================================================================

_guideline_cache: Dict[str, Dict[str, Any]] = {}


def load_all_guidelines() -> Dict[str, Dict[str, Any]]:
    """Load all guideline + evaluator JSON files into memory.

    Returns a dict keyed by guideline ID (e.g. 'NG84') with values:
        {
            "guideline": { "nodes": [...], "edges": [...] },
            "evaluator": { "n1": {...}, ... },
            "merged_evaluator": { ... }  # combined evaluator entries
        }
    """
    global _guideline_cache
    if _guideline_cache:
        return _guideline_cache

    data: Dict[str, Dict[str, Any]] = {}

    for stem, gid in _FILENAME_TO_ID.items():
        guideline_path = _GUIDELINES_DIR / f"{stem}.json"
        evaluator_path = _EVALUATORS_DIR / f"{stem}_eval.json"

        if not guideline_path.exists():
            logger.warning("Guideline file not found: %s", guideline_path)
            continue

        with open(guideline_path) as f:
            guideline = json.load(f)

        evaluator = {}
        if evaluator_path.exists():
            with open(evaluator_path) as f:
                evaluator = json.load(f)

        # Build merged evaluator (flatten if nested under keys)
        merged = {}
        if isinstance(evaluator, dict):
            for key, val in evaluator.items():
                if isinstance(val, dict) and "variable" not in val and "type" not in val:
                    # Nested structure — merge inner keys
                    merged.update(val)
                else:
                    merged[key] = val

        data[gid] = {
            "guideline": guideline,
            "evaluator": evaluator,
            "merged_evaluator": merged if merged else evaluator,
        }

    node_count = sum(
        len(d["guideline"].get("nodes", [])) for d in data.values()
    )
    logger.info(
        "Loaded %d guidelines with %d total nodes", len(data), node_count
    )

    _guideline_cache = data
    return data


def get_guideline(guideline_id: str) -> Optional[Dict[str, Any]]:
    """Get a single guideline's data by ID. Loads all if not cached."""
    data = load_all_guidelines()
    return data.get(guideline_id)


# ===================================================================
# Blood pressure parsing
# ===================================================================


def parse_bp(bp_string) -> Optional[Tuple[int, int]]:
    """Parse a blood pressure string like '180/120' into (systolic, diastolic)."""
    if bp_string is None:
        return None
    if isinstance(bp_string, (list, tuple)) and len(bp_string) == 2:
        try:
            return (int(bp_string[0]), int(bp_string[1]))
        except (ValueError, TypeError):
            return None
    if not isinstance(bp_string, str):
        return None
    match = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", str(bp_string))
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


# ===================================================================
# Condition evaluation
# ===================================================================


def _compare(value, threshold, op):
    """Apply a comparison operator between value and threshold."""
    ops = {
        ">=": lambda v, t: v >= t,
        "<=": lambda v, t: v <= t,
        ">": lambda v, t: v > t,
        "<": lambda v, t: v < t,
        "==": lambda v, t: v == t,
        "!=": lambda v, t: v != t,
    }
    fn = ops.get(op)
    return fn(value, threshold) if fn else None


def evaluate_single_condition(condition_spec, variables):
    """Evaluate a single condition specification against patient variables.

    Returns True/False for boolean conditions, None if a required variable
    is missing, or a string (edge label) for treatment_type conditions.
    """
    if condition_spec is None:
        return None

    ctype = condition_spec.get("type")

    # Shorthand nested 'and' key
    if "and" in condition_spec and ctype is None and "variable" not in condition_spec:
        sub_conditions = condition_spec["and"]
        results = [evaluate_single_condition(c, variables) for c in sub_conditions]
        if any(r is None for r in results):
            return None
        return all(results)

    # Simple variable check (no 'type' field)
    if ctype is None and "variable" in condition_spec:
        var_name = condition_spec["variable"]
        if var_name not in variables:
            return None
        val = variables[var_name]
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            lower = val.lower()
            if lower in ("true", "yes", "1"):
                return True
            if lower in ("false", "no", "0", ""):
                return False
            return True
        if isinstance(val, (int, float)):
            return bool(val)
        return bool(val)

    # Numeric compare (also used for age_compare)
    if ctype in ("numeric_compare", "age_compare"):
        var_name = condition_spec["variable"]
        if var_name not in variables:
            return None
        val = variables[var_name]
        if val is None:
            return None
        try:
            val = float(val)
        except (ValueError, TypeError):
            return None
        threshold = float(condition_spec["threshold"])
        return _compare(val, threshold, condition_spec["op"])

    # Blood pressure compare
    if ctype == "bp_compare":
        var_name = condition_spec["variable"]
        if var_name not in variables:
            return None
        bp_val = parse_bp(variables[var_name])
        bp_thresh = parse_bp(condition_spec["threshold"])
        if bp_val is None or bp_thresh is None:
            return None
        op = condition_spec["op"]
        sys_v, dia_v = bp_val
        sys_t, dia_t = bp_thresh
        if op == ">=":
            return sys_v >= sys_t or dia_v >= dia_t
        elif op == ">":
            return sys_v > sys_t or dia_v > dia_t
        elif op == "<=":
            return sys_v <= sys_t and dia_v <= dia_t
        elif op == "<":
            return sys_v < sys_t and dia_v < dia_t
        elif op == "==":
            return sys_v == sys_t and dia_v == dia_t
        return None

    # Blood pressure range
    if ctype == "bp_range":
        var_name = condition_spec["variable"]
        if var_name not in variables:
            return None
        bp_val = parse_bp(variables[var_name])
        if bp_val is None:
            return None
        sys_v, dia_v = bp_val
        sys_min = condition_spec.get("systolic_min", 0)
        sys_max = condition_spec.get("systolic_max", 999)
        dia_min = condition_spec.get("diastolic_min", 0)
        dia_max = condition_spec.get("diastolic_max", 999)
        return (sys_min <= sys_v <= sys_max) and (dia_min <= dia_v <= dia_max)

    # AND logic
    if ctype == "and":
        sub_conditions = condition_spec.get("conditions", [])
        results = [evaluate_single_condition(c, variables) for c in sub_conditions]
        if any(r is None for r in results):
            return None
        return all(results)

    # OR logic
    if ctype == "or":
        sub_conditions = condition_spec.get("conditions", [])
        results = [evaluate_single_condition(c, variables) for c in sub_conditions]
        if any(r is True for r in results):
            return True
        if all(r is None for r in results):
            return None
        if any(r is None for r in results):
            return None
        return False

    # Treatment type map (NG222)
    if ctype == "treatment_type":
        var_name = condition_spec["variable"]
        if var_name not in variables:
            return None
        val = variables[var_name]
        if val is None:
            return None
        treatment_map = condition_spec.get("map", {})
        if val in treatment_map:
            return treatment_map[val]
        val_lower = str(val).lower().strip()
        for key, label in treatment_map.items():
            if key.lower().strip() == val_lower:
                return label
        return None

    return None


def evaluate_condition(node_id: str, evaluator: dict, variables: dict):
    """Look up a node in the evaluator and evaluate its condition."""
    if not evaluator or node_id not in evaluator:
        return None
    return evaluate_single_condition(evaluator[node_id], variables)


# ===================================================================
# Graph traversal
# ===================================================================


def traverse_guideline_graph(
    nodes: list, edges: list, evaluator: dict, variables: dict
) -> Dict[str, Any]:
    """Walk a guideline decision tree and determine which actions are reached.

    Returns:
        {
            "reached_actions": [str, ...],
            "path": [(node_id, node_text, decision), ...],
            "missing_variables": [str, ...]
        }
    """
    if not nodes:
        return {"reached_actions": [], "path": [], "missing_variables": []}

    node_map = {n["id"]: n for n in nodes}
    edges_from: Dict[str, List[Tuple[str, str]]] = {}
    incoming = set()
    for e in edges:
        src = e["from"]
        tgt = e["to"]
        label = e.get("label", "")
        edges_from.setdefault(src, []).append((tgt, label))
        incoming.add(tgt)

    all_node_ids = [n["id"] for n in nodes]
    roots = [nid for nid in all_node_ids if nid not in incoming]
    if not roots:
        roots = [all_node_ids[0]]

    reached_actions: List[str] = []
    path: List[Tuple[str, str, str]] = []
    missing_variables: List[str] = []
    visited = set()
    max_steps = len(nodes) * 3

    queue = list(roots)
    step_count = 0

    while queue and step_count < max_steps:
        step_count += 1
        current_id = queue.pop(0)

        if current_id in visited:
            continue
        visited.add(current_id)

        if current_id not in node_map:
            continue

        node = node_map[current_id]
        node_type = node.get("type", "")
        node_text = node.get("text", "")

        if node_type == "action":
            reached_actions.append(node_text)
            path.append((current_id, node_text, "action"))
            for tgt, label in edges_from.get(current_id, []):
                if label == "next":
                    queue.append(tgt)

        elif node_type == "condition":
            result = evaluate_condition(current_id, evaluator, variables)

            if result is None:
                path.append((current_id, node_text, "missing_variable"))
                _collect_missing_vars(current_id, evaluator, variables, missing_variables)

            elif isinstance(result, str):
                path.append((current_id, node_text, f"treatment_type:{result}"))
                matched = False
                for tgt, label in edges_from.get(current_id, []):
                    if label == result:
                        queue.append(tgt)
                        matched = True
                if not matched:
                    path.append((current_id, node_text, f"no_matching_edge:{result}"))

            elif result is True:
                path.append((current_id, node_text, "yes"))
                for tgt, label in edges_from.get(current_id, []):
                    if label in ("yes", "next"):
                        queue.append(tgt)

            elif result is False:
                path.append((current_id, node_text, "no"))
                for tgt, label in edges_from.get(current_id, []):
                    if label == "no":
                        queue.append(tgt)
        else:
            path.append((current_id, node_text, f"unknown_type:{node_type}"))

    # Deduplicate missing variables preserving order
    seen = set()
    unique_missing = []
    for v in missing_variables:
        if v not in seen:
            seen.add(v)
            unique_missing.append(v)

    return {
        "reached_actions": reached_actions,
        "path": path,
        "missing_variables": unique_missing,
    }


def _collect_missing_vars(node_id, evaluator, variables, missing_list):
    """Extract variable names needed by a node that are not in variables."""
    if not evaluator or node_id not in evaluator:
        return
    _collect_missing_from_spec(evaluator[node_id], variables, missing_list)


def _collect_missing_from_spec(spec, variables, missing_list):
    """Recursively collect missing variable names from a condition spec."""
    if spec is None:
        return

    if "variable" in spec:
        var_name = spec["variable"]
        if var_name not in variables:
            missing_list.append(var_name)

    if "and" in spec and isinstance(spec["and"], list):
        for sub in spec["and"]:
            _collect_missing_from_spec(sub, variables, missing_list)

    ctype = spec.get("type")
    if ctype in ("and", "or") and "conditions" in spec:
        for sub in spec["conditions"]:
            _collect_missing_from_spec(sub, variables, missing_list)

    if ctype == "treatment_type" and "variable" in spec:
        var_name = spec["variable"]
        if var_name not in variables:
            missing_list.append(var_name)


def get_all_variables_from_evaluator(evaluator: dict) -> List[str]:
    """Extract all variable names referenced anywhere in an evaluator."""
    if not evaluator:
        return []
    all_vars: set = set()
    for node_id, spec in evaluator.items():
        _collect_vars_from_spec(spec, all_vars)
    return sorted(all_vars)


def _collect_vars_from_spec(spec, var_set):
    """Recursively collect all variable names from a condition spec."""
    if spec is None or not isinstance(spec, dict):
        return
    if "variable" in spec:
        var_set.add(spec["variable"])
    if "and" in spec and isinstance(spec["and"], list):
        for sub in spec["and"]:
            _collect_vars_from_spec(sub, var_set)
    if "conditions" in spec and isinstance(spec["conditions"], list):
        for sub in spec["conditions"]:
            _collect_vars_from_spec(sub, var_set)


def get_missing_variables_for_next_step(
    nodes: list, edges: list, evaluator: dict, known_vars: dict
) -> List[str]:
    """Traverse the graph and return variable names needed to advance."""
    result = traverse_guideline_graph(nodes, edges, evaluator, known_vars)
    return result["missing_variables"]


# ===================================================================
# JSON extraction from LLM output
# ===================================================================


def extract_json_from_text(text: str) -> dict:
    """Extract JSON from model output that may contain extra text."""
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    result = {}
    pattern = r'["\'](\w+)["\']\s*:\s*([^,}\n]+)'
    matches = re.findall(pattern, text)
    for key, value in matches:
        value = value.strip().strip("\"'")
        if value.lower() == "true":
            result[key] = True
        elif value.lower() == "false":
            result[key] = False
        elif value.lower() in ("null", "none"):
            result[key] = None
        elif value.isdigit():
            result[key] = int(value)
        else:
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value

    return result if result else {}


# ===================================================================
# Variable extraction helpers (regex-based post-processing)
# ===================================================================


def fix_variable_extraction(extracted: dict, scenario_text: str) -> dict:
    """Regex-based variable extraction from scenario text to fill gaps."""
    fixed = extracted.copy()
    text_lower = scenario_text.lower()

    # Age
    if "age" not in fixed or fixed["age"] is None:
        for pattern in [
            r"(\d+)\s*(?:year|yr|y\.?o\.?)",
            r"age[:\s]+(\d+)",
            r"(\d+)\s*(?:month|mo)\s+old",
        ]:
            match = re.search(pattern, text_lower)
            if match:
                age_val = int(match.group(1))
                if "month" in match.group(0) or "mo" in match.group(0):
                    fixed["age"] = age_val / 12.0
                else:
                    fixed["age"] = age_val
                break

    # Gender
    if "gender" not in fixed or fixed["gender"] is None:
        if re.search(r"\b(male|man|boy|father|husband|mr\.?)\b", text_lower):
            fixed["gender"] = "male"
        elif re.search(
            r"\b(female|woman|girl|mother|wife|mrs\.?|ms\.?|pregnant)\b", text_lower
        ):
            fixed["gender"] = "female"

    # GCS Score
    if "gcs_score" not in fixed or fixed["gcs_score"] is None:
        gcs_match = re.search(r"gcs[:\s]+(\d+)", text_lower)
        if gcs_match:
            fixed["gcs_score"] = int(gcs_match.group(1))

    # Blood Pressure
    if "clinic_bp" not in fixed or fixed["clinic_bp"] is None:
        for pattern in [
            r"(\d{2,3})\s*/\s*(\d{2,3})",
            r"bp[:\s]+(\d{2,3})\s*/\s*(\d{2,3})",
            r"blood pressure[:\s]+(\d{2,3})\s*/\s*(\d{2,3})",
        ]:
            match = re.search(pattern, text_lower)
            if match:
                fixed["clinic_bp"] = f"{match.group(1)}/{match.group(2)}"
                break

    if "bp" not in fixed or fixed["bp"] is None:
        if "clinic_bp" in fixed and fixed["clinic_bp"]:
            fixed["bp"] = fixed["clinic_bp"]

    # Gestational Age
    if "gestational_age" not in fixed or fixed["gestational_age"] is None:
        gest_match = re.search(r"(\d+)\s+weeks?\s+(?:pregnant|gestation)", text_lower)
        if gest_match:
            fixed["gestational_age"] = int(gest_match.group(1))

    # Fever
    if "fever" not in fixed or fixed["fever"] is None:
        if re.search(r"fever|febrile|temperature\s+\d", text_lower):
            temp_match = re.search(r"(\d{2}\.?\d?)\s*[°c]", text_lower)
            if temp_match:
                fixed["fever"] = float(temp_match.group(1))
            else:
                fixed["fever"] = True
        elif re.search(r"no\s+fever|afebrile|apyrexial", text_lower):
            fixed["fever"] = False

    # Head injury
    if "head_injury_present" not in fixed or fixed["head_injury_present"] is None:
        if re.search(r"head\s+injur|hit\s+head|head\s+trauma|struck\s+head", text_lower):
            fixed["head_injury_present"] = True

    # Recurrent UTI
    if "recurrent_uti" not in fixed or fixed["recurrent_uti"] is None:
        if re.search(r"recurrent\s+uti|multiple\s+utis?", text_lower):
            fixed["recurrent_uti"] = True

    # Diabetes
    if "diabetes" not in fixed or fixed["diabetes"] is None:
        if re.search(r"\bdiabetes|diabetic|type [12] diabetes", text_lower):
            fixed["diabetes"] = True
        elif re.search(r"no\s+diabetes|non-diabetic", text_lower):
            fixed["diabetes"] = False

    return fixed


def fix_variable_extraction_v2(extracted: dict, scenario_text: str) -> dict:
    """Additional edge case handling — call AFTER fix_variable_extraction()."""
    fixed = extracted.copy()
    text_lower = scenario_text.lower()

    negation_patterns = [
        (r"no\s+vomiting", "vomiting", False),
        (r"no\s+fever", "fever", False),
        (r"no\s+headache", "headache", False),
        (r"denies\s+chest\s+pain", "emergency_signs", False),
        (r"no\s+loss\s+of\s+consciousness", "loc", False),
        (r"no\s+visual\s+disturbance", "emergency_signs", False),
    ]

    for pattern, var_name, correct_value in negation_patterns:
        if re.search(pattern, text_lower):
            if var_name in fixed:
                fixed[var_name] = correct_value

    # NG222 — Better remission detection
    if "remission" not in fixed or fixed.get("remission") is None:
        if re.search(r"well-controlled|stable mood|asymptomatic", text_lower):
            fixed["remission"] = "full"

    # NG91 — High risk for very young children
    if "high_risk" not in fixed or fixed.get("high_risk") is None:
        age = fixed.get("age")
        if age is not None and age < 1:
            fixed["high_risk"] = True

    # NG84 — Infer Centor score from components
    if "centor_score" not in fixed or fixed.get("centor_score") is None:
        score = 0
        if fixed.get("fever"):
            score += 1
        if fixed.get("tonsillar_exudate") or fixed.get("purulent_tonsils"):
            score += 1
        if fixed.get("tender_lymph_nodes"):
            score += 1
        if fixed.get("cough") is False:
            score += 1
        if score > 0:
            fixed["centor_score"] = score

    return fixed


# ===================================================================
# LLM output helpers
# ===================================================================


def extract_best_question(raw_response: str) -> str:
    """Multi-strategy question extraction from LLM output."""
    text = raw_response.strip()

    # Strategy 1: Quoted question
    quoted = re.findall(r'"([^"]*\?)"', text)
    for q in quoted:
        if len(q) > 15 and q.lower() != "we?":
            return q

    # Strategy 2: "Question:" marker
    if "Question:" in text:
        parts = text.split("Question:")
        candidate = parts[-1].strip().split("\n")[0].strip()
        if "?" in candidate:
            candidate = candidate[: candidate.rindex("?") + 1]
            if len(candidate) > 15:
                return candidate

    # Strategy 3: Real question sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    questions = [
        s.strip()
        for s in sentences
        if "?" in s
        and len(s.strip()) > 15
        and not s.strip().lower().startswith("we?")
        and not s.strip().lower().startswith("we need")
    ]
    if questions:
        return max(questions, key=len)

    # Strategy 4: rindex to find last ?
    if "?" in text:
        last_q = text[: text.rindex("?") + 1]
        for marker in [". ", "\n", ": "]:
            idx = last_q.rfind(marker)
            if idx > 0:
                candidate = last_q[idx + len(marker) :].strip()
                if len(candidate) > 15:
                    return candidate
        if len(last_q) > 15:
            return last_q

    # Fallback
    for s in sentences:
        s = s.strip()
        if len(s) > 20 and not s.lower().startswith("we "):
            return s

    return text[:100] if text else "Could you provide more information?"


def build_patient_record(scenario: str) -> str:
    """Pre-parse scenario into structured PATIENT RECORD for better extraction."""
    scenario_lower = scenario.lower()
    patient_lines = []

    age_match = re.search(
        r"(\d{1,3})\s*(?:year|yr|yo)(?:\s+old)?", scenario_lower
    )
    if not age_match:
        age_match = re.search(r"(?:age|aged)[:\s]+(\d{1,3})", scenario_lower)
    if age_match:
        patient_lines.append(f"Age: {age_match.group(1)} years")

    if re.search(r"\b(male|man|boy)\b", scenario_lower) and not re.search(
        r"\b(female|woman|girl)\b", scenario_lower
    ):
        patient_lines.append("Gender: Male")
    elif re.search(r"\b(female|woman|girl|pregnant|pregnancy)\b", scenario_lower):
        patient_lines.append("Gender: Female")

    conditions = []
    if any(
        kw in scenario_lower
        for kw in ["diabetes", "diabetic", "type 2 dm", "t2dm"]
    ):
        conditions.append("diabetes")
    if any(
        kw in scenario_lower
        for kw in ["hypertension", "high blood pressure", "htn"]
    ):
        conditions.append("hypertension")
    if any(
        kw in scenario_lower
        for kw in ["ckd", "chronic kidney", "renal impairment"]
    ):
        conditions.append("chronic kidney disease")
    if conditions:
        patient_lines.append(f"Medical History: {', '.join(conditions)}")

    return "\n".join(patient_lines) if patient_lines else "No documented medical history"


# Variable descriptions for better extraction prompts
VAR_DESCRIPTIONS = {
    "age": "age (patient's age in years)",
    "mechanism": "mechanism (how injury occurred: fall, assault, RTC, etc.)",
    "vomiting_count": "vomiting_count (number of vomiting episodes as integer)",
    "gcs_score": "gcs_score (Glasgow Coma Scale score 3-15)",
    "emergency_signs": "emergency_signs (true if life-threatening symptoms)",
    "loss_of_consciousness": "loss_of_consciousness (true/false)",
    "fever": "fever (true if fever present, or temperature value)",
    "duration": "duration (duration in days as integer)",
    "clinic_bp": "clinic_bp (blood pressure reading as string e.g. '140/90')",
    "gestational_age": "gestational_age (weeks of pregnancy as integer)",
    "gender": "gender (male/female)",
    "recurrent_uti": "recurrent_uti (true/false)",
    "head_injury_present": "head_injury_present (true/false)",
    "bite_type": "bite_type (type of animal: cat, dog, human, etc.)",
    "broken_skin": "broken_skin (true/false - whether skin is broken)",
    "high_risk_area": "high_risk_area (true/false - hand, face, genitals, etc.)",
    "treatment_completed": "treatment_completed (true/false)",
    "acute_treatment": "acute_treatment (type of treatment received)",
    "ear_pain": "ear_pain (true/false)",
    "newly_diagnosed": "newly_diagnosed (true/false)",
    "iop": "iop (intraocular pressure as integer)",
    "iop_level": "iop_level (intraocular pressure as integer)",
    "family_history_glaucoma": "family_history_glaucoma (true/false)",
}


def format_recommendation_template(
    guideline_id: str,
    scenario: str,
    actions: List[str],
    known_vars: dict,
) -> str:
    """Template-based formatting of action nodes into recommendation text.

    No LLM call needed — formats directly from action node text.
    """
    # Deduplicate actions preserving order
    seen: set = set()
    unique_actions = []
    for a in actions:
        a_lower = a.strip().lower()
        if a_lower not in seen:
            seen.add(a_lower)
            unique_actions.append(a.strip())

    # Build recommendation text
    verb_starts = {
        "do", "refer", "give", "offer", "advise", "note", "consider",
        "prescribe", "seek", "review", "continue", "step", "annual",
        "measure", "perform", "diagnose",
    }

    if len(unique_actions) == 1:
        actions_text = unique_actions[0]
    elif len(unique_actions) == 2:
        second = unique_actions[1]
        if (
            second[0].isupper()
            and second.split()[0].lower() not in verb_starts
        ):
            second = second[0].lower() + second[1:]
        actions_text = f"{unique_actions[0]}. Additionally, {second}"
    else:
        actions_text = " ".join(
            f"({i}) {a}" for i, a in enumerate(unique_actions, 1)
        )

    recommendation = f"Based on NICE {guideline_id}, {actions_text}"

    # Append patient context
    age_match = re.search(r"(\d{1,3})\s*(?:year|yr|yo)", scenario.lower())
    if age_match:
        age = age_match.group(1)
        if age not in recommendation:
            recommendation += f" (Patient: {age} years old)"

    if recommendation and recommendation[-1] not in ".!":
        recommendation += "."

    return recommendation
