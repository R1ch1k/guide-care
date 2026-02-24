from typing import Any, Dict, Optional

from app.orchestration.utils import log_step


def _config(conversation_id: str) -> dict:
    return {"configurable": {"thread_id": conversation_id}}


async def process_user_turn(
    *,
    graph: Any,
    patient_id: str,
    conversation_id: str,
    user_message: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Process a single user turn through LangGraph.
    Returns one assistant event (clarification question or final answer) or None.
    """
    text = user_message.get("content", "")

    update = {
        "patient_id": patient_id,
        "conversation_id": conversation_id,
        "last_user_message": text,
        "conversation_history": [user_message],  # reducer appends
        "assistant_event": {},  # clear stale event from previous turn
    }

    log_step(conversation_id, "turn_start", user_chars=len(text))

    state = await graph.ainvoke(update, config=_config(conversation_id))

    if not isinstance(state, dict):
        return None

    evt = state.get("assistant_event") or {}
    if evt.get("type") == "clarification_question":
        return {
            "type": "clarification_question",
            "content": evt.get("content", ""),
            "meta": evt.get("meta") or {},
            "selected_guideline": state.get("selected_guideline"),
            "extracted_variables": state.get("extracted_variables") or {},
            "status": "in_progress",
        }

    final = state.get("final_recommendation")
    if final:
        return {
            "type": "final",
            "content": final,
            "meta": {"citation": state.get("citation", "")},
            "final_recommendation": final,
            "selected_guideline": state.get("selected_guideline"),
            "extracted_variables": state.get("extracted_variables") or {},
            "pathway_walked": state.get("pathway_walked") or [],
            "status": "completed",
        }

    return None
