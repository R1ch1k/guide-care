import os
import httpx
import logging
from app.core.config import settings

logger = logging.getLogger("langgraph")

async def call_langgraph(patient_id: str, conversation_id: str, message: dict, guideline: str = None):
    """
    Placeholder for LangGraph integration.
    Sends a POST request to configured LANGGRAPH_API_URL with authentication header.
    """
    if not settings.LANGGRAPH_API_URL or not settings.LANGGRAPH_WORKFLOW_ID:
        logger.debug("LangGraph not configured; skipping call")
        return None

    url = settings.LANGGRAPH_API_URL.rstrip("/") + f"/{settings.LANGGRAPH_WORKFLOW_ID}/invoke"
    headers = {}
    if settings.LANGGRAPH_API_KEY:
        headers["Authorization"] = f"Bearer {settings.LANGGRAPH_API_KEY}"
    payload = {
        "patient_id": patient_id,
        "conversation_id": str(conversation_id),
        "message": message,
        "selected_guideline": guideline,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            logger.info("LangGraph invoked: %s", resp.status_code)
            return resp.json()
        except Exception as e:
            logger.exception("LangGraph invocation failed: %s", e)
            return None
