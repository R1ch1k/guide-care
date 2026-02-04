import asyncio
import logging
from typing import Dict, Set
from fastapi import WebSocket
from app.db.session import AsyncSessionLocal
from app.db.models import Conversation
from app.langgraph import call_langgraph
from datetime import datetime
import json
import uuid

logger = logging.getLogger("ws_manager")

# Mapping patient_id -> set of WebSocket
class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, Set[WebSocket]] = {}
        self.locks: Dict[str, asyncio.Lock] = {}

    async def connect(self, patient_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active.setdefault(patient_id, set()).add(websocket)
        self.locks.setdefault(patient_id, asyncio.Lock())
        logger.info("WebSocket connected for patient %s", patient_id)

    async def disconnect(self, patient_id: str, websocket: WebSocket):
        conns = self.active.get(patient_id, set())
        if websocket in conns:
            conns.remove(websocket)
        logger.info("WebSocket disconnected for patient %s", patient_id)

    async def broadcast(self, patient_id: str, message: dict):
        conns = self.active.get(patient_id, set())
        for ws in list(conns):
            try:
                await ws.send_json(message)
            except Exception:
                logger.exception("Failed to send message to websocket")

    async def handle_incoming_message(self, patient_id: str, message: dict):
        """
        Called when a websocket client sends a message.
        Persist to DB (conversations.messages), create conv if needed,
        call LangGraph hook (async), and broadcast to other clients.
        """
        # enrich message with timestamp and id
        message_record = {
            "id": str(uuid.uuid4()),
            "role": message.get("role", "user"),
            "content": message.get("content"),
            "meta": message.get("meta", {}),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        # Persist to DB
        async with AsyncSessionLocal() as db:
            # Find existing in-progress conversation for patient, or create one
            q = await db.execute(
                "SELECT id, messages, status FROM conversations WHERE patient_id = :pid ORDER BY updated_at DESC LIMIT 1",
                {"pid": patient_id}
            )
            row = q.fetchone()
            conv = None
            if row:
                # Use SQLAlchemy ORM for updates
                conv = await db.get(Conversation, row[0])
            if not conv:
                conv = Conversation(patient_id=patient_id, messages=[message_record])
                db.add(conv)
                await db.commit()
                await db.refresh(conv)
            else:
                msgs = conv.messages or []
                msgs.append(message_record)
                conv.messages = msgs
                conv.updated_at = datetime.utcnow()
                await db.commit()
                await db.refresh(conv)

            # Fire LangGraph asynchronously (do not block)
            asyncio.create_task(call_langgraph(patient_id, conv.id, message_record, conv.selected_guideline))

            # Broadcast the saved message to all connected clients for this patient
            await self.broadcast(patient_id, {"type": "message", "message": message_record, "conversation_id": str(conv.id)})

manager = ConnectionManager()
