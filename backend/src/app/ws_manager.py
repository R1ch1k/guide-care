import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Set, Union
from uuid import UUID, uuid4

from fastapi import WebSocket
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db.models import Conversation
from app.db.session import AsyncSessionLocal
from app.langgraph import call_langgraph

logger = logging.getLogger("ws_manager")


class ConnectionManager:
    """
    patient_id (str UUID) -> set[WebSocket]
    patient_id (str UUID) -> asyncio.Lock
    """
    def __init__(self) -> None:
        self.active: Dict[str, Set[WebSocket]] = {}
        self.locks: Dict[str, asyncio.Lock] = {}

    def _key(self, patient_id: Union[str, UUID]) -> str:
        return str(patient_id)

    async def connect(self, patient_id: Union[str, UUID], websocket: WebSocket) -> None:
        key = self._key(patient_id)
        await websocket.accept()
        self.active.setdefault(key, set()).add(websocket)
        self.locks.setdefault(key, asyncio.Lock())
        logger.info("WebSocket connected for patient %s", key)

    async def disconnect(self, patient_id: Union[str, UUID], websocket: WebSocket) -> None:
        key = self._key(patient_id)
        conns = self.active.get(key)
        if not conns:
            return
        conns.discard(websocket)
        if not conns:
            # cleanup
            self.active.pop(key, None)
            self.locks.pop(key, None)
        logger.info("WebSocket disconnected for patient %s", key)

    async def broadcast(self, patient_id: Union[str, UUID], message: dict) -> None:
        key = self._key(patient_id)
        conns = self.active.get(key, set())
        dead: Set[WebSocket] = set()

        for ws in list(conns):
            try:
                await ws.send_json(message)
            except Exception:
                dead.add(ws)
                logger.exception("Failed to send message to websocket (patient=%s)", key)

        # prune dead sockets
        for ws in dead:
            conns.discard(ws)
        if not conns and key in self.active:
            self.active.pop(key, None)
            self.locks.pop(key, None)

    async def handle_incoming_message(self, patient_id: Union[str, UUID], message: dict) -> None:
        """
        Persist message to latest in-progress conversation (or create one),
        trigger LangGraph asynchronously, broadcast saved message.
        """
        # normalize patient_id to UUID (DB uses UUID)
        try:
            pid = patient_id if isinstance(patient_id, UUID) else UUID(str(patient_id))
        except Exception:
            await self.broadcast(patient_id, {"type": "error", "detail": "Invalid patient_id"})
            return

        patient_key = str(pid)
        lock = self.locks.setdefault(patient_key, asyncio.Lock())

        # enrich message with timestamp and id
        message_record = {
            "id": str(uuid4()),
            "role": message.get("role", "user"),
            "content": message.get("content"),
            "meta": message.get("meta") or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        async with lock:
            async with AsyncSessionLocal() as db:
                try:
                    # Find most recent in-progress conversation
                    result = await db.execute(
                        select(Conversation)
                        .where(Conversation.patient_id == pid)
                        .where(Conversation.status == "in_progress")
                        .order_by(Conversation.updated_at.desc())
                        .limit(1)
                    )
                    conv = result.scalars().first()

                    if not conv:
                        conv = Conversation(
                            patient_id=pid,
                            messages=[message_record],
                            status="in_progress",
                        )
                        db.add(conv)
                        await db.commit()
                        await db.refresh(conv)
                    else:
                        msgs = conv.messages or []
                        msgs.append(message_record)
                        conv.messages = msgs
                        # keep updated_at moving even if server-side onupdate is finicky
                        conv.updated_at = datetime.now(timezone.utc)
                        await db.commit()
                        await db.refresh(conv)

                except IntegrityError:
                    await db.rollback()
                    # likely FK violation if patient doesn't exist
                    logger.exception("DB integrity error while saving message (patient=%s)", patient_key)
                    await self.broadcast(patient_key, {"type": "error", "detail": "Patient not found"})
                    return
                except Exception:
                    await db.rollback()
                    logger.exception("DB error while saving message (patient=%s)", patient_key)
                    await self.broadcast(patient_key, {"type": "error", "detail": "Failed to save message"})
                    return

        # Fire LangGraph asynchronously (do not block websocket)
        asyncio.create_task(
            call_langgraph(
                patient_key,
                str(conv.id),
                message_record,
                conv.selected_guideline,
            )
        )

        # Broadcast saved message to all connected clients for this patient
        await self.broadcast(
            patient_key,
            {
                "type": "message",
                "message": message_record,
                "conversation_id": str(conv.id),
            },
        )


manager = ConnectionManager()
