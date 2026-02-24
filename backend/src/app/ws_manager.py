import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Set
from uuid import UUID, uuid4

from fastapi import WebSocket
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db.models import Conversation
from app.db.session import AsyncSessionLocal
from app.orchestration.runner import process_user_turn

logger = logging.getLogger("ws_manager")


class ConnectionManager:
    def __init__(self) -> None:
        self.active: Dict[str, Set[WebSocket]] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
        self._orch_graph = None

    def set_orchestrator(self, graph) -> None:
        self._orch_graph = graph
        logger.info("Orchestrator connected to ws_manager")

    async def connect(self, patient_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active.setdefault(patient_id, set()).add(websocket)
        self.locks.setdefault(patient_id, asyncio.Lock())
        logger.info("WebSocket connected for patient %s", patient_id)

    async def disconnect(self, patient_id: str, websocket: WebSocket) -> None:
        conns = self.active.get(patient_id)
        if not conns:
            return
        conns.discard(websocket)
        if not conns:
            self.active.pop(patient_id, None)
            self.locks.pop(patient_id, None)
        logger.info("WebSocket disconnected for patient %s", patient_id)

    async def broadcast(self, patient_id: str, message: dict) -> None:
        conns = self.active.get(patient_id, set())
        dead: Set[WebSocket] = set()
        for ws in list(conns):
            try:
                await ws.send_json(message)
            except Exception:
                dead.add(ws)
                logger.exception("Failed to send websocket message (patient=%s)", patient_id)

        for ws in dead:
            conns.discard(ws)
        if not conns and patient_id in self.active:
            self.active.pop(patient_id, None)
            self.locks.pop(patient_id, None)

    async def _get_or_create_conversation(self, db, pid: UUID) -> Conversation:
        result = await db.execute(
            select(Conversation)
            .where(Conversation.patient_id == pid)
            .where(Conversation.status == "in_progress")
            .order_by(Conversation.updated_at.desc())
            .limit(1)
        )
        conv = result.scalars().first()
        if conv:
            return conv

        conv = Conversation(patient_id=pid, messages=[], status="in_progress")
        db.add(conv)
        await db.commit()
        await db.refresh(conv)
        return conv

    async def handle_incoming_message(self, patient_id: str, message: dict) -> None:
        if self._orch_graph is None:
            await self.broadcast(patient_id, {"type": "error", "detail": "Orchestrator not initialized"})
            return

        try:
            pid = UUID(patient_id)
        except Exception:
            await self.broadcast(patient_id, {"type": "error", "detail": "Invalid patient_id"})
            return

        lock = self.locks.setdefault(patient_id, asyncio.Lock())

        user_record = {
            "id": str(uuid4()),
            "role": message.get("role", "user"),
            "content": message.get("content", ""),
            "meta": message.get("meta") or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Persist user message
        async with lock:
            async with AsyncSessionLocal() as db:
                try:
                    conv = await self._get_or_create_conversation(db, pid)

                    msgs = conv.messages or []
                    msgs.append(user_record)
                    conv.messages = msgs
                    conv.updated_at = datetime.now(timezone.utc)

                    await db.commit()
                    await db.refresh(conv)
                except IntegrityError:
                    await db.rollback()
                    logger.exception("Integrity error saving user message")
                    await self.broadcast(patient_id, {"type": "error", "detail": "Patient not found"})
                    return
                except Exception:
                    await db.rollback()
                    logger.exception("DB error saving user message")
                    await self.broadcast(patient_id, {"type": "error", "detail": "Failed to save message"})
                    return

        # Broadcast user message
        await self.broadcast(
            patient_id,
            {"type": "message", "message": user_record, "conversation_id": str(conv.id)},
        )

        # Run orchestration for this turn (no DB lock)
        try:
            event = await process_user_turn(
                graph=self._orch_graph,
                patient_id=patient_id,
                conversation_id=str(conv.id),
                user_message=user_record,
            )
        except Exception:
            logger.exception("Orchestration failed")
            await self.broadcast(patient_id, {"type": "error", "detail": "Orchestration failed"})
            return

        if not event:
            return

        assistant_msg = {
            "id": str(uuid4()),
            "role": "assistant",
            "content": event.get("content", ""),
            "meta": event.get("meta") or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Persist assistant output + structured fields
        async with lock:
            async with AsyncSessionLocal() as db:
                try:
                    conv2 = await db.get(Conversation, UUID(str(conv.id)))
                    if conv2:
                        msgs = conv2.messages or []
                        msgs.append(assistant_msg)
                        conv2.messages = msgs

                        if event.get("final_recommendation"):
                            conv2.final_recommendation = event["final_recommendation"]
                        if event.get("selected_guideline"):
                            conv2.selected_guideline = event["selected_guideline"]
                        if event.get("extracted_variables") is not None:
                            conv2.extracted_variables = event["extracted_variables"]
                        if event.get("status"):
                            conv2.status = event["status"]

                        conv2.updated_at = datetime.now(timezone.utc)
                        await db.commit()
                except Exception:
                    await db.rollback()
                    logger.exception("Failed to persist assistant event")

        await self.broadcast(
            patient_id,
            {
                "type": event.get("type", "assistant_event"),
                "message": assistant_msg,
                "conversation_id": str(conv.id),
                "payload": event,
            },
        )


manager = ConnectionManager()
