import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.session import async_engine, get_async_session, init_db
from app.api import patients, conversations
from app.ws_manager import manager
from app.seed import seed_if_empty

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("guidecare-backend")

app = FastAPI(title="GuideCare Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(patients.router, prefix="/patients", tags=["patients"])
app.include_router(conversations.router, prefix="/conversations", tags=["conversations"])

@app.on_event("startup")
async def on_startup():
    # Create tables
    await init_db()
    # Seed DB if empty
    await seed_if_empty()

@app.websocket("/ws/chat/{patient_id}")
async def websocket_chat(websocket: WebSocket, patient_id: str):
    await manager.connect(patient_id, websocket)
    try:
        while True:
            data = await websocket.receive_json()
            # Expect structure: {"role": "user"|"assistant", "content": "...", "meta": {...}}
            await manager.handle_incoming_message(patient_id, data)
    except WebSocketDisconnect:
        await manager.disconnect(patient_id, websocket)
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
        await manager.disconnect(patient_id, websocket)
