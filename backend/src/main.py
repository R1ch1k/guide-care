import logging
import os
from uuid import UUID

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.api import conversations, patients
from app.db.session import init_db
from app.seed import seed_if_empty
from app.ws_manager import manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("guidecare-backend")

app = FastAPI(title="GuideCare Backend")
app.state.graph = build_graph()
app.state.orch_deps = build_orchestration_deps()

# If CORS_ORIGINS is set, use it and allow credentials.
# Otherwise, allow "*" but disable credentials (browser-safe).
cors_origins_env = os.getenv("CORS_ORIGINS", "").strip()
if cors_origins_env:
    allow_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
    allow_credentials = True
else:
    allow_origins = ["*"]
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(patients.router, prefix="/patients", tags=["patients"])
app.include_router(conversations.router, prefix="/conversations", tags=["conversations"])


@app.on_event("startup")
async def on_startup():
    await init_db()
    await seed_if_empty()


@app.websocket("/ws/chat/{patient_id}")
async def websocket_chat(websocket: WebSocket, patient_id: UUID):
    await manager.connect(patient_id, websocket)
    try:
        while True:
            data = await websocket.receive_json()
            await manager.handle_incoming_message(patient_id, data)
    except WebSocketDisconnect:
        await manager.disconnect(patient_id, websocket)
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
        await manager.disconnect(patient_id, websocket)
