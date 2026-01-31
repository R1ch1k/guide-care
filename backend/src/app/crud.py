from datetime import date, datetime
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Patient, Conversation
from app.schemas import PatientCreate
from uuid import UUID
import uuid

async def compute_age(dob: date) -> int:
    today = date.today()
    years = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return years

async def create_patient(db: AsyncSession, payload: PatientCreate) -> Patient:
    age = await compute_age(payload.date_of_birth)
    patient = Patient(
        nhs_number=payload.nhs_number,
        first_name=payload.first_name,
        last_name=payload.last_name,
        date_of_birth=payload.date_of_birth,
        age=age,
        gender=payload.gender,
        conditions=payload.conditions or [],
        medications=[m.dict() for m in (payload.medications or [])],
        allergies=payload.allergies or [],
        recent_vitals=payload.recent_vitals or {},
        clinical_notes=payload.clinical_notes or []
    )
    db.add(patient)
    await db.commit()
    await db.refresh(patient)
    return patient

async def get_patient(db: AsyncSession, patient_id: UUID):
    q = await db.execute(select(Patient).where(Patient.id == patient_id))
    return q.scalars().first()

async def list_patients(db: AsyncSession, limit: int = 100):
    q = await db.execute(select(Patient).limit(limit))
    return q.scalars().all()

async def start_conversation(db: AsyncSession, patient_id: UUID, selected_guideline: str = None):
    conv = Conversation(patient_id=patient_id, selected_guideline=selected_guideline, messages=[])
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv

async def get_conversation(db: AsyncSession, conv_id: UUID):
    q = await db.execute(select(Conversation).where(Conversation.id == conv_id))
    return q.scalars().first()

async def append_message_to_conversation(db: AsyncSession, conv: Conversation, message: dict):
    # append to messages array and save
    current = conv.messages or []
    current.append(message)
    conv.messages = current
    conv.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(conv)
    return conv
