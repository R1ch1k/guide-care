from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session
from app.schemas import PatientCreate, PatientOut
from app import crud

router = APIRouter()


@router.get("", response_model=list[PatientOut])
async def list_patients(db: AsyncSession = Depends(get_async_session)):
    return await crud.list_patients(db)


@router.get("/{patient_id}", response_model=PatientOut)
async def get_patient(patient_id: UUID, db: AsyncSession = Depends(get_async_session)):
    p = await crud.get_patient(db, patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    return p


@router.get("/{patient_id}/context")
async def patient_context(patient_id: UUID, db: AsyncSession = Depends(get_async_session)):
    p = await crud.get_patient(db, patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")

    return {
        "id": str(p.id),
        "nhs_number": p.nhs_number,
        "name": f"{p.first_name} {p.last_name}",
        "age": p.age,
        "gender": p.gender,
        "conditions": p.conditions,
        "medications": p.medications,
        "allergies": p.allergies,
        "recent_vitals": p.recent_vitals,
        "clinical_notes": p.clinical_notes,
    }


@router.post("", response_model=PatientOut, status_code=201)
async def create_patient(payload: PatientCreate, db: AsyncSession = Depends(get_async_session)):
    return await crud.create_patient(db, payload)
