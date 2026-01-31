from pydantic import BaseModel, Field
from typing import List, Optional, Any
from uuid import UUID
from datetime import date, datetime

class Medication(BaseModel):
    name: str
    dose: Optional[str] = None

class PatientCreate(BaseModel):
    nhs_number: str
    first_name: str
    last_name: str
    date_of_birth: date
    gender: Optional[str] = None
    conditions: Optional[List[str]] = []
    medications: Optional[List[Medication]] = []
    allergies: Optional[List[str]] = []
    recent_vitals: Optional[dict] = {}
    clinical_notes: Optional[List[dict]] = []

class PatientOut(BaseModel):
    id: UUID
    nhs_number: str
    first_name: str
    last_name: str
    date_of_birth: date
    age: int
    gender: Optional[str] = None
    conditions: List[str] = []
    medications: List[Any] = []
    allergies: List[str] = []
    recent_vitals: dict = {}
    clinical_notes: List[Any] = []
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class ConversationCreate(BaseModel):
    patient_id: UUID
    selected_guideline: Optional[str] = None

class Message(BaseModel):
    role: str
    content: str
    meta: Optional[dict] = None
    timestamp: Optional[datetime] = None

class ConversationOut(BaseModel):
    id: UUID
    patient_id: UUID
    messages: List[Any]
    selected_guideline: Optional[str] = None
    extracted_variables: dict = {}
    final_recommendation: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
