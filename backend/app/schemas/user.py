"""
app/schemas/user.py
Pydantic models for user-related request/response data.
"""
from datetime import datetime

from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    first_name:     str           = Field(..., min_length=1, max_length=100)
    last_name:      str           = Field(..., min_length=1, max_length=100)
    nickname:       str | None    = Field(None, max_length=100)
    favorite_color: str | None    = Field(None, max_length=50)
    favorite_food:  str | None    = Field(None, max_length=100)
    pet_name:       str | None    = Field(None, max_length=100)


class UserOut(BaseModel):
    id:             str
    first_name:     str
    last_name:      str
    nickname:       str | None
    favorite_color: str | None
    favorite_food:  str | None
    pet_name:       str | None
    created_at:     datetime

    model_config = {"from_attributes": True}


class FactSheetOut(BaseModel):
    """Minimal card data for the fact sheet view."""
    id:             str
    first_name:     str
    last_name:      str
    nickname:       str | None
    favorite_color: str | None
    favorite_food:  str | None
    pet_name:       str | None
    created_at:     datetime
    verified:       bool = True   # always True — only issued after successful verification

    model_config = {"from_attributes": True}
