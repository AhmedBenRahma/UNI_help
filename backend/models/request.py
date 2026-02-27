"""Pydantic request schemas."""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat request payload."""

    message: str = Field(..., min_length=1)
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    generate_email: bool = False
    email_type: str | None = None
    student_context: str | None = None
