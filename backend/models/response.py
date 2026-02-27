"""Pydantic response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Retrieved source chunk."""

    filename: str
    page: int
    excerpt: str
    score: float


class EmailDraft(BaseModel):
    """Generated standardized email draft."""

    subject: str
    body: str
    email_type: str
    recipient_hint: str


class ChatResponse(BaseModel):
    """Chat API response."""

    answer: str
    sources: list[Source]
    confidence: float = Field(ge=0.0, le=1.0)
    has_answer: bool
    email_draft: EmailDraft | None = None
    session_id: str


class UploadResponse(BaseModel):
    """Upload endpoint response."""

    indexed_chunks: int
    filename: str
    status: str


class AdminDocumentsResponse(BaseModel):
    """Admin documents list response."""

    documents: list[dict[str, str | int | float | None]]


class HealthResponse(BaseModel):
    """Health endpoint response."""

    status: str
    vectorstore_loaded: bool
    document_count: int
    model: str
