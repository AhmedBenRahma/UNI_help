"""Health API route."""

from __future__ import annotations

from fastapi import APIRouter, Request, status

from backend.models.response import HealthResponse

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health(request: Request) -> HealthResponse:
    """Return service health information."""
    vector_store = request.app.state.vector_store
    return HealthResponse(
        status="ok",
        vectorstore_loaded=vector_store.is_loaded,
        document_count=len(vector_store.list_documents()),
        model=request.app.state.settings.MODEL_NAME,
    )
