"""Administrative API routes for document management."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Header, HTTPException, Request, UploadFile, status

from backend.core.logging import get_logger
from backend.models.response import AdminDocumentsResponse, UploadResponse

router = APIRouter(prefix="/api/admin", tags=["admin"])
logger = get_logger(__name__)


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_200_OK)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    x_admin_key: str = Header(..., alias="X-Admin-Key"),
) -> UploadResponse:
    """Upload and index a document into FAISS."""
    settings = request.app.state.settings
    if x_admin_key != settings.ADMIN_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid admin key.")

    allowed_extensions = {".pdf", ".docx", ".txt"}
    extension = Path(file.filename or "").suffix.lower()
    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unsupported file type. Allowed: PDF, DOCX, TXT.",
        )

    documents_path = Path(settings.DOCUMENTS_PATH)
    documents_path.mkdir(parents=True, exist_ok=True)
    destination = documents_path / (file.filename or "uploaded_file")

    logger.info("upload_started", filename=file.filename)
    content = await file.read()
    destination.write_bytes(content)

    try:
        result = request.app.state.ingestion_service.ingest_file(destination)
    except ValueError as exc:
        logger.error("upload_validation_error", filename=file.filename, error=str(exc))
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("upload_runtime_error", filename=file.filename, error=str(exc))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    logger.info("upload_completed", filename=file.filename, indexed_chunks=result["indexed_chunks"])
    return UploadResponse(
        indexed_chunks=result["indexed_chunks"],
        filename=file.filename or destination.name,
        status="indexed",
    )


@router.get("/documents", response_model=AdminDocumentsResponse, status_code=status.HTTP_200_OK)
async def list_documents(request: Request) -> AdminDocumentsResponse:
    """List indexed documents and their metadata."""
    logger.info("list_documents_started")
    documents = request.app.state.vector_store.list_documents()
    logger.info("list_documents_completed", document_count=len(documents))
    return AdminDocumentsResponse(documents=documents)
