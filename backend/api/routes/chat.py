"""Chat API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from backend.core.logging import get_logger
from backend.models.request import ChatRequest
from backend.models.response import ChatResponse, EmailDraft

router = APIRouter(prefix="/api", tags=["chat"])
logger = get_logger(__name__)


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Process user chat messages with RAG and optional email drafting."""
    rag_pipeline = request.app.state.rag_pipeline
    email_generator = request.app.state.email_generator

    logger.info("chat_request_received", session_id=payload.session_id)

    try:
        result = rag_pipeline.invoke(question=payload.message)
    except ValueError as exc:
        logger.error("chat_request_validation_error", error=str(exc))
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("chat_request_runtime_error", error=str(exc))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    email_draft: EmailDraft | None = None
    if payload.generate_email and result["has_answer"]:
        if not payload.email_type:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="email_type is required when generate_email is true.",
            )

        try:
            draft = email_generator.generate_email(
                email_type=payload.email_type,
                student_input=payload.student_context or payload.message,
                assistant_answer=result["answer"],
            )
            email_draft = EmailDraft(**draft)
        except ValueError as exc:
            logger.error("email_generation_validation_error", error=str(exc))
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        except RuntimeError as exc:
            logger.error("email_generation_runtime_error", error=str(exc))
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    logger.info("chat_request_completed", session_id=payload.session_id, has_answer=result["has_answer"])
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
        has_answer=result["has_answer"],
        email_draft=email_draft,
        session_id=payload.session_id,
    )
