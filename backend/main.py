"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.admin import router as admin_router
from backend.api.routes.chat import router as chat_router
from backend.api.routes.health import router as health_router
from backend.core.config import get_settings
from backend.core.logging import configure_logging, get_logger
from backend.services.email_generator import EmailGeneratorService
from backend.services.ingestion import IngestionService
from backend.services.rag_pipeline import RAGPipelineService
from backend.services.vector_store import VectorStoreService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup long-lived app services."""
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    logger = get_logger(__name__)
    logger.info("app_startup_begin")

    vector_store = VectorStoreService(settings)
    vector_store.load()

    ingestion_service = IngestionService(settings=settings, vector_store=vector_store)
    rag_pipeline = RAGPipelineService(settings=settings, vector_store=vector_store)
    email_generator = EmailGeneratorService(settings=settings, templates_path=Path("email_templates"))

    app.state.settings = settings
    app.state.vector_store = vector_store
    app.state.ingestion_service = ingestion_service
    app.state.rag_pipeline = rag_pipeline
    app.state.email_generator = email_generator

    logger.info("app_startup_complete")
    yield
    logger.info("app_shutdown_complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    app = FastAPI(title="UniHelp AI Assistant", version="1.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_router)
    app.include_router(admin_router)
    app.include_router(health_router)
    return app


app = create_app()
