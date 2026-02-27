"""FAISS vector store wrapper service."""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from backend.core.config import Settings
from backend.core.exceptions import VectorStoreError
from backend.core.logging import get_logger


class VectorStoreService:
    """Wrapper around FAISS for loading, saving, and searching."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__)
        self.embeddings = OpenAIEmbeddings(
            model=self.settings.EMBEDDING_MODEL,
            api_key=self.settings.OPENAI_API_KEY,
        )
        self.index_path = Path(self.settings.VECTORSTORE_PATH)
        self._vector_store: FAISS | None = None

    @property
    def is_loaded(self) -> bool:
        """Return whether vector store is loaded in memory."""
        return self._vector_store is not None

    def build(self, documents: list[Document]) -> None:
        """Build FAISS index from documents."""
        self.logger.info("vectorstore_build_started", document_count=len(documents))
        if not documents:
            self._vector_store = None
            self.logger.info("vectorstore_build_skipped", reason="no_documents")
            return

        try:
            self._vector_store = FAISS.from_documents(documents, self.embeddings)
            self.logger.info("vectorstore_build_completed", document_count=len(documents))
        except (ValueError, RuntimeError) as exc:
            self.logger.error("vectorstore_build_failed", error=str(exc))
            raise VectorStoreError("Failed to build vector store.") from exc

    def load(self) -> bool:
        """Load FAISS index from disk if available."""
        self.logger.info("vectorstore_load_started", path=str(self.index_path))
        if not self.index_path.exists():
            self._vector_store = None
            self.logger.info("vectorstore_not_found", path=str(self.index_path))
            return False

        try:
            self._vector_store = FAISS.load_local(
                folder_path=str(self.index_path),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
            self.logger.info("vectorstore_load_completed", path=str(self.index_path))
            return True
        except (ValueError, RuntimeError, OSError) as exc:
            self._vector_store = None
            self.logger.error("vectorstore_load_failed", error=str(exc))
            return False

    def save(self) -> None:
        """Persist current FAISS index to disk."""
        self.logger.info("vectorstore_save_started", path=str(self.index_path))
        if self._vector_store is None:
            self.logger.info("vectorstore_save_skipped", reason="index_not_loaded")
            return

        try:
            self.index_path.mkdir(parents=True, exist_ok=True)
            self._vector_store.save_local(str(self.index_path))
            self.logger.info("vectorstore_save_completed", path=str(self.index_path))
        except OSError as exc:
            self.logger.error("vectorstore_save_failed", error=str(exc))
            raise VectorStoreError("Failed to save vector store.") from exc

    def add_documents(self, documents: list[Document]) -> None:
        """Add new documents to an existing or new index and persist."""
        self.logger.info("vectorstore_add_documents_started", document_count=len(documents))
        if not documents:
            self.logger.info("vectorstore_add_documents_skipped", reason="no_documents")
            return

        try:
            if self._vector_store is None:
                self.build(documents)
            else:
                self._vector_store.add_documents(documents)

            self.save()
            self.logger.info("vectorstore_add_documents_completed", document_count=len(documents))
        except (ValueError, RuntimeError, VectorStoreError) as exc:
            self.logger.error("vectorstore_add_documents_failed", error=str(exc))
            raise VectorStoreError("Failed to add documents to vector store.") from exc

    def similarity_search_with_score(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Run similarity search and return tuples of document and score."""
        self.logger.info("vectorstore_similarity_search_started", top_k=k)
        if self._vector_store is None and not self.load():
            self.logger.info("vectorstore_similarity_search_empty", reason="index_not_found")
            return []

        if self._vector_store is None:
            return []

        try:
            results = self._vector_store.similarity_search_with_score(query, k=k)
            self.logger.info("vectorstore_similarity_search_completed", result_count=len(results))
            return results
        except (ValueError, RuntimeError) as exc:
            self.logger.error("vectorstore_similarity_search_failed", error=str(exc))
            raise VectorStoreError("Similarity search failed.") from exc

    def list_documents(self) -> list[dict[str, str | int | float | None]]:
        """Return unique indexed documents metadata."""
        if self._vector_store is None and not self.load():
            return []

        if self._vector_store is None:
            return []

        doc_entries: dict[str, dict[str, str | int | float | None]] = {}
        for doc in self._vector_store.docstore._dict.values():  # pylint: disable=protected-access
            metadata = doc.metadata
            filename = str(metadata.get("filename", "unknown"))
            if filename not in doc_entries:
                doc_entries[filename] = {
                    "filename": filename,
                    "upload_timestamp": metadata.get("upload_timestamp"),
                    "content_hash": metadata.get("content_hash"),
                }

        return list(doc_entries.values())

    def existing_hashes(self) -> set[str]:
        """Return content hashes already present in the index."""
        if self._vector_store is None and not self.load():
            return set()

        if self._vector_store is None:
            return set()

        hashes: set[str] = set()
        for doc in self._vector_store.docstore._dict.values():  # pylint: disable=protected-access
            content_hash = doc.metadata.get("content_hash")
            if isinstance(content_hash, str):
                hashes.add(content_hash)
        return hashes
