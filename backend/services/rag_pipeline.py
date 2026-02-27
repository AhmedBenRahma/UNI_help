"""LCEL-based RAG pipeline service."""

from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from backend.core.config import Settings
from backend.core.logging import get_logger
from backend.models.response import Source
from backend.services.vector_store import VectorStoreService

FALLBACK_ANSWER = (
    "I don't have information about this in the official documents.\n"
    "Please contact the administration directly."
)


class RAGPipelineService:
    """Retrieval-augmented generation pipeline using LCEL."""

    def __init__(self, settings: Settings, vector_store: VectorStoreService) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.logger = get_logger(__name__)
        self.llm = ChatOpenAI(model=self.settings.MODEL_NAME, api_key=self.settings.OPENAI_API_KEY, temperature=0)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are UniHelp, an AI assistant for {university_name}. "
                    "Answer ONLY using the provided context. "
                    "If the answer is not in the context, reply exactly:\n"
                    "'I don't have information about this in the official documents.\n"
                    "Please contact the administration directly.'\n"
                    "Never invent or assume information.",
                ),
                ("human", "Question:\n{question}\n\nContext:\n{context}"),
            ]
        )
        self.parser = StrOutputParser()
        self.generation_chain = self.prompt | self.llm | self.parser

    def invoke(self, question: str) -> dict[str, Any]:
        """Run retrieval, generation, and structured formatting."""
        self.logger.info("rag_invoke_started", question=question)
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        retrieved = self.vector_store.similarity_search_with_score(question, k=self.settings.MAX_RETRIEVED_DOCS)
        if not retrieved:
            self.logger.info("rag_no_documents_found")
            return {
                "answer": FALLBACK_ANSWER,
                "sources": [],
                "confidence": 0.0,
                "has_answer": False,
            }

        context = self._build_context(retrieved)
        raw_answer = self.generation_chain.invoke(
            {
                "university_name": self.settings.UNIVERSITY_NAME,
                "question": question,
                "context": context,
            }
        )

        sources = self._build_sources(retrieved)
        confidence = self._compute_confidence(retrieved)
        has_answer = self._compute_has_answer(raw_answer, confidence)
        final_answer = raw_answer if has_answer else FALLBACK_ANSWER

        response = {
            "answer": final_answer,
            "sources": sources,
            "confidence": confidence,
            "has_answer": has_answer,
        }
        self.logger.info("rag_invoke_completed", confidence=confidence, has_answer=has_answer)
        return response

    def _build_context(self, retrieved: list[tuple[Any, float]]) -> str:
        """Build textual context from retrieved chunks."""
        blocks: list[str] = []
        for doc, score in retrieved:
            filename = doc.metadata.get("filename", "unknown")
            page_number = doc.metadata.get("page_number", 1)
            snippet = doc.page_content.strip()
            blocks.append(f"[Source: {filename} | Page: {page_number} | Score: {score:.4f}]\n{snippet}")
        return "\n\n".join(blocks)

    def _build_sources(self, retrieved: list[tuple[Any, float]]) -> list[Source]:
        """Convert retrieved chunks into strict Source schema objects."""
        sources: list[Source] = []
        for doc, raw_score in retrieved:
            normalized_score = self._normalize_similarity(raw_score)
            sources.append(
                Source(
                    filename=str(doc.metadata.get("filename", "unknown")),
                    page=int(doc.metadata.get("page_number", 1)),
                    excerpt=doc.page_content.strip()[:250],
                    score=round(normalized_score, 4),
                )
            )
        return sources

    def _compute_confidence(self, retrieved: list[tuple[Any, float]]) -> float:
        """Compute normalized confidence from average similarity score."""
        if not retrieved:
            return 0.0

        normalized = [self._normalize_similarity(score) for _, score in retrieved]
        confidence = sum(normalized) / len(normalized)
        return round(max(0.0, min(1.0, confidence)), 4)

    def _normalize_similarity(self, score: float) -> float:
        """Normalize FAISS distance-like score to 0-1 confidence."""
        if score < 0:
            return 0.0
        return 1.0 / (1.0 + float(score))

    def _compute_has_answer(self, answer: str, confidence: float) -> bool:
        """Determine if answer is valid based on fallback and threshold."""
        if answer.strip() == FALLBACK_ANSWER:
            return False
        return confidence >= self.settings.CONFIDENCE_THRESHOLD
