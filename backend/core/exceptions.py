"""Custom application exception classes."""


class UniHelpError(Exception):
    """Base exception for UniHelp-specific errors."""


class VectorStoreError(UniHelpError):
    """Raised when vector store operations fail."""


class IngestionError(UniHelpError):
    """Raised when document ingestion fails."""


class EmailGenerationError(UniHelpError):
    """Raised when email generation fails."""
