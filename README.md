# UniHelp AI University Administrative Assistant

Production-ready AI assistant for university administrative Q&A with document-grounded responses, confidence scoring, and standardized email draft generation.

## 1) Architecture Diagram (ASCII)

```text
┌─────────────────────────────── Frontend (frontend/index.html) ────────────────────────────────┐
│  Chat UI (dark theme) • Source chips • Confidence bar • Email modal                            │
└───────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                │ HTTP/JSON
┌───────────────────────────────────────────────▼──────────────────────────────────────────────────┐
│                               FastAPI Backend (backend/main.py)                                 │
│                                                                                                  │
│  /api/chat ─────────► RAGPipelineService (LCEL) ───────────────► ChatOpenAI (gpt-4o-mini)      │
│       │                          │                                 ▲                              │
│       │                          └── similarity_search_with_score ─┘                              │
│       │                                     │                                                     │
│       │                                     ▼                                                     │
│       │                              FAISS Vector Store                                           │
│       │                          (OpenAIEmbeddings text-embedding-3-small)                       │
│       │                                                                                            │
│       └── optional email draft ─► EmailGeneratorService (LLM + Jinja2 template)                  │
│                                                                                                  │
│  /api/admin/upload ─► IngestionService (PDF/DOCX/TXT, chunking, metadata, dedup by MD5)         │
│  /api/admin/documents ─► indexed metadata list                                                    │
│  /api/health ─► health + model + vectorstore status                                               │
└───────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                │
                                  data/documents + data/vectorstore
```
## 🖼️ Demo

![Demo 1](Demo%201.png)
![Demo 2](Demo%202.png)
![Demo 3](Demo%203.png)

## 2) Quick Start

```bash
git clone <(https://github.com/AhmedBenRahma/UNI_help.git)>
cp .env.example .env
docker-compose up --build
```

Open:

- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/api/health`

## 3) How to Add Documents

1. Use admin upload endpoint (`POST /api/admin/upload`) with `multipart/form-data`.
2. Supported files: `.pdf`, `.docx`, `.txt`.
3. Include header: `X-Admin-Key: <ADMIN_KEY from .env>`.
4. The system extracts metadata (`filename`, `page_number`, `upload_timestamp`) and deduplicates by MD5 content hash.
5. FAISS index is automatically persisted to `data/vectorstore` after upload.

## 4) API Reference

| Method | Endpoint               | Description                                                       | Request                                                                                                | Response                                                                             |
| ------ | ---------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| POST   | `/api/chat`            | Ask administrative questions and optionally generate email drafts | `{"message": "...", "session_id": "..."}` + optional `generate_email`, `email_type`, `student_context` | `ChatResponse`                                                                       |
| POST   | `/api/admin/upload`    | Upload and index document                                         | `multipart/form-data` + `X-Admin-Key` header                                                           | `{"indexed_chunks": N, "filename": "...", "status": "indexed"}`                      |
| GET    | `/api/admin/documents` | List indexed documents metadata                                   | None                                                                                                   | `{"documents": [...]}`                                                               |
| GET    | `/api/health`          | Service health and vectorstore status                             | None                                                                                                   | `{"status":"ok","vectorstore_loaded":bool,"document_count":N,"model":"gpt-4o-mini"}` |

## 5) How Confidence Scoring Works

- Retrieval uses `similarity_search_with_score()` from FAISS.
- Each retrieved chunk score is normalized using: `normalized = 1 / (1 + distance)`.
- Confidence is the average of normalized scores across retrieved chunks.
- Final confidence is clamped to `[0.0, 1.0]`.
- `has_answer` is `true` only when:
  - model output is not the strict fallback sentence, and
  - confidence is `>= CONFIDENCE_THRESHOLD` (default `0.75`).

## 6) Deployment Notes

- Uses Python 3.11 slim multi-stage Docker build.
- Persistent data mounted via `./data:/app/data` to keep FAISS index across restarts.
- Logs can be collected from stdout (JSON via structlog) and optionally persisted under `logs/`.
- Optional reverse proxy: `docker-compose --profile proxy up --build` exposes Nginx on port `8080`.
- For production:
  - set strong `ADMIN_KEY`,
  - lock down `CORS_ORIGINS`,
  - use HTTPS termination at proxy/load balancer,
  - inject secrets via secure secret manager.
