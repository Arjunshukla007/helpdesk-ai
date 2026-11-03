# HelpDesk AI — IT Support Knowledge Assistant

A local-first knowledge assistant for IT runbooks and support documents. It supports PDF, Markdown and TXT uploads, hybrid retrieval, source-backed troubleshooting guidance, feedback and L2 escalation.

## Live application

**[🚀 Try HelpDesk AI — Streamlit Live Demo](https://appdesk-ai-vl32hcynngexuyv9xohrnz.streamlit.app/)**

Recruiters and reviewers can open the live application directly to explore the interface and try the support assistant. Availability and features depend on the currently deployed Streamlit version.

## RAG architecture

`PDF/Markdown/TXT → overlapping chunks → local sentence-transformer embeddings → dense retrieval + BM25 → reciprocal-rank fusion → optional cross-encoder reranking → answer with source references`

- **Local embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (downloads the model on first use, then runs locally).
- **Dense retrieval:** FAISS inner-product index when available; normalized NumPy cosine-search fallback otherwise.
- **Lexical retrieval:** BM25 for exact technical terms, commands and product names.
- **Fusion:** Reciprocal Rank Fusion combines dense and lexical rankings.
- **Optional reranking:** local CrossEncoder, disabled by default.
- **Answering:** extractive, source-backed answers by default; optional OpenAI generation grounded in retrieved chunks.
- **Privacy:** uploaded files remain in the current Streamlit session's knowledge-base copy. When OpenAI generation is enabled, retrieved context is sent to OpenAI to draft the answer; embeddings and retrieval remain local.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The embedding model downloads on first run. Retrieval runs locally after the model is available.

## Optional OpenAI generation

Copy `.env.example` to `.env` and set `ENABLE_OPENAI_GENERATION=true` and `OPENAI_API_KEY`. The Streamlit sidebar also includes a toggle. Without an API key, keep generation disabled and use extractive answers. If OpenAI generation fails, the application falls back to a local extractive answer.

Never commit `.env` or an API key; `.gitignore` excludes local environment files.

## Optional reranking

Set `RAG_USE_RERANKER=true` to load `cross-encoder/ms-marco-MiniLM-L-6-v2` locally. This downloads an additional model on first use. Keep it disabled for a lighter default setup.

## Tests and evaluation

Run `pytest -q`. Tests use a deterministic injected embedder and mocked generation, so they do not download models or call OpenAI. `recall_at_k` and `reciprocal_rank` helpers support evaluation against a labeled query set. No performance or quality benchmark is claimed until measured on a documented dataset.

## Preserved features

- Built-in Markdown knowledge base and private per-session document uploads.
- PDF, Markdown and TXT ingestion.
- Troubleshooting steps, extracted commands and related issues.
- Retrieved source chunk inspection.
- Positive/negative feedback and L2 escalation ticket generation.
- Streamlit interface and Docker entry point.

## Security

Keep API credentials in environment variables or Streamlit secrets. Retrieved document text is treated as untrusted context in the generation prompt. Review any suggested commands before execution.
