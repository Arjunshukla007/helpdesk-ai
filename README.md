# HelpDesk AI — IT Support Knowledge Assistant

> **Try the project:** [🚀 Open the live HelpDesk AI demo](https://appdesk-ai-vl32hcynngexuyv9xohrnz.streamlit.app/)  
> **Source code:** [Browse this GitHub repository](https://github.com/Arjunshukla007/helpdesk-ai) · **[Setup and run locally](#run-locally)**

A local-first knowledge assistant for IT runbooks and support documents. It supports PDF, Markdown and TXT uploads, hybrid retrieval, source-backed troubleshooting guidance, feedback and L2 escalation.

## Recruiter quick start

1. Open the [live demo](https://appdesk-ai-vl32hcynngexuyv9xohrnz.streamlit.app/).
2. Try a sample question such as **“Docker container keeps restarting, what should I check?”**
3. Upload a PDF, Markdown or TXT runbook from the sidebar and select **Index uploaded files**.
4. Review the generated guidance and expand **Sources** to inspect the retrieved document chunks.
5. Use **Yes, fixed** or **Not fixed** to try the feedback and L2 ticket workflow.

No installation or API key is needed to try the hosted demo. The live site is the quickest way to review the user experience; the local setup below lets you run the source code yourself.

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

**Requirements:** Python 3.10 or newer and Git.

```bash
git clone https://github.com/Arjunshukla007/helpdesk-ai.git
cd helpdesk-ai

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). The embedding model downloads on first run. Retrieval runs locally after the model is available.

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

## Development status

The advanced local-RAG upgrade is also available in the [open draft pull request](https://github.com/Arjunshukla007/helpdesk-ai/pull/1). Check the pull request for the proposed branch changes and review status before assuming those changes are part of the default branch.
