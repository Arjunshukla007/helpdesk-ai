# 🛠️ HelpDesk AI — IT Support Knowledge Assistant

**🔗 Live demo: [Help Desk AI](https://appdesk-ai1-jzkkbtrqxupzykjimorwtu.streamlit.app/)** (click → ask a question, no login, no API key)

A Retrieval-Augmented Generation (RAG) app that answers IT support questions from runbooks and SOPs **with source citations**. Upload your own PDFs/Markdown/TXT and query them instantly.

## Why this is useful in the IT industry
L1/L2 support teams lose hours searching wikis and old tickets. HelpDesk AI gives instant, grounded answers from the team's own documents, and says *"I don't know"* instead of guessing.

## Features
- Semantic search with local embeddings (`all-MiniLM-L6-v2`) + **FAISS** vector index
- Extractive answers: **no hallucination**, every answer is traceable to a source chunk
- Confidence score + latency shown for each query
- Upload your own PDF / MD / TXT docs and index them live
- **Zero API keys, zero cost**: runs on CPU
- Dockerised, unit tested, CI via GitHub Actions

## Architecture
`Docs → chunking → MiniLM embeddings → FAISS index → top-k retrieval → sentence ranking → answer + sources`

## Run locally (2 commands)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Or with Docker:
```bash
docker build -t helpdesk-ai . && docker run -p 8501:8501 helpdesk-ai
```

## Tech stack
Python • Streamlit • sentence-transformers • FAISS • pypdf • Docker • pytest • GitHub Actions

## Author
Arjun Shukla — arjunshukla085@gmail.com

## Add your own help desk entries
Add sections to any `.md` file in `data/kb/` (or create a new one). Use this pattern, and put the words people actually type in the title:
```
## Your issue title (extra search words, synonyms)
First fix, written as a clear sentence.

Next step or fallback, such as a command in `backticks`.
```
Every sentence becomes a numbered solution step and every `backtick` command becomes a copyable command block.
