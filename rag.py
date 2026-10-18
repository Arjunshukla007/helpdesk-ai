"""Lightweight RAG engine: chunk -> TF-IDF vectors (scikit-learn) -> cosine search -> extractive answer.
No API key, no torch, no model download. Answers come only from your documents (no hallucination)."""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def chunk_text(text: str, max_chars: int = 600) -> list[str]:
    """Split on blank lines, then merge small pieces up to max_chars."""
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for p in parts:
        has_body = any(l.strip() and not l.lstrip().startswith("#") for l in current.splitlines())
        starts_section = p.lstrip().startswith("#") and has_body  # new chunk per section
        if current and (len(current) + len(p) > max_chars or starts_section):
            chunks.append(current.strip())
            current = ""
        current += p + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


def split_sentences(text: str) -> list[str]:
    lines = [l for l in text.splitlines() if not l.lstrip().startswith("#")]  # drop headings
    sents = re.split(r"(?<=[.!?])\s+|\n+", "\n".join(lines))
    return [s.strip(" -*") for s in sents if len(s.strip()) > 15]


def _new_vectorizer() -> TfidfVectorizer:
    # char n-grams inside words handle "restart/restarting", "commit/commits" without a stemmer
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=True)


class KnowledgeBase:
    def __init__(self):
        self.chunks: list[dict] = []
        self.vectorizer = None
        self.matrix = None

    def _rebuild(self):
        self.vectorizer = _new_vectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def add_documents(self, docs: list[tuple[str, str]]) -> int:
        new = []
        for name, text in docs:
            title = "General"
            for c in chunk_text(text):
                heads = re.findall(r"^#{1,6}\s+(.+)$", c, re.M)
                if heads:
                    title = heads[-1].strip()  # most specific heading in this chunk
                new.append({"source": name, "title": title, "text": c})
        if not new:
            return 0
        self.chunks += new
        self._rebuild()
        return len(new)

    def search(self, query: str, k: int = 4):
        if self.matrix is None:
            return []
        sims = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        top = np.argsort(-sims)[:k]
        return [{**self.chunks[i], "score": float(sims[i])} for i in top if sims[i] > 0]

    def answer(self, query: str, k: int = 4, threshold: float = 0.15):
        hits = self.search(query, k)
        if not hits or hits[0]["score"] < threshold:
            return {"answer": None, "sources": hits, "confidence": hits[0]["score"] if hits else 0.0}
        close = [h for h in hits[:2] if h["score"] >= 0.85 * hits[0]["score"]]
        sents = [s for h in close for s in split_sentences(h["text"])]
        if not sents:
            return {"answer": None, "sources": hits, "confidence": hits[0]["score"]}
        qv = self.vectorizer.transform([query])
        sims = (self.vectorizer.transform(sents) @ qv.T).toarray().ravel()
        top = [i for i in np.argsort(-sims)[:3] if sims[i] >= threshold * 0.4]
        top = top or list(range(min(2, len(sents))))  # fallback: first sentences of best chunk
        lines = [f"- {sents[i]}" for i in top]
        return {"answer": "\n".join(lines) or None, "sources": hits, "confidence": hits[0]["score"]}

    def solve(self, query: str, k: int = 6, threshold: float = 0.15):
        """Identify the problem (best matching article) and return the full fix:
        numbered steps, commands to run, and related issues."""
        hits = self.search(query, k)
        if not hits or hits[0]["score"] < threshold:
            return {"found": False, "confidence": hits[0]["score"] if hits else 0.0, "sources": hits}
        best = hits[0]
        key = (best["source"], best["title"])
        section = [c for c in self.chunks if (c["source"], c["title"]) == key]
        steps = [s for c in section for s in split_sentences(c["text"])]
        commands = list(dict.fromkeys(re.findall(r"`([^`\n]+)`", " ".join(c["text"] for c in section))))
        related, seen = [], {key}
        for h in hits[1:]:
            k2 = (h["source"], h["title"])
            if k2 not in seen and h["score"] >= threshold * 0.6:
                related.append({"title": h["title"], "source": h["source"]})
                seen.add(k2)
            if len(related) == 2:
                break
        return {"found": True, "title": best["title"], "source": best["source"], "steps": steps,
                "commands": commands, "related": related, "confidence": best["score"], "sources": hits}


def pretty_title(title: str) -> str:
    """Drop the '(synonyms, for search)' part of a heading for display."""
    return re.sub(r"\s*\(.*?\)\s*", " ", title).strip()
