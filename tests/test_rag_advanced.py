import numpy as np

from rag import KnowledgeBase, RAGConfig, chunk_text, pretty_title, recall_at_k, reciprocal_rank


class TinyEmbedder:
    """Deterministic test embedder; avoids downloading a model in unit tests."""
    vocab = ["vpn", "password", "reset", "network", "dns", "printer", "restart", "account"]

    def encode(self, texts, **kwargs):
        matrix = []
        for text in texts:
            words = set(text.lower().replace(".", " ").split())
            vector = np.array([float(term in words) for term in self.vocab], dtype=np.float32)
            matrix.append(vector)
        return np.vstack(matrix)


def make_kb():
    config = RAGConfig(use_faiss=False, chunk_size=500, chunk_overlap=40)
    return KnowledgeBase(embedder=TinyEmbedder(), config=config)


def test_chunk_text_returns_chunks_and_handles_empty():
    assert chunk_text("") == []
    chunks = chunk_text("# VPN\n\nRestart the VPN client.\n\nCheck network settings.", 200, 20)
    assert chunks
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_add_documents_deduplicates_and_searches():
    kb = make_kb()
    text = "# VPN troubleshooting\n\nRestart the VPN client and check network settings."
    assert kb.add_documents([("vpn.md", text)]) >= 1
    assert kb.add_documents([("vpn.md", text)]) == 0
    results = kb.search("VPN restart", k=2)
    assert results
    assert results[0]["source"] == "vpn.md"
    assert "dense_score" in results[0] and "lexical_score" in results[0]


def test_answer_preserves_source_and_extractive_mode():
    kb = make_kb()
    kb.add_documents([("vpn.md", "# VPN troubleshooting\n\nRestart the VPN client to restore the connection.")])
    result = kb.answer("How do I restart VPN?", threshold=0)
    assert result["answer"]
    assert result["sources"][0]["source"] == "vpn.md"
    assert result["mode"] == "extractive"


def test_solve_returns_legacy_shape():
    kb = make_kb()
    kb.add_documents([("vpn.md", "# VPN troubleshooting\n\nRestart the VPN client. Run `ipconfig /flushdns` to clear DNS cache.")])
    result = kb.solve("VPN restart", threshold=0)
    assert result["found"] is True
    assert result["source"] == "vpn.md"
    assert isinstance(result["commands"], list)


def test_no_match_is_safe():
    kb = make_kb()
    kb.add_documents([("vpn.md", "# VPN\n\nRestart the VPN client.")])
    result = kb.answer("unknown unrelated topic", threshold=0.99)
    assert result["answer"] is None


def test_metrics_and_title_helpers():
    assert recall_at_k(["a", "b", "c"], {"b", "d"}, k=2) == 0.5
    assert reciprocal_rank(["x", "b"], {"b"}) == 0.5
    assert pretty_title("VPN issue (synonyms, search terms)") == "VPN issue"


def test_solve_can_use_optional_openai_generation(monkeypatch):
    kb = make_kb()
    kb.add_documents([("vpn.md", "# VPN troubleshooting\n\nRestart the VPN client to restore the connection.")])
    monkeypatch.setattr(kb, "_openai_answer", lambda query, hits: "Restart the VPN client. [S1]")
    result = kb.solve("VPN restart", threshold=0, use_llm=True)
    assert result["found"] is True
    assert result["answer_mode"] == "openai"
    assert "[S1]" in result["answer"]


def test_openai_failure_falls_back_to_extractive_answer(monkeypatch):
    kb = make_kb()
    kb.add_documents([("vpn.md", "# VPN troubleshooting\n\nRestart the VPN client to restore the connection.")])
    def fail_generation(query, hits):
        raise RuntimeError("API unavailable")
    monkeypatch.setattr(kb, "_openai_answer", fail_generation)
    result = kb.solve("VPN restart", threshold=0, use_llm=True)
    assert result["found"] is True
    assert result["answer_mode"] == "extractive"
    assert result["answer"]
    assert "API unavailable" in result["generation_error"]
