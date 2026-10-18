from rag import chunk_text, split_sentences

def test_chunking_respects_limit():
    text = "\n\n".join(["word " * 40] * 10)
    assert all(len(c) < 900 for c in chunk_text(text, 600))

def test_empty_text():
    assert chunk_text("   ") == []

def test_sentence_split():
    assert len(split_sentences("Restart the service now. Then check the logs for errors.")) == 2


def test_solve_returns_problem_steps_commands():
    from rag import KnowledgeBase
    kb = KnowledgeBase()
    kb.add_documents([("t.md", "# T\n\n## Printer is not printing (printer stuck)\nCheck that the printer is powered on. Run `net stop spooler` to restart it.\n\n## VPN will not connect\nRestart the VPN client and try again.")])
    res = kb.solve("printer stuck")
    assert res["found"] and res["title"].startswith("Printer") and res["commands"] == ["net stop spooler"]
    assert len(res["steps"]) == 2
    assert kb.solve("capital of France")["found"] is False
