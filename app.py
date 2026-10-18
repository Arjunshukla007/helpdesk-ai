import copy
import glob
import os
import time
from datetime import datetime
from urllib.parse import quote

import streamlit as st
from pypdf import PdfReader

from rag import KnowledgeBase, pretty_title

st.set_page_config(page_title="HelpDesk AI", page_icon="🛠️", layout="wide")
SUPPORT_EMAIL = "helpdesk@example.com"  # change to your L2 support / team mailbox
KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "kb")


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_base_kb() -> KnowledgeBase:
    """Built-in docs only. Shared and read-only: user uploads never go into this object."""
    kb = KnowledgeBase()
    docs = []
    for path in sorted(glob.glob(os.path.join(KB_DIR, "*.md"))):
        with open(path, encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read()))
    kb.add_documents(docs)
    return kb


def read_upload(f) -> str:
    if f.name.lower().endswith(".pdf"):
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(f).pages)
    return f.read().decode("utf-8", errors="ignore")


base_kb = load_base_kb()
# Each visitor gets a private copy, so uploaded files are never visible to other users
if "kb" not in st.session_state:
    st.session_state.kb = copy.deepcopy(base_kb)
    st.session_state.uploaded = []
kb = st.session_state.kb
st.session_state.setdefault("tickets", [])
st.session_state.setdefault("current_ticket", None)

st.title("🛠️ HelpDesk AI — IT Support Knowledge Assistant")
st.caption("Ask IT questions and get answers with sources. Retrieval-based, no API key, no paid service.")

with st.sidebar:
    st.header("📄 Add your own docs")
    st.caption("Uploads are private to your session and are not stored.")
    files = st.file_uploader("Runbooks / SOPs (PDF, MD, TXT)", type=["pdf", "md", "txt"], accept_multiple_files=True)
    if files and st.button("Index uploaded files"):
        new = [f for f in files if f.name not in st.session_state.uploaded]
        n = kb.add_documents([(f.name, read_upload(f)) for f in new])
        st.session_state.uploaded += [f.name for f in new]
        st.success(f"Indexed {n} new chunks." if new else "Those files are already indexed.")
    if st.session_state.uploaded and st.button("Remove uploaded docs"):
        st.session_state.kb = copy.deepcopy(base_kb)
        st.session_state.uploaded = []
        st.rerun()
    st.divider()
    st.metric("Chunks in knowledge base", len(kb.chunks))
    if not base_kb.chunks:
        st.error("Built-in knowledge base is empty. Make sure the data/kb folder is in your repo.")
    st.markdown("**Built-in topics:** Docker, Git, Linux, SQL, networking, VPN, printers, Wi-Fi, Outlook")

samples = ["Docker container keeps restarting, what should I check?",
           "How do I undo my last git commit?",
           "Disk is full on a Linux server, how to find large files?",
           "Why is my SQL query slow?",
           "My VPN is not connecting"]
st.write("**Try a sample question:**")
cols = st.columns(len(samples))
picked = None
for c, s in zip(cols, samples):
    if c.button(s, use_container_width=True):
        picked = s

def confidence_label(score: float) -> str:
    return "High" if score >= 0.30 else "Medium" if score >= 0.20 else "Low"


def ticket_text(query: str, res: dict, tid: str) -> str:
    tried = "\n".join(f"  {i}. {s}" for i, s in enumerate(res["steps"], 1))
    return (f"Ticket: {tid}\n"
            f"Subject: {pretty_title(res['title'])} - user issue not resolved\n\n"
            f"Issue reported by user:\n  {query}\n\n"
            f"Matched knowledge base article: {pretty_title(res['title'])} ({res['source']})\n\n"
            f"Steps already tried:\n{tried}\n\n"
            f"Status: Not resolved. Please escalate to L2 support.")


def render_result(last: dict):
    query, res = last["query"], last["res"]
    st.chat_message("user").write(query)
    with st.chat_message("assistant"):
        if not res["found"]:
            st.warning("I couldn't find a reliable answer in the knowledge base. "
                       "Try rephrasing, or upload a relevant doc in the sidebar.")
        else:
            st.markdown("#### 🔍 Problem identified")
            st.markdown(f"**{pretty_title(res['title'])}**")
            st.caption(f"Confidence: {confidence_label(res['confidence'])} ({res['confidence']:.2f}) "
                       f"• Source: {res['source']} • {res['ms']:.0f} ms")
            st.markdown("#### ✅ Recommended solution")
            for i, step in enumerate(res["steps"], 1):
                st.markdown(f"{i}. {step}")
            if res["commands"]:
                st.markdown("#### 💻 Commands you may need")
                for cmd in res["commands"]:
                    st.code(cmd, language="bash")
            if res["related"]:
                st.markdown("#### 🔗 Related issues")
                for r in res["related"]:
                    st.markdown(f"- {pretty_title(r['title'])}  _({r['source']})_")

            st.markdown("#### Did this fix your problem?")
            c1, c2, _ = st.columns([1, 1, 4])
            if c1.button("✅ Yes, fixed"):
                st.session_state.feedback = "yes"
            if c2.button("❌ Not fixed"):
                st.session_state.feedback = "no"
                tid = f"TKT-{datetime.now():%y%m%d}-{len(st.session_state.tickets) + 1:03d}"
                ticket = {"id": tid, "issue": pretty_title(res["title"]), "status": "Escalated to L2",
                          "text": ticket_text(query, res, tid)}
                st.session_state.tickets.append(ticket)
                st.session_state.current_ticket = ticket
            if st.session_state.get("feedback") == "yes":
                st.success("Great, glad it's resolved! 🎉")
            elif st.session_state.get("feedback") == "no" and st.session_state.current_ticket:
                t = st.session_state.current_ticket
                st.info(f"Ticket **{t['id']}** created. Send it to L2 support so they can take over:")
                st.code(t["text"], language=None)
                subject = t["text"].splitlines()[1].replace("Subject: ", "")
                b1, b2, _ = st.columns([2, 2, 3])
                b1.link_button("📧 Email to L2 support",
                               f"mailto:{SUPPORT_EMAIL}?subject={quote(t['id'] + ' ' + subject)}&body={quote(t['text'])}")
                b2.download_button("⬇️ Download ticket (.txt)", t["text"], file_name=f"{t['id']}.txt")
                st.markdown("**What happens next:** the L2 engineer reads this ticket (the issue and the steps "
                            "you already tried are included, so you don't repeat yourself) and replies by email. "
                            f"Keep the ticket ID **{t['id']}** for follow-up.")
        with st.expander("📚 Sources (retrieved chunks)"):
            for h in res["sources"]:
                st.markdown(f"**{h['source']}** — similarity `{h['score']:.2f}`")
                st.text(h["text"])


query = st.chat_input("Describe your IT problem...") or picked
if query:
    t0 = time.time()
    res = kb.solve(query)
    res["ms"] = (time.time() - t0) * 1000
    st.session_state.last = {"query": query, "res": res}
    st.session_state.feedback = None
    st.session_state.current_ticket = None
if "last" in st.session_state:
    render_result(st.session_state.last)

with st.sidebar:
    if st.session_state.tickets:
        st.divider()
        st.subheader("🎫 My tickets (this session)")
        for t in st.session_state.tickets:
            st.markdown(f"**{t['id']}** · {t['issue']}  \n_{t['status']}_")
