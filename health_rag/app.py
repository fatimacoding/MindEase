import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindEase · Mental Health Assistant",
    page_icon="🧠",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

/* Root variables */
:root {
    --bg: #0f1117;
    --card: #1a1d27;
    --accent: #7c9ff5;
    --accent2: #a78bfa;
    --text: #e8eaf2;
    --muted: #8b8fa8;
    --user-bubble: #1e2235;
    --bot-bubble: #151929;
    --border: rgba(124,159,245,0.15);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 780px; }

/* Hero header */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, #0f1117 0%, #13162a 100%);
    border-radius: 20px;
    border: 1px solid var(--border);
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(124,159,245,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; right: -40px;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(167,139,250,0.10) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    background: linear-gradient(135deg, #7c9ff5, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem;
    position: relative; z-index: 1;
}
.hero p {
    color: var(--muted);
    font-size: 0.95rem;
    font-weight: 300;
    margin: 0;
    position: relative; z-index: 1;
}

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(124,159,245,0.08);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    color: var(--accent);
    margin-top: 0.8rem;
    position: relative; z-index: 1;
}
.dot {
    width: 7px; height: 7px;
    background: #4ade80;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}

/* Chat messages */
.msg-wrapper { margin-bottom: 1rem; }

.msg-user {
    display: flex;
    justify-content: flex-end;
    align-items: flex-end;
    gap: 8px;
}
.msg-bot {
    display: flex;
    justify-content: flex-start;
    align-items: flex-start;
    gap: 8px;
}

.bubble-user {
    background: linear-gradient(135deg, #2a3256, #1e2847);
    border: 1px solid rgba(124,159,245,0.2);
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    max-width: 78%;
    font-size: 0.9rem;
    line-height: 1.6;
    color: var(--text);
}

.bubble-bot {
    background: var(--bot-bubble);
    border: 1px solid var(--border);
    border-radius: 18px 18px 18px 4px;
    padding: 14px 18px;
    max-width: 82%;
    font-size: 0.9rem;
    line-height: 1.7;
    color: var(--text);
}

.avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
}
.avatar-bot {
    background: linear-gradient(135deg, #7c9ff5, #a78bfa);
    color: white;
}
.avatar-user {
    background: rgba(124,159,245,0.15);
    border: 1px solid var(--border);
    color: var(--accent);
}

/* Source box */
.source-box {
    margin-top: 8px;
    padding: 8px 12px;
    background: rgba(167,139,250,0.06);
    border-left: 3px solid var(--accent2);
    border-radius: 0 8px 8px 0;
    font-size: 0.78rem;
    color: var(--muted);
}
.source-box strong { color: var(--accent2); }

/* Input area */
.stTextInput > div > div > input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(124,159,245,0.12) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c9ff5, #a78bfa) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.4rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Disclaimer */
.disclaimer {
    background: rgba(251,191,36,0.06);
    border: 1px solid rgba(251,191,36,0.18);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.78rem;
    color: #a0916b;
    margin-bottom: 1.2rem;
}

/* Divider */
hr { border-color: var(--border); }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyAgx7vFbXu28oaE9o3uP_ylT6QWBriFwOE"
KAGGLE_CSV_URL = (
    "https://github.com/kqin050/Chatbot/raw/refs/heads/main/Mental_Health_FAQ.csv"
)
# Mirrors the Kaggle "Mental Health FAQ for Chatbot" dataset
# (kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3

genai.configure(api_key=GEMINI_API_KEY)

# ── Data & index loading (cached) ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rag_system():
    """Load dataset, build embeddings, build FAISS index."""
    # 1. Load dataset
    r = requests.get(KAGGLE_CSV_URL, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))

    # Normalise column names (the CSV has 'Questions' and 'Answers')
    df.columns = [c.strip() for c in df.columns]
    q_col = [c for c in df.columns if "question" in c.lower()][0]
    a_col = [c for c in df.columns if "answer" in c.lower()][0]
    df = df[[q_col, a_col]].dropna()
    df.columns = ["question", "answer"]

    # 2. Build documents = "Q: ... A: ..." chunks
    docs = (
        df["question"].astype(str) + "\n" + df["answer"].astype(str)
    ).tolist()

    # 3. Embed
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = embedder.encode(docs, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    # 4. FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return df, docs, embedder, index

# ── RAG query ─────────────────────────────────────────────────────────────────
def retrieve(query: str, embedder, index, docs, top_k=TOP_K):
    q_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)
    scores, idxs = index.search(q_vec, top_k)
    return [(docs[i], float(scores[0][j])) for j, i in enumerate(idxs[0])]


def generate_answer(query: str, context_chunks: list) -> str:
    context = "\n\n---\n\n".join([c for c, _ in context_chunks])
    prompt = f"""You are MindEase, a compassionate and knowledgeable mental health support assistant.
Answer the user's question using ONLY the information from the provided context below.
Be warm, empathetic, and supportive. If the context does not contain enough information, 
say so gently and suggest seeking professional help.
Do NOT make up information outside the context.

CONTEXT:
{context}

USER QUESTION:
{query}

YOUR ANSWER:"""
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ready" not in st.session_state:
    st.session_state.ready = False

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🧠 MindEase</h1>
    <p>Your compassionate mental health companion — powered by AI & evidence-based FAQs</p>
    <span class="status-badge"><span class="dot"></span>RAG System Active</span>
</div>
""", unsafe_allow_html=True)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
⚠️ <strong>Disclaimer:</strong> MindEase is an educational AI prototype and does not replace professional mental health care. 
If you are in crisis, please contact a licensed professional or a crisis helpline immediately.
</div>
""", unsafe_allow_html=True)

# ── Load system ───────────────────────────────────────────────────────────────
with st.spinner("🔄 Loading knowledge base and building vector index…"):
    try:
        df, docs, embedder, index = load_rag_system()
        st.session_state.ready = True
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
<div class="msg-wrapper msg-user">
    <div class="bubble-user">{msg["content"]}</div>
    <div class="avatar avatar-user">👤</div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class="msg-wrapper msg-bot">
    <div class="avatar avatar-bot">🧠</div>
    <div>
        <div class="bubble-bot">{msg["content"]}</div>
        {f'<div class="source-box"><strong>📚 Sources used:</strong> {msg["sources"]}</div>' if msg.get("sources") else ""}
    </div>
</div>""", unsafe_allow_html=True)

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input(
        label="Ask a question",
        placeholder="e.g. What is anxiety? How can I manage stress?",
        label_visibility="collapsed",
        key="input_box"
    )
with col2:
    send = st.button("Send →")

# ── Suggested questions ───────────────────────────────────────────────────────
st.markdown("<p style='color:#8b8fa8;font-size:0.78rem;margin-top:0.5rem'>💡 Try asking:</p>", unsafe_allow_html=True)
suggestions = [
    "What is depression?",
    "How to manage anxiety?",
    "What is PTSD?",
    "How does therapy help?",
]
cols = st.columns(len(suggestions))
for i, s in enumerate(suggestions):
    if cols[i].button(s, key=f"sug_{i}"):
        user_input = s
        send = True

# ── Process ───────────────────────────────────────────────────────────────────
if send and user_input.strip():
    query = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("MindEase is thinking…"):
        try:
            chunks = retrieve(query, embedder, index, docs)
            answer = generate_answer(query, chunks)
            # Build source summary (first 60 chars of each chunk)
            src_labels = " | ".join([f"\"{c[:55].strip()}…\"" for c, _ in chunks])
        except Exception as e:
            answer = f"I'm sorry, I encountered an error: {e}"
            src_labels = ""

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": src_labels,
    })
    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#8b8fa8;font-size:0.72rem;margin-top:2rem;padding-top:1rem;border-top:1px solid rgba(124,159,245,0.1)'>
    MindEase · CCAI 435 Deep Learning Project · University of Jeddah<br>
    Dataset: <a href='https://www.kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot' 
    style='color:#7c9ff5;text-decoration:none' target='_blank'>Mental Health FAQ (Kaggle)</a> · 
    Powered by Gemini 1.5 Flash + FAISS + Sentence Transformers
</div>
""", unsafe_allow_html=True)
