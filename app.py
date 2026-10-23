"""
Streamlit Web Application for Project 3: Gas Turbine Maintenance Assistant.
Grounded RAG Assistant over GE Document GER-3620 (Operating & Maintenance Considerations).
"""

import os
import streamlit as st
from pathlib import Path

from config import (
    PDF_PATH,
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    DEFAULT_PDF_NAME,
    TOP_K,
    NOT_FOUND_RESPONSE,
)
from rag_pipeline import query_rag, get_vectorstore

# Page Configuration
st.set_page_config(
    page_title="Gas Turbine Maintenance Assistant",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, professional styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .status-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 1.2rem;
    }
    .answer-box {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #0284c7;
        border-radius: 8px;
        padding: 18px 22px;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .not-found-box {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        border-left: 5px solid #e11d48;
        border-radius: 8px;
        padding: 18px 22px;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        font-weight: 600;
        color: #9f1239;
    }
    .source-tag {
        display: inline-block;
        background: #e0f2fe;
        color: #0369a1;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .evidence-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 10px;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def check_database_status() -> tuple[bool, int, str]:
    """Checks whether ChromaDB exists and has indexed chunks."""
    if not CHROMA_DIR.exists():
        return False, 0, "Vector database directory missing."
    try:
        store = get_vectorstore()
        data = store.get()
        count = len(data["ids"])
        if count == 0:
            return False, 0, "Vector store is empty."
        return True, count, "Ready"
    except Exception as e:
        return False, 0, str(e)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/General_Electric_logo.svg/320px-General_Electric_logo.svg.png",
        width=60,
    )
    st.title("Settings & Info")
    st.markdown("---")

    # Ingestion Status
    db_ok, chunk_count, status_msg = check_database_status()
    st.markdown("### 📊 Knowledge Base Status")
    if db_ok:
        st.success(f"● **Vector Store Active**\n\nIndexed Chunks: `{chunk_count}`")
    else:
        st.error(f"⚠️ **Vector Store Inactive**\n\n{status_msg}")
        st.info("Run `python ingest.py` to index the PDF document.")

    st.caption(f"**Document:** `{DEFAULT_PDF_NAME}` (60 pp)")
    st.caption(f"**Embeddings:** `{EMBEDDING_MODEL_NAME}` (Local)")

    st.markdown("---")
    st.markdown("### 🤖 LLM Configuration")

    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Optional: Enter key to use live Google Gemini Flash. If empty, the assistant uses deterministic grounded extraction.",
    )

    model_choice = st.selectbox(
        "Model Selection",
        options=["gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash"],
        index=0,
    )

    top_k_val = st.slider("Retrieved Chunks (Top-K)", min_value=2, max_value=8, value=TOP_K)

    st.markdown("---")
    st.markdown(
        """
        **About this Project:**  
        Project 3 for GenAI Course.  
        Grounds responses strictly on **GE GER-3620 Rev P** (Heavy-Duty Gas Turbine Operating & Maintenance Considerations).  
        Returns **`Not found`** whenever sufficient factual evidence is absent.
        """
    )


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown("<div class='main-title'>⚙️ Gas Turbine Maintenance Assistant</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Ask technical operating and maintenance questions grounded in <b>GE GER-3620 Rev P</b>.</div>",
    unsafe_allow_html=True,
)

if not PDF_PATH.exists():
    st.error(
        f"🚨 Target technical PDF not found at `{PDF_PATH}`. "
        "Please place `GER-3620-P.pdf` in the `data/` folder and run `python ingest.py`."
    )

# ---------------------------------------------------------
# Benchmark Example Questions
# ---------------------------------------------------------
st.markdown("##### 💡 Select or Copy Example Benchmark Questions:")
example_questions = [
    "What are the three main types of inspection?",
    "What does a maintenance factor of two mean?",
    "What is the purchase price of this gas turbine?",
    "What is the melting point of titanium?",
    "What is the recommended combustion-inspection interval in hours?",
]

# Quick Selection Buttons
col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
selected_prompt = None

if col_b1.button("📋 Question 1", help=example_questions[0]):
    selected_prompt = example_questions[0]
if col_b2.button("📋 Question 2", help=example_questions[1]):
    selected_prompt = example_questions[1]
if col_b3.button("❓ Question 3 (OOD)", help=example_questions[2]):
    selected_prompt = example_questions[2]
if col_b4.button("❓ Question 4 (OOD)", help=example_questions[3]):
    selected_prompt = example_questions[3]
if col_b5.button("📋 Question 5", help=example_questions[4]):
    selected_prompt = example_questions[4]

# Initialize session state for user question
if "user_question" not in st.session_state:
    st.session_state.user_question = example_questions[0]

if selected_prompt:
    st.session_state.user_question = selected_prompt

# ---------------------------------------------------------
# Question Input Form
# ---------------------------------------------------------
with st.form("query_form"):
    user_query = st.text_input(
        "Enter your maintenance or operating question:",
        value=st.session_state.user_question,
        placeholder="e.g. What are the three main types of inspection?",
    )
    submit_button = st.form_submit_button("🔍 Ask Assistant", use_container_width=True)

# ---------------------------------------------------------
# Execution & Result Display
# ---------------------------------------------------------
if submit_button and user_query:
    if not db_ok:
        st.error("ChromaDB is not ready. Please run `python ingest.py` first.")
    else:
        with st.spinner("Analyzing GER-3620 technical document..."):
            result = query_rag(
                question=user_query,
                top_k=top_k_val,
                api_key=api_key_input if api_key_input.strip() else None,
                model_name=model_choice,
            )

        st.markdown("### 💬 Assistant Response")

        answer_text = result["answer"]
        is_found = result["found"]

        if not is_found or answer_text.strip() == NOT_FOUND_RESPONSE:
            st.markdown(
                f"<div class='not-found-box'>❌ <b>{NOT_FOUND_RESPONSE}</b><br>"
                f"<span style='font-size:0.88rem; font-weight:normal; color:#475569;'>"
                f"The requested information is not established or supported by the available evidence in GER-3620."
                f"</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='answer-box'>{answer_text}</div>",
                unsafe_allow_html=True,
            )

        # Citations / Sources
        sources = result.get("sources", [])
        if sources:
            st.markdown("#### 📑 Grounded Citations")
            badge_html = ""
            for s in sources:
                doc_p = s.get("doc_page", "N/A")
                pdf_p = s.get("page", "N/A")
                sec = s.get("section", "General")
                badge_html += f"<span class='source-tag'>Doc Page {doc_p} (PDF p. {pdf_p}) • {sec}</span>"
            st.markdown(badge_html, unsafe_allow_html=True)

        # Expandable Evidence Section
        chunks = result.get("retrieved_chunks", [])
        with st.expander(f"🔎 View Retrieved Evidence ({len(chunks)} Chunks)", expanded=False):
            if not chunks:
                st.write("No excerpts retrieved.")
            for i, c in enumerate(chunks, 1):
                meta = c.get("metadata", {})
                dist = c.get("distance", "N/A")
                st.markdown(
                    f"<div class='evidence-card'>"
                    f"<b>Chunk {i}</b> | "
                    f"<b>Doc Page:</b> {meta.get('doc_page')} | "
                    f"<b>PDF Page:</b> {meta.get('page')} | "
                    f"<b>Section:</b> {meta.get('section')} | "
                    f"<b>Distance:</b> {dist}"
                    f"<pre style='white-space: pre-wrap; font-size: 0.85rem; margin-top: 6px;'>"
                    f"{c.get('content')}"
                    f"</pre>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.caption(f"Pipeline Engine: `{result.get('llm_used')}`")
