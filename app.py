"""
Streamlit UI for the Adaptive RAG Academic Chatbot.
"""

import os
import time
from html import escape
from datetime import datetime
from typing import Any, Dict, List

import streamlit as st

from config import EMBEDDING_MODEL, OLLAMA_MODEL
from src.pipeline import RAGPipeline


APP_TITLE = "Adaptive RAG Academic Chatbot"
UPLOAD_DIR = "data"
PIPELINE_CACHE_VERSION = "2026-06-27-adaptive-dashboard-v2"


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_pipeline(version: str) -> RAGPipeline:
    """Load the reusable RAG pipeline once per Streamlit session."""

    _ = version
    return RAGPipeline()


pipeline = load_pipeline(PIPELINE_CACHE_VERSION)


def initialize_state() -> None:
    """Initialize Streamlit session state."""

    defaults = {
        "database_ready": False,
        "uploaded_signature": None,
        "build_stats": {
            "pages": 0,
            "chunks": 0,
            "embeddings": 0,
            "vectors": 0,
            "files": 0,
            "indexing_time": 0.0
        },
        "processing_logs": [],
        "chat_history": [],
        "last_result": None,
        "preview_page": None,
        "dark_mode": False
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if not st.session_state.database_ready:
        db_stats = pipeline.get_database_stats()

        if db_stats["vector_count"] > 0:
            st.session_state.database_ready = True
            st.session_state.build_stats = {
                "pages": db_stats["page_count"],
                "chunks": db_stats["metadata_count"],
                "embeddings": db_stats["metadata_count"],
                "vectors": db_stats["vector_count"],
                "files": 0,
                "indexing_time": 0.0
            }


def apply_theme(dark_mode: bool) -> None:
    """Inject app-specific CSS."""

    if dark_mode:
        background = "#0f172a"
        surface = "#111827"
        soft_surface = "#1f2937"
        text = "#e5e7eb"
        muted = "#9ca3af"
        border = "#334155"
        card_shadow = "0 18px 45px rgba(0, 0, 0, 0.28)"
    else:
        background = "#f6f9fc"
        surface = "#ffffff"
        soft_surface = "#eef5ff"
        text = "#0f172a"
        muted = "#64748b"
        border = "#dbe7f3"
        card_shadow = "0 14px 36px rgba(15, 23, 42, 0.08)"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {background};
            color: {text};
            font-family: Inter, "Segoe UI", sans-serif;
        }}
        [data-testid="stSidebar"] {{
            background: {surface};
            border-right: 1px solid {border};
        }}
        .main .block-container {{
            padding-top: 2rem;
            max-width: 1280px;
        }}
        .app-hero {{
            padding: 1.25rem 0 1.4rem 0;
        }}
        .app-kicker {{
            color: #2563eb;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}
        .app-title {{
            color: {text};
            font-size: clamp(2rem, 4vw, 3.2rem);
            line-height: 1.05;
            font-weight: 850;
            margin: 0.25rem 0 0.5rem 0;
        }}
        .app-subtitle {{
            color: {muted};
            font-size: 1.1rem;
            max-width: 760px;
        }}
        .card {{
            background: {surface};
            border: 1px solid {border};
            border-radius: 8px;
            box-shadow: {card_shadow};
            padding: 1.1rem 1.2rem;
            margin-bottom: 1rem;
        }}
        .small-card {{
            background: {surface};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 1rem;
            min-height: 112px;
            box-shadow: 0 8px 24px rgba(37, 99, 235, 0.06);
        }}
        .metric-label {{
            color: {muted};
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .metric-value {{
            color: {text};
            font-size: 1.55rem;
            font-weight: 820;
            margin-top: 0.35rem;
        }}
        .answer-card {{
            background: linear-gradient(180deg, {surface}, {soft_surface});
            border: 1px solid {border};
            border-left: 5px solid #2563eb;
            border-radius: 8px;
            box-shadow: {card_shadow};
            padding: 1.25rem;
            margin-top: 1rem;
        }}
        .answer-card h3 {{
            color: {text};
            margin: 0 0 0.75rem 0;
        }}
        .answer-text {{
            color: {text};
            line-height: 1.7;
            font-size: 1rem;
            white-space: pre-wrap;
        }}
        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            border-radius: 999px;
            padding: 0.35rem 0.7rem;
            font-size: 0.83rem;
            font-weight: 750;
        }}
        .status-ready {{
            background: #dcfce7;
            color: #166534;
        }}
        .status-waiting {{
            background: #dbeafe;
            color: #1d4ed8;
        }}
        .confidence-high {{
            background: #dcfce7;
            color: #166534;
            border: 1px solid #86efac;
        }}
        .confidence-medium {{
            background: #ffedd5;
            color: #9a3412;
            border: 1px solid #fdba74;
        }}
        .confidence-low {{
            background: #fee2e2;
            color: #991b1b;
            border: 1px solid #fca5a5;
        }}
        .sidebar-logo {{
            width: 48px;
            height: 48px;
            border-radius: 8px;
            display: grid;
            place-items: center;
            background: #2563eb;
            color: white;
            font-size: 1.5rem;
            font-weight: 850;
            margin-bottom: 0.6rem;
        }}
        .sidebar-title {{
            color: {text};
            font-size: 1.15rem;
            font-weight: 850;
            line-height: 1.2;
            margin-bottom: 1rem;
        }}
        .example-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}
        .example-chip {{
            border: 1px solid {border};
            border-radius: 999px;
            color: #1d4ed8;
            background: {surface};
            padding: 0.45rem 0.7rem;
            font-size: 0.85rem;
            font-weight: 700;
        }}
        textarea {{
            border-radius: 8px !important;
        }}
        div[data-testid="stMetric"] {{
            background: {surface};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 0.9rem;
            box-shadow: 0 8px 24px rgba(37, 99, 235, 0.05);
        }}
        .stButton > button {{
            border-radius: 8px;
            border: 1px solid #bfdbfe;
            color: #1d4ed8;
            background: {surface};
            font-weight: 750;
        }}
        .stButton > button:hover {{
            border-color: #2563eb;
            color: #1d4ed8;
            background: #eff6ff;
        }}
        .stButton > button[kind="primary"],
        button[data-testid="stBaseButton-primary"] {{
            border-color: #2563eb;
            background: #2563eb;
            color: #ffffff;
        }}
        .stButton > button[kind="primary"]:hover,
        button[data-testid="stBaseButton-primary"]:hover {{
            border-color: #1d4ed8;
            background: #1d4ed8;
            color: #ffffff;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def add_log(message: str) -> None:
    """Add a timestamped processing log entry."""

    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.processing_logs.append(f"[{timestamp}] {message}")


def uploaded_signature(files: List[Any]) -> str:
    """Create a lightweight signature for uploaded files."""

    return "|".join(
        f"{file.name}:{file.size}"
        for file in files
    )


def save_uploaded_files(files: List[Any]) -> List[str]:
    """Persist uploaded PDFs and return local paths."""

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    paths = []

    for file in files:
        path = os.path.join(UPLOAD_DIR, file.name)

        with open(path, "wb") as output:
            output.write(file.getbuffer())

        paths.append(path)

    return paths


def confidence_class(level: str) -> str:
    """Map confidence level to a CSS class."""

    level = level.lower()

    if level == "high":
        return "confidence-high"

    if level == "medium":
        return "confidence-medium"

    return "confidence-low"


def render_metric_card(label: str, value: Any) -> None:
    """Render a compact metric card."""

    st.markdown(
        f"""
        <div class="small-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_sidebar() -> List[Any]:
    """Render the sidebar and return uploaded files."""

    with st.sidebar:
        st.markdown('<div class="sidebar-logo">A</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-title">Adaptive RAG<br>Academic Chatbot</div>',
            unsafe_allow_html=True
        )

        st.session_state.dark_mode = st.toggle(
            "Dark mode",
            value=st.session_state.dark_mode
        )

        uploaded_files = st.file_uploader(
            "Upload academic PDFs",
            type=["pdf"],
            accept_multiple_files=True
        )

        current_signature = uploaded_signature(uploaded_files) if uploaded_files else None
        force_rebuild = st.checkbox("Force rebuild", value=False)

        build_disabled = not uploaded_files

        if st.button(
            "Build / Rebuild Database",
            type="primary",
            disabled=build_disabled,
            use_container_width=True
        ):
            if not uploaded_files:
                st.warning("Upload at least one PDF before building the database.")
            elif (
                st.session_state.database_ready
                and current_signature == st.session_state.uploaded_signature
                and not force_rebuild
            ):
                st.info("This PDF set is already indexed. Enable force rebuild to rebuild it.")
            else:
                st.session_state.processing_logs = []

                try:
                    paths = save_uploaded_files(uploaded_files)
                    progress_bar = st.progress(0)
                    status = st.empty()

                    def update_progress(stage: str, progress: float, message: str) -> None:
                        progress_bar.progress(progress)
                        status.info(message)
                        add_log(f"{stage.title()}: {message}")

                    with st.spinner("Indexing academic documents..."):
                        stats = pipeline.build_database(
                            paths,
                            progress_callback=update_progress
                        )

                    st.session_state.database_ready = True
                    st.session_state.uploaded_signature = current_signature
                    st.session_state.build_stats = stats
                    add_log(
                        f"Ready: indexed {stats['chunks']} chunks in "
                        f"{stats['indexing_time']} seconds"
                    )
                    st.success("Database ready.")
                except Exception as exc:
                    st.session_state.database_ready = False
                    st.error(f"Could not build the database: {exc}")
                    add_log(f"Error: {exc}")

        if st.session_state.database_ready:
            st.markdown(
                '<span class="status-pill status-ready">Ready</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="status-pill status-waiting">Waiting for PDFs</span>',
                unsafe_allow_html=True
            )

        st.divider()
        st.caption("Model Information")
        st.write(f"Embedding: `{EMBEDDING_MODEL}`")
        st.write(f"LLM: `{OLLAMA_MODEL}`")

        stats = st.session_state.build_stats
        st.caption("Database Status")
        st.metric("Total Pages", stats["pages"])
        st.metric("Total Chunks", stats["chunks"])
        st.metric("Vector Count", stats["vectors"])

        if stats["indexing_time"]:
            st.write(f"Indexing Time: `{stats['indexing_time']} sec`")

        with st.expander("Processing Logs", expanded=False):
            if st.session_state.processing_logs:
                for entry in st.session_state.processing_logs[-12:]:
                    st.text(entry)
            else:
                st.caption("No processing logs yet.")

        if st.button("Clear Database", use_container_width=True):
            pipeline.clear_database()
            st.session_state.database_ready = False
            st.session_state.build_stats = {
                "pages": 0,
                "chunks": 0,
                "embeddings": 0,
                "vectors": 0,
                "files": 0,
                "indexing_time": 0.0
            }
            st.session_state.uploaded_signature = None
            st.session_state.last_result = None
            add_log("Cleared vector database")
            st.success("Database cleared.")

    return uploaded_files


def render_examples() -> None:
    """Render example question buttons."""

    examples = [
        "What are the Program Outcomes?",
        "Explain Course Objectives.",
        "What is the syllabus for AI?",
        "What are the COs of DBMS?"
    ]

    st.markdown('<div class="example-row">', unsafe_allow_html=True)
    cols = st.columns(4)

    for index, example in enumerate(examples):
        if cols[index].button(example, use_container_width=True):
            st.session_state.question = example

    st.markdown("</div>", unsafe_allow_html=True)


def render_stats(result: Dict[str, Any] | None) -> None:
    """Render the statistics dashboard."""

    build_stats = st.session_state.build_stats

    metric_values = [
        ("Pages", build_stats["pages"]),
        ("Chunks", build_stats["chunks"]),
        ("Embeddings", build_stats["embeddings"]),
        (
            "Retrieved",
            result["retrieved_chunks"] if result else 0
        ),
        (
            "Selected",
            result["selected_chunks"] if result else 0
        ),
        (
            "Response",
            f"{result['response_time']} sec" if result else "0 sec"
        ),
        (
            "Model",
            result["model"] if result else OLLAMA_MODEL
        ),
        ("Embedding", EMBEDDING_MODEL.split("/")[-1])
    ]

    cols = st.columns(4)

    for index, (label, value) in enumerate(metric_values):
        with cols[index % 4]:
            render_metric_card(label, value)


def answer_to_text(result: Dict[str, Any]) -> str:
    """Convert an answer result into a downloadable text report."""

    citations = result["citations"]["citation_text"]
    confidence = result["confidence"]

    return f"""Adaptive RAG Academic Chatbot

Question:
{result["question"]}

Answer:
{result["answer"]}

Supporting Pages:
{citations}

Confidence:
{confidence["level"]} ({confidence["score"]:.4f})

LLM:
{result["model"]}

Response Time:
{result["response_time"]} sec
"""


def render_answer(result: Dict[str, Any]) -> None:
    """Render the answer, citations, confidence, and context."""

    confidence = result["confidence"]
    level = confidence["level"]
    css_class = confidence_class(level)
    safe_answer = escape(result["answer"])

    st.markdown(
        f"""
        <div class="answer-card">
            <h3>Answer</h3>
            <div class="answer-text">{safe_answer}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    meta_cols = st.columns(5)
    meta_cols[0].metric("LLM Used", result["model"])
    meta_cols[1].metric("Response Time", f"{result['response_time']} sec")
    meta_cols[2].metric("Retrieved", result["retrieved_chunks"])
    meta_cols[3].metric("Selected", result["selected_chunks"])
    meta_cols[4].metric(
        "Question Type",
        result["retrieval_policy"]["query_type"].title()
    )

    citation_col, confidence_col = st.columns([1.2, 0.8])

    with citation_col:
        st.markdown("#### Citation Card")
        pages = result["citations"]["pages"]

        if pages:
            page_cols = st.columns(min(len(pages), 4))

            for index, page in enumerate(pages):
                if page_cols[index % len(page_cols)].button(
                    f"Page {page}",
                    key=f"page-{page}-{index}"
                ):
                    st.session_state.preview_page = page
        else:
            st.info("No citations available.")

        st.caption(result["citations"]["citation_text"])

    with confidence_col:
        st.markdown("#### Confidence")
        st.markdown(
            f"""
            <div class="card {css_class}">
                <div class="metric-label">{level}</div>
                <div class="metric-value">{confidence["score"]:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.download_button(
        "Download Answer as TXT",
        data=answer_to_text(result),
        file_name="adaptive_rag_answer.txt",
        mime="text/plain"
    )

    with st.expander("Retrieved Context", expanded=False):
        for rank, chunk in enumerate(result["selected_context"], start=1):
            source = chunk.get("source_file", "Uploaded PDF")
            safe_text = escape(chunk["text"])

            st.markdown(
                f"""
                <div class="card">
                    <div class="metric-label">
                        Rank {rank} | Page {chunk['page_number']} |
                        Score {chunk['score']:.4f} | {escape(source)}
                    </div>
                    <div class="answer-text">{safe_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    if st.session_state.preview_page is not None:
        st.markdown("#### Page Preview")
        preview_chunks = [
            chunk
            for chunk in result["selected_context"]
            if chunk.get("page_number") == st.session_state.preview_page
        ]

        if preview_chunks:
            for chunk in preview_chunks:
                st.info(chunk["text"])
        else:
            st.warning("The selected page is not part of the current retrieved context.")


def render_chat_history() -> None:
    """Render session chat history."""

    with st.expander("Chat History", expanded=False):
        if not st.session_state.chat_history:
            st.caption("No questions asked yet.")
            return

        for item in reversed(st.session_state.chat_history):
            st.markdown(f"**Q:** {item['question']}")
            st.caption(
                f"{item['confidence']['level']} confidence | "
                f"{item['response_time']} sec | {item['model']}"
            )
            st.write(item["answer"])
            st.divider()


def main() -> None:
    """Render the complete application."""

    initialize_state()
    apply_theme(st.session_state.dark_mode)
    render_sidebar()

    st.markdown(
        """
        <section class="app-hero">
            <div class="app-kicker">Academic Question Answering System</div>
            <h1 class="app-title">Adaptive RAG Academic Chatbot</h1>
            <p class="app-subtitle">
                Ask questions from your uploaded academic documents.
            </p>
        </section>
        """,
        unsafe_allow_html=True
    )

    render_examples()

    if "question" not in st.session_state:
        st.session_state.question = ""

    question = st.text_area(
        "Ask a question",
        key="question",
        height=118,
        placeholder="Example: What are the Program Outcomes?"
    )

    action_col, clear_col = st.columns([0.22, 0.78])
    ask_clicked = action_col.button(
        "Ask",
        type="primary",
        use_container_width=True
    )

    if clear_col.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.last_result = None
        st.success("Chat history cleared.")

    if ask_clicked:
        if not st.session_state.database_ready:
            st.error("No database found. Upload PDFs and build the database first.")
        elif not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Retrieving context and generating answer..."):
                    start = time.time()
                    result = pipeline.ask(question.strip())
                    result["answering_time"] = round(time.time() - start, 2)

                st.session_state.last_result = result
                st.session_state.chat_history.append(result)
            except RuntimeError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not generate an answer: {exc}")

    render_stats(st.session_state.last_result)

    if st.session_state.last_result:
        render_answer(st.session_state.last_result)
    else:
        st.info("Build a database, then ask a question to see grounded answers, citations, and confidence.")

    render_chat_history()


if __name__ == "__main__":
    main()
