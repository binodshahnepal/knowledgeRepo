import os
import pandas as pd
import streamlit as st

from core.config import (
    APP_TITLE,
    APP_ICON,
    SUPPORTED_FILE_TYPES,
    DEFAULT_EMBEDDING_MODEL,
    VECTOR_DIR,
    UPLOAD_DIR,
    EXPORT_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    ensure_directories,
)
from core.helpers import (
    sanitize_filename,
    truncate_text,
    save_text_export,
    get_timestamp,
    safe_json_dumps,
)
from core.file_loaders import load_file_content
from core.splitter import split_text
from core.embeddings_store import VectorStore
from core.llm_service import LLMService
from core.db_service import DBService
from core.sql_guard import is_safe_select_query
from core.summarizer import summarize_file_metadata, dataframe_profile


ensure_directories()

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        .main > div {padding-top: 0.8rem;}
        .block-container {padding-top: 1rem; padding-bottom: 1rem;}
        .metric-card {
            background: #111827;
            padding: 16px;
            border-radius: 14px;
            border: 1px solid #2d3748;
            margin-bottom: 10px;
        }
        .source-card {
            border: 1px solid #2f2f2f;
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 10px;
            background: rgba(255,255,255,0.02);
        }
        .section-title {
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .small-note {
            color: #9ca3af;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "nav" not in st.session_state:
    st.session_state.nav = "Overview"

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore.load(VECTOR_DIR)

if "file_chat_history" not in st.session_state:
    st.session_state.file_chat_history = []

if "db_chat_history" not in st.session_state:
    st.session_state.db_chat_history = []

if "schema_info" not in st.session_state:
    st.session_state.schema_info = None

if "db_connection_string" not in st.session_state:
    st.session_state.db_connection_string = ""

if "file_summaries" not in st.session_state:
    st.session_state.file_summaries = []

llm = None
try:
    llm = LLMService()
except Exception as e:
    llm_error = str(e)
else:
    llm_error = None


with st.sidebar:
    st.title(f"{APP_ICON} Assistant")
    st.caption("Enterprise file + database Q&A")

    nav = st.radio(
        "Navigation",
        ["Overview", "File Knowledge Base", "Database Knowledge Base", "Diagnostics"]
    )
    st.session_state.nav = nav

    st.divider()
    st.subheader("Runtime Settings")
    st.write(f"Embedding model: `{DEFAULT_EMBEDDING_MODEL}`")
    st.write(f"Chunk size: `{CHUNK_SIZE}`")
    st.write(f"Chunk overlap: `{CHUNK_OVERLAP}`")
    st.write(f"Top-k retrieval: `{TOP_K}`")

    st.divider()
    st.subheader("Maintenance")

    if st.button("Clear File Chat", use_container_width=True):
        st.session_state.file_chat_history = []
        st.success("File chat cleared.")

    if st.button("Clear DB Chat", use_container_width=True):
        st.session_state.db_chat_history = []
        st.success("Database chat cleared.")

    if st.button("Reset Vector Index", use_container_width=True):
        for file_name in ["index.faiss", "documents.pkl", "meta.pkl"]:
            path = os.path.join(VECTOR_DIR, file_name)
            if os.path.exists(path):
                os.remove(path)
        st.session_state.vector_store = None
        st.success("Vector index reset.")

    if st.button("Clear Loaded Schema", use_container_width=True):
        st.session_state.schema_info = None
        st.success("Loaded schema cleared.")


st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("Upload files, build searchable knowledge, connect databases, and answer business questions.")

if llm_error:
    st.error(f"LLM initialization failed: {llm_error}")


def render_overview():
    st.subheader("Overview")

    col1, col2, col3 = st.columns(3)
    indexed_chunks = len(st.session_state.vector_store.documents) if st.session_state.vector_store else 0
    schema_tables = len(st.session_state.schema_info) if st.session_state.schema_info else 0

    with col1:
        st.metric("Indexed Chunks", indexed_chunks)
    with col2:
        st.metric("Schema Tables", schema_tables)
    with col3:
        st.metric("File Summaries", len(st.session_state.file_summaries))

    st.markdown("### What this app supports")
    st.write(
        """
- Document Q&A over uploaded files
- Spreadsheet and structured file understanding
- Database schema inspection
- Text-to-SQL for safe read-only queries
- Source-grounded answers
"""
    )

    st.markdown("### File support")
    st.write(", ".join(SUPPORTED_FILE_TYPES))
    st.markdown("### Important note")
    st.info(".bak files should be restored into SQL Server first, then queried from the Database Knowledge Base.")


def render_file_kb():
    st.subheader("File Knowledge Base")

    col_left, col_right = st.columns([1, 1.35], gap="large")

    with col_left:
        st.markdown('<div class="section-title">Upload and Build</div>', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload one or more files",
            type=SUPPORTED_FILE_TYPES,
            accept_multiple_files=True
        )

        if st.button("Build / Rebuild Knowledge Base", use_container_width=True):
            if not uploaded_files:
                st.warning("Please upload at least one file.")
            else:
                try:
                    all_chunks = []
                    summaries = []

                    progress = st.progress(0, text="Processing files...")

                    for idx, uploaded_file in enumerate(uploaded_files, start=1):
                        safe_name = sanitize_filename(uploaded_file.name)
                        save_path = os.path.join(UPLOAD_DIR, safe_name)

                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        content = load_file_content(save_path)
                        summaries.append(summarize_file_metadata(safe_name, content))

                        chunks = split_text(content, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
                        for chunk_id, chunk in enumerate(chunks, start=1):
                            all_chunks.append(
                                {
                                    "source": safe_name,
                                    "chunk_id": chunk_id,
                                    "text": chunk
                                }
                            )

                        progress.progress(idx / len(uploaded_files), text=f"Processed {safe_name}")

                    vector_store = VectorStore(DEFAULT_EMBEDDING_MODEL)
                    vector_store.build(all_chunks)
                    vector_store.save(VECTOR_DIR)

                    st.session_state.vector_store = vector_store
                    st.session_state.file_summaries = summaries

                    st.success(
                        f"Knowledge base built with {len(all_chunks)} chunks from {len(uploaded_files)} files."
                    )
                except Exception as e:
                    st.error(f"Build failed: {e}")

        st.markdown(
            '<p class="small-note">Use Database Knowledge Base for restored SQL Server .bak data.</p>',
            unsafe_allow_html=True
        )

        st.markdown("### File Summary Dashboard")
        if st.session_state.file_summaries:
            summary_df = pd.DataFrame(st.session_state.file_summaries)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.info("No file summaries available yet.")

        if st.session_state.file_summaries and llm:
            if st.button("Generate Overall File Summary", use_container_width=True):
                combined_preview = "\n\n".join(
                    [f"{item['file_name']}:\n{item['preview']}" for item in st.session_state.file_summaries]
                )
                try:
                    overall_summary = llm.summarize_text(combined_preview[:10000])
                    st.session_state.file_chat_history.append(
                        {"role": "assistant", "content": f"**Overall file summary**\n\n{overall_summary}"}
                    )
                    st.success("Overall summary added to file chat.")
                except Exception as e:
                    st.error(f"Summary generation failed: {e}")

    with col_right:
        st.markdown('<div class="section-title">Ask Questions</div>', unsafe_allow_html=True)

        for msg in st.session_state.file_chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("Ask a question from uploaded files...", key="file_chat_input_v3")

        if question:
            st.session_state.file_chat_history.append({"role": "user", "content": question})

            with st.chat_message("user"):
                st.markdown(question)

            if not llm:
                answer = "LLM service is not available. Please check your Gemini API key."
                results = []
            elif not st.session_state.vector_store:
                answer = "Please build or load the file knowledge base first."
                results = []
            else:
                try:
                    results = st.session_state.vector_store.search(question, top_k=TOP_K)
                    if not results:
                        answer = "I could not find relevant context in the indexed files."
                    else:
                        answer = llm.answer_from_context(question, results)
                except Exception as e:
                    answer = f"Error while answering the question: {e}"
                    results = []

            st.session_state.file_chat_history.append({"role": "assistant", "content": answer})

            with st.chat_message("assistant"):
                st.markdown(answer)

            if results:
                st.markdown("### Retrieved Sources")
                for item in results:
                    st.markdown(
                        f"""
<div class="source-card">
<b>Source:</b> {item['source']}<br>
<b>Chunk:</b> {item['chunk_id']}<br>
<b>Distance:</b> {round(item.get('distance', 0.0), 4)}
</div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.code(truncate_text(item["text"], 800))

        if st.session_state.file_chat_history:
            export_text = "\n\n".join(
                [f"{msg['role'].upper()}:\n{msg['content']}" for msg in st.session_state.file_chat_history]
            )
            st.download_button(
                label="Download File Chat",
                data=export_text,
                file_name=f"file_chat_{get_timestamp()}.txt",
                mime="text/plain",
                use_container_width=True
            )


def render_db_kb():
    st.subheader("Database Knowledge Base")

    col_left, col_right = st.columns([1, 1.35], gap="large")

    with col_left:
        st.markdown('<div class="section-title">Connection and Schema</div>', unsafe_allow_html=True)

        db_type = st.selectbox("Database Type", ["SQL Server", "PostgreSQL", "MySQL", "SQLite"])

        default_conn = ""
        if db_type == "SQL Server":
            default_conn = "mssql+pyodbc://username:password@server/database?driver=ODBC+Driver+17+for+SQL+Server"
        elif db_type == "PostgreSQL":
            default_conn = "postgresql+psycopg2://username:password@host:5432/database"
        elif db_type == "MySQL":
            default_conn = "mysql+pymysql://username:password@host:3306/database"
        elif db_type == "SQLite":
            default_conn = "sqlite:///example.db"

        connection_string = st.text_input(
            "Connection String",
            value=st.session_state.db_connection_string or default_conn
        )
        st.session_state.db_connection_string = connection_string

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Test Connection", use_container_width=True):
                try:
                    db = DBService(connection_string)
                    db.test_connection()
                    st.success("Connection successful.")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

        with c2:
            if st.button("Load Schema", use_container_width=True):
                try:
                    db = DBService(connection_string)
                    schema_info = db.get_schema_info()
                    st.session_state.schema_info = schema_info
                    st.success(f"Loaded {len(schema_info)} tables.")
                except Exception as e:
                    st.error(f"Schema loading failed: {e}")

        st.markdown(
            '<p class="small-note">SQL Server .bak files require restore-first workflow.</p>',
            unsafe_allow_html=True
        )

        if st.session_state.schema_info:
            st.markdown("### Schema Overview")
            schema_df = pd.DataFrame(
                [
                    {"table": table, "columns": len(cols)}
                    for table, cols in st.session_state.schema_info.items()
                ]
            )
            st.dataframe(schema_df, use_container_width=True, hide_index=True)

            with st.expander("Full Schema JSON"):
                st.code(safe_json_dumps(st.session_state.schema_info), language="json")

    with col_right:
        st.markdown('<div class="section-title">Ask Database Questions</div>', unsafe_allow_html=True)

        for msg in st.session_state.db_chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        db_question = st.chat_input("Ask a business question about your database...", key="db_chat_input_v3")

        if db_question:
            st.session_state.db_chat_history.append({"role": "user", "content": db_question})

            with st.chat_message("user"):
                st.markdown(db_question)

            if not llm:
                final_answer = "LLM service is not available. Please check your Gemini API key."
                result_df = None

            elif not st.session_state.schema_info:
                final_answer = "Please load the database schema first."
                result_df = None

            else:
                try:
                    db = DBService(connection_string)
                    schema_text = DBService.schema_to_text(st.session_state.schema_info)
                    generated_sql = llm.generate_sql(db_question, schema_text)

                    safe, reason = is_safe_select_query(generated_sql)
                    if not safe:
                        raise ValueError(f"Generated SQL blocked: {reason}")

                    result_df = db.run_select_query(generated_sql)

                    if len(result_df) > 500:
                        result_df = result_df.head(500)

                    preview_text = (
                        result_df.head(20).to_csv(index=False)
                        if not result_df.empty
                        else "No rows returned."
                    )
                    explanation = llm.explain_table_result(db_question, preview_text)

                    final_answer = (
                        f"**Generated SQL**\n"
                        f"```sql\n{generated_sql}\n```\n\n"
                        f"**Explanation**\n{explanation}"
                    )

                except Exception as e:
                    final_answer = f"Database query failed: {e}"
                    result_df = None

            st.session_state.db_chat_history.append({"role": "assistant", "content": final_answer})

            with st.chat_message("assistant"):
                st.markdown(final_answer)

            if result_df is not None:
                st.markdown("### Query Result")
                st.dataframe(result_df, use_container_width=True)

                profile = dataframe_profile(result_df)
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows", profile["rows"])
                c2.metric("Columns", profile["columns"])
                c3.metric("Visible Preview Limit", len(result_df))

                csv_data = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Result as CSV",
                    data=csv_data,
                    file_name=f"db_result_{get_timestamp()}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        if st.session_state.db_chat_history:
            export_text = "\n\n".join(
                [f"{msg['role'].upper()}:\n{msg['content']}" for msg in st.session_state.db_chat_history]
            )
            st.download_button(
                label="Download DB Chat",
                data=export_text,
                file_name=f"db_chat_{get_timestamp()}.txt",
                mime="text/plain",
                use_container_width=True
            )


def render_diagnostics():
    st.subheader("Diagnostics")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### File Index Status")
        if st.session_state.vector_store:
            st.success("Vector store loaded.")
            st.write(f"Indexed documents: {len(st.session_state.vector_store.documents)}")
        else:
            st.info("No vector store loaded.")

        st.markdown("### LLM Status")
        if llm:
            st.success("LLM service initialized.")
        else:
            st.error("LLM service not initialized.")

    with c2:
        st.markdown("### Database Schema Status")
        if st.session_state.schema_info:
            st.success(f"Schema loaded with {len(st.session_state.schema_info)} tables.")
        else:
            st.info("No schema loaded.")

        st.markdown("### Upload Folder")
        uploaded_files = os.listdir(UPLOAD_DIR) if os.path.exists(UPLOAD_DIR) else []
        st.write(f"Stored uploads: {len(uploaded_files)}")
        if uploaded_files:
            st.write(uploaded_files[:20])


if st.session_state.nav == "Overview":
    render_overview()
elif st.session_state.nav == "File Knowledge Base":
    render_file_kb()
elif st.session_state.nav == "Database Knowledge Base":
    render_db_kb()
elif st.session_state.nav == "Diagnostics":
    render_diagnostics()