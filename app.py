import streamlit as st
import time
from datetime import datetime
import database
from vector_store import VectorStore
from ingestion import parse_document
from chunking import process_extracted_pages
from llm import stream_grounded_answer

# 1. Page Configuration
st.set_page_config(
    page_title="UniMind | Intelligent Campus Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern UI & SaaS Theme System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background-color: #0B0F17;
    }

    /* Top Hero Banner */
    .hero-wrapper {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.2rem 0 1.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }

    .live-status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }

    .status-pulse {
        width: 7px;
        height: 7px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10B981;
    }

    /* Metric & File Cards */
    .stat-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
    }

    .stat-label {
        color: #94A3B8;
        font-size: 0.7rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    .stat-value {
        color: #F8FAFC;
        font-size: 1.15rem;
        font-weight: 600;
        margin-top: 0.2rem;
    }

    /* Citation Cards & Badges */
    .source-card {
        background: #0F172A;
        border-left: 3px solid #6366F1;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }

    .source-meta-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.35rem;
    }

    .score-badge {
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 9999px;
    }

    .score-high {
        background: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .score-mid {
        background: rgba(245, 158, 11, 0.18);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .telemetry-tag {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Pill buttons */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 20px !important;
        font-size: 0.82rem !important;
        background-color: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        transition: all 0.2s ease-in-out !important;
    }

    div[data-testid="stHorizontalBlock"] button:hover {
        border-color: #818CF8 !important;
        color: #818CF8 !important;
        background-color: rgba(49, 46, 129, 0.25) !important;
    }

    /* Floating Glowing Action Button */
    div.st-key-floating_guide_btn {
        position: fixed !important;
        bottom: 24px !important;
        right: 24px !important;
        width: 48px !important;
        height: 48px !important;
        z-index: 999999 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    div.st-key-floating_guide_btn > div {
        width: 48px !important;
        height: 48px !important;
    }

    div.st-key-floating_guide_btn button {
        width: 48px !important;
        height: 48px !important;
        min-width: 48px !important;
        min-height: 48px !important;
        max-width: 48px !important;
        max-height: 48px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 100%) !important;
        color: white !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 1.25rem !important;
        cursor: pointer !important;
        box-shadow: 0 0 16px rgba(99, 102, 241, 0.65), 0 4px 10px rgba(0, 0, 0, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    div.st-key-floating_guide_btn button:hover {
        transform: scale(1.1) translateY(-2px) !important;
        box-shadow: 0 0 25px rgba(168, 85, 247, 0.85), 0 6px 16px rgba(0, 0, 0, 0.5) !important;
        color: #ffffff !important;
        border-color: rgba(255, 255, 255, 0.5) !important;
    }

    div.st-key-floating_guide_btn button:focus,
    div.st-key-floating_guide_btn button:active {
        color: #ffffff !important;
        border-color: #818CF8 !important;
        outline: none !important;
    }

    div.st-key-floating_guide_btn button p {
        font-size: 1.25rem !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. State & Resource Initialization
try:
    database.init_db()
except Exception as e:
    st.error(f"Database error: {e}")

@st.cache_resource
def get_store():
    return VectorStore()

try:
    store = get_store()
except Exception as e:
    st.error(f"Vector Store error: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "has_seen_guide" not in st.session_state:
    st.session_state.has_seen_guide = False

# 4. Modals & Dialogs
@st.dialog("👋 Welcome to UniMind Knowledge Assistant!")
def show_welcome_guide():
    st.markdown("""
    **UniMind** is your institutional AI knowledge engine. It reads official university documents 
    (regulations, syllabus, circulars) and answers questions with verifiable citations without guessing.

    ---
    ### 🧭 How to use:
    1. **Upload Documents (Sidebar):** Upload official PDFs, DOCX, or TXT circulars.
    2. **Ask Any Question:** Type your question or click one of the suggested query chips.
    3. **Verified Answers:** The assistant quotes exact passages and tells you the exact document and page number.
    4. **Inspect & Download:** Check source citations, test confidence match scores, and export your transcript.

    ---
    💡 *Tip: If information isn't present in your files, the model will clearly state it has no record of it.*
    """)
    if st.button("🚀 Let's Get Started", type="primary", use_container_width=True):
        st.session_state.has_seen_guide = True
        st.rerun()

@st.dialog("📄 Document Content Inspector")
def inspect_document_dialog(filename: str):
    chunks = database.get_chunks_by_filename(filename)
    if not chunks:
        st.warning("No chunk records found for this document.")
        return

    st.markdown(f"### Inspecting: `{filename}`")
    st.caption(f"Total Chunks: {len(chunks)} | Pages: {len(set(c['page_number'] for c in chunks))}")

    # Build full document text for download
    full_text = "\n\n--- Next Chunk ---\n\n".join(
        [f"[Page {c['page_number']} | Chunk {c['chunk_index']}]:\n{c['text_content']}" for c in chunks]
    )
    
    st.download_button(
        label="📥 Download Parsed Document (.txt)",
        data=full_text,
        file_name=f"{filename}_parsed.txt",
        mime="text/plain",
        use_container_width=True
    )
    
    st.markdown("---")
    st.markdown("#### Stored Text Excerpts:")
    for c in chunks:
        with st.expander(f"Page {c['page_number']} — Segment #{c['chunk_index']}"):
            st.text(c["text_content"])

# Trigger welcome modal on initial launch
if not st.session_state.has_seen_guide:
    show_welcome_guide()

# 5. Sidebar: Repositories, Telemetry & File Management
with st.sidebar:
    st.markdown("### 🏛️ Knowledge Hub")
    st.caption("Institutional document database & status.")

    docs = database.list_uploaded_documents()
    total_docs = len(docs)
    total_vectors = store.index.ntotal if hasattr(store, "index") else 0

    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Files</div>
            <div class="stat-value">{total_docs}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stat2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Chunks</div>
            <div class="stat-value">{total_vectors}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Ingest New Files")
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if st.button("🚀 Process & Index", use_container_width=True, type="primary"):
        if uploaded_files:
            progress_bar = st.progress(0, text="Ingesting files...")
            success_count = 0
            for idx, file in enumerate(uploaded_files):
                filename = file.name
                try:
                    if hasattr(file, "seek"):
                        file.seek(0)
                    file_bytes = file.read()
                    if not file_bytes:
                        st.warning(f"File '{filename}' is empty.")
                        continue
                    pages = parse_document(file_bytes, filename)
                    chunks = process_extracted_pages(pages, chunk_size=1500, overlap=200)
                    if chunks:
                        store.add_documents(chunks, filename)
                        success_count += 1
                    else:
                        st.warning(f"No extractable text found in '{filename}'.")
                except Exception as err:
                    st.error(f"Error indexing {filename}: {err}")
                finally:
                    progress_bar.progress((idx + 1) / len(uploaded_files))
            if success_count > 0:
                st.toast(f"Indexed {success_count} file(s)!", icon="✅")
                time.sleep(0.6)
                st.rerun()
        else:
            st.warning("Select files to upload.")

    st.markdown("---")
    st.markdown("#### Active Documents")
    if docs:
        for doc_name in docs:
            c1, c2, c3 = st.columns([0.65, 0.18, 0.17])
            with c1:
                st.markdown(f"<div style='font-size: 0.82rem; font-weight: 500; padding-top: 4px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;'>📄 {doc_name}</div>", unsafe_allow_html=True)
            with c2:
                if st.button("👁️", key=f"preview_{doc_name}", help=f"Inspect {doc_name}"):
                    inspect_document_dialog(doc_name)
            with c3:
                if st.button("🗑️️", key=f"del_{doc_name}", help=f"Delete {doc_name}"):
                    removed = store.delete_document(doc_name)
                    st.toast(f"Removed '{doc_name}' ({removed} chunks evicted)", icon="🗑️")
                    st.rerun()
    else:
        st.caption("No files indexed yet.")

    st.markdown("---")
    if st.button("🧹 Clear Chat Feed", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 6. Hero Header
st.markdown("""
<div class="hero-wrapper">
    <div>
        <h1 class="hero-title">UniMind Intelligence Engine</h1>
        <div class="hero-subtitle">Deterministic, citation-backed academic query assistant powered by Qwen 2.5 Coder.</div>
    </div>
    <div>
        <span class="live-status"><span class="status-pulse"></span>System Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 7. Suggested Inquiry Chips
st.caption("Suggested inquiries:")
chip_col1, chip_col2, chip_col3 = st.columns(3)
active_query = None

if chip_col1.button("📋 Minimum Attendance Rules", use_container_width=True):
    active_query = "What is the minimum attendance required and what are the condonation rules?"
if chip_col2.button("🎯 Passing & Grading Policy", use_container_width=True):
    active_query = "What is the passing minimum marks for end semester examinations?"
if chip_col3.button("📚 Library Limits & Fines", use_container_width=True):
    active_query = "How many books can an undergraduate student borrow and what is the late fine?"

st.markdown("<br>", unsafe_allow_html=True)

# Helper function to generate Markdown export string
def create_transcript_markdown(query: str, answer: str, sources: list, latency: float) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    doc = f"# UniMind Query Transcript\n*Timestamp: {timestamp}*\n\n"
    doc += f"## User Query\n{query}\n\n"
    doc += f"## Assistant Answer\n{answer}\n\n"
    doc += f"**Telemetry:** Generated in {latency:.2f}s | Scanned {len(sources)} passages\n\n"
    doc += "## Verified Source Evidence\n"
    for i, s in enumerate(sources):
        doc += f"### {i+1}. {s.get('filename')} (Page {s.get('page_number')})\n"
        doc += f"- **Confidence Match:** {s.get('similarity_score', 0)}%\n"
        doc += f"- **Excerpt:**\n> {s.get('text_content', '')}\n\n"
    return doc

# 8. Render Message Feed
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Telemetry & Sources
        if msg.get("sources"):
            latency = msg.get("latency", 0.0)
            st.markdown(
                f"<div class='telemetry-tag'>⚡ <b>Response Telemetry:</b> Generated in {latency:.2f}s • {len(msg['sources'])} evidence passages scanned</div>",
                unsafe_allow_html=True
            )
            with st.expander("🔍 Verified Institutional Citations"):
                for src in msg["sources"]:
                    score = src.get("similarity_score", 0.0)
                    score_class = "score-high" if score >= 60.0 else "score-mid"
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-meta-row">
                            <span style="color: #818CF8; font-weight: 600; font-size: 0.82rem;">📄 {src.get('filename')} • Page {src.get('page_number')}</span>
                            <span class="score-badge {score_class}">Confidence: {score}% Match</span>
                        </div>
                        <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.45;">{src.get('text_content', '')[:280]}...</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Export Transcript Button
            transcript_md = create_transcript_markdown(
                query=msg.get("query_prompt", "Inquiry"),
                answer=msg["content"],
                sources=msg["sources"],
                latency=latency
            )
            st.download_button(
                label="📥 Export Answer & Sources (.md)",
                data=transcript_md,
                file_name=f"unimind_answer_{idx}.md",
                mime="text/markdown",
                key=f"dl_{idx}"
            )

# 9. Handle User Chat Input
raw_user_input = st.chat_input("Ask about examination policies, course prerequisites, syllabus...")
user_input = (raw_user_input or active_query or "").strip()

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    start_time = time.time()

    with st.spinner("Scanning institutional vector embeddings..."):
        try:
            retrieved_chunks = store.search(user_input, top_k=3)
        except Exception as e:
            retrieved_chunks = []
            st.error(f"Search retrieval error: {e}")

    with st.chat_message("assistant"):
        if not retrieved_chunks:
            fallback = "The institutional knowledge base does not contain any relevant documents to answer this question. Please upload the relevant documentation via the sidebar."
            st.markdown(fallback)
            st.session_state.messages.append({"role": "assistant", "content": fallback})
        else:
            try:
                response_stream = stream_grounded_answer(user_input, retrieved_chunks)

                def token_generator():
                    for chunk in response_stream:
                        if isinstance(chunk, str):
                            yield chunk
                        elif hasattr(chunk, "choices") and chunk.choices:
                            delta = chunk.choices[0].delta.content
                            if delta:
                                yield delta

                complete_response = st.write_stream(token_generator())
                inference_latency = time.time() - start_time
                is_error = complete_response.strip().startswith("⚠️")

                if not is_error:
                    st.markdown(
                        f"<div class='telemetry-tag'>⚡ <b>Response Telemetry:</b> Generated in {inference_latency:.2f}s • {len(retrieved_chunks)} evidence passages scanned</div>",
                        unsafe_allow_html=True
                    )
                    with st.expander("🔍 Verified Institutional Citations"):
                        for src in retrieved_chunks:
                            score = src.get("similarity_score", 0.0)
                            score_class = "score-high" if score >= 60.0 else "score-mid"
                            st.markdown(f"""
                            <div class="source-card">
                                <div class="source-meta-row">
                                    <span style="color: #818CF8; font-weight: 600; font-size: 0.82rem;">📄 {src.get('filename')} • Page {src.get('page_number')}</span>
                                    <span class="score-badge {score_class}">Confidence: {score}% Match</span>
                                </div>
                                <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.45;">{src.get('text_content', '')[:280]}...</div>
                            </div>
                            """, unsafe_allow_html=True)

                    transcript_md = create_transcript_markdown(
                        query=user_input,
                        answer=complete_response,
                        sources=retrieved_chunks,
                        latency=inference_latency
                    )
                    st.download_button(
                        label="📥 Export Answer & Sources (.md)",
                        data=transcript_md,
                        file_name=f"unimind_answer_{len(st.session_state.messages)}.md",
                        mime="text/markdown",
                        key=f"dl_live_{len(st.session_state.messages)}"
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": complete_response,
                    "sources": retrieved_chunks if not is_error else [],
                    "latency": inference_latency,
                    "query_prompt": user_input
                })
            except Exception as e:
                err_msg = f"⚠️ Inference failure: {str(e)}"
                st.error(err_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err_msg,
                    "sources": []
                })

# 10. Floating Action Button (Quick Guide)
if st.button("❓", key="floating_guide_btn", help="How to Use UniMind Guide"):
    show_welcome_guide()