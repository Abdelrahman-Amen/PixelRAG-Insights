"""
PixelRAG Streamlit App
Clean chat UI like GPT/Gemini with optional PDF viewer toggle.
Run: streamlit run app.py
"""

import base64
import time
from pathlib import Path

import streamlit as st

from config import INDEX_DIR, N_DOCS, UPLOAD_DIR
from pixelrag_pipeline import answer, build, get_paths, is_built, start_server, stop_server

st.set_page_config(
    page_title="PixelRAG",
    page_icon="📄",
    layout="wide",
)

# ── Force sidebar always open ─────────────────────────────
st.markdown("""
<style>
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] { 
    min-width: 300px !important; 
    max-width: 300px !important;
    transform: none !important;
}
</style>
""", unsafe_allow_html=True)

# ── Custom CSS ────────────────────────────────────────────

st.markdown("""
<style>
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"] { min-width: 300px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* hide default streamlit header */
#MainMenu, footer, header {visibility: hidden;}

/* full height chat */
.block-container { padding-top: 1rem; padding-bottom: 0; }

/* PDF iframe */
.pdf-frame {
    width: 100%;
    height: 80vh;
    border: 1px solid #333;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────
for k, v in {
    "ready": False,
    "article_dir": None,
    "pdf_name": None,
    "pdf_path": None,
    "history": [],
    "show_pdf": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Pipeline runner ───────────────────────────────────────
def run_pipeline(pdf_path: Path):
    index_dir = INDEX_DIR / pdf_path.stem
    with st.status(
        "⚙️ Building index — first time only, please wait..." if not is_built(index_dir)
        else "🚀 Loading...",
        expanded=True
    ) as status:
        try:
            if not is_built(index_dir):
                st.write("📄 Rendering pages...")
                idx_dir, _ = build(pdf_path)
                st.write("✅ Index built")
            else:
                idx_dir = index_dir
            st.write("🚀 Starting search server...")
            stop_server()
            time.sleep(1)
            start_server(idx_dir)
            _, art_dir, _, _ = get_paths(pdf_path)
            st.session_state.article_dir = art_dir
            st.session_state.ready       = True
            status.update(label="✅ Ready to chat!", state="complete")
        except Exception as e:
            status.update(label=f"❌ Error: {e}", state="error")


def pdf_to_b64(pdf_path: Path) -> str:
    return base64.b64encode(pdf_path.read_bytes()).decode()


# ── Top bar ───────────────────────────────────────────────
st.markdown("## 📄 PixelRAG")

# ── Sidebar — upload ──────────────────────────────────────
with st.sidebar:
    st.header("📁 Upload PDF")
    uploaded = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

# ── Handle upload ─────────────────────────────────────────
if uploaded:
    pdf_path = UPLOAD_DIR / uploaded.name
    pdf_path.write_bytes(uploaded.getvalue())

    if st.session_state.pdf_name != uploaded.name:
        st.session_state.update({
            "ready": False,
            "article_dir": None,
            "history": [],
            "pdf_name": uploaded.name,
            "pdf_path": pdf_path,
            "show_pdf": False,
        })

    if not st.session_state.ready:
        run_pipeline(pdf_path)

st.divider()

# ── Main layout ───────────────────────────────────────────
if not uploaded:
    st.markdown("""
    <div style='text-align:center; margin-top:15vh; color:#888;'>
        <h2>👆 Upload a PDF to get started</h2>
        <p>Ask questions about any PDF — charts, tables, images, scanned docs</p>
    </div>
    """, unsafe_allow_html=True)

elif not st.session_state.ready:
    st.info("Processing PDF...")

else:
    # ── PDF toggle button ─────────────────────────────────
    btn_col, title_col = st.columns([1, 8])
    with btn_col:
        if st.button(
            "📖 Hide PDF" if st.session_state.show_pdf else "📖 Show PDF",
            use_container_width=True
        ):
            st.session_state.show_pdf = not st.session_state.show_pdf
            st.rerun()

    with title_col:
        st.caption(f"📄 {uploaded.name.replace('.pdf', '')}")

    # ── PDF viewer (collapsible) ──────────────────────────
    if st.session_state.show_pdf:
        b64 = pdf_to_b64(st.session_state.pdf_path)
        st.markdown(f"""
        <iframe
            src="data:application/pdf;base64,{b64}"
            class="pdf-frame"
        ></iframe>
        """, unsafe_allow_html=True)
        st.divider()

    # ── Chat history ──────────────────────────────────────
    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["query"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            with st.expander(f"🔍 Retrieved pages: {item['pages']}"):
                cols = st.columns(len(item["pages"]))
                for i, (page, score) in enumerate(zip(item["pages"], item["scores"])):
                    with cols[i]:
                        st.metric(f"Page {page}", f"{score:.4f}")
                        art_dir = st.session_state.article_dir
                        for ext in ["jpg", "png"]:
                            img_path = art_dir / f"tile_{page-1:04d}.{ext}"
                            if img_path.exists():
                                st.image(str(img_path))
                                break

    # ── Input ─────────────────────────────────────────────
    query = st.chat_input("Ask a question about the PDF...")
    if query:
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = answer(
                        query,
                        st.session_state.article_dir,
                        n_docs=N_DOCS,
                    )
                    st.write(result["answer"])
                    st.session_state.history.append(result)
                except Exception as e:
                    st.error(f"Error: {e}")
        st.rerun()