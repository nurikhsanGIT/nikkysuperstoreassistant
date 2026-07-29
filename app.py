try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import time
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from graph.workflow import agent_graph
from utils.evaluator import ResponseEvaluator
from rag.loader import DocumentLoader
from rag.splitter import DocumentSplitter
from rag.vector_store import VectorStoreManager
from utils.config import DOCUMENTS_DIR, CHROMA_DB_DIR
from utils.data_loader import SuperstoreDataLoader
import logging

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Josjis Superstore AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700&display=swap');

/* ── Reset & Core Theme ── */
*, html, body {
    box-sizing: border-box;
}

body, .stApp, p, div, h1, h2, h3, h4, h5, h6, span, input, textarea {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Preserve Material Icon Fonts */
[data-testid="stIconMaterial"], 
[data-testid="collapsedControl"] *,
[data-testid="stSidebarCollapseButton"] *,
[data-testid="stHeader"] *,
.material-symbols-outlined {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons', sans-serif !important;
}

html, body {
    overflow-x: hidden !important;
    max-width: 100vw !important;
}

/* ── Streamlit Container Padding ── */
.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

/* ── App background ── */
.stApp {
    background: #0b0f17 !important;
    color: #f1f5f9 !important;
    background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.1) 0px, transparent 50%) !important;
    overflow-x: hidden !important;
}
.stAppHeader { background: transparent !important; }

/* ── Streamlit chrome & Sidebar Toggle ── */
footer { visibility: hidden; }
.stAppHeader { background: rgba(11, 15, 23, 0.7) !important; backdrop-filter: blur(10px); z-index: 99999; }

[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] button {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 10px !important;
    color: #a5b4fc !important;
    padding: 6px 10px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
}
[data-testid="collapsedControl"]:hover,
[data-testid="stSidebarCollapseButton"] button:hover {
    background: #6366f1 !important;
    color: #ffffff !important;
    border-color: #818cf8 !important;
}

/* ── Responsive Columns Layout (Mobile only) ── */
@media (max-width: 768px) {
    [data-testid="stColumn"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    .main .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-top: 1rem !important;
    }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }

/* ── Chat input ── */
.stChatInput textarea {
    background: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: #f8fafc !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
}
.stChatInput textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(17, 24, 39, 0.7) !important;
    gap: 4px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding: 4px;
    border-radius: 12px;
    overflow-x: auto !important;
    flex-wrap: nowrap !important;
    white-space: nowrap !important;
    -webkit-overflow-scrolling: touch;
}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
    display: none;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border-radius: 8px !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    border: none !important;
    flex-shrink: 0 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.2)) !important;
    color: #a5b4fc !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.15) !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 14px !important; width: 100% !important; }
[data-testid="stDataFrame"] {
    background: #131b2e !important;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    overflow-x: auto !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    padding: 10px 18px !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
    background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%) !important;
}

/* ── SIDEBAR CUSTOM ── */
.sb-brand {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 20px 18px;
    margin: 16px 12px 14px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.sb-brand-name {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    margin: 0;
    letter-spacing: -0.02em;
}
.sb-brand-tag {
    font-size: 0.68rem !important;
    color: #a5b4fc !important;
    margin: 4px 0 0 0;
    letter-spacing: 0.1em;
    font-weight: 700 !important;
}
.sb-stat {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 12px 14px;
    margin: 0 12px 8px;
    backdrop-filter: blur(8px);
}
.sb-stat-label {
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0;
}
.sb-stat-val {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: #38bdf8 !important;
    margin: 3px 0 0 0;
    line-height: 1.2;
}
.sb-stat-sub {
    font-size: 0.68rem !important;
    color: #94a3b8 !important;
    margin: 2px 0 0 0;
}
.sb-section {
    padding: 12px 16px 6px;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.sb-file {
    margin: 0 12px 4px;
    padding: 8px 12px;
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    font-size: 0.78rem !important;
    color: #cbd5e1 !important;
    display: flex;
    align-items: center;
    gap: 8px;
    word-break: break-word;
}
.sb-divider {
    height: 1px;
    background: rgba(255, 255, 255, 0.08);
    margin: 14px 12px;
}

/* Status Badge */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34d399;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    margin: 0 12px 12px;
}
.status-dot {
    width: 7px;
    height: 7px;
    background: #34d399;
    border-radius: 50%;
    box-shadow: 0 0 8px #34d399;
}

/* ── PAGE HEADER ── */
.page-hdr {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 28px 0 22px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 24px;
}
.page-hdr-icon {
    width: 52px;
    height: 52px;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    flex-shrink: 0;
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.2);
}
.page-hdr-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.02em;
}
.page-hdr-sub {
    font-size: 0.84rem;
    color: #94a3b8;
    margin: 4px 0 0 0;
}

/* ── CHAT AREA ── */
.chat-area {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 12px 4px;
}
.chat-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 24px;
    text-align: center;
    gap: 12px;
    background: rgba(30, 41, 59, 0.25);
    border: 1px dashed rgba(255, 255, 255, 0.1);
    border-radius: 16px;
}
.chat-empty-icon {
    width: 58px;
    height: 58px;
    background: rgba(99, 102, 241, 0.15);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6rem;
    margin: 0 auto 4px;
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #818cf8;
}
.chat-empty-text {
    font-size: 0.95rem;
    color: #f1f5f9;
    margin: 0;
    font-weight: 600;
}
.chat-empty-hint {
    font-size: 0.8rem;
    color: #94a3b8;
    margin: 0;
    line-height: 1.7;
}

.msg-user-wrap { display: flex; flex-direction: column; align-items: flex-end; gap: 4px; }
.msg-bot-wrap  { display: flex; flex-direction: column; align-items: flex-start; gap: 4px; }
.msg-label {
    font-size: 0.66rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0 4px;
}
.msg-label-u { color: #818cf8; }
.msg-label-b { color: #34d399; }

.msg-user {
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
    border-radius: 18px 18px 4px 18px;
    padding: 14px 18px;
    color: #ffffff;
    font-size: 0.9rem;
    line-height: 1.65;
    max-width: 86%;
    word-wrap: break-word;
    overflow-wrap: anywhere;
    box-shadow: 0 4px 16px rgba(79, 70, 229, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.12);
}

.msg-bot {
    background: #131b2e;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px 18px 18px 18px;
    padding: 16px 20px;
    color: #e2e8f0;
    font-size: 0.9rem;
    line-height: 1.7;
    max-width: 94%;
    word-wrap: break-word;
    overflow-wrap: anywhere;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}
.msg-bot b, .msg-bot strong { color: #ffffff; }
.msg-bot ul, .msg-bot ol { padding-left: 20px; margin: 8px 0; }
.msg-bot li { margin-bottom: 4px; }

/* ── PANEL LABELS ── */
.panel-label {
    font-size: 0.7rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 12px;
}

/* ── STAT CARDS ── */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 16px; }
.stat-card {
    background: #131b2e;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 2px solid #6366f1;
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
}
.stat-card-label {
    font-size: 0.65rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0;
}
.stat-card-val {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.35rem;
    font-weight: 700;
    color: #38bdf8;
    margin: 6px 0 0 0;
    line-height: 1.1;
}
.chart-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: #94a3b8;
    margin: 16px 0 8px 0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── EVALUATION METRICS TABLE ── */
.eval-table {
    width: 100% !important;
    border: none;
    margin-top: 4px;
    border-collapse: collapse;
}
.eval-label-col {
    width: 130px;
    border: none;
    padding: 4px 0;
    color: #94a3b8;
}

/* ── MOBILE MEDIA QUERIES (< 640px) ── */
@media (max-width: 640px) {
    .page-hdr {
        gap: 12px;
        padding: 16px 0 16px 0;
        margin-bottom: 16px;
    }
    .page-hdr-icon {
        width: 44px;
        height: 44px;
        font-size: 1.2rem;
        border-radius: 12px;
    }
    .page-hdr-title {
        font-size: 1.2rem !important;
    }
    .page-hdr-sub {
        font-size: 0.78rem !important;
    }
    .msg-user {
        max-width: 95% !important;
        padding: 10px 14px !important;
        font-size: 0.85rem !important;
    }
    .msg-bot {
        max-width: 98% !important;
        padding: 12px 14px !important;
        font-size: 0.85rem !important;
    }
    .chat-empty {
        padding: 32px 14px !important;
    }
    .chat-empty-icon {
        width: 46px !important;
        height: 46px !important;
        font-size: 1.3rem !important;
    }
    .chat-empty-text {
        font-size: 0.88rem !important;
    }
    .chat-empty-hint {
        font-size: 0.75rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 6px 12px !important;
        font-size: 0.78rem !important;
    }
    .eval-label-col {
        width: 105px !important;
        font-size: 0.78rem !important;
    }
}

/* ── SMALL MOBILE MEDIA QUERIES (< 480px) ── */
@media (max-width: 480px) {
    .stat-grid {
        grid-template-columns: 1fr !important;
        gap: 8px !important;
    }
    .stat-card {
        padding: 12px 14px !important;
    }
    .eval-table tr {
        display: flex !important;
        flex-direction: column !important;
        margin-bottom: 6px !important;
        border-bottom: 1px dashed rgba(255,255,255,0.06) !important;
        padding-bottom: 4px !important;
    }
    .eval-table td {
        width: 100% !important;
        display: block !important;
        padding: 1px 0 !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_state" not in st.session_state:
    st.session_state.last_state = {}
if "processing_query" not in st.session_state:
    st.session_state.processing_query = None

# ─── Load data (cached) ────────────────────────────────────────────────────────
fin = SuperstoreDataLoader.get_finance_summary()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand header
    st.markdown("""
    <div class="sb-brand">
        <p class="sb-brand-name">🛒 Josjis Enterprise AI</p>
        <p class="sb-brand-tag">MULTI-AGENT BUSINESS ASSISTANT</p>
    </div>
    <div class="status-pill">
        <div class="status-dot"></div> AI Engine: Active (llama3.2:1b)
    </div>
    """, unsafe_allow_html=True)

    # Stats
    if fin:
        st.markdown("<p class='sb-section'>Ringkasan Dataset</p>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sb-stat">
            <p class="sb-stat-label">Total Omzet</p>
            <p class="sb-stat-val">${fin.get('total_sales',0)/1e6:.2f}M</p>
            <p class="sb-stat-sub">dari {fin.get('total_orders',0):,} transaksi</p>
        </div>
        <div class="sb-stat">
            <p class="sb-stat-label">Total Profit</p>
            <p class="sb-stat-val">${fin.get('total_profit',0)/1e3:.1f}K</p>
            <p class="sb-stat-sub">margin {fin.get('total_profit',0)/max(fin.get('total_sales',1),1)*100:.1f}%</p>
        </div>
        <div class="sb-stat">
            <p class="sb-stat-label">Pelanggan Unik</p>
            <p class="sb-stat-val">{fin.get('total_customers',0):,}</p>
            <p class="sb-stat-sub">rata-rata diskon {fin.get('avg_discount',0):.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    # Dokumen Referensi dengan Preview
    st.markdown("<div class='sb-divider'></div>", unsafe_allow_html=True)
    st.markdown("<p class='sb-section'>Dokumen Referensi</p>", unsafe_allow_html=True)
    if os.path.exists(DOCUMENTS_DIR):
        files = [f for f in os.listdir(DOCUMENTS_DIR) if not f.startswith(".")]
        for f in files:
            file_path = os.path.join(DOCUMENTS_DIR, f)
            with st.expander(f"📄 {f}"):
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as doc_file:
                        content = doc_file.read(300)
                        st.caption(content + ("..." if len(content) >= 300 else ""))
                except Exception:
                    st.caption("Preview tidak dapat dimuat.")

    st.markdown("<div class='sb-divider'></div>", unsafe_allow_html=True)

    if st.button("🔄 Perbarui Basis Pengetahuan", use_container_width=True):
        with st.spinner("Memperbarui..."):
            loader = DocumentLoader()
            docs = loader.load_directory(DOCUMENTS_DIR)
            splitter = DocumentSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)
            vm = VectorStoreManager()
            vm.reset_store()
            vm.add_documents(chunks)
            st.success("Selesai!")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Export Chat Button
    if st.session_state.chat_history:
        chat_export_txt = "=== LAPORAN AI JOSJIS ASSISTANT ===\n\n"
        for msg in st.session_state.chat_history:
            role = "ANDA" if msg["role"] == "user" else "ASSISTANT"
            raw_text = msg["text"].replace("<br>", "\n")
            if "<div style=" in raw_text:
                raw_text = raw_text.split("<div style=")[0].strip()
            chat_export_txt += f"[{role}]\n{raw_text}\n\n" + ("="*40) + "\n\n"
        
        st.download_button(
            label="📥 Unduh Laporan Chat",
            data=chat_export_txt,
            file_name="laporan_josjis_assistant.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_state = {}
        st.rerun()

# ─── PAGE HEADER ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hdr">
    <div class="page-hdr-icon">🛒</div>
    <div>
        <p class="page-hdr-title">Josjis Superstore Assistant</p>
        <p class="page-hdr-sub">Tanyakan tentang produk, penjualan, keuangan, atau layanan pelanggan</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── LAYOUT ───────────────────────────────────────────────────────────────────
col_chat, col_dash = st.columns([1.15, 0.85], gap="large")

# ─── CHAT PANEL ───────────────────────────────────────────────────────────────
with col_chat:
    st.markdown("<p class='panel-label'>Percakapan</p>", unsafe_allow_html=True)

    # Chat history area
    chat_box = st.container(height=440, border=False)
    with chat_box:
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="chat-empty">
                <div class="chat-empty-icon">💬</div>
                <p class="chat-empty-text">Mulai percakapan</p>
                <p class="chat-empty-hint">
                    Contoh pertanyaan:<br>
                    "produk terlaris bulan ini"<br>
                    "total omzet dan profit"<br>
                    "kebijakan refund"
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            msgs_html = '<div class="chat-area">'
            for chat in st.session_state.chat_history:
                if chat["role"] == "user":
                    msgs_html += f"""
                    <div class="msg-user-wrap">
                        <span class="msg-label msg-label-u">Anda</span>
                        <div class="msg-user">{chat["text"]}</div>
                    </div>"""
                else:
                    text = chat["text"].replace("\n", "<br>")
                    msgs_html += f"""
                    <div class="msg-bot-wrap">
                        <span class="msg-label msg-label-b">Assistant</span>
                        <div class="msg-bot">{text}</div>
                    </div>"""
            msgs_html += "</div>"
            st.markdown(msgs_html, unsafe_allow_html=True)

    # Chat input
    if user_query := st.chat_input("Tanyakan sesuatu..."):
        st.session_state.chat_history.append({"role": "user", "text": user_query})
        st.session_state.processing_query = user_query
        st.rerun()
        
    if st.session_state.get("processing_query"):
        with st.spinner("Memproses..."):
            query = st.session_state.processing_query
            start_time = time.time()
            state_input = {
                "user_query": query,
                "user_id": "user",
                "session_id": "session",
                "chat_history": st.session_state.chat_history[-6:],
                "intent": "",
                "tasks": [],
                "sql_results": [],
                "rag_results": [],
                "tool_results": [],
                "agent_outputs": [],
                "findings": "",
                "recommendations": "",
                "confidence": 1.0,
                "need_replan": False,
                "retry_count": 0,
                "final_answer": ""
            }
            try:
                result_state = agent_graph.invoke(state_input)
                st.session_state.last_state = result_state
            except Exception as e:
                logging.error(f"Workflow error: {e}")
                result_state = {}
                st.session_state.last_state = {}
            
            end_time = time.time()
            response_time = end_time - start_time

        ans = result_state.get("final_answer", "Maaf, tidak dapat memproses pertanyaan tersebut saat ini.")
        
        # Ekstrak nama agen yang bekerja
        agent_badges_html = ""
        tasks_run = result_state.get("tasks", [])
        if tasks_run:
            agents = list(set([t.get("agent", "Unknown").capitalize() for t in tasks_run]))
            badges = "".join([f'<span style="background:rgba(99,102,241,0.18); color:#a5b4fc; padding:4px 12px; border-radius:20px; font-size:0.72rem; font-weight:700; margin-right:6px; border:1px solid rgba(99,102,241,0.35); box-shadow:0 2px 8px rgba(99,102,241,0.15);">🤖 {a} Agent</span>' for a in agents])
            agent_badges_html = f'<div style="margin-bottom:12px; display:flex; flex-wrap:wrap; gap:4px;">{badges}</div>'
        
        # Ekstraksi RAG context untuk evaluasi halusinasi
        rag_context = ""
        if result_state.get("rag_results"):
            contexts = []
            for r in result_state["rag_results"]:
                if isinstance(r, dict) and "content" in r:
                    contexts.append(r["content"])
                elif isinstance(r, dict) and "page_content" in r:
                    contexts.append(r["page_content"])
                else:
                    contexts.append(str(r))
            rag_context = " ".join(contexts)
        elif result_state.get("findings"):
            rag_context = result_state.get("findings")
        
        # Evaluasi
        eval_metrics = ResponseEvaluator.evaluate(query, ans, rag_context, response_time)
        
        sources_html = ", ".join(eval_metrics['sources_used']) if eval_metrics['sources_used'] else "Tidak ada (Pengetahuan Umum)"
        metrics_html = f'''
<div style="background:rgba(15,23,42,0.6); padding:14px 16px; border-radius:12px; border:1px solid rgba(255,255,255,0.08); margin-top:14px; font-size:0.83rem;">
    <div style="font-weight:700; color:#818cf8; font-size:0.8rem; letter-spacing:0.04em; margin-bottom:8px; display:flex; align-items:center; gap:6px;">
        📊 METRIK EVALUASI MODEL
    </div>
    <table class="eval-table">
        <tr><td class="eval-label-col">🎯 <b>Akurasi</b></td><td style="border:none; padding:4px 0; color:#38bdf8; font-weight:600;">: {eval_metrics['accuracy']}%</td></tr>
        <tr><td class="eval-label-col">📏 <b>Efektivitas</b></td><td style="border:none; padding:4px 0; color:#38bdf8; font-weight:600;">: {eval_metrics['effectiveness']}%</td></tr>
        <tr><td class="eval-label-col">⚡ <b>Kecepatan</b></td><td style="border:none; padding:4px 0; color:#34d399; font-weight:600;">: {eval_metrics['efficiency_seconds']}s <span style="color:#64748b; font-weight:normal;">({eval_metrics['efficiency_rating']})</span></td></tr>
        <tr><td class="eval-label-col">🧠 <b>Halusinasi Risk</b></td><td style="border:none; padding:4px 0; color:#cbd5e1;">: {eval_metrics['hallucination_rating']}</td></tr>
        <tr><td class="eval-label-col" style="vertical-align:top;">📚 <b>Sumber Data</b></td><td style="border:none; padding:4px 0; color:#a5b4fc;">: {sources_html}</td></tr>
    </table>
</div>
'''
        final_ans = agent_badges_html + ans + metrics_html

        st.session_state.chat_history.append({"role": "assistant", "text": final_ans})
        st.session_state.processing_query = None
        st.rerun()

# ─── DASHBOARD PANEL ──────────────────────────────────────────────────────────
with col_dash:
    st.markdown("<p class='panel-label'>Data & Insight</p>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📦  Produk", "📈  Penjualan", "⭐  Kepuasan"])

    # ── Tab Produk ──
    with tab1:
        col_search1, col_search2 = st.columns([0.6, 0.4])
        with col_search1:
            kw_search = st.text_input("🔍 Cari produk", key="prod_kw_search", placeholder="Ketik nama produk...")
        with col_search2:
            cat_filter = st.selectbox("Kategori", ["Semua", "Furniture", "Office Supplies", "Technology"], key="prod_cat_filter")

        cat_param = None if cat_filter == "Semua" else cat_filter
        filtered_df = SuperstoreDataLoader.get_products(
            keyword=kw_search if kw_search else None,
            category=cat_param,
            top_n=15
        )
        if not filtered_df.empty:
            disp = filtered_df[["Product_Name", "Category", "Total_Sales", "Total_Quantity"]].copy()
            disp.columns = ["Produk", "Kategori", "Penjualan ($)", "Qty"]
            disp["Penjualan ($)"] = disp["Penjualan ($)"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(disp, use_container_width=True, hide_index=True, height=330)
        else:
            st.info("Tidak ada produk yang cocok.")

    # ── Tab Penjualan ──
    with tab2:
        if fin:
            st.markdown(f"""
            <div class="stat-grid">
                <div class="stat-card">
                    <p class="stat-card-label">Total Omzet</p>
                    <p class="stat-card-val">${fin.get('total_sales',0)/1e6:.2f}M</p>
                </div>
                <div class="stat-card">
                    <p class="stat-card-label">Total Profit</p>
                    <p class="stat-card-val">${fin.get('total_profit',0)/1e3:.1f}K</p>
                </div>
                <div class="stat-card">
                    <p class="stat-card-label">Unit Terjual</p>
                    <p class="stat-card-val">{fin.get('total_quantity',0):,}</p>
                </div>
                <div class="stat-card">
                    <p class="stat-card-label">Rata Diskon</p>
                    <p class="stat-card-val">{fin.get('avg_discount',0):.1f}%</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        by_cat = SuperstoreDataLoader.get_finance_by_category()
        if not by_cat.empty:
            st.markdown("<p class='chart-title'>Omzet per Kategori</p>", unsafe_allow_html=True)
            fig_cat = px.bar(
                by_cat,
                x="Category",
                y="Total_Sales",
                color="Total_Sales",
                color_continuous_scale=["#4f46e5", "#6366f1", "#38bdf8"]
            )
            fig_cat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=5, r=5, t=10, b=10),
                height=180,
                showlegend=False,
                coloraxis_showscale=False,
                font=dict(color="#94a3b8", family="Plus Jakarta Sans"),
                xaxis=dict(title="", showgrid=False, tickfont=dict(size=11, color="#cbd5e1")),
                yaxis=dict(title="", showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickprefix="$"),
                hoverlabel=dict(bgcolor="#1e293b", font_size=12, font_family="Plus Jakarta Sans")
            )
            fig_cat.update_traces(
                hovertemplate="<b>%{x}</b><br>Omzet: $%{y:,.2f}<extra></extra>"
            )
            st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar": False})

        by_reg = SuperstoreDataLoader.get_sales_by_region()
        if not by_reg.empty:
            st.markdown("<p class='chart-title'>Omzet Per Region</p>", unsafe_allow_html=True)
            by_reg_sorted = by_reg.sort_values("Total_Sales", ascending=True)
            fig_reg = px.bar(
                by_reg_sorted,
                x="Total_Sales",
                y="Region",
                orientation="h",
                color="Total_Sales",
                color_continuous_scale=["#4f46e5", "#6366f1", "#38bdf8", "#34d399"],
                custom_data=["Total_Profit"]
            )
            fig_reg.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=5, r=10, t=10, b=10),
                height=320,
                showlegend=False,
                coloraxis_showscale=False,
                font=dict(color="#94a3b8", family="Plus Jakarta Sans"),
                xaxis=dict(title="", showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickprefix="$"),
                yaxis=dict(title="", showgrid=False, tickfont=dict(size=11, color="#cbd5e1")),
                hoverlabel=dict(bgcolor="#1e293b", font_size=12, font_family="Plus Jakarta Sans")
            )
            fig_reg.update_traces(
                hovertemplate="<b>Region %{y}</b><br>Omzet: $%{x:,.2f}<br>Profit: $%{customdata[0]:,.2f}<extra></extra>"
            )
            st.plotly_chart(fig_reg, use_container_width=True, config={"displayModeBar": False})

    # ── Tab Kepuasan ──
    with tab3:
        csat = SuperstoreDataLoader.get_csat_summary()
        if csat:
            st.markdown(f"""
            <div class="stat-grid">
                <div class="stat-card">
                    <p class="stat-card-label">Avg CSAT</p>
                    <p class="stat-card-val">{csat.get('avg_csat',0):.2f}<span style="font-size:0.7rem;color:#374151"> /5</span></p>
                </div>
                <div class="stat-card">
                    <p class="stat-card-label">Total Kasus</p>
                    <p class="stat-card-val">{csat.get('total_complaints',0)/1e3:.1f}K</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            dist = csat.get("csat_distribution", {})
            if dist:
                st.markdown("<p class='chart-title'>Distribusi Skor CSAT</p>", unsafe_allow_html=True)
                dist_df = pd.DataFrame([
                    {"Skor": f"{k} ⭐", "Jumlah": v} for k, v in sorted(dist.items())
                ])
                fig_csat = px.bar(
                    dist_df,
                    x="Skor",
                    y="Jumlah",
                    color="Jumlah",
                    color_continuous_scale=["#059669", "#10b981", "#34d399"]
                )
                fig_csat.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=5, r=5, t=10, b=10),
                    height=180,
                    showlegend=False,
                    coloraxis_showscale=False,
                    font=dict(color="#94a3b8", family="Plus Jakarta Sans"),
                    xaxis=dict(title="", showgrid=False, tickfont=dict(size=11, color="#cbd5e1")),
                    yaxis=dict(title="", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                    hoverlabel=dict(bgcolor="#1e293b", font_size=12, font_family="Plus Jakarta Sans")
                )
                fig_csat.update_traces(
                    hovertemplate="<b>%{x}</b><br>Jumlah: %{y:,} kasus<extra></extra>"
                )
                st.plotly_chart(fig_csat, use_container_width=True, config={"displayModeBar": False})

        complaints = SuperstoreDataLoader.get_complaints_by_category(top_n=5)
        if not complaints.empty:
            st.markdown("<p class='chart-title'>Top Kategori Komplain</p>", unsafe_allow_html=True)
            cd = complaints[["category", "Total", "Avg_CSAT"]].copy()
            cd.columns = ["Kategori", "Jumlah", "CSAT"]
            cd["CSAT"] = cd["CSAT"].round(2)
            st.dataframe(cd, use_container_width=True, hide_index=True, height=210)
