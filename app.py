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
    page_title="Nikky Superstore AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700&display=swap');

/* ── Reset & Core Theme ── */
*, html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    box-sizing: border-box;
}

/* ── App background ── */
.stApp {
    background: #0b0f17 !important;
    color: #f1f5f9 !important;
    background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.1) 0px, transparent 50%) !important;
}
.stAppHeader { background: transparent !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

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
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border-radius: 8px !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.2)) !important;
    color: #a5b4fc !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.15) !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 14px !important; }
[data-testid="stDataFrame"] {
    background: #131b2e !important;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    overflow: hidden;
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
        <p class="sb-brand-name">🛒 Nikky Enterprise AI</p>
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

    # Dokumen
    st.markdown("<div class='sb-divider'></div>", unsafe_allow_html=True)
    st.markdown("<p class='sb-section'>Dokumen Referensi</p>", unsafe_allow_html=True)
    if os.path.exists(DOCUMENTS_DIR):
        files = [f for f in os.listdir(DOCUMENTS_DIR) if not f.startswith(".")]
        for f in files:
            st.markdown(f"<div class='sb-file'>📄 {f}</div>", unsafe_allow_html=True)

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

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_state = {}
        st.rerun()

# ─── PAGE HEADER ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hdr">
    <div class="page-hdr-icon">🛒</div>
    <div>
        <p class="page-hdr-title">Nikky Superstore Assistant</p>
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
    <table style="width:100%; border:none; margin-top:4px; border-collapse:collapse;">
        <tr><td style="width:140px; border:none; padding:4px 0; color:#94a3b8;">🎯 <b>Akurasi</b></td><td style="border:none; padding:4px 0; color:#38bdf8; font-weight:600;">: {eval_metrics['accuracy']}%</td></tr>
        <tr><td style="border:none; padding:4px 0; color:#94a3b8;">📏 <b>Efektivitas</b></td><td style="border:none; padding:4px 0; color:#38bdf8; font-weight:600;">: {eval_metrics['effectiveness']}%</td></tr>
        <tr><td style="border:none; padding:4px 0; color:#94a3b8;">⚡ <b>Kecepatan</b></td><td style="border:none; padding:4px 0; color:#34d399; font-weight:600;">: {eval_metrics['efficiency_seconds']}s <span style="color:#64748b; font-weight:normal;">({eval_metrics['efficiency_rating']})</span></td></tr>
        <tr><td style="border:none; padding:4px 0; color:#94a3b8;">🧠 <b>Halusinasi Risk</b></td><td style="border:none; padding:4px 0; color:#cbd5e1;">: {eval_metrics['hallucination_rating']}</td></tr>
        <tr><td style="border:none; padding:4px 0; vertical-align:top; color:#94a3b8;">📚 <b>Sumber Data</b></td><td style="border:none; padding:4px 0; color:#a5b4fc;">: {sources_html}</td></tr>
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
        top_df = SuperstoreDataLoader.get_top_products(top_n=10)
        if not top_df.empty:
            disp = top_df[["Product_Name", "Category", "Total_Sales", "Total_Quantity"]].copy()
            disp.columns = ["Produk", "Kategori", "Penjualan ($)", "Qty"]
            disp["Penjualan ($)"] = disp["Penjualan ($)"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(disp, use_container_width=True, hide_index=True, height=380)

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
            st.bar_chart(by_cat.set_index("Category")["Total_Sales"], height=160, color="#3b82f6")

        by_reg = SuperstoreDataLoader.get_sales_by_region()
        if not by_reg.empty:
            st.markdown("<p class='chart-title'>Per Region</p>", unsafe_allow_html=True)
            rd = by_reg[["Region", "Total_Sales", "Total_Profit"]].copy()
            rd.columns = ["Region", "Penjualan ($)", "Profit ($)"]
            rd["Penjualan ($)"] = rd["Penjualan ($)"].apply(lambda x: f"${x:,.0f}")
            rd["Profit ($)"] = rd["Profit ($)"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(rd, use_container_width=True, hide_index=True, height=175)

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
                dist_df = pd.DataFrame.from_dict(
                    {str(k): v for k, v in sorted(dist.items())},
                    orient="index", columns=["Jumlah"]
                )
                st.bar_chart(dist_df, height=160, color="#10b981")

        complaints = SuperstoreDataLoader.get_complaints_by_category(top_n=5)
        if not complaints.empty:
            st.markdown("<p class='chart-title'>Top Kategori Komplain</p>", unsafe_allow_html=True)
            cd = complaints[["category", "Total", "Avg_CSAT"]].copy()
            cd.columns = ["Kategori", "Jumlah", "CSAT"]
            cd["CSAT"] = cd["CSAT"].round(2)
            st.dataframe(cd, use_container_width=True, hide_index=True, height=210)
