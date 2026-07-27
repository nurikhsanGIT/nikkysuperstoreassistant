import json
import logging
from llm.ollama_model import OllamaModel

logger = logging.getLogger(__name__)


class Planner:
    """Enterprise Planner: tentukan agent yang tepat berdasarkan query user."""

    def __init__(self):
        self.llm = OllamaModel(temperature=0.1).get_llm()

    def generate_plan(self, query: str, chat_history: list = None) -> list:
        q = query.lower().strip()

        # ─── PRIORITY KEYWORD ROUTING (tidak perlu panggil LLM) ──────────────
        tasks = self._keyword_routing(q, query)
        if tasks:
            logger.info(f"Planner: keyword routing → {[t['agent'] for t in tasks]}")
            return tasks

        # Format recent conversation context
        history_str = ""
        if chat_history:
            recent = chat_history[-6:]
            lines = []
            for m in recent:
                role = "User" if m.get("role") == "user" else "Assistant"
                text = m.get("text", m.get("content", ""))
                if "<div style=" in text:
                    text = text.split("<div style=")[0].strip()
                lines.append(f"{role}: {text}")
            history_str = "\n".join(lines)

        # ─── LLM ROUTING (sebagai fallback jika keyword tidak cocok) ─────────
        prompt = f"""
Anda adalah AI Enterprise Planner untuk POS Nikky Superstore.
Tentukan agent mana yang harus menangani pertanyaan user berikut.

Riwayat Percakapan Sebelumnya:
{history_str if history_str else "Belum ada riwayat."}

Specialist Agents yang tersedia:
1. inventory  : info produk, daftar barang, kategori, stok, quantity
2. sales      : laporan penjualan, transaksi, region, segmen pelanggan
3. finance    : omzet, profit/laba, pendapatan, pengeluaran, keuangan, diskon
4. customer   : FAQ, SOP, kebijakan, refund, komplain, CSAT, kepuasan pelanggan
5. marketing  : produk terlaris, slow moving, strategi promosi, diskon
6. purchasing : rencana pembelian ke supplier

Aturan:
- Gunakan riwayat percakapan untuk memahami konteks pertanyaan jika user menggunakan rujukan (misal: "dia", "produk tersebut", "pelanggan itu").
- Pilih 1-2 agent yang paling relevan saja.
- WAJIB output berupa JSON array valid. Contoh: [{{"agent": "inventory", "task": "{query}"}}]
- Nilai "task" HARUS berisi pertanyaan user secara lengkap, BUKAN label pendek.
- Hanya keluarkan JSON, tanpa penjelasan, tanpa markdown.

Pertanyaan User Saat Ini: "{query}"
"""

        try:
            res = self.llm.invoke(prompt)
            clean = res.content.strip()
            # Bersihkan markdown code block jika ada
            for marker in ["```json", "```"]:
                if marker in clean:
                    clean = clean.split(marker)[1].split("```")[0].strip()

            tasks = json.loads(clean)
            if isinstance(tasks, list) and tasks:
                # Pastikan task_desc berisi user_query, bukan label pendek
                for t in tasks:
                    if len(t.get("task", "")) < 10:
                        t["task"] = query
                return tasks
        except Exception as e:
            logger.error(f"Planner LLM failed: {e}")

        # ─── FALLBACK jika LLM juga gagal ────────────────────────────────────
        logger.warning("Planner fallback: defaulting to inventory agent.")
        return [{"agent": "inventory", "task": query}]

    def _keyword_routing(self, q: str, original_query: str) -> list:
        """
        Routing berbasis keyword — lebih cepat dan akurat dibanding LLM untuk query umum.
        Mengembalikan list kosong jika tidak ada keyword yang cocok.
        """
        tasks = []

        # Finance / Keuangan
        if any(k in q for k in [
            "omzet", "laba", "profit", "keuangan", "pendapatan", "revenue",
            "pengeluaran", "diskon", "discount", "total penjualan"
        ]):
            tasks.append({"agent": "finance", "task": original_query})

        # Marketing / Promosi
        if any(k in q for k in [
            "terlaris", "best seller", "slow moving", "lambat", "promosi",
            "marketing", "campaign", "tidak laku", "paling laku"
        ]):
            tasks.append({"agent": "marketing", "task": original_query})

        # Sales / Transaksi
        if any(k in q for k in [
            "penjualan", "transaksi", "invoice", "region", "wilayah",
            "segmen", "segment", "order", "sales"
        ]):
            tasks.append({"agent": "sales", "task": original_query})

        # Inventory / Produk
        if any(k in q for k in [
            "barang", "produk", "product", "inventory", "stok", "stock",
            "kategori", "category", "daftar", "info", "data barang",
            "furniture", "technology", "office", "quantity"
        ]):
            # Hindari duplikasi jika sudah ada sales/marketing agent
            if not any(t["agent"] in ["sales", "marketing"] for t in tasks):
                tasks.append({"agent": "inventory", "task": original_query})

        # Customer / FAQ / SOP / General Meta
        if any(k in q for k in [
            "pelanggan", "customer", "refund", "komplain", "keluhan",
            "sop", "faq", "kebijakan", "csat", "kepuasan", "garansi", "return",
            "agen", "sistem", "siapa", "ai", "halo", "bantuan", "hi"
        ]):
            tasks.append({"agent": "customer", "task": original_query})

        # Purchasing
        if any(k in q for k in [
            "supplier", "vendor", "purchasing", "pembelian", "restock", "order ke supplier"
        ]):
            tasks.append({"agent": "purchasing", "task": original_query})

        return tasks
