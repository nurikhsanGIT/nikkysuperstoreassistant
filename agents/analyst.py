import time
import logging
from agents.base import BaseAgent
from graph.state import EnterpriseState

logger = logging.getLogger(__name__)

class BusinessAnalystAgent(BaseAgent):
    """Business Analyst Agent aggregates findings and recommendations from various agent runs."""
    
    def __init__(self):
        super().__init__(
            name="Business Analyst Agent",
            role_description="Spesialis menyusun rekomendasi bisnis, temuan, dan mencari akar penyebab masalah dari berbagai output agen."
        )

    def execute(self, state: EnterpriseState, task_desc: str) -> dict:
        start_time = time.time()
        
        # Gather all specialist agent outputs
        outputs = state.get("agent_outputs", [])
        reports = []
        for out in outputs:
            reports.append(f"### Agent: {out.get('agent_name')}\n{out.get('response')}")
        
        context_data = "\n\n".join(reports)
        user_query = state.get("user_query", "").lower()

        # If the user is asking a simple lookup question, skip the generic business analysis template
        # and just synthesize/pass through the specialist answers directly
        simple_lookup_keywords = [
            "inventory", "stok", "stock", "barang", "sebutkan", "daftar",
            "harga", "berapa", "lihat", "tampilkan", "cek", "check"
        ]
        is_simple_lookup = any(k in user_query for k in simple_lookup_keywords)

        if is_simple_lookup and len(outputs) <= 2:
            # For simple lookups, just synthesize and present the specialist data clearly
            prompt = f"""Anda adalah asisten POS Nikky Superstore yang ramah dan informatif.

Seorang pengguna bertanya: "{state.get('user_query', '')}"

Berikut data yang sudah dikumpulkan oleh agen spesialis kami:
{context_data}

Sajikan jawaban secara langsung, jelas, dan ringkas dalam Bahasa Indonesia.
JIKA ada data barang (nama produk, harga, stok), tampilkan sebagai daftar poin-poin.
JANGAN membuat template laporan bisnis jika ini hanya pertanyaan lookup sederhana."""
        else:
            # For complex business analysis questions, use the full business report template
            prompt = f"""
Anda adalah Business Analyst Agent dari Nikky Superstore.
Tugas Anda adalah merangkum hasil kerja dari agen-agen spesialis POS, lalu menyusun analisis bisnis yang rapi.

Berikut laporan dari para agen spesialis:
{context_data}

Silakan analisis data di atas dan buat 3 bagian dalam Bahasa Indonesia secara profesional:
1. Findings (Temuan-temuan utama)
2. Root Cause (Akar permasalahan jika ada penurunan penjualan atau stok tidak sinkron)
3. Recommendations (Rekomendasi taktis/strategis)

Format jawaban Anda dengan Markdown yang bersih dan informatif. Gunakan data nyata dari laporan agen, JANGAN hanya menulis pernyataan generik.
        """
        
        try:
            res = self.llm.invoke(prompt)
            response_text = res.content
        except Exception as e:
            logger.error(f"Analyst Agent LLM call failed: {e}")
            response_text = "Gagal menyusun analisis bisnis dari laporan agen."

        # Parse findings and recommendations placeholder or standard splitting
        findings = response_text
        recommendations = "Rekomendasi bisnis terintegrasi."
        
        return {
            "agent_name": self.name,
            "response": response_text,
            "findings": findings,
            "recommendations": recommendations,
            "response_time": time.time() - start_time,
            "context_used": context_data
        }

