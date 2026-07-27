import time
import json
import logging
from agents.base import BaseAgent
from graph.state import EnterpriseState
from utils.data_loader import SuperstoreDataLoader

logger = logging.getLogger(__name__)


class FinanceAgent(BaseAgent):
    """Finance Agent: analisis keuangan dari superstore.csv (Sales, Profit, Discount)."""

    def __init__(self):
        super().__init__(
            name="Finance Agent",
            role_description="Spesialis melacak revenue, profit, dan analisis keuangan dari dataset Superstore."
        )

    def execute(self, state: EnterpriseState, task_desc: str) -> dict:
        start_time = time.time()
        user_query = state.get("user_query", task_desc)
        q = user_query.lower()

        # Ambil data relevan
        finance_summary = SuperstoreDataLoader.get_finance_summary()
        by_category = SuperstoreDataLoader.get_finance_by_category()
        by_region = SuperstoreDataLoader.get_sales_by_region()

        data_context = {
            "ringkasan_keuangan": finance_summary,
            "per_kategori": by_category.to_dict(orient="records"),
            "per_region": by_region.to_dict(orient="records"),
        }
        llm_input = json.dumps(data_context, ensure_ascii=False)

        llm_prompt = (
            f"Anda adalah Finance Agent dari Nikky Superstore.\n"
            f"Berikut data keuangan lengkap dari dataset Superstore:\n{llm_input}\n\n"
            f"Keterangan field:\n"
            f"- total_sales: total pendapatan/omzet (USD)\n"
            f"- total_profit: total keuntungan bersih (USD)\n"
            f"- total_quantity: total unit terjual\n"
            f"- avg_discount: rata-rata diskon (%)\n"
            f"- total_orders: jumlah order unik\n"
            f"- total_customers: jumlah pelanggan unik\n"
            f"- per_kategori: omzet & profit per kategori produk\n"
            f"- per_region: omzet & profit per wilayah\n\n"
            f"Tolong jawab pertanyaan user: '{user_query}'\n"
            f"Jawab secara profesional dalam Bahasa Indonesia. "
            f"Gunakan istilah bisnis (omzet, laba, pendapatan). "
            f"JANGAN mengarang angka di luar data yang diberikan."
        )

        try:
            res = self.llm.invoke(llm_prompt)
            final_response = getattr(res, "content", str(res))
        except Exception as e:
            logger.error(f"Finance Agent LLM failed: {e}")
            s = finance_summary
            final_response = (
                f"**Ringkasan Keuangan Nikky Superstore:**\n"
                f"- Total Omzet: ${s.get('total_sales', 0):,.2f}\n"
                f"- Total Laba: ${s.get('total_profit', 0):,.2f}\n"
                f"- Total Unit Terjual: {s.get('total_quantity', 0):,}\n"
                f"- Rata-rata Diskon: {s.get('avg_discount', 0):.1f}%\n"
                f"- Total Order: {s.get('total_orders', 0):,}\n"
                f"- Total Pelanggan: {s.get('total_customers', 0):,}"
            )

        return {
            "agent_name": self.name,
            "response": final_response,
            "sql_executed": "pandas query on superstore.csv",
            "rows_retrieved": len(SuperstoreDataLoader.get_superstore()),
            "response_time": time.time() - start_time,
            "context_used": json.dumps(data_context, ensure_ascii=False)[:500]
        }
