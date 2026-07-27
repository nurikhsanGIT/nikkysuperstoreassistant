import time
import json
import logging
from agents.base import BaseAgent
from graph.state import EnterpriseState
from utils.data_loader import SuperstoreDataLoader

logger = logging.getLogger(__name__)


class SalesAgent(BaseAgent):
    """Sales Agent: analisis transaksi penjualan dari superstore.csv."""

    def __init__(self):
        super().__init__(
            name="Sales Agent",
            role_description="Spesialis melacak transaksi penjualan, region, dan segmen pelanggan dari dataset Superstore."
        )

    def execute(self, state: EnterpriseState, task_desc: str) -> dict:
        start_time = time.time()
        user_query = state.get("user_query", task_desc)
        q = user_query.lower()

        # Pilih data yang relevan
        if any(k in q for k in ["region", "wilayah", "area"]):
            df = SuperstoreDataLoader.get_sales_by_region()
            data_label = "penjualan per region"
        elif any(k in q for k in ["segmen", "segment", "pelanggan", "customer"]):
            df = SuperstoreDataLoader.get_sales_by_segment()
            data_label = "penjualan per segmen pelanggan"
        elif any(k in q for k in ["terlaris", "best seller", "top produk", "paling laku"]):
            df = SuperstoreDataLoader.get_top_products(top_n=10)
            data_label = "produk terlaris"
        else:
            # Default: ringkasan + region
            finance = SuperstoreDataLoader.get_finance_summary()
            region_df = SuperstoreDataLoader.get_sales_by_region()
            data_context = {
                "summary": finance,
                "by_region": region_df.to_dict(orient="records")
            }
            llm_input = json.dumps(data_context, ensure_ascii=False)

            llm_prompt = (
                f"Anda adalah Sales Agent untuk toko Nikky Superstore.\n"
                f"Berikut adalah data penjualan dari dataset Superstore:\n{llm_input}\n\n"
                f"Keterangan:\n"
                f"- total_sales: total nilai penjualan (USD)\n"
                f"- total_profit: total keuntungan (USD)\n"
                f"- total_orders: jumlah order unik\n"
                f"- by_region: breakdown penjualan per wilayah\n\n"
                f"Tolong jawab pertanyaan user: '{user_query}'\n"
                f"Jawab dalam Bahasa Indonesia yang profesional dan informatif."
            )
            try:
                res = self.llm.invoke(llm_prompt)
                final_response = getattr(res, "content", str(res))
            except Exception as e:
                logger.error(f"Sales Agent LLM failed: {e}")
                s = data_context["summary"]
                final_response = (
                    f"**Ringkasan Penjualan Superstore:**\n"
                    f"- Total Penjualan: ${s.get('total_sales', 0):,.2f}\n"
                    f"- Total Profit: ${s.get('total_profit', 0):,.2f}\n"
                    f"- Total Orders: {s.get('total_orders', 0):,}\n"
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

        if df.empty:
            return {
                "agent_name": self.name,
                "response": "Tidak ada data penjualan yang ditemukan.",
                "sql_executed": "pandas query on superstore.csv",
                "rows_retrieved": 0,
                "response_time": time.time() - start_time,
                "context_used": ""
            }

        data_context = df.to_dict(orient="records")
        llm_input = json.dumps(data_context, ensure_ascii=False)

        llm_prompt = (
            f"Anda adalah Sales Agent untuk toko Nikky Superstore.\n"
            f"Berikut adalah data {data_label} dari dataset:\n{llm_input}\n\n"
            f"Tolong jawab pertanyaan user: '{user_query}'\n"
            f"Jawab dalam Bahasa Indonesia yang profesional. "
            f"Sajikan data dalam format daftar poin-poin yang mudah dibaca. "
            f"JANGAN mengarang data di luar yang diberikan."
        )

        try:
            res = self.llm.invoke(llm_prompt)
            final_response = getattr(res, "content", str(res))
        except Exception as e:
            logger.error(f"Sales Agent LLM failed: {e}")
            lines = [f"**Data {data_label}:**"]
            for row in data_context:
                line = " | ".join(f"{k}: {v}" for k, v in row.items())
                lines.append(f"- {line}")
            final_response = "\n".join(lines)

        return {
            "agent_name": self.name,
            "response": final_response,
            "sql_executed": "pandas query on superstore.csv",
            "rows_retrieved": len(df),
            "response_time": time.time() - start_time,
            "context_used": json.dumps(data_context, ensure_ascii=False)[:500]
        }
