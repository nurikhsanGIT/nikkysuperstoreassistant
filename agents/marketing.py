import time
import json
import logging
from agents.base import BaseAgent
from graph.state import EnterpriseState
from utils.data_loader import SuperstoreDataLoader

logger = logging.getLogger(__name__)


class MarketingAgent(BaseAgent):
    """Marketing Agent: analisis produk terlaris, slow moving, dan rekomendasi promosi."""

    def __init__(self):
        super().__init__(
            name="Marketing Agent",
            role_description="Spesialis merancang strategi promosi berdasarkan data produk terlaris dan slow moving dari dataset Superstore."
        )

    def execute(self, state: EnterpriseState, task_desc: str) -> dict:
        start_time = time.time()
        user_query = state.get("user_query", task_desc)
        q = user_query.lower()

        # Tentukan data yang dibutuhkan
        if any(k in q for k in ["slow", "lambat", "tidak laku", "diskon slow"]):
            df_top = None
            df_slow = SuperstoreDataLoader.get_slow_products(bottom_n=10)
            data_label = "produk slow-moving (perlu promosi)"
            data_context = {"slow_moving": df_slow.to_dict(orient="records")}
        elif any(k in q for k in ["terlaris", "best seller", "laku", "populer"]):
            df_top = SuperstoreDataLoader.get_top_products(top_n=10)
            df_slow = None
            data_label = "produk terlaris (best seller)"
            data_context = {"top_products": df_top.to_dict(orient="records")}
        else:
            # Default: gabungan top & slow untuk buat strategi marketing
            df_top = SuperstoreDataLoader.get_top_products(top_n=5)
            df_slow = SuperstoreDataLoader.get_slow_products(bottom_n=5)
            finance = SuperstoreDataLoader.get_finance_summary()
            data_context = {
                "top_products": df_top.to_dict(orient="records"),
                "slow_moving": df_slow.to_dict(orient="records"),
                "finance_summary": finance,
            }
            data_label = "analisis marketing lengkap"

        llm_input = json.dumps(data_context, ensure_ascii=False)

        llm_prompt = (
            f"Anda adalah Marketing Agent dari Nikky Superstore.\n"
            f"Berikut data {data_label} dari dataset Superstore:\n{llm_input}\n\n"
            f"Keterangan field:\n"
            f"- Product_Name: nama produk\n"
            f"- Category / Sub_Category: kategori produk\n"
            f"- Total_Sales: total nilai penjualan (USD)\n"
            f"- Total_Quantity: total unit terjual\n"
            f"- Total_Profit: total profit (USD)\n\n"
            f"Tolong jawab pertanyaan user: '{user_query}'\n"
            f"Berikan analisis dan rekomendasi strategi marketing/promosi yang konkret dalam Bahasa Indonesia. "
            f"Sebutkan nama produk nyata dari data di atas. "
            f"JANGAN mengarang produk atau angka yang tidak ada dalam data."
        )

        try:
            res = self.llm.invoke(llm_prompt)
            final_response = getattr(res, "content", str(res))
        except Exception as e:
            logger.error(f"Marketing Agent LLM failed: {e}")
            lines = [f"**Data {data_label}:**"]
            for key, records in data_context.items():
                if isinstance(records, list):
                    lines.append(f"\n*{key}:*")
                    for row in records[:5]:
                        name = row.get("Product_Name", row.get("Category", "-"))
                        sales = row.get("Total_Sales", 0)
                        qty = row.get("Total_Quantity", 0)
                        lines.append(f"  - {name} | Sales: ${sales:,.2f} | Qty: {qty:,}")
            final_response = "\n".join(lines)

        return {
            "agent_name": self.name,
            "response": final_response,
            "sql_executed": "pandas query on superstore.csv",
            "rows_retrieved": sum(len(v) for v in data_context.values() if isinstance(v, list)),
            "response_time": time.time() - start_time,
            "context_used": json.dumps(data_context, ensure_ascii=False)[:500]
        }
