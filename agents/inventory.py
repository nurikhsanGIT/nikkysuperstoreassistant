import time
import json
import logging
from agents.base import BaseAgent
from graph.state import EnterpriseState
from utils.data_loader import SuperstoreDataLoader

logger = logging.getLogger(__name__)


class InventoryAgent(BaseAgent):
    """Inventory Agent: menampilkan data produk dari superstore.csv via pandas."""

    def __init__(self):
        super().__init__(
            name="Inventory Agent",
            role_description="Spesialis mengawasi daftar produk, kategori, stok (quantity), dan harga dari dataset Superstore."
        )

    def execute(self, state: EnterpriseState, task_desc: str) -> dict:
        start_time = time.time()
        user_query = state.get("user_query", task_desc)
        q = user_query.lower()

        # Tentukan filter berdasarkan keyword user
        category_filter = None
        sub_cat_filter = None
        keyword_filter = None
        top_n = 20

        if any(k in q for k in ["furniture", "furnitur"]):
            category_filter = "Furniture"
        elif any(k in q for k in ["technology", "teknologi", "elektronik"]):
            category_filter = "Technology"
        elif any(k in q for k in ["office", "supplies", "alat tulis"]):
            category_filter = "Office Supplies"

        if any(k in q for k in ["terlaris", "best seller", "paling laku", "top"]):
            df = SuperstoreDataLoader.get_top_products(top_n=10)
        elif any(k in q for k in ["slow", "lambat", "tidak laku"]):
            df = SuperstoreDataLoader.get_slow_products(bottom_n=10)
        else:
            df = SuperstoreDataLoader.get_products(
                keyword=keyword_filter,
                category=category_filter,
                sub_category=sub_cat_filter,
                top_n=top_n
            )

        if df.empty:
            return {
                "agent_name": self.name,
                "response": "Tidak ada data produk yang ditemukan sesuai permintaan.",
                "sql_executed": "pandas query on superstore.csv",
                "rows_retrieved": 0,
                "response_time": time.time() - start_time,
                "context_used": ""
            }

        data_context = df.to_dict(orient="records")
        llm_input = json.dumps(data_context, ensure_ascii=False)

        llm_prompt = (
            f"Anda adalah Inventory Agent untuk toko Nikky Superstore.\n"
            f"Berikut adalah data produk dari dataset Superstore (maks {top_n} item):\n{llm_input}\n\n"
            f"Keterangan field:\n"
            f"- Product_Name: nama produk\n"
            f"- Category: kategori produk (Furniture, Technology, Office Supplies)\n"
            f"- Sub_Category: sub-kategori\n"
            f"- Total_Sales: total nilai penjualan produk (USD)\n"
            f"- Total_Profit: total keuntungan (USD)\n"
            f"- Total_Quantity: jumlah unit terjual\n\n"
            f"Tolong jawab pertanyaan user: '{user_query}'\n"
            f"Sajikan dalam bentuk daftar poin-poin yang rapi menggunakan Bahasa Indonesia. "
            f"Sertakan nama produk, kategori, dan data relevan. "
            f"JANGAN mengarang data yang tidak ada di daftar di atas."
        )

        try:
            llm_result = self.llm.invoke(llm_prompt)
            final_response = getattr(llm_result, "content", str(llm_result))
        except Exception as e:
            logger.error(f"Inventory Agent LLM call failed: {e}")
            # Format manual sebagai fallback
            lines = []
            for i, row in enumerate(data_context[:15], 1):
                lines.append(
                    f"{i}. **{row.get('Product_Name', '-')}** "
                    f"(Kategori: {row.get('Category', '-')}, Sub: {row.get('Sub_Category', '-')}) | "
                    f"Sales: ${row.get('Total_Sales', 0):,.2f} | "
                    f"Qty: {row.get('Total_Quantity', 0):,}"
                )
            final_response = "\n".join(lines)

        return {
            "agent_name": self.name,
            "response": final_response,
            "sql_executed": "pandas query on superstore.csv",
            "rows_retrieved": len(df),
            "response_time": time.time() - start_time,
            "context_used": json.dumps(data_context[:5], ensure_ascii=False)
        }
