import time
import logging
from agents.base import BaseAgent
from graph.state import EnterpriseState

logger = logging.getLogger(__name__)

class PurchasingAgent(BaseAgent):
    """Purchasing Agent handles restocking and vendor interaction plans."""
    
    def __init__(self):
        super().__init__(
            name="Purchasing Agent",
            role_description="Spesialis merencanakan pembelian stock produk dari supplier."
        )

    def execute(self, state: EnterpriseState, task_desc: str) -> dict:
        start_time = time.time()
        
        from utils.data_loader import SuperstoreDataLoader
        import json
        
        user_query = state.get("user_query", task_desc).lower()
        import re
        if any(k in user_query for k in ["bawah", "kurang dari", "sedikit", "low stock", "stok"]):
            threshold = 10
            match = re.search(r'(?:bawah|kurang dari|<)\s*(\d+)', user_query)
            if match:
                threshold = int(match.group(1))
            products_df = SuperstoreDataLoader.get_low_stock_products(max_qty=threshold, top_n=10)
        else:
            products_df = SuperstoreDataLoader.get_top_products(top_n=5)

        context_data = products_df.to_dict(orient="records") if not products_df.empty else []
        llm_input = json.dumps(context_data, ensure_ascii=False)

        prompt = f"""
Anda adalah AI Purchasing Agent dari Josjis Super Store.
Berikut adalah daftar produk yang relevan untuk restock / analisis stok:
{llm_input}


Tugas Anda adalah merencanakan pembelian stok kembali ke supplier. 
PERATURAN KETAT:
1. HANYA rekomendasikan produk dari data di atas.
2. DILARANG KERAS mengarang produk fiktif seperti pakaian wanita, baju kaos, sepatu, dll. Toko ini HANYA menjual Furniture, Technology, dan Office Supplies.
3. Jawab dengan ringkas dan profesional dalam Bahasa Indonesia.

Tugas/Pertanyaan: {task_desc}
        """
        try:
            res = self.llm.invoke(prompt)
            response_text = res.content
        except Exception as e:
            logger.error(f"Purchasing Agent LLM call failed: {e}")
            response_text = "Gagal memproses tugas purchasing."

        return {
            "agent_name": self.name,
            "response": response_text,
            "response_time": time.time() - start_time,
            "context_used": ""
        }

