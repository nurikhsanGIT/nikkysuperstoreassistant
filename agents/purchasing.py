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
        
        # Berikan data produk terlaris sebagai acuan restock
        top_products = SuperstoreDataLoader.get_top_products(top_n=5)
        context_data = top_products.to_dict(orient="records") if not top_products.empty else []
        llm_input = json.dumps(context_data, ensure_ascii=False)

        prompt = f"""
Anda adalah AI Purchasing Agent dari Josjis Super Store.
Berikut adalah daftar produk dengan penjualan tertinggi saat ini yang relevan untuk restock:
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

