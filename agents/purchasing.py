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
        
        prompt = f"""
Anda adalah AI Purchasing Agent dari Nikky Superstore.
Tugas Anda adalah merencanakan pembelian stok kembali dan mengoordinasikan pengadaan barang.
Tugas/Pertanyaan: {task_desc}

Jawab dengan ringkas dan profesional dalam Bahasa Indonesia.
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

