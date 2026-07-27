import time
import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from agents.base import BaseAgent
from graph.state import EnterpriseState
from utils.data_loader import SuperstoreDataLoader

logger = logging.getLogger(__name__)


class CustomerAgent(BaseAgent):
    """Customer Service Agent: jawab FAQ/SOP via RAG, diperkaya data CSAT dari CSV."""

    def __init__(self):
        super().__init__(
            name="Customer Service Agent",
            role_description="Spesialis melayani pelanggan, menjawab pertanyaan FAQ/SOP, dan menganalisis tingkat kepuasan pelanggan (CSAT)."
        )

    def execute(self, state: EnterpriseState, task_desc: str) -> dict:
        start_time = time.time()
        user_query = state.get("user_query", task_desc)
        q = user_query.lower()

        # Ambil konteks RAG dari ChromaDB (SOP/FAQ dokumen)
        rag_context = self.rag_tool.get_relevant_context(user_query, k=3)

        # Tambahkan data CSAT dari Customer_support_data.csv jika relevan
        csat_context = ""
        if any(k in q for k in ["csat", "kepuasan", "rating", "komplain", "keluhan", "kategori"]):
            try:
                csat_summary = SuperstoreDataLoader.get_csat_summary()
                complaints = SuperstoreDataLoader.get_complaints_by_category(top_n=5)
                csat_context = (
                    f"\n\nData CSAT (Customer Satisfaction) dari sistem:\n"
                    f"- Rata-rata CSAT Score: {csat_summary.get('avg_csat', 'N/A')}/5.0\n"
                    f"- Total komplain tercatat: {csat_summary.get('total_complaints', 0):,}\n"
                    f"- Kategori komplain terbanyak: {csat_summary.get('top_category', 'N/A')}\n"
                )
                if not complaints.empty:
                    csat_context += "\nTop 5 Kategori Komplain:\n"
                    for _, row in complaints.iterrows():
                        csat_context += (
                            f"  - {row.get('category', '-')}: "
                            f"{row.get('Total', 0):,} kasus, "
                            f"avg CSAT {row.get('Avg_CSAT', 0):.2f}\n"
                        )
            except Exception as e:
                logger.warning(f"Failed to load CSAT data: {e}")

        combined_context = rag_context + csat_context

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Anda adalah AI Customer Service Agent dari Nikky Superstore.\n"
                "Informasi Sistem: Anda adalah bagian dari Enterprise AI Assistant Nikky Frozen yang memiliki 6 agen spesialis (Manager, Customer, Inventory, Finance, Sales, Marketing).\n"
                "Tugas Anda: menjawab pertanyaan pelanggan dengan ramah, informatif, dan ringkas "
                "sesuai data SOP/FAQ perusahaan dan data kepuasan pelanggan berikut:\n"
                "-----------------\n"
                "{context}\n"
                "-----------------\n"
                "Jawab pertanyaan berdasarkan konteks di atas secara jujur. "
                "Untuk pertanyaan tentang identitas/jumlah agen, gunakan Informasi Sistem di atas. "
                "Jika tidak ada di dokumen/data, katakan tidak tahu secara sopan.\n"
                "JANGAN menyebutkan kata 'RAG', 'retriever', 'konteks dokumen', atau 'database' dalam respon Anda."
            )),
            ("human", "{query}")
        ])

        try:
            chain = prompt | self.llm
            response = chain.invoke({"context": combined_context, "query": user_query})
            final_response = response.content
        except Exception as e:
            logger.error(f"Customer Service Agent execution failed: {e}")
            final_response = "Maaf, terjadi kendala saat memproses jawaban dari FAQ/SOP."

        return {
            "agent_name": self.name,
            "response": final_response,
            "response_time": time.time() - start_time,
            "context_used": combined_context[:500]
        }
