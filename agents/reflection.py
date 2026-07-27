import time
import logging
import json
from agents.base import BaseAgent
from graph.state import EnterpriseState

logger = logging.getLogger(__name__)

class ReflectionAgent(BaseAgent):
    """Reflection Agent evaluates quality of overall generated answer and sets replan directives."""
    
    def __init__(self):
        super().__init__(
            name="Reflection Agent",
            role_description="Spesialis menguji kelengkapan data, akurasi, dan menentukan apakah butuh replanning."
        )

    def execute(self, state: EnterpriseState, task_desc: str) -> dict:
        start_time = time.time()
        
        user_query = state.get("user_query", "")
        findings = state.get("findings", "")
        
        prompt = f"""
Anda adalah Reflection Agent dari Nikky Superstore.
Tugas Anda adalah menilai apakah data/laporan analisis bisnis yang diajukan sudah menjawab pertanyaan user dengan cukup lengkap dan meyakinkan.

Pertanyaan User: {user_query}
Analisis Bisnis:
{findings}

Kembalikan jawaban dalam format JSON valid dengan field berikut:
- "confidence": nilai desimal dari 0.0 sampai 1.0 mewakili tingkat keyakinan Anda terhadap keakuratan data.
- "need_replan": boolean (true/false) apakah kita perlu merencanakan ulang/menambah data karena kurang lengkap.
- "feedback": saran singkat untuk replanning jika need_replan bernilai true.

Hanya keluarkan JSON valid tanpa penjelasan tambahan.
        """
        
        confidence = 0.9
        need_replan = False
        feedback = "Data dirasa sudah cukup menjawab."
        
        try:
            res = self.llm.invoke(prompt)
            clean_res = res.content.strip()
            if "```json" in clean_res:
                clean_res = clean_res.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_res:
                clean_res = clean_res.split("```")[1].split("```")[0].strip()
            
            data = json.loads(clean_res)
            confidence = float(data.get("confidence", 0.9))
            need_replan = bool(data.get("need_replan", False))
            feedback = data.get("feedback", "")
        except Exception as e:
            logger.error(f"Reflection Agent processing failed: {e}")

        return {
            "agent_name": self.name,
            "confidence": confidence,
            "need_replan": need_replan,
            "feedback": feedback,
            "response": f"Confidence: {confidence}, Replan needed: {need_replan}",
            "response_time": time.time() - start_time,
            "context_used": ""
        }

