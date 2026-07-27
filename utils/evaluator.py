from typing import List
import time

class ResponseEvaluator:
    """Evaluates the quality, accuracy, efficiency, and hallucination metrics of the Agent's response."""
    
    @staticmethod
    def evaluate(query: str, response: str, context: str, response_time: float) -> dict:
        # 1. Hallucination check
        # Simple keyword heuristic check: count overlapping words of length > 4 from response in the context
        if not context or context == "No document records found in database.":
            hallucination_score = 100.0  # If no context was provided, it's a zero-shot response
            hallucination_label = "N/A (No Context)"
        else:
            response_words = [w.lower().strip(",.?!()\"'") for w in response.split() if len(w) > 4]
            context_lower = context.lower()
            if response_words:
                matches = sum(1 for w in response_words if w in context_lower)
                overlap_ratio = matches / len(response_words)
                # If more than 30% of key words in output exist in retrieved text, low chance of hallucination
                if overlap_ratio > 0.35:
                    hallucination_score = 0.0
                    hallucination_label = "Low (Grounded)"
                elif overlap_ratio > 0.15:
                    hallucination_score = 50.0
                    hallucination_label = "Medium (Possible Hallucination)"
                else:
                    hallucination_score = 100.0
                    hallucination_label = "High (Not grounded in context)"
            else:
                hallucination_score = 0.0
                hallucination_label = "Low"

        # 2. Accuracy check (Heuristic based on query and context relevance)
        # Check if key query terms exist in the response
        query_words = [w.lower().strip(",.?!()\"'") for w in query.split() if len(w) > 3]
        if query_words:
            matched_query_words = sum(1 for w in query_words if w in response.lower())
            accuracy = (matched_query_words / len(query_words)) * 100
        else:
            accuracy = 100.0
        
        # 3. Effectiveness check (Score based on sentence length and directness)
        response_len = len(response.split())
        if response_len > 15 and response_len < 120:
            effectiveness = 100.0  # Perfect length
        elif response_len >= 120:
            effectiveness = 80.0   # Too verbose
        else:
            effectiveness = 50.0   # Too brief
            
        # 4. Efficiency
        # Threshold: < 2.0s is Excellent, < 5.0s is Average, >= 5.0s is Slow
        if response_time < 2.0:
            efficiency_rating = "Excellent"
        elif response_time < 5.0:
            efficiency_rating = "Good"
        else:
            efficiency_rating = "Slow"

        # 5. Explainability (Extract sources)
        sources = []
        import re
        source_matches = re.findall(r"\[Source:\s*([^\]]+)\]", context)
        if source_matches:
            sources = list(set(source_matches))
        else:
            sources = ["General Knowledge (No direct document source)"]

        return {
            "accuracy": round(accuracy, 2),
            "effectiveness": effectiveness,
            "efficiency_seconds": round(response_time, 2),
            "efficiency_rating": efficiency_rating,
            "hallucination_score": hallucination_score,
            "hallucination_rating": hallucination_label,
            "sources_used": sources
        }
