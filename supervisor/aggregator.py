import logging

logger = logging.getLogger(__name__)

class Aggregator:
    """Combines structured outputs or reports into consolidated variables."""
    @staticmethod
    def aggregate_logs(outputs: list) -> str:
        parts = []
        for out in outputs:
            parts.append(f"[{out.get('agent_name')}]: {out.get('response')}")
        return "\n".join(parts)
