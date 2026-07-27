import logging
from graph.state import EnterpriseState

logger = logging.getLogger(__name__)


def router_decision(state: EnterpriseState) -> str:
    """
    Decides next step after manager node.

    Returns:
        - "analyst"   : query kompleks, perlu Business Analyst untuk sintesis
        - "formatter" : query sederhana, langsung format & kirim ke user
    """
    needs_analyst = state.get("needs_analyst", False)
    tasks = state.get("tasks", [])

    # Paksa analyst jika > 1 agent dipanggil (query multi-domain)
    if needs_analyst or len(tasks) > 1:
        logger.info(f"Router: {len(tasks)} tasks → routing to analyst.")
        return "analyst"

    # Query sederhana (1 agent) → langsung formatter
    logger.info("Router: Single-agent task → routing to formatter (skip analyst).")
    return "formatter"
