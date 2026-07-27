import logging
from graph.state import EnterpriseState
from planner.planner import Planner
from supervisor.manager import Manager
from agents.analyst import BusinessAnalystAgent
from agents.reflection import ReflectionAgent

logger = logging.getLogger(__name__)

def planner_node(state: EnterpriseState) -> dict:
    logger.info("Entering Planner Node")
    planner = Planner()
    tasks = planner.generate_plan(state["user_query"])
    return {"tasks": tasks}

def manager_node(state: EnterpriseState) -> dict:
    logger.info("Entering Manager Node")
    manager = Manager()
    agent_outputs = manager.execute_tasks(state)

    # Store sql, rag, tool results for transparency
    sql_results = []
    rag_results = []
    tool_results = []

    for out in agent_outputs:
        if "sql_executed" in out:
            sql_results.append({"agent": out["agent_name"], "sql": out["sql_executed"], "rows": out.get("rows_retrieved", 0)})
        if out.get("context_used"):
            rag_results.append({"agent": out["agent_name"], "context": out["context_used"]})
        tool_results.append({"agent": out["agent_name"], "response_time": out.get("response_time", 0.0)})

    # Determine if analyst step is needed (more than one task implies need for synthesis)
    needs_analyst = len(state.get("tasks", [])) > 1

    return {
        "agent_outputs": agent_outputs,
        "sql_results": sql_results,
        "rag_results": rag_results,
        "tool_results": tool_results,
        "needs_analyst": needs_analyst,
        "need_replan": False,
        "retry_count": state.get("retry_count", 0)
    }

def formatter_node(state: EnterpriseState) -> dict:
    """Create the final user‑facing answer.
    For now we simply concatenate the outputs of specialist agents.
    """
    logger.info("Entering Formatter Node")
    # If analyst already produced a final answer, use it
    if state.get("final_answer"):
        answer = state["final_answer"]
    else:
        # Simple aggregation of agent outputs
        parts = []
        for out in state.get("agent_outputs", []):
            parts.append(out.get("response", ""))
        answer = "\n".join(filter(None, parts))
    return {"final_answer": answer}

def analyst_node(state: EnterpriseState) -> dict:
    logger.info("Entering Business Analyst Node")
    analyst = BusinessAnalystAgent()
    res = analyst.run_lifecycle(state, "Analyze the outputs from all specialist agents.")
    return {
        "findings": res.get("findings", ""),
        "recommendations": res.get("recommendations", ""),
        "final_answer": res.get("response", "")
    }

def reflection_node(state: EnterpriseState) -> dict:
    logger.info("Entering Reflection Node")
    reflection = ReflectionAgent()
    res = reflection.run_lifecycle(state, "Evaluate accuracy and sufficiency.")
    return {
        "confidence": res.get("confidence", 0.9),
        "need_replan": res.get("need_replan", False)
    }
