from langgraph.graph import StateGraph, END
from graph.state import EnterpriseState
from graph.nodes import planner_node, manager_node, analyst_node, reflection_node, formatter_node
from graph.router import router_decision

# Define Graph
workflow = StateGraph(EnterpriseState)

# Add Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("manager", manager_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("reflection", reflection_node)
workflow.add_node("formatter", formatter_node)

# Set Entry Point
workflow.set_entry_point("planner")

# planner → manager selalu
workflow.add_edge("planner", "manager")

# manager → analyst (jika kompleks) ATAU langsung formatter (jika simple)
workflow.add_conditional_edges(
    "manager",
    router_decision,
    {
        "analyst": "analyst",
        "formatter": "formatter",
    }
)

# analyst → reflection → formatter
workflow.add_edge("analyst", "reflection")
workflow.add_edge("reflection", "formatter")
workflow.add_edge("formatter", END)

agent_graph = workflow.compile()
