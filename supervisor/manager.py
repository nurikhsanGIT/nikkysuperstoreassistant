import logging
from typing import Dict, Any, List
from agents.sales import SalesAgent
from agents.inventory import InventoryAgent
from agents.finance import FinanceAgent
from agents.purchasing import PurchasingAgent
from agents.customer import CustomerAgent
from agents.marketing import MarketingAgent
from graph.state import EnterpriseState

logger = logging.getLogger(__name__)

class Manager:
    """Supervisor Manager responsible for routing tasks to the appropriate Specialist Agents sequentially."""
    
    def __init__(self):
        self.agents = {
            "sales": SalesAgent(),
            "inventory": InventoryAgent(),
            "finance": FinanceAgent(),
            "purchasing": PurchasingAgent(),
            "customer": CustomerAgent(),
            "marketing": MarketingAgent()
        }

    def execute_tasks(self, state: EnterpriseState) -> List[Dict[str, Any]]:
        tasks = state.get("tasks", [])
        agent_outputs = []
        
        for t in tasks:
            agent_name = t.get("agent")
            task_desc = t.get("task")
            
            if agent_name in self.agents:
                logger.info(f"Manager assigning task: {task_desc} to specialist: {agent_name}")
                agent = self.agents[agent_name]
                try:
                    result = agent.run_lifecycle(state, task_desc)
                    agent_outputs.append(result)
                except Exception as e:
                    logger.error(f"Failed to execute agent {agent_name}: {e}")
            else:
                logger.warning(f"No agent registered for name: {agent_name}")
                
        return agent_outputs
