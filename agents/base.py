import logging
from typing import Dict, Any
from llm.ollama_model import OllamaModel
from tools.sql_tool import SQLTool
from tools.rag_tool import RAGTool
from tools.laravel_tool import LaravelTool
from graph.state import EnterpriseState

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base class for all specialist agents defining the standard lifecycle."""
    
    def __init__(self, name: str, role_description: str):
        self.name = name
        self.role_description = role_description
        self.llm = OllamaModel(temperature=0.1).get_llm()
        self.rag_tool = RAGTool()
        self.sql_tool = SQLTool()
        self.laravel_tool = LaravelTool()

    def plan(self, state: EnterpriseState, task_desc: str) -> Dict[str, Any]:
        """Lifecycle step 1: Plan the action or strategy for the task."""
        return {"agent": self.name, "phase": "plan", "status": "done"}

    def retrieve(self, state: EnterpriseState, task_desc: str) -> Dict[str, Any]:
        """Lifecycle step 2: Retrieve any necessary database context or documents."""
        return {"agent": self.name, "phase": "retrieve", "status": "done"}

    def execute(self, state: EnterpriseState, task_desc: str) -> Dict[str, Any]:
        """Lifecycle step 3: Process the task using model and tool contexts."""
        raise NotImplementedError("Specialist agent must implement execute()")

    def reflect(self, state: EnterpriseState, response: str) -> Dict[str, Any]:
        """Lifecycle step 4: Reflect on the output self-consistency or validity."""
        return {"agent": self.name, "phase": "reflect", "status": "done"}

    def finish(self, state: EnterpriseState, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Lifecycle step 5: Finalize and format the output."""
        return execution_result
        
    def run_lifecycle(self, state: EnterpriseState, task_desc: str) -> Dict[str, Any]:
        """Executes the standard lifecycle for the agent."""
        logger.info(f"Running lifecycle for {self.name} with task: {task_desc}")
        self.plan(state, task_desc)
        self.retrieve(state, task_desc)
        exec_res = self.execute(state, task_desc)
        self.reflect(state, exec_res.get("response", ""))
        return self.finish(state, exec_res)
