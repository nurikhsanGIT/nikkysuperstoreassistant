from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from graph.workflow import agent_graph
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Nikky Frozen Enterprise AI API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    text: str

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[ChatMessage]] = []

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    logger.info(f"Received query: {req.query}")
    try:
        start_time = time.time()
        
        # Invoke LangGraph workflow
        state_input = {
            "user_query": req.query,
            "user_id": "api_user",
            "session_id": "api_session",
            "intent": "",
            "tasks": [],
            "sql_results": [],
            "rag_results": [],
            "tool_results": [],
            "agent_outputs": [],
            "findings": "",
            "recommendations": "",
            "confidence": 1.0,
            "need_replan": False,
            "retry_count": 0,
            "final_answer": ""
        }
        
        result_state = agent_graph.invoke(state_input)
        response_time = time.time() - start_time
        
        return {
            "query": req.query,
            "next_agent": result_state.get("tasks", [{"agent": "general"}])[0].get("agent", "general") if result_state.get("tasks") else "general",
            "agent_responses": result_state.get("agent_outputs", []),
            "final_response": result_state.get("final_answer", ""),
            "response_time": response_time
        }
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
