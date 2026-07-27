import logging
from typing import List
from langchain_core.documents import Document
from rag.retriever import DocumentRetriever

logger = logging.getLogger(__name__)

class RAGTool:
    """Wrapper class for retrieving context documents from ChromaDB."""
    
    def __init__(self):
        self.retriever = DocumentRetriever()

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        try:
            logger.info(f"Retrieving documents via RAGTool for: {query}")
            return self.retriever.retrieve(query, k=k)
        except Exception as e:
            logger.error(f"RAGTool retrieval error: {e}")
            return []
            
    def get_relevant_context(self, query: str, k: int = 3) -> str:
        docs = self.retrieve(query, k=k)
        if not docs:
            return "No document records found in database."
        
        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            context_parts.append(f"[Source: {source}]\n{doc.page_content}")
        
        return "\n\n".join(context_parts)
