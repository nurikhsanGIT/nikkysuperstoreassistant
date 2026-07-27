from typing import List
from langchain_core.documents import Document
from rag.vector_store import VectorStoreManager
import logging

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Retrieves document chunks matching queries from ChromaDB."""
    
    def __init__(self):
        self.vector_manager = VectorStoreManager()

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        vector_store = self.vector_manager.get_vector_store()
        if vector_store is None:
            logger.warning("Vector store is not initialized. Returning empty list.")
            return []
        
        try:
            results = vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents for query '{query}': {e}")
            return []
