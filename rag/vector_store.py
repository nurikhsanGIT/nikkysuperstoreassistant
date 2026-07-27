import os
import shutil
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rag.embedding import EmbeddingModel
from utils.config import CHROMA_DB_DIR
import logging

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages the ChromaDB vector database creation, loading, and reset operations."""
    
    def __init__(self):
        self.embedding_model = EmbeddingModel().get_embeddings()
        self.vector_store = None
        self.initialize_store()

    def initialize_store(self):
        try:
            self.vector_store = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=self.embedding_model
            )
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")

    def add_documents(self, documents: List[Document]):
        if not documents:
            return
        if self.vector_store is None:
            self.initialize_store()
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Successfully added {len(documents)} document chunks to ChromaDB.")
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")

    def reset_store(self):
        """Delete current vector database and reinitialize empty."""
        try:
            if os.path.exists(CHROMA_DB_DIR):
                # Close connection if possible (Chroma closes automatically or when garbage collected)
                self.vector_store = None
                shutil.rmtree(CHROMA_DB_DIR)
                logger.info("ChromaDB persistent directory deleted.")
            self.initialize_store()
        except Exception as e:
            logger.error(f"Failed to reset ChromaDB: {e}")

    def get_vector_store(self) -> Chroma:
        if self.vector_store is None:
            self.initialize_store()
        return self.vector_store
