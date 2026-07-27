from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.config import EMBEDDING_MODEL_NAME

class EmbeddingModel:
    """Class to manage HuggingFace Sentence Transformers embeddings."""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        return self.embeddings
