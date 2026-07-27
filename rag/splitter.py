from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentSplitter:
    """Class to split LangChain documents into smaller chunks for vector database storage."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)
