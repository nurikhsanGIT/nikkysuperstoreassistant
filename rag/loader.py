import os
import pandas as pd
from typing import List
from langchain_core.documents import Document
from pypdf import PdfReader
import logging

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Class to load different document types (PDF, CSV, TXT) into LangChain Document format."""
    
    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        documents = []
        try:
            reader = PdfReader(file_path)
            filename = os.path.basename(file_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": filename, "page": page_num + 1, "type": "pdf"}
                    ))
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
        return documents

    @staticmethod
    def load_csv(file_path: str) -> List[Document]:
        documents = []
        try:
            df = pd.read_csv(file_path)
            filename = os.path.basename(file_path)
            for idx, row in df.iterrows():
                # Convert row into a text representation
                content = ", ".join([f"{col}: {val}" for col, val in row.items()])
                documents.append(Document(
                    page_content=content,
                    metadata={"source": filename, "row": idx + 1, "type": "csv"}
                ))
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {e}")
        return documents

    @staticmethod
    def load_txt(file_path: str) -> List[Document]:
        documents = []
        try:
            filename = os.path.basename(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            if content.strip():
                documents.append(Document(
                    page_content=content,
                    metadata={"source": filename, "type": "txt"}
                ))
        except Exception as e:
            logger.error(f"Error loading TXT {file_path}: {e}")
        return documents

    def load_directory(self, dir_path: str) -> List[Document]:
        all_documents = []
        if not os.path.exists(dir_path):
            return all_documents

        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isdir(file_path):
                continue
            
            ext = filename.split(".")[-1].lower()
            if ext == "pdf":
                all_documents.extend(self.load_pdf(file_path))
            elif ext == "csv":
                all_documents.extend(self.load_csv(file_path))
            elif ext in ["txt", "md"]:
                all_documents.extend(self.load_txt(file_path))
                
        return all_documents
