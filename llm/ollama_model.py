import os
import logging
from langchain_ollama import ChatOllama
from utils.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from langchain_core.language_models.chat_models import BaseChatModel

class OllamaModel:
    """Manages the connection to the LLM instance (Groq Cloud or local Ollama)."""
    
    def __init__(self, temperature: float = 0.2):
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            try:
                import streamlit as st
                if "GROQ_API_KEY" in st.secrets:
                    groq_api_key = st.secrets["GROQ_API_KEY"]
            except Exception:
                pass
        
        if groq_api_key:
            try:
                from langchain_groq import ChatGroq
                self.llm = ChatGroq(
                    model_name="llama-3.1-8b-instant",
                    temperature=temperature,
                    api_key=groq_api_key
                )
            except ImportError:
                logging.error("langchain-groq not installed. Fallback to Ollama.")
                self.llm = ChatOllama(
                    base_url=OLLAMA_BASE_URL,
                    model=OLLAMA_MODEL,
                    temperature=temperature
                )
        else:
            self.llm = ChatOllama(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                temperature=temperature
            )

    def get_llm(self) -> BaseChatModel:
        return self.llm
