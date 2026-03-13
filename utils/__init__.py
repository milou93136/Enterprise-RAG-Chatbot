"""
Package utils — Logique métier du chatbot RAG.

Modules :
  - document_processor : Chargement, nettoyage et chunking des documents.
  - ai_engine          : Embeddings, vector store et chaîne RAG LangChain.
"""

from utils.document_processor import process_uploaded_file
from utils.ai_engine import initialize_rag_pipeline, ask_question

__all__ = [
    "process_uploaded_file",
    "initialize_rag_pipeline",
    "ask_question",
]
