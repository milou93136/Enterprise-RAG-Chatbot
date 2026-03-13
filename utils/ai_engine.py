"""
ai_engine.py
============
Responsabilité : Vectorisation, stockage, recherche et génération de réponses.

CONCEPT CLÉ — LA VECTOR SEARCH (Recherche Vectorielle)
-------------------------------------------------------
Un embedding est la représentation numérique d'un texte sous forme de vecteur
(liste de nombres à virgule flottante) dans un espace de haute dimension.

Des textes sémantiquement proches auront des vecteurs proches dans cet espace
(distance cosinus faible). Cela permet une recherche par SENS et non par mots-clés.

Exemple :
  "Comment fonctionne un moteur ?"  →  [0.12, -0.45, 0.78, ...]
  "Quel est le principe d'un moteur ?" →  [0.11, -0.44, 0.76, ...]  ← proche !
  "Quelle est la météo demain ?"    →  [0.90,  0.23, -0.12, ...]  ← loin

Pipeline RAG complet :
                                                   ┌─────────────┐
  Question utilisateur ──► Embedding ──► Vecteur ──► Vector DB   │
                                                   │  (ChromaDB) │
                                                   └──────┬──────┘
                                                          │ Top-K chunks
                                                          ▼
                           ┌──────────────────────────────────────┐
                           │  LLM  (GPT-3.5 / Mistral)            │
                           │  Prompt = Question + Contexte (RAG)  │
                           └──────────────────┬───────────────────┘
                                              │
                                              ▼
                                    Réponse ancrée dans le document
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings


# ---------------------------------------------------------------------------
# Paramètres de recherche vectorielle
# ---------------------------------------------------------------------------
TOP_K_RESULTS = 4          # Nombre de chunks retournés par la recherche
CHROMA_PERSIST_DIR = "./chroma_db"   # Répertoire de persistance ChromaDB
COLLECTION_NAME = "rag_documents"   # Nom de la collection dans ChromaDB


# ---------------------------------------------------------------------------
# Sélection du modèle d'embeddings
# ---------------------------------------------------------------------------

def get_embeddings(use_openai: bool = True) -> Embeddings:
    """
    Retourne le modèle d'embeddings configuré.

    Deux modes disponibles :
      - OpenAI (text-embedding-3-small) : haute qualité, nécessite une clé API.
      - HuggingFace (all-MiniLM-L6-v2) : 100% local, gratuit, légèrement moins
        performant mais suffisant pour la majorité des cas d'usage.

    Args:
        use_openai: True pour OpenAI, False pour HuggingFace local.

    Returns:
        Instance du modèle d'embeddings.
    """
    if use_openai:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )
    else:
        # Modèle local HuggingFace — téléchargé automatiquement au premier lancement
        # Poids ~90 MB, aucune clé API requise
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


# ---------------------------------------------------------------------------
# Sélection du LLM
# ---------------------------------------------------------------------------

def get_llm(
    use_openai: bool = True,
    temperature: float = 0.0,
) -> BaseChatModel:
    """
    Retourne le LLM configuré pour la génération de réponses.

    temperature=0.0 : réponses déterministes, idéal pour la factualité (RAG).
    Augmenter la température rend les réponses plus créatives/variées.

    Args:
        use_openai: True pour GPT-3.5-turbo (OpenAI), False pour un modèle
                    local via Ollama (ex: Mistral).
        temperature: Créativité du modèle (0.0 = factuel, 1.0 = créatif).

    Returns:
        Instance du LLM.
    """
    if use_openai:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=temperature,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )
    else:
        # Ollama doit être installé et le modèle téléchargé localement.
        # Commande : ollama pull mistral
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model="mistral",
            temperature=temperature,
        )


# ---------------------------------------------------------------------------
# Gestion de la base vectorielle (ChromaDB)
# ---------------------------------------------------------------------------

def build_vector_store(
    chunks: List[Document],
    embeddings: Embeddings,
) -> Chroma:
    """
    Crée ou écrase la base vectorielle ChromaDB à partir des chunks.

    Le processus de vectorisation :
      1. Pour chaque chunk, le modèle d'embeddings génère un vecteur.
      2. Ces vecteurs sont indexés dans ChromaDB (structure HNSW).
      3. La base est persistée sur disque pour les sessions suivantes.

    Args:
        chunks: Liste de Documents (sortie de document_processor).
        embeddings: Modèle d'embeddings à utiliser.

    Returns:
        Instance ChromaDB chargée et prête pour la recherche.
    """
    # Chroma.from_documents vectorise ET indexe les chunks en une seule étape
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return vector_store


def load_vector_store(embeddings: Embeddings) -> Optional[Chroma]:
    """
    Charge une base vectorielle existante depuis le disque.

    Args:
        embeddings: Modèle d'embeddings (doit être le même que lors de la création).

    Returns:
        Instance ChromaDB ou None si aucune base n'existe.
    """
    if not os.path.exists(CHROMA_PERSIST_DIR):
        return None

    try:
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        # Vérifie que la collection contient bien des données
        if vector_store._collection.count() == 0:
            return None
        return vector_store
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Création de la chaîne RAG (Retrieval-Augmented Generation)
# ---------------------------------------------------------------------------

def build_rag_chain(
    vector_store: Chroma,
    llm: BaseChatModel,
) -> ConversationalRetrievalChain:
    """
    Assemble la chaîne RAG conversationnelle avec LangChain.

    Architecture de la chaîne :
      Question  ──►  Retriever (Vector Search)  ──►  Top-K chunks
                                                          │
      Historique de conversation ─────────────────────────┤
                                                          ▼
                                                   LLM (Génération)
                                                          │
                                                          ▼
                                                       Réponse

    Le Retriever est configuré en mode MMR (Maximal Marginal Relevance) :
    il sélectionne des chunks pertinents ET diversifiés pour éviter la
    redondance dans le contexte fourni au LLM.

    Args:
        vector_store: Base vectorielle peuplée.
        llm: Modèle de langage pour la génération.

    Returns:
        Chaîne LangChain prête à recevoir des questions.
    """
    # Le Retriever est l'interface entre ChromaDB et la chaîne LangChain.
    # MMR = Maximal Marginal Relevance : équilibre pertinence et diversité.
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": TOP_K_RESULTS,
            "fetch_k": TOP_K_RESULTS * 2,  # Candidats pré-sélectionnés pour MMR
        },
    )

    # Mémoire conversationnelle : conserve l'historique des échanges.
    # Permet au LLM de comprendre les questions de suivi (ex : "et lui ?")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # Assemblage de la chaîne complète
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,   # Retourne les chunks utilisés
        verbose=False,
    )

    return chain


# ---------------------------------------------------------------------------
# Fonction principale de traitement d'une question
# ---------------------------------------------------------------------------

def ask_question(
    chain: ConversationalRetrievalChain,
    question: str,
) -> Tuple[str, List[Document]]:
    """
    Pose une question à la chaîne RAG et retourne la réponse + les sources.

    Args:
        chain: Chaîne RAG configurée (sortie de build_rag_chain).
        question: Question en langage naturel de l'utilisateur.

    Returns:
        Tuple (réponse_texte, liste_de_chunks_sources).
    """
    result: Dict[str, Any] = chain.invoke({"question": question})

    answer: str = result.get("answer", "Je n'ai pas trouvé de réponse.")
    source_docs: List[Document] = result.get("source_documents", [])

    return answer, source_docs


# ---------------------------------------------------------------------------
# Pipeline complet (point d'entrée pour app.py)
# ---------------------------------------------------------------------------

def initialize_rag_pipeline(
    chunks: List[Document],
    use_openai: bool = True,
) -> ConversationalRetrievalChain:
    """
    Pipeline complet d'initialisation RAG : embeddings → vector store → chaîne.

    Args:
        chunks: Chunks issus de document_processor.process_uploaded_file.
        use_openai: Sélecteur OpenAI vs HuggingFace / Ollama.

    Returns:
        Chaîne RAG prête à l'emploi.
    """
    embeddings = get_embeddings(use_openai=use_openai)
    vector_store = build_vector_store(chunks, embeddings)
    llm = get_llm(use_openai=use_openai)
    chain = build_rag_chain(vector_store, llm)
    return chain
