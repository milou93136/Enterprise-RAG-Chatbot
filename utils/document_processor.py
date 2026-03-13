"""
document_processor.py
=====================
Responsabilité : Chargement, nettoyage et découpage (chunking) des documents.

CONCEPT CLÉ — LE CHUNKING (Découpage du texte)
------------------------------------------------
Un LLM possède une "fenêtre de contexte" limitée (ex: 8 000 tokens).
Il est donc impossible d'envoyer un document PDF entier de 100 pages.

Solution : on découpe le document en petits morceaux appelés "chunks".
Chaque chunk est une portion de texte cohérente (ex: 500 mots avec un
chevauchement de 50 mots entre deux chunks adjacents).

Le chevauchement (overlap) est crucial : il évite de couper une phrase
ou une idée en plein milieu, garantissant ainsi que le contexte est
préservé aux frontières des chunks.

                   Document complet
          ┌─────────────────────────────┐
          │  ...texte...  │  ...texte...│
          └──────┬────────┴──────┬──────┘
                 │    CHUNKING   │
         ┌───────▼───┐       ┌───▼───────┐
         │  Chunk 1  │  ◄──► │  Chunk 2  │  (overlap)
         └───────────┘       └───────────┘
"""

import os
import re
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader


# ---------------------------------------------------------------------------
# Paramètres de chunking — modifiables selon le cas d'usage
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1000       # Taille cible de chaque chunk en caractères
CHUNK_OVERLAP = 150     # Nombre de caractères partagés entre deux chunks


def load_document(file_path: str) -> List[Document]:
    """
    Charge un document PDF ou texte depuis le chemin indiqué.

    Args:
        file_path: Chemin absolu vers le fichier (.pdf ou .txt).

    Returns:
        Liste d'objets Document LangChain (1 par page pour les PDF).

    Raises:
        ValueError: Si le format de fichier n'est pas supporté.
        FileNotFoundError: Si le fichier n'existe pas.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    elif extension in (".txt", ".md"):
        # encoding="utf-8" pour gérer les accents et caractères spéciaux
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
    else:
        raise ValueError(
            f"Format non supporté : '{extension}'. "
            "Utilisez un fichier .pdf, .txt ou .md."
        )

    return documents


def clean_text(text: str) -> str:
    """
    Nettoie le texte brut extrait d'un PDF.

    Les PDF produisent souvent des artefacts : espaces multiples,
    sauts de ligne superflus, caractères spéciaux parasites, etc.

    Args:
        text: Texte brut à nettoyer.

    Returns:
        Texte nettoyé.
    """
    # Remplace les séquences de whitespace multiples par un seul espace
    text = re.sub(r"\s+", " ", text)
    # Supprime les tirets de césure en fin de ligne (artefacts PDF)
    text = re.sub(r"-\s+", "", text)
    # Supprime les espaces en début et fin de chaîne
    text = text.strip()
    return text


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Découpe les documents en chunks sémantiquement cohérents.

    Utilise RecursiveCharacterTextSplitter de LangChain, qui essaie de
    couper le texte en respectant l'ordre de priorité suivant :
      1. Double saut de ligne  (\\n\\n)  → séparateur de paragraphe
      2. Saut de ligne simple  (\\n)     → séparateur de ligne
      3. Espace                (' ')    → séparateur de mot
      4. Caractère vide        ('')     → en dernier recours

    Ce découpage "intelligent" préserve au maximum la cohérence sémantique
    de chaque chunk, ce qui améliore la qualité de la recherche vectorielle.

    Args:
        documents: Liste de Documents LangChain (sortie de load_document).

    Returns:
        Liste de Documents découpés, prêts pour la vectorisation.
    """
    # Nettoyage du texte de chaque document avant découpage
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    # Initialisation du splitter avec les paramètres définis plus haut
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    # Ajout d'un index à chaque chunk pour la traçabilité
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_total"] = len(chunks)

    return chunks


def process_uploaded_file(file_path: str) -> List[Document]:
    """
    Pipeline complet : charge → nettoie → découpe un document.

    Cette fonction est le point d'entrée unique utilisé par app.py.

    Args:
        file_path: Chemin du fichier uploadé par l'utilisateur.

    Returns:
        Liste de chunks Document prêts à être vectorisés.
    """
    # Étape 1 : Chargement
    raw_documents = load_document(file_path)

    # Étape 2 : Découpage en chunks
    chunks = chunk_documents(raw_documents)

    return chunks
