"""
app.py — Assistant Documentaire Intelligent (RAG)
==================================================
Point d'entrée Streamlit de l'application.

Lancement :
    streamlit run app.py

Flux utilisateur :
  1. L'utilisateur choisit le backend (OpenAI ou HuggingFace local).
  2. Il uploade un document PDF ou texte.
  3. Le document est traité (chunking + vectorisation).
  4. Il peut poser des questions en langage naturel dans le chat.
  5. Le RAG retrouve les passages pertinents et génère une réponse ancrée.
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from utils.document_processor import process_uploaded_file
from utils.ai_engine import initialize_rag_pipeline, ask_question

# ---------------------------------------------------------------------------
# Configuration initiale
# ---------------------------------------------------------------------------

# Charge les variables d'environnement depuis .env (clé OpenAI, etc.)
load_dotenv()

st.set_page_config(
    page_title="DocChat RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Initialisation du session state Streamlit
# ---------------------------------------------------------------------------
# Le session_state persiste les données entre les reruns de l'application.
# Sans lui, chaque interaction Streamlit efface toutes les variables Python.

def init_session_state() -> None:
    """Initialise les clés du session_state si elles n'existent pas encore."""
    defaults = {
        "messages": [],          # Historique du chat affiché
        "rag_chain": None,       # Chaîne LangChain initialisée
        "doc_processed": False,  # Flag : document traité ?
        "doc_name": "",          # Nom du fichier uploadé
        "chunk_count": 0,        # Nombre de chunks générés
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ---------------------------------------------------------------------------
# Barre latérale — Configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Configuration")
    st.divider()

    # Choix du backend IA
    st.subheader("🤖 Modèle IA")
    backend = st.radio(
        "Sélectionner le backend",
        options=["OpenAI (GPT-3.5)", "Local (HuggingFace + Ollama)"],
        index=0,
        help=(
            "OpenAI : requiert une clé API, meilleure qualité.\n\n"
            "Local : 100% gratuit, nécessite Ollama installé localement."
        ),
    )
    use_openai = backend.startswith("OpenAI")

    # Champ clé API OpenAI (masqué si mode local)
    if use_openai:
        st.subheader("🔑 Clé API OpenAI")
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.environ.get("OPENAI_API_KEY", ""),
            placeholder="sk-...",
            help="Votre clé API OpenAI. Elle sera utilisée uniquement pendant cette session.",
        )
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input
    else:
        st.info(
            "Mode local activé.\n\n"
            "Assurez-vous qu'**Ollama** est lancé et que le modèle **mistral** "
            "est téléchargé (`ollama pull mistral`)."
        )

    st.divider()

    # Informations sur le document traité
    st.subheader("📊 Document actif")
    if st.session_state.doc_processed:
        st.success(f"✅ {st.session_state.doc_name}")
        st.metric("Chunks générés", st.session_state.chunk_count)
    else:
        st.warning("Aucun document chargé.")

    # Bouton de réinitialisation
    if st.button("🗑️ Réinitialiser la session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        # Supprime la base vectorielle persistée
        if os.path.exists("./chroma_db"):
            import shutil
            shutil.rmtree("./chroma_db")
        st.rerun()

    st.divider()
    st.caption(
        "**Enterprise RAG Chatbot** — Portfolio project\n\n"
        "Stack : LangChain · ChromaDB · Streamlit"
    )


# ---------------------------------------------------------------------------
# En-tête principal
# ---------------------------------------------------------------------------

st.title("📄 Assistant Documentaire Intelligent")
st.markdown(
    "Uploadez un document **PDF ou texte** et posez-lui des questions "
    "en langage naturel grâce à la technique **RAG** "
    "*(Retrieval-Augmented Generation)*."
)
st.divider()


# ---------------------------------------------------------------------------
# Zone d'upload du document
# ---------------------------------------------------------------------------

col_upload, col_info = st.columns([3, 2], gap="large")

with col_upload:
    st.subheader("1️⃣ Charger un document")
    uploaded_file = st.file_uploader(
        "Déposez votre fichier ici",
        type=["pdf", "txt", "md"],
        help="Formats supportés : PDF, TXT, Markdown (.md)",
        label_visibility="collapsed",
    )

    if uploaded_file is not None and not st.session_state.doc_processed:
        with st.spinner("Traitement du document en cours..."):
            # Sauvegarde temporaire du fichier uploadé
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                # Étape 1 : Chunking du document
                st.info("🔪 Découpage du document en chunks...")
                chunks = process_uploaded_file(tmp_path)
                st.session_state.chunk_count = len(chunks)

                # Étape 2 : Vectorisation et initialisation de la chaîne RAG
                st.info("🧠 Vectorisation et indexation des chunks...")
                rag_chain = initialize_rag_pipeline(
                    chunks=chunks,
                    use_openai=use_openai,
                )

                # Sauvegarde dans le session state
                st.session_state.rag_chain = rag_chain
                st.session_state.doc_processed = True
                st.session_state.doc_name = uploaded_file.name
                st.session_state.messages = []  # Reset du chat

            except Exception as e:
                st.error(f"❌ Erreur lors du traitement : {e}")
            finally:
                os.unlink(tmp_path)  # Nettoyage du fichier temporaire

        if st.session_state.doc_processed:
            st.success(
                f"✅ Document traité avec succès ! "
                f"**{st.session_state.chunk_count} chunks** générés."
            )
            st.rerun()

with col_info:
    st.subheader("2️⃣ Comprendre le RAG")
    with st.expander("Comment ça fonctionne ?", expanded=True):
        st.markdown(
            """
**1. Chunking** — Le document est découpé en petits morceaux (chunks)
de ~1000 caractères avec un chevauchement (overlap) de 150 caractères.

**2. Embeddings** — Chaque chunk est converti en vecteur numérique
représentant son sens sémantique.

**3. Vector Search** — Votre question est vectorisée et comparée aux
chunks pour trouver les plus pertinents (distance cosinus).

**4. Génération** — Le LLM génère une réponse en se basant *uniquement*
sur les chunks retrouvés → pas d'hallucination !
            """
        )


# ---------------------------------------------------------------------------
# Interface de chat
# ---------------------------------------------------------------------------

st.divider()
st.subheader("3️⃣ Dialoguer avec votre document")

if not st.session_state.doc_processed:
    st.info("👆 Uploadez un document ci-dessus pour commencer à poser des questions.")
else:
    # Affichage de l'historique du chat
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Affiche les sources si disponibles (messages assistant)
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 Sources utilisées", expanded=False):
                        for i, source in enumerate(message["sources"], 1):
                            page = source.metadata.get("page", "N/A")
                            chunk_idx = source.metadata.get("chunk_index", "?")
                            st.markdown(
                                f"**Source {i}** — Page {page} · Chunk #{chunk_idx}"
                            )
                            st.caption(source.page_content[:300] + "...")
                            st.divider()

    # Zone de saisie de la question
    question = st.chat_input(
        f"Posez une question sur '{st.session_state.doc_name}'...",
        key="chat_input",
    )

    if question:
        # Affichage immédiat de la question utilisateur
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        # Génération de la réponse RAG
        with st.chat_message("assistant"):
            with st.spinner("Recherche et génération en cours..."):
                try:
                    answer, sources = ask_question(
                        st.session_state.rag_chain,
                        question,
                    )
                    st.markdown(answer)

                    # Affichage des sources dans un expander
                    if sources:
                        with st.expander("📚 Sources utilisées", expanded=False):
                            for i, source in enumerate(sources, 1):
                                page = source.metadata.get("page", "N/A")
                                chunk_idx = source.metadata.get("chunk_index", "?")
                                st.markdown(
                                    f"**Source {i}** — Page {page} · Chunk #{chunk_idx}"
                                )
                                st.caption(source.page_content[:300] + "...")
                                st.divider()

                    # Sauvegarde dans l'historique
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as e:
                    error_msg = f"❌ Erreur : {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })

        st.rerun()
