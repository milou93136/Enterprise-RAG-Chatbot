# DocChat RAG — Assistant Documentaire Intelligent

> Posez des questions à vos documents PDF en langage naturel, sans hallucinations.

## Description du Projet

Ce projet explore l'utilisation des **LLMs dans un contexte métier** en implémentant
un système **RAG (Retrieval-Augmented Generation)** de bout en bout.

L'objectif est de résoudre le problème des *hallucinations* de l'IA en lui fournissant
une base de connaissances externe et vérifiable : le document de l'utilisateur.
Le LLM ne peut répondre qu'en se basant sur le contenu réel du fichier chargé.

---

## Architecture technique

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIPELINE RAG                             │
│                                                                 │
│  Document PDF/TXT                                               │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────┐    Chunking     ┌───────────────────────┐  │
│  │ document_       │  (1000 chars    │ Chunks textuels       │  │
│  │ processor.py    │───► +150 overlap│ [chunk_1, chunk_2...] │  │
│  └─────────────────┘                └──────────┬────────────┘  │
│                                                │               │
│                                                ▼               │
│  ┌─────────────────┐   Embeddings   ┌───────────────────────┐  │
│  │ ai_engine.py    │  (OpenAI ou    │ ChromaDB              │  │
│  │                 │───► HuggingFace│ (vecteurs persistés)  │  │
│  └─────────────────┘                └──────────┬────────────┘  │
│                                                │               │
│       Question utilisateur ──► Embedding ──────┤               │
│                                                │ Vector Search │
│                                                ▼               │
│                               Top-K chunks pertinents          │
│                                                │               │
│                                                ▼               │
│                               LLM (GPT-3.5 / Mistral)          │
│                               Prompt = Question + Contexte      │
│                                                │               │
│                                                ▼               │
│                                          Réponse ancrée        │
└─────────────────────────────────────────────────────────────────┘
```

### Points techniques clés

- **Data Ingestion** : Pipeline de chargement et nettoyage de documents PDF avec
  `PyPDFLoader` de LangChain. Gestion des artefacts PDF (césures, espaces superflus).

- **Chunking** : Découpage intelligent avec `RecursiveCharacterTextSplitter`
  (1000 chars, 150 chars d'overlap) pour préserver la cohérence sémantique.

- **Vectorisation** : Transformation du texte en vecteurs numériques (embeddings)
  via `text-embedding-3-small` (OpenAI) ou `all-MiniLM-L6-v2` (HuggingFace local).

- **Vector Search** : Indexation et recherche sémantique via **ChromaDB** avec
  algorithme **MMR** (Maximal Marginal Relevance) pour des résultats pertinents et diversifiés.

- **Orchestration** : Chaîne `ConversationalRetrievalChain` de **LangChain** avec
  mémoire conversationnelle pour gérer les questions de suivi.

- **Interface** : Web-app réactive sous **Streamlit** avec affichage des sources.

---

## Structure du projet

```
Enterprise-RAG-Chatbot/
│
├── app.py                     # Interface Streamlit (point d'entrée)
│
├── utils/
│   ├── __init__.py            # Package Python
│   ├── document_processor.py  # Chargement, nettoyage, chunking
│   └── ai_engine.py           # Embeddings, ChromaDB, chaîne RAG
│
├── chroma_db/                 # Base vectorielle persistée (auto-généré)
├── requirements.txt           # Dépendances Python
├── .env.example               # Template des variables d'environnement
├── .gitignore
└── README.md
```

---

## Installation et lancement

### Prérequis

- Python 3.10 ou 3.11
- Git

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-username/Enterprise-RAG-Chatbot.git
cd Enterprise-RAG-Chatbot
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# OU
.venv\Scripts\activate         # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Configuration

### Option A — Backend OpenAI (Recommandé)

1. Obtenez une clé API sur [platform.openai.com](https://platform.openai.com/api-keys).
2. Créez votre fichier `.env` :

```bash
cp .env.example .env
```

3. Éditez `.env` et ajoutez votre clé :

```
OPENAI_API_KEY=sk-votre-clé-ici
```

> **Coût estimé** : ~$0.001 par question (modèle `gpt-3.5-turbo` + `text-embedding-3-small`).

---

### Option B — Backend local 100% gratuit (HuggingFace + Ollama)

Aucune clé API requise. Tout fonctionne en local.

#### Installer Ollama

```bash
# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# macOS
brew install ollama
```

#### Télécharger le modèle Mistral

```bash
ollama pull mistral
```

#### Lancer le serveur Ollama

```bash
ollama serve
```

> **Note** : Le modèle d'embeddings HuggingFace (`all-MiniLM-L6-v2`, ~90 MB)
> est téléchargé automatiquement au premier lancement.

---

## Lancement de l'application

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans votre navigateur à l'adresse
`http://localhost:8501`.

---

## Guide d'utilisation

1. **Configurer le backend** dans la barre latérale (OpenAI ou Local).
2. **Uploader un document** PDF ou texte via la zone de dépôt.
3. Attendre le traitement (chunking + vectorisation, ~10-30 secondes selon la taille).
4. **Poser des questions** en langage naturel dans le chat.
5. Consulter les **sources utilisées** pour vérifier les passages du document.

---

## Pourquoi ce projet ?

Ce projet démontre la capacité à **intégrer des technologies d'IA de pointe**
dans une application concrète et fonctionnelle.

Il illustre :
- La maîtrise de l'**architecture RAG**, pattern fondamental en IA appliquée.
- L'utilisation de **LangChain** pour orchestrer des pipelines IA complexes.
- La connaissance des **embeddings** et de la **recherche vectorielle**.
- Le développement d'une interface utilisateur avec **Streamlit**.
- Une approche **modulaire et documentée** du code (portfolio-ready).

Ce projet complète des compétences en développement logiciel par une expertise
en **architecture IA moderne**, particulièrement recherchée dans l'industrie.

---

## Technologies utilisées

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| Orchestration | LangChain 0.3 | Pipeline RAG, mémoire, chaînes |
| LLM | OpenAI GPT-3.5 / Mistral | Génération de réponses |
| Embeddings | OpenAI / all-MiniLM-L6-v2 | Vectorisation du texte |
| Vector DB | ChromaDB | Stockage et recherche vectorielle |
| Interface | Streamlit | UI web interactive |
| PDF | PyPDF | Extraction de texte PDF |

---

## Licence

MIT — Libre d'utilisation pour usage personnel et commercial.
