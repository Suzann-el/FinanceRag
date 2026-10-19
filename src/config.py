"""
FinanceRAG — Configuration centralisée
======================================
Charge les variables d'environnement et expose les constantes du projet.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# ── Chemins racine ──────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
VECTORSTORE_DIR = ROOT_DIR / "vectorstore"

# Créer les dossiers si absents
for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, VECTORSTORE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Chargement .env ──────────────────────────────────────────────────────────
load_dotenv(ROOT_DIR / ".env")

# ── Clé API ─────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_API_KEY:
    logger.warning("ANTHROPIC_API_KEY non définie — les appels LLM échoueront.")

# ── Modèle ───────────────────────────────────────────────────────────────────
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
# ── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
# Modèle léger 90MB — suffisant pour docs bancaires FR/EN

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 100))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", 5))
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", 0.3))

# ── Vector Store ─────────────────────────────────────────────────────────────
VECTORSTORE_PATH: str = os.getenv(
    "VECTORSTORE_PATH",
    str(VECTORSTORE_DIR / "faiss_index")
)

# ── Streamlit ─────────────────────────────────────────────────────────────────
STREAMLIT_PAGE_TITLE: str = os.getenv(
    "STREAMLIT_PAGE_TITLE",
    "FinanceRAG — Assistant Documentaire Bancaire"
)

# ── Prompt système ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Tu es un assistant expert en finance bancaire et assurance.
Tu réponds UNIQUEMENT à partir des documents fournis dans le contexte.

Règles strictes :
1. Si la réponse n'est pas dans les documents, réponds : "Je ne trouve pas cette information dans les documents disponibles."
2. Cite toujours la source (nom du document, page si disponible).
3. Réponds en français par défaut, sauf si la question est posée en anglais.
4. Sois précis, factuel, et structure ta réponse avec des puces si nécessaire.
5. Ne fabrique jamais de données chiffrées ou réglementaires.

Format de réponse :
- Réponse directe à la question
- Sources : [Nom du document — page X]
"""

logger.info(f"Config chargée — modèle : {CLAUDE_MODEL} | chunks : {CHUNK_SIZE}/{CHUNK_OVERLAP}")
