"""
FinanceRAG — Chargement et parsing des documents
=================================================
Supporte : PDF, DOCX, TXT
Extrait le texte brut + métadonnées (source, page, type).
"""

from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

# LangChain document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_core.documents import Document


# ── Extensions supportées ────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def load_document(file_path: str | Path) -> List[Document]:
    """
    Charge un fichier et retourne une liste de Documents LangChain.
    Chaque Document contient : page_content (str) + metadata (dict).

    Args:
        file_path: Chemin vers le fichier à charger.

    Returns:
        Liste de Documents (1 par page pour les PDFs, 1 pour DOCX/TXT).

    Raises:
        ValueError: Si l'extension n'est pas supportée.
        FileNotFoundError: Si le fichier n'existe pas.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Extension non supportée : {ext}. Supportées : {SUPPORTED_EXTENSIONS}")

    logger.info(f"Chargement : {path.name} ({ext})")

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(str(path))
            docs = loader.load()  # 1 Document par page

        elif ext == ".docx":
            loader = Docx2txtLoader(str(path))
            docs = loader.load()

        elif ext in (".txt", ".md"):
            loader = TextLoader(str(path), encoding="utf-8")
            docs = loader.load()

        else:
            docs = []

    except Exception as e:
        logger.error(f"Erreur lors du chargement de {path.name} : {e}")
        raise

    # Enrichir les métadonnées
    for i, doc in enumerate(docs):
        doc.metadata.update({
            "source": path.name,
            "file_path": str(path),
            "file_type": ext.lstrip("."),
            "total_pages": len(docs),
        })
        if ext == ".pdf":
            doc.metadata["page"] = i + 1

    logger.success(f"✓ {path.name} → {len(docs)} segment(s) chargé(s)")
    return docs


def load_directory(directory: str | Path) -> List[Document]:
    """
    Charge tous les documents supportés depuis un répertoire.

    Args:
        directory: Chemin du répertoire contenant les documents.

    Returns:
        Liste consolidée de tous les Documents.
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"Répertoire introuvable : {dir_path}")

    all_docs: List[Document] = []
    files = [f for f in dir_path.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        logger.warning(f"Aucun document supporté dans {dir_path}")
        return []

    logger.info(f"Chargement de {len(files)} fichier(s) depuis {dir_path.name}/")

    for file in sorted(files):
        try:
            docs = load_document(file)
            all_docs.extend(docs)
        except Exception as e:
            logger.warning(f"Fichier ignoré ({file.name}) : {e}")

    logger.success(f"✓ Total : {len(all_docs)} segments depuis {len(files)} fichier(s)")
    return all_docs


def get_document_stats(docs: List[Document]) -> Dict[str, Any]:
    """
    Calcule des statistiques sur un corpus de Documents.

    Returns:
        Dictionnaire avec : nb_documents, nb_segments, sources, avg_chars
    """
    if not docs:
        return {"nb_documents": 0, "nb_segments": 0, "sources": [], "avg_chars": 0}

    sources = list({doc.metadata.get("source", "inconnu") for doc in docs})
    total_chars = sum(len(doc.page_content) for doc in docs)

    return {
        "nb_documents": len(sources),
        "nb_segments": len(docs),
        "sources": sorted(sources),
        "avg_chars": round(total_chars / len(docs)),
        "total_chars": total_chars,
    }
