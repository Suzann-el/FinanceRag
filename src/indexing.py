"""
FinanceRAG — Phase 1 : Indexation (build vectorstore)
======================================================
Pipeline complet :
  1. Chargement des documents (PDF / DOCX / TXT)
  2. Découpage en chunks (RecursiveCharacterTextSplitter)
  3. Génération des embeddings (sentence-transformers multilingue)
  4. Stockage dans un index FAISS persistant sur disque

Usage CLI :
    python -m src.indexing --docs_dir data/raw --output vectorstore/faiss_index
"""

import argparse
from pathlib import Path
from typing import List

from loguru import logger
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    VECTORSTORE_PATH,
    DATA_RAW_DIR,
)
from src.document_loader import load_directory, get_document_stats


# ── Séparateurs adaptés aux docs financiers ───────────────────────────────────
# Priorité : sections → paragraphes → phrases → mots
SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Découpe les Documents en chunks de taille fixe avec overlap.

    Stratégie :
    - Chunk size 800 : assez grand pour conserver le contexte d'une clause
    - Overlap 100 : évite de couper une phrase importante à cheval entre 2 chunks
    - Séparateurs sémantiques : on découpe d'abord aux paragraphes

    Args:
        docs: Liste brute de Documents (1 par page PDF typiquement).

    Returns:
        Liste de chunks (Documents enrichis avec chunk_id, chunk_index).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
        add_start_index=True,  # ajoute metadata["start_index"]
    )

    chunks = splitter.split_documents(docs)

    # Ajouter un identifiant unique par chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        source = chunk.metadata.get("source", "doc")
        page = chunk.metadata.get("page", "")
        chunk.metadata["chunk_label"] = f"{source} — p.{page}" if page else source

    logger.info(f"Découpage : {len(docs)} segment(s) → {len(chunks)} chunk(s)")
    logger.info(f"  Taille moyenne : {sum(len(c.page_content) for c in chunks) // max(len(chunks),1)} chars/chunk")

    return chunks


def build_vectorstore(
    chunks: List[Document],
    save_path: str | None = None,
) -> FAISS:
    """
    Génère les embeddings et construit l'index FAISS.

    Modèle d'embedding :
    - paraphrase-multilingual-mpnet-base-v2 (Sentence-Transformers)
    - 768 dimensions, supporte 50+ langues dont FR et EN
    - Bon équilibre vitesse / qualité pour des docs techniques

    Args:
        chunks: Liste de Documents découpés.
        save_path: Si fourni, sauvegarde l'index sur disque.

    Returns:
        Index FAISS chargeable pour les requêtes.
    """
    logger.info(f"Chargement du modèle d'embedding : {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder="./models_cache",
    )

    logger.info(f"Génération des embeddings pour {len(chunks)} chunks...")
    # Traitement par batch pour éviter les problèmes mémoire
    batch_size = 64
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    logger.info("Construction de l'index FAISS...")
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
    )

    logger.success(f"✓ Index FAISS construit — {len(chunks)} vecteurs")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(save_path)
        logger.success(f"✓ Index sauvegardé : {save_path}")

    return vectorstore


def load_vectorstore(load_path: str | None = None) -> FAISS:
    """
    Charge un index FAISS existant depuis le disque.

    Args:
        load_path: Chemin de l'index. Utilise VECTORSTORE_PATH par défaut.

    Returns:
        Index FAISS prêt pour les requêtes.

    Raises:
        FileNotFoundError: Si l'index n'existe pas.
    """
    path = load_path or VECTORSTORE_PATH

    if not Path(path).exists():
        raise FileNotFoundError(
            f"Index FAISS introuvable : {path}\n"
            "Lancez d'abord : python -m src.indexing"
        )

    logger.info(f"Chargement de l'index FAISS : {path}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,  # fichiers locaux sûrs
    )

    logger.success(f"✓ Index FAISS chargé depuis {path}")
    return vectorstore


def run_indexing_pipeline(
    docs_dir: str | Path | None = None,
    output_path: str | None = None,
) -> FAISS:
    """
    Pipeline d'indexation complet : load → split → embed → save.

    Args:
        docs_dir: Répertoire des documents sources.
        output_path: Chemin de sauvegarde de l'index.

    Returns:
        Index FAISS prêt à l'emploi.
    """
    docs_dir = docs_dir or DATA_RAW_DIR
    output_path = output_path or VECTORSTORE_PATH

    logger.info("=" * 60)
    logger.info("FinanceRAG — Pipeline d'indexation")
    logger.info("=" * 60)

    # 1. Chargement
    docs = load_directory(docs_dir)
    if not docs:
        raise ValueError(f"Aucun document trouvé dans {docs_dir}")

    stats = get_document_stats(docs)
    logger.info(f"Corpus : {stats['nb_documents']} doc(s), {stats['nb_segments']} segment(s)")
    logger.info(f"Sources : {', '.join(stats['sources'])}")

    # 2. Découpage
    chunks = split_documents(docs)

    # 3. Embeddings + FAISS
    vectorstore = build_vectorstore(chunks, save_path=output_path)

    logger.info("=" * 60)
    logger.success("Indexation terminée ✓")
    logger.info(f"  Documents : {stats['nb_documents']}")
    logger.info(f"  Chunks    : {len(chunks)}")
    logger.info(f"  Index     : {output_path}")
    logger.info("=" * 60)

    return vectorstore


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinanceRAG — Indexation des documents")
    parser.add_argument(
        "--docs_dir", default=str(DATA_RAW_DIR),
        help="Répertoire des documents à indexer"
    )
    parser.add_argument(
        "--output", default=VECTORSTORE_PATH,
        help="Chemin de sauvegarde de l'index FAISS"
    )
    args = parser.parse_args()

    run_indexing_pipeline(docs_dir=args.docs_dir, output_path=args.output)
