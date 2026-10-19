"""
FinanceRAG — Retriever : recherche sémantique dans FAISS
=========================================================
Encapsule la logique de récupération des chunks pertinents
avant l'appel au LLM.

Stratégies disponibles :
- similarity_search : recherche par similarité cosinus (défaut)
- mmr : Maximal Marginal Relevance — diversifie les résultats
"""

from typing import List, Tuple, Optional
from loguru import logger

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import TOP_K_RETRIEVAL, SIMILARITY_THRESHOLD


class FinanceRetriever:
    """
    Retriever sémantique pour FinanceRAG.

    Attributes:
        vectorstore: Index FAISS chargé.
        top_k: Nombre de chunks à retourner.
        threshold: Score de similarité minimum (0–1).
        strategy: "similarity" ou "mmr".
    """

    def __init__(
        self,
        vectorstore: FAISS,
        top_k: int = TOP_K_RETRIEVAL,
        threshold: float = SIMILARITY_THRESHOLD,
        strategy: str = "similarity",
    ):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.threshold = threshold
        self.strategy = strategy
        logger.info(
            f"Retriever initialisé — stratégie : {strategy} | "
            f"top_k : {top_k} | seuil : {threshold}"
        )

    def retrieve(self, query: str) -> List[Document]:
        """
        Recherche les chunks les plus pertinents pour une requête.

        Le processus :
        1. Encode la requête avec le même modèle d'embedding
        2. Calcule la similarité cosinus avec tous les vecteurs FAISS
        3. Retourne les top_k résultats au-dessus du seuil

        Args:
            query: Question de l'utilisateur en langage naturel.

        Returns:
            Liste de Documents triés par pertinence décroissante.
        """
        if not query.strip():
            logger.warning("Requête vide reçue")
            return []

        logger.info(f"Requête : '{query[:80]}...' " if len(query) > 80 else f"Requête : '{query}'")

        try:
            if self.strategy == "mmr":
                # MMR : équilibre pertinence + diversité
                # Utile quand plusieurs docs couvrent le même sujet
                docs = self.vectorstore.max_marginal_relevance_search(
                    query,
                    k=self.top_k,
                    fetch_k=self.top_k * 3,  # candidats à diversifier
                    lambda_mult=0.7,  # 0=diversité max, 1=pertinence max
                )
                # MMR ne retourne pas les scores → filtrage par seuil non applicable
                filtered = docs

            else:
                # Similarity search avec scores
                docs_with_scores: List[Tuple[Document, float]] = (
                    self.vectorstore.similarity_search_with_score(
                        query, k=self.top_k
                    )
                )

                # FAISS retourne la distance L2 (plus petit = plus proche)
                # On convertit en score de similarité [0, 1]
                filtered = []
                for doc, distance in docs_with_scores:
                    # Normalisation approximative : score = 1 / (1 + distance)
                    similarity = 1.0 / (1.0 + distance)
                    doc.metadata["similarity_score"] = round(similarity, 4)
                    doc.metadata["distance"] = round(float(distance), 4)

                    if similarity >= self.threshold:
                        filtered.append(doc)
                        logger.debug(
                            f"  ✓ {doc.metadata.get('source','?')} "
                            f"score={similarity:.3f}"
                        )
                    else:
                        logger.debug(
                            f"  ✗ {doc.metadata.get('source','?')} "
                            f"score={similarity:.3f} (sous seuil)"
                        )

            logger.info(f"→ {len(filtered)} chunk(s) pertinent(s) trouvé(s)")
            return filtered

        except Exception as e:
            logger.error(f"Erreur retrieval : {e}")
            return []

    def format_context(self, docs: List[Document]) -> str:
        """
        Formate les chunks récupérés en un bloc de contexte pour le LLM.

        Format :
            [Source: nom_fichier — page X]
            contenu du chunk...
            ---

        Args:
            docs: Documents récupérés par retrieve().

        Returns:
            Chaîne de contexte prête à injecter dans le prompt.
        """
        if not docs:
            return ""

        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source", "Document inconnu")
            page = doc.metadata.get("page", "")
            score = doc.metadata.get("similarity_score", "")

            header = f"[Source: {source}"
            if page:
                header += f" — page {page}"
            if score:
                header += f" | pertinence: {score:.0%}"
            header += "]"

            context_parts.append(f"{header}\n{doc.page_content.strip()}")

        return "\n\n---\n\n".join(context_parts)

    def get_sources(self, docs: List[Document]) -> List[str]:
        """
        Extrait les sources uniques d'une liste de Documents.

        Returns:
            Liste de chaînes "nom_fichier (page X)" dédupliquées.
        """
        sources = set()
        for doc in docs:
            source = doc.metadata.get("source", "Inconnu")
            page = doc.metadata.get("page")
            label = f"{source} (p. {page})" if page else source
            sources.add(label)
        return sorted(sources)
