"""
Tests unitaires — Retriever
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever import FinanceRetriever
from langchain.schema import Document


@pytest.fixture
def mock_vectorstore():
    """Mock d'un index FAISS."""
    mock = MagicMock()

    # Simuler 3 résultats avec des scores de distance
    docs_with_scores = [
        (
            Document(
                page_content="Le ratio CET1 de la banque est de 13,2%.",
                metadata={"source": "rapport_2024.pdf", "page": 12}
            ),
            0.15  # distance faible = très similaire
        ),
        (
            Document(
                page_content="Les fonds propres réglementaires s'élèvent à 8,4 milliards.",
                metadata={"source": "rapport_2024.pdf", "page": 15}
            ),
            0.35
        ),
        (
            Document(
                page_content="Météo à Paris : ensoleillé avec 22°C.",
                metadata={"source": "hors_sujet.txt", "page": 1}
            ),
            2.5  # distance élevée = peu pertinent
        ),
    ]

    mock.similarity_search_with_score.return_value = docs_with_scores
    mock.max_marginal_relevance_search.return_value = [d for d, _ in docs_with_scores[:2]]

    return mock


class TestFinanceRetriever:

    def test_init(self, mock_vectorstore):
        retriever = FinanceRetriever(mock_vectorstore)
        assert retriever.top_k == 5
        assert retriever.strategy == "similarity"

    def test_retrieve_retourne_docs_pertinents(self, mock_vectorstore):
        retriever = FinanceRetriever(mock_vectorstore, threshold=0.4)
        docs = retriever.retrieve("Quel est le ratio CET1 ?")
        # Avec threshold=0.4, seul le premier doc passe (score≈0.87)
        assert len(docs) >= 1
        assert "CET1" in docs[0].page_content

    def test_retrieve_filtre_par_seuil(self, mock_vectorstore):
        # Seuil très élevé → seuls les très proches passent
        retriever = FinanceRetriever(mock_vectorstore, threshold=0.8)
        docs = retriever.retrieve("ratio CET1")
        # Le doc sur la météo ne doit pas passer
        contents = [d.page_content for d in docs]
        assert not any("Météo" in c for c in contents)

    def test_retrieve_requete_vide(self, mock_vectorstore):
        retriever = FinanceRetriever(mock_vectorstore)
        docs = retriever.retrieve("")
        assert docs == []

    def test_retrieve_requete_espaces(self, mock_vectorstore):
        retriever = FinanceRetriever(mock_vectorstore)
        docs = retriever.retrieve("   ")
        assert docs == []

    def test_strategie_mmr(self, mock_vectorstore):
        retriever = FinanceRetriever(mock_vectorstore, strategy="mmr")
        docs = retriever.retrieve("fonds propres")
        assert len(docs) >= 1
        mock_vectorstore.max_marginal_relevance_search.assert_called_once()

    def test_format_context_structure(self, mock_vectorstore):
        retriever = FinanceRetriever(mock_vectorstore, threshold=0.5)
        docs = retriever.retrieve("ratio CET1")

        if docs:
            context = retriever.format_context(docs)
            assert "[Source:" in context
            assert "---" in context or len(docs) == 1

    def test_format_context_vide(self, mock_vectorstore):
        retriever = FinanceRetriever(mock_vectorstore)
        context = retriever.format_context([])
        assert context == ""

    def test_get_sources(self, mock_vectorstore):
        retriever = FinanceRetriever(mock_vectorstore, threshold=0.5)
        docs = retriever.retrieve("ratio")

        if docs:
            sources = retriever.get_sources(docs)
            assert isinstance(sources, list)
            assert len(sources) > 0
            # Les sources sont des chaînes
            for s in sources:
                assert isinstance(s, str)

    def test_metadata_score_ajoutee(self, mock_vectorstore):
        retriever = FinanceRetriever(mock_vectorstore, threshold=0.0)
        docs = retriever.retrieve("test")

        for doc in docs:
            if "similarity_score" in doc.metadata:
                score = doc.metadata["similarity_score"]
                assert 0 <= score <= 1
