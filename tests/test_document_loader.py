"""
Tests unitaires — DocumentLoader
"""

import pytest
from pathlib import Path
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_loader import load_document, load_directory, get_document_stats


@pytest.fixture
def sample_txt_file(tmp_path):
    """Crée un fichier texte temporaire pour les tests."""
    content = """Rapport Annuel 2024 — Banque Exemple

Section 1 : Résultats financiers
Le produit net bancaire (PNB) s'établit à 2,4 milliards d'euros.
Les commissions progressent de 12% à 890 millions d'euros.

Section 2 : Gestion des risques
Le ratio CET1 atteint 13,2%, bien au-dessus du minimum réglementaire de 8%.
Le coût du risque reste maîtrisé à 35 points de base.

Section 3 : Perspectives
Objectif de ROE supérieur à 12% pour 2025.
"""
    file_path = tmp_path / "rapport_annuel.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_directory(tmp_path, sample_txt_file):
    """Crée un répertoire avec plusieurs fichiers."""
    # Deuxième fichier
    content2 = "Conditions Générales Assurance Vie\nGarantie décès : 100 000 €\nFrais de gestion : 0,75% par an"
    (tmp_path / "conditions_generales.txt").write_text(content2, encoding="utf-8")
    return tmp_path


class TestLoadDocument:

    def test_charge_fichier_txt(self, sample_txt_file):
        docs = load_document(sample_txt_file)
        assert len(docs) >= 1
        assert len(docs[0].page_content) > 0

    def test_metadonnees_correctes(self, sample_txt_file):
        docs = load_document(sample_txt_file)
        assert docs[0].metadata["source"] == "rapport_annuel.txt"
        assert docs[0].metadata["file_type"] == "txt"

    def test_fichier_inexistant(self):
        with pytest.raises(FileNotFoundError):
            load_document("/chemin/qui/nexiste/pas.pdf")

    def test_extension_non_supportee(self, tmp_path):
        fake_file = tmp_path / "data.csv"
        fake_file.write_text("col1,col2\n1,2")
        with pytest.raises(ValueError, match="Extension non supportée"):
            load_document(fake_file)

    def test_contenu_non_vide(self, sample_txt_file):
        docs = load_document(sample_txt_file)
        total_chars = sum(len(d.page_content) for d in docs)
        assert total_chars > 100


class TestLoadDirectory:

    def test_charge_repertoire(self, sample_directory):
        docs = load_directory(sample_directory)
        assert len(docs) >= 2

    def test_repertoire_inexistant(self):
        with pytest.raises(FileNotFoundError):
            load_directory("/chemin/inexistant")

    def test_repertoire_vide(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        docs = load_directory(empty_dir)
        assert docs == []


class TestGetDocumentStats:

    def test_stats_corpus(self, sample_directory):
        docs = load_directory(sample_directory)
        stats = get_document_stats(docs)

        assert stats["nb_documents"] >= 2
        assert stats["nb_segments"] >= 2
        assert stats["avg_chars"] > 0
        assert len(stats["sources"]) >= 2

    def test_stats_corpus_vide(self):
        stats = get_document_stats([])
        assert stats["nb_documents"] == 0
        assert stats["nb_segments"] == 0
        assert stats["sources"] == []
