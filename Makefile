# FinanceRAG — Makefile
# Usage : make <cible>

.PHONY: install setup index run test lint clean help

PYTHON = python
PIP = pip

## ── Installation ────────────────────────────────────────────────────────────

install:
	@echo "📦 Installation des dépendances..."
	$(PIP) install -r requirements.txt --break-system-packages
	@echo "✓ Installation terminée"

setup: install
	@echo "⚙️  Configuration initiale..."
	@if [ ! -f .env ]; then cp .env.example .env; echo "✓ .env créé depuis .env.example (ajoutez votre ANTHROPIC_API_KEY)"; fi
	@mkdir -p data/raw data/processed vectorstore docs
	@echo "✓ Structure de dossiers créée"

## ── Pipeline ────────────────────────────────────────────────────────────────

index:
	@echo "🔍 Indexation des documents..."
	$(PYTHON) -m src.indexing --docs_dir data/raw --output vectorstore/faiss_index
	@echo "✓ Index FAISS créé"

run:
	@echo "🚀 Lancement de l'application Streamlit..."
	streamlit run app/app.py

run-dev:
	@echo "🛠  Lancement en mode développement..."
	streamlit run app/app.py --server.fileWatcherType auto

## ── Tests ───────────────────────────────────────────────────────────────────

test:
	@echo "🧪 Exécution des tests..."
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:
	@echo "📊 Tests avec couverture..."
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "✓ Rapport disponible : htmlcov/index.html"

## ── Qualité ─────────────────────────────────────────────────────────────────

lint:
	@echo "🔍 Vérification du code..."
	$(PYTHON) -m flake8 src/ app/ tests/ --max-line-length=100 --ignore=E203,W503

## ── Nettoyage ───────────────────────────────────────────────────────────────

clean:
	@echo "🧹 Nettoyage..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Nettoyage terminé"

clean-index:
	@echo "🗑️  Suppression de l'index FAISS..."
	rm -rf vectorstore/faiss_index
	@echo "✓ Index supprimé"

## ── Aide ────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "FinanceRAG — Commandes disponibles :"
	@echo "─────────────────────────────────────"
	@echo "  make setup      : Installation complète + configuration"
	@echo "  make install    : Installer les dépendances Python"
	@echo "  make index      : Indexer les documents de data/raw/"
	@echo "  make run        : Lancer l'app Streamlit"
	@echo "  make test       : Exécuter les tests"
	@echo "  make test-cov   : Tests + rapport de couverture"
	@echo "  make clean      : Nettoyer les fichiers temporaires"
	@echo "  make clean-index: Supprimer l'index FAISS"
	@echo ""
