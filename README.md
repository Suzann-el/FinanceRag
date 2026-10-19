# 🏦 FinanceRAG — Chatbot Documentaire Bancaire & Assurance

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Claude API](https://img.shields.io/badge/Claude-3.5%20Sonnet-orange.svg)](https://anthropic.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green.svg)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/FAISS-1.8-purple.svg)](https://faiss.ai)

**FinanceRAG** est un chatbot documentaire spécialisé dans les documents financiers (rapports annuels, conditions générales, réglementations bancaires). Il utilise la technologie RAG (Retrieval-Augmented Generation) pour répondre avec précision à des questions métier en citant ses sources.

---
#Lien vers le chatbot : https://financerag-5.streamlit.app/
## 🎯 Cas d'usage

| Secteur | Documents supportés | Exemples de questions |
|---------|--------------------|-----------------------|
| **Banque** | Rapports annuels, politiques de crédit | "Quel est le ratio CET1 ?" |
| **Assurance** | Conditions générales, tableaux de garanties | "Quels sont les frais de gestion ?" |
| **Conformité** | Règlements internes, circulaires | "Quelles sont les obligations KYC ?" |
| **Finance** | Prospectus, notices d'information | "Comment est calculé le TAEG ?" |

---

## 🏗️ Architecture

```
Utilisateur
    │
    ▼
[Question en langage naturel]
    │
    ▼
[Embedding (sentence-transformers)]
    │
    ▼
[FAISS — Recherche sémantique]
    │ top_k chunks pertinents
    ▼
[Construction du prompt]
    │ contexte + question
    ▼
[Claude 3.5 Sonnet — Génération]
    │
    ▼
[Réponse sourcée + citations]
```

---

## 📁 Structure du projet

```
FinanceRAG/
├── src/
│   ├── config.py          # Configuration centralisée (.env)
│   ├── document_loader.py # Chargement PDF/DOCX/TXT
│   ├── indexing.py        # Chunking + embeddings + FAISS
│   ├── retriever.py       # Recherche sémantique
│   └── rag_pipeline.py    # Pipeline complet + Claude API
│
├── app/
│   └── app.py             # Interface Streamlit
│
├── data/
│   ├── raw/               # Documents sources à indexer
│   └── processed/         # (réservé)
│
├── vectorstore/           # Index FAISS (généré automatiquement)
│
├── notebooks/
│   └── 01_exploration.ipynb  # Tests et évaluation
│
├── tests/
│   ├── test_document_loader.py
│   └── test_retriever.py
│
├── requirements.txt
├── .env.example
└── Makefile
```

---

## 🚀 Installation et démarrage

### 1. Cloner et installer

```bash
git clone https://github.com/Suzann-el/FinanceRAG.git
cd FinanceRAG

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou : venv\Scripts\activate  # Windows

# Installation complète
make setup
```

### 2. Configurer la clé API

```bash
# Éditer le fichier .env
nano .env
# Renseigner : ANTHROPIC_API_KEY=sk-ant-...
```


### 3. Ajouter vos documents

Copiez vos PDFs/DOCXs dans `data/raw/` :

```bash
cp ~/Documents/rapport_annuel_2024.pdf data/raw/
cp ~/Documents/conditions_generales.pdf data/raw/
```

Des documents d'exemple sont fournis dans `data/raw/exemple_*.txt`.

### 4. Indexer les documents

```bash
make index
# ou directement :
python -m src.indexing
```

### 5. Lancer l'application

```bash
make run
# → http://localhost:8501
```

---

## ⚙️ Configuration

Variables d'environnement dans `.env` :

| Variable | Défaut | Description |
|----------|--------|-------------|
| `ANTHROPIC_API_KEY` | — | **Obligatoire** — clé API Anthropic |
| `CLAUDE_MODEL` | `claude-3-5-sonnet-20241022` | Modèle Claude |
| `CHUNK_SIZE` | `800` | Taille des chunks (caractères) |
| `CHUNK_OVERLAP` | `100` | Chevauchement entre chunks |
| `TOP_K_RETRIEVAL` | `5` | Nombre de chunks récupérés |
| `SIMILARITY_THRESHOLD` | `0.6` | Seuil de pertinence (0-1) |

---

## 🧪 Tests

```bash
# Tous les tests
make test

# Avec rapport de couverture
make test-cov
```

---

## 📊 Choix techniques

### Modèle d'embedding
**`paraphrase-multilingual-mpnet-base-v2`** — Justification :
- Support natif du français + anglais (documents bancaires mixtes)
- 768 dimensions : bon équilibre précision / vitesse
- Open source, pas de coût d'API pour l'indexation
- Disponible via `sentence-transformers` (no dependency hell)

### Chunking strategy
- **Taille 800 chars** : assez large pour préserver le contexte d'une clause contractuelle
- **Overlap 100 chars** : évite les coupures sur des formules (ex : calcul TAEG en 2 parties)
- **Séparateurs sémantiques** : découpage prioritaire aux paragraphes, jamais en plein mot

### Vector store : FAISS vs Chroma vs Pinecone
| | FAISS | Chroma | Pinecone |
|---|---|---|---|
| Coût | Gratuit | Gratuit | Payant |
| Hébergement | Local | Local | Cloud |
| Performance | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Scalabilité | 🔄 Fichier | 🔄 Fichier | ☁️ Infinie |
| **Choix** | ✅ Local, rapide, gratuit | — | — |

### RAG vs Fine-tuning
**RAG choisi** car :
- Documents fréquemment mis à jour → pas besoin de ré-entraîner
- Traçabilité des sources (conformité bancaire critique)
- Coût : €0 d'entraînement vs €1000+ pour fine-tuning
- Contrôle : réponse uniquement sur les documents chargés

---

## 👤 Auteur

**Saoussan EL HAOUZI** — Data Scientist  
[GitHub](https://github.com/Suzann-el) | Paris, France

---

*Projet portfolio — Démonstration de RAG sur documents financiers*
