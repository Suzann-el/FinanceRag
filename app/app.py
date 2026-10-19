"""
FinanceRAG — Application Streamlit
====================================
Interface chatbot documentaire bancaire & assurance.

Lancement :
    streamlit run app/app.py

Structure de l'interface :
- Sidebar : configuration, upload de documents, stats
- Zone principale : chat conversationnel avec streaming
- Bas de page : sources des dernières réponses
"""

import sys
from pathlib import Path

# Ajouter la racine du projet au path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from loguru import logger

from src.config import VECTORSTORE_PATH, DATA_RAW_DIR, STREAMLIT_PAGE_TITLE
from src.indexing import run_indexing_pipeline, load_vectorstore
from src.rag_pipeline import FinanceRAGPipeline


# ── Configuration Streamlit ───────────────────────────────────────────────────
st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Couleurs principales */
    :root {
        --navy: #1B2A4A;
        --gold: #C9A84C;
        --light-bg: #F8F9FA;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1B2A4A 0%, #2C3E6B 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #C9A84C;
    }

    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }

    .main-header p {
        margin: 0.3rem 0 0;
        opacity: 0.85;
        font-size: 0.95rem;
    }

    /* Bulle de chat utilisateur */
    .user-bubble {
        background: #1B2A4A;
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }

    /* Bulle de chat assistant */
    .assistant-bubble {
        background: white;
        color: #1B2A4A;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        max-width: 90%;
        border: 1px solid #E8ECF4;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }

    /* Badge source */
    .source-badge {
        background: #F0F4FF;
        color: #1B2A4A;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.78rem;
        border: 1px solid #C9A84C;
        display: inline-block;
        margin: 0.2rem;
    }

    /* Métriques sidebar */
    .metric-card {
        background: white;
        border: 1px solid #E8ECF4;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        margin: 0.3rem 0;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1B2A4A;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #666;
    }

    /* Questions suggérées */
    .stButton button {
        background: white;
        color: #1B2A4A;
        border: 1px solid #C9A84C;
        border-radius: 20px;
        padding: 0.4rem 1rem;
        font-size: 0.85rem;
        transition: all 0.2s;
    }

    .stButton button:hover {
        background: #1B2A4A;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ── Initialisation de la session ───────────────────────────────────────────────
def init_session_state():
    """Initialise les variables de session Streamlit."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "index_built" not in st.session_state:
        st.session_state.index_built = Path(VECTORSTORE_PATH).exists()
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    if "nb_queries" not in st.session_state:
        st.session_state.nb_queries = 0


@st.cache_resource
def get_pipeline() -> FinanceRAGPipeline:
    """
    Charge le pipeline RAG (mis en cache pour éviter le rechargement à chaque interaction).
    @st.cache_resource : exécuté une seule fois par session serveur.
    """
    return FinanceRAGPipeline()


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    """Rendu de la barre latérale : config + upload + stats."""
    with st.sidebar:
        # Logo / titre
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0;'>
            <div style='font-size:2.5rem'>🏦</div>
            <div style='font-weight:700; color:#1B2A4A; font-size:1.1rem'>FinanceRAG</div>
            <div style='font-size:0.8rem; color:#666'>Assistant Documentaire</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Section : Indexation ─────────────────────────────────────────────
        st.subheader("📂 Documents")

        uploaded_files = st.file_uploader(
            "Ajouter des documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Ajoutez vos rapports annuels, règlements, conditions générales..."
        )

        if uploaded_files:
            # Sauvegarder les fichiers uploadés
            for f in uploaded_files:
                save_path = DATA_RAW_DIR / f.name
                save_path.write_bytes(f.read())
                st.success(f"✓ {f.name}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Indexer", use_container_width=True, type="primary"):
                with st.spinner("Indexation en cours..."):
                    try:
                        run_indexing_pipeline()
                        st.session_state.index_built = True
                        st.session_state.pipeline = None  # Force rechargement
                        st.success("Index créé ✓")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erreur : {e}")

        with col2:
            status = "🟢 Prêt" if st.session_state.index_built else "🔴 Non indexé"
            st.info(status)

        st.divider()

        # ── Section : Paramètres ─────────────────────────────────────────────
        with st.expander("⚙️ Paramètres", expanded=False):
            top_k = st.slider("Chunks récupérés (top_k)", 1, 10, 5)
            strategy = st.radio(
                "Stratégie de recherche",
                ["similarity", "mmr"],
                help="MMR diversifie les résultats, similarity maximise la pertinence"
            )

        st.divider()

        # ── Section : Statistiques ────────────────────────────────────────────
        st.subheader("📊 Session")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Questions", st.session_state.nb_queries)
        with col2:
            tokens_k = st.session_state.total_tokens // 1000
            st.metric("Tokens", f"{tokens_k}k" if tokens_k > 0 else "0")

        if st.button("🗑️ Effacer l'historique", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_tokens = 0
            st.session_state.nb_queries = 0
            if st.session_state.pipeline:
                st.session_state.pipeline.reset_history()
            st.rerun()

        st.divider()

        # ── Exemples de questions ─────────────────────────────────────────────
        st.subheader("💡 Questions suggérées")

        example_questions = [
            "Quels sont les frais de gestion annuels ?",
            "Quelle est la politique de remboursement ?",
            "Quels sont les critères d'éligibilité ?",
            "Comment est calculé le taux d'intérêt ?",
            "Quelles sont les garanties incluses ?",
        ]

        for q in example_questions:
            if st.button(q, use_container_width=True, key=f"example_{q[:20]}"):
                st.session_state.pending_question = q
                st.rerun()

        return top_k, strategy


# ── Zone principale ───────────────────────────────────────────────────────────
def render_chat(pipeline: FinanceRAGPipeline):
    """Rendu de la zone de chat principale."""

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 FinanceRAG — Assistant Documentaire Bancaire</h1>
        <p>Posez vos questions sur vos documents financiers. Réponses sourcées et précises.</p>
    </div>
    """, unsafe_allow_html=True)

    # Afficher l'historique
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

            # Afficher les sources pour les réponses assistant
            if msg["role"] == "assistant" and "sources" in msg:
                if msg["sources"]:
                    st.markdown("**Sources :**")
                    for source in msg["sources"]:
                        st.markdown(
                            f'<span class="source-badge">📄 {source}</span>',
                            unsafe_allow_html=True
                        )

    # Champ de saisie
    prompt = st.chat_input(
        "Posez votre question sur vos documents financiers...",
        disabled=not st.session_state.index_built
    )

    # Question depuis les boutons exemples
    if hasattr(st.session_state, "pending_question"):
        prompt = st.session_state.pending_question
        del st.session_state.pending_question

    if not st.session_state.index_built:
        st.info(
            "💡 **Commencez par indexer vos documents.**\n\n"
            "Uploadez vos PDFs/DOCXs dans la sidebar et cliquez sur **Indexer**."
        )
        return

    if prompt:
        # Afficher la question utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Générer la réponse
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                # Streaming de la réponse
                with st.spinner("Recherche dans les documents..."):
                    response = pipeline.query(prompt, use_history=True)

                # Affichage progressif (simulation streaming)
                for i in range(0, len(response.answer), 10):
                    full_response = response.answer[:i + 10]
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(response.answer)

                # Sources
                if response.sources:
                    st.markdown("**Sources :**")
                    for source in response.sources:
                        st.markdown(
                            f'<span class="source-badge">📄 {source}</span>',
                            unsafe_allow_html=True
                        )

                # Métriques tokens (discret)
                st.caption(
                    f"🔢 {response.total_tokens} tokens "
                    f"({response.input_tokens} in / {response.output_tokens} out)"
                )

                # Mise à jour des stats
                st.session_state.total_tokens += response.total_tokens
                st.session_state.nb_queries += 1

                # Sauvegarder dans l'historique
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                })

            except Exception as e:
                error_msg = f"❌ Erreur : {str(e)}"
                response_placeholder.error(error_msg)
                logger.error(f"Erreur pipeline : {e}")


# ── Point d'entrée ─────────────────────────────────────────────────────────────
def main():
    init_session_state()

    # Sidebar
    top_k, strategy = render_sidebar()

    # Charger le pipeline si l'index existe
    if st.session_state.index_built:
        if st.session_state.pipeline is None:
            try:
                st.session_state.pipeline = get_pipeline()
            except Exception as e:
                st.error(f"Erreur lors du chargement du pipeline : {e}")
                return

        render_chat(st.session_state.pipeline)
    else:
        render_chat(None)


if __name__ == "__main__":
    main()
