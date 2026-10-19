"""
FinanceRAG — Phase 2 : Pipeline RAG complet
============================================
Orchestre le flux :
  Requête utilisateur
      → Retrieval (FAISS)
      → Construction du prompt
      → Appel Claude API (Anthropic)
      → Réponse sourcée

Pattern : RAG simple (pas de chain-of-thought, pas d'agent)
Justification : Pour des questions sur des règlements/docs financiers,
la précision > la créativité. On veut de la récitation sourcée.
"""

from typing import List, Optional, Generator
from dataclasses import dataclass, field

import anthropic
from loguru import logger

from src.config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    SYSTEM_PROMPT,
    TOP_K_RETRIEVAL,
)
from src.retriever import FinanceRetriever
from src.indexing import load_vectorstore


@dataclass
class RAGResponse:
    """
    Réponse structurée du pipeline RAG.

    Attributes:
        answer: Réponse générée par Claude.
        sources: Liste des sources citées.
        retrieved_docs: Chunks utilisés (pour debug/audit).
        query: Question originale.
        model: Modèle Claude utilisé.
        input_tokens: Tokens en entrée.
        output_tokens: Tokens en sortie.
    """
    answer: str
    sources: List[str]
    query: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    retrieved_docs: List = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __str__(self) -> str:
        sources_str = "\n".join(f"  • {s}" for s in self.sources)
        return (
            f"Réponse :\n{self.answer}\n\n"
            f"Sources :\n{sources_str}\n\n"
            f"Tokens : {self.total_tokens} (in={self.input_tokens}, out={self.output_tokens})"
        )


class FinanceRAGPipeline:
    """
    Pipeline RAG pour l'assistance documentaire bancaire/assurance.

    Workflow :
    1. L'utilisateur pose une question
    2. Le retriever cherche les passages pertinents dans FAISS
    3. On construit un prompt avec le contexte et la question
    4. Claude génère une réponse sourcée et factuelle
    5. On retourne la réponse + les sources + les métriques

    Example:
        pipeline = FinanceRAGPipeline()
        response = pipeline.query("Quel est le taux de frais de gestion ?")
        print(response.answer)
    """

    def __init__(
        self,
        vectorstore_path: Optional[str] = None,
        top_k: int = TOP_K_RETRIEVAL,
        strategy: str = "similarity",
    ):
        """
        Initialise le pipeline en chargeant l'index FAISS et le client Anthropic.

        Args:
            vectorstore_path: Chemin de l'index FAISS (défaut : config).
            top_k: Nombre de chunks à récupérer.
            strategy: "similarity" ou "mmr".
        """
        # Client Anthropic
        if not ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY manquante. "
                "Copiez .env.example en .env et renseignez votre clé."
            )
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = CLAUDE_MODEL

        # Chargement de l'index FAISS
        vectorstore = load_vectorstore(vectorstore_path)

        # Retriever
        self.retriever = FinanceRetriever(
            vectorstore=vectorstore,
            top_k=top_k,
            strategy=strategy,
        )

        # Historique de conversation (optionnel, pour mode chat)
        self.conversation_history: List[dict] = []

        logger.success("FinanceRAGPipeline initialisé ✓")

    def query(
        self,
        question: str,
        use_history: bool = False,
    ) -> RAGResponse:
        """
        Pose une question et retourne une réponse sourcée.

        Args:
            question: Question en langage naturel.
            use_history: Si True, inclut l'historique dans le prompt (mode chat).

        Returns:
            RAGResponse avec answer, sources, métriques.
        """
        logger.info(f"Question reçue : {question[:100]}")

        # 1. Retrieval
        docs = self.retriever.retrieve(question)

        if not docs:
            logger.warning("Aucun document pertinent trouvé")
            return RAGResponse(
                answer=(
                    "Je ne trouve pas d'information pertinente dans les documents disponibles. "
                    "Veuillez reformuler votre question ou vérifier que les documents sont bien indexés."
                ),
                sources=[],
                query=question,
                model=self.model,
            )

        # 2. Construction du contexte
        context = self.retriever.format_context(docs)
        sources = self.retriever.get_sources(docs)

        # 3. Construction du prompt utilisateur
        user_message = self._build_user_message(question, context)

        # 4. Messages (avec ou sans historique)
        messages = []
        if use_history and self.conversation_history:
            messages.extend(self.conversation_history[-6:])  # 3 derniers tours
        messages.append({"role": "user", "content": user_message})

        # 5. Appel Claude API
        logger.info(f"Appel Claude ({self.model})...")
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                messages=messages,
            )

            answer = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            logger.success(
                f"Réponse générée — {input_tokens} tokens in / {output_tokens} tokens out"
            )

        except anthropic.APIConnectionError as e:
            logger.error(f"Erreur connexion API : {e}")
            raise
        except anthropic.RateLimitError as e:
            logger.error(f"Rate limit atteint : {e}")
            raise
        except anthropic.APIStatusError as e:
            logger.error(f"Erreur API ({e.status_code}) : {e.message}")
            raise

        # 6. Mise à jour de l'historique
        if use_history:
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": answer})

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            retrieved_docs=docs,
        )

    def stream_query(
        self,
        question: str,
    ) -> Generator[str, None, None]:
        """
        Version streaming de query() — yield les tokens un par un.
        Utilisé par Streamlit pour l'effet "typing".

        Args:
            question: Question en langage naturel.

        Yields:
            Tokens de la réponse au fur et à mesure.
        """
        docs = self.retriever.retrieve(question)

        if not docs:
            yield "Je ne trouve pas d'information pertinente dans les documents disponibles."
            return

        context = self.retriever.format_context(docs)
        user_message = self._build_user_message(question, context)

        with self.client.messages.stream(
            model=self.model,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                yield text

    def _build_user_message(self, question: str, context: str) -> str:
        """
        Construit le message utilisateur avec le contexte injecté.

        Format RAG standard :
        - Contexte documentaire en premier (anchor)
        - Question claire à la fin
        - Instructions de citation explicites
        """
        return f"""Voici les extraits de documents pertinents pour répondre à la question :

<context>
{context}
</context>

Question : {question}

Instructions :
- Réponds uniquement à partir du contexte ci-dessus
- Cite les sources entre crochets [Source: ...]
- Si l'information n'est pas dans le contexte, dis-le clairement
"""

    def reset_history(self):
        """Efface l'historique de conversation."""
        self.conversation_history = []
        logger.info("Historique de conversation réinitialisé")
