"""
Retriever for Jupiter FAQ Bot

Handles document retrieval and ranking for RAG:
- Semantic search using ChromaDB
- Result reranking and filtering
- Context preparation for LLM
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger as log

from config.settings import settings
from src.database.chroma_client import ChromaDBClient
from src.database.data_models import CategoryEnum, LanguageEnum


@dataclass
class RetrievedDocument:
    """Structure for retrieved document with metadata"""

    id: str
    question: str
    answer: str
    category: str
    source_url: str
    similarity_score: float
    confidence_score: float
    rank: int = 0
    relevance_score: float = 0.0


@dataclass
class RetrievalResult:
    """Complete retrieval result with context"""

    query: str
    documents: list[RetrievedDocument]
    total_found: int
    retrieval_time_ms: int
    language: LanguageEnum
    suggested_category: str | None = None
    context_text: str = ""


class Retriever:
    """Handles document retrieval and ranking for RAG pipeline"""

    def __init__(self):
        self.chroma_client = ChromaDBClient()
        self.similarity_threshold = settings.model.similarity_threshold
        self.top_k = settings.model.top_k_results

        log.info("Retriever initialized")

    def retrieve(
        self,
        query: str,
        language: LanguageEnum = None,
        category_filter: str = None,
        top_k: int = None,
        similarity_threshold: float = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query

        Args:
            query: User query text
            language: Query language for filtering
            category_filter: Category to filter results
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score

        Returns:
            RetrievalResult with ranked documents and context
        """
        start_time = datetime.now()

        # Use defaults if not provided
        if top_k is None:
            top_k = self.top_k
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold

        try:
            # Detect language if not provided
            if language is None:
                language = self._detect_language(query)

            # Suggest category based on query
            suggested_category = self._suggest_category(query)

            # Use suggested category if no filter provided
            if category_filter is None and suggested_category:
                category_filter = suggested_category

            # Search ChromaDB
            search_results = self.chroma_client.search_similar(
                query=query,
                n_results=top_k * 2,  # Get more to allow for filtering
                category_filter=category_filter,
            )

            # Convert to RetrievedDocument objects
            documents = []
            for i, result in enumerate(search_results["results"]):
                if result["similarity_score"] >= similarity_threshold:
                    doc = RetrievedDocument(
                        id=result["id"],
                        question=result["question"],
                        answer=result["answer"],
                        category=result["category"],
                        source_url=result["source_url"],
                        similarity_score=result["similarity_score"],
                        confidence_score=result["confidence_score"],
                        rank=i + 1,
                    )
                    documents.append(doc)

            # Rerank documents
            documents = self._rerank_documents(query, documents)

            # Take only top_k after reranking
            documents = documents[:top_k]

            # Generate context text
            context_text = self._generate_context(documents)

            # Calculate retrieval time
            end_time = datetime.now()
            retrieval_time_ms = int((end_time - start_time).total_seconds() * 1000)

            result = RetrievalResult(
                query=query,
                documents=documents,
                total_found=len(documents),
                retrieval_time_ms=retrieval_time_ms,
                language=language,
                suggested_category=suggested_category,
                context_text=context_text,
            )

            log.info(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
            return result

        except Exception as e:
            log.error(f"Error during retrieval: {e}")
            return RetrievalResult(
                query=query,
                documents=[],
                total_found=0,
                retrieval_time_ms=0,
                language=language or LanguageEnum.ENGLISH,
                suggested_category=None,
                context_text="",
            )

    def _detect_language(self, query: str) -> LanguageEnum:
        """Detect query language"""
        import re

        # Simple language detection
        hindi_patterns = [
            r"[\u0900-\u097F]",  # Devanagari script
            r"\b(kya|hai|kaise|kahan|kab|kyun|mera|mujhe|karna|chahiye)\b",
            r"\b(paisa|bank|account|card|payment|transfer)\b.*\b(kaise|kya|hai)\b",
        ]

        english_words = len(re.findall(r"\b[a-zA-Z]+\b", query))
        hindi_indicators = sum(1 for pattern in hindi_patterns if re.search(pattern, query.lower()))

        if hindi_indicators > 0 and english_words > 0:
            return LanguageEnum.HINGLISH
        elif hindi_indicators > 0:
            return LanguageEnum.HINDI
        else:
            return LanguageEnum.ENGLISH

    def _suggest_category(self, query: str) -> str | None:
        """Suggest most relevant category based on query keywords"""
        query_lower = query.lower()

        # Category keyword mappings
        category_keywords = {
            CategoryEnum.CARDS: [
                "card",
                "credit",
                "debit",
                "pin",
                "cvv",
                "edge",
                "rupay",
                "visa",
                "block",
                "unblock",
                "activate",
                "limit",
                "statement",
                "bill",
            ],
            CategoryEnum.PAYMENTS: [
                "payment",
                "pay",
                "upi",
                "transfer",
                "send",
                "money",
                "transaction",
                "qr",
                "scan",
                "bill",
                "recharge",
                "merchant",
                "refund",
            ],
            CategoryEnum.ACCOUNTS: [
                "account",
                "balance",
                "savings",
                "salary",
                "corporate",
                "pots",
                "open",
                "close",
                "statement",
                "passbook",
                "dormant",
            ],
            CategoryEnum.INVESTMENTS: [
                "invest",
                "mutual",
                "fund",
                "sip",
                "portfolio",
                "returns",
                "gold",
                "digifold",
                "fd",
                "fixed",
                "deposit",
                "rd",
                "recurring",
            ],
            CategoryEnum.LOANS: [
                "loan",
                "personal",
                "borrow",
                "emi",
                "interest",
                "credit",
                "eligibility",
                "apply",
                "repay",
                "tenure",
                "amount",
            ],
            CategoryEnum.REWARDS: [
                "reward",
                "points",
                "cashback",
                "offer",
                "benefit",
                "loyalty",
                "redeem",
                "program",
                "bonus",
                "scratch",
            ],
            CategoryEnum.KYC: [
                "kyc",
                "verification",
                "document",
                "aadhaar",
                "pan",
                "identity",
                "verify",
                "upload",
                "selfie",
                "compliance",
            ],
            CategoryEnum.TRACK: [
                "track",
                "expense",
                "budget",
                "spending",
                "category",
                "insight",
                "analysis",
                "report",
                "summary",
                "breakdown",
            ],
            CategoryEnum.TECHNICAL: [
                "app",
                "login",
                "password",
                "otp",
                "error",
                "bug",
                "issue",
                "update",
                "notification",
                "sync",
                "backup",
            ],
        }

        # Score each category
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                category_scores[category] = score

        # Return category with highest score
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            return best_category.value

        return None

    def _rerank_documents(
        self, query: str, documents: list[RetrievedDocument]
    ) -> list[RetrievedDocument]:
        """Rerank documents based on additional relevance signals"""

        for doc in documents:
            relevance_score = self._calculate_relevance_score(query, doc)
            doc.relevance_score = relevance_score

        # Sort by combined score (similarity + relevance + confidence)
        def combined_score(doc: RetrievedDocument) -> float:
            return (
                doc.similarity_score * 0.5 + doc.relevance_score * 0.3 + doc.confidence_score * 0.2
            )

        # Sort in descending order
        documents.sort(key=combined_score, reverse=True)

        # Update ranks
        for i, doc in enumerate(documents):
            doc.rank = i + 1

        return documents

    def _calculate_relevance_score(self, query: str, doc: RetrievedDocument) -> float:
        """Calculate additional relevance score based on content analysis"""
        score = 0.0
        query_lower = query.lower()
        question_lower = doc.question.lower()
        answer_lower = doc.answer.lower()

        # Exact keyword matches in question (higher weight)
        query_words = set(query_lower.split())
        question_words = set(question_lower.split())
        answer_words = set(answer_lower.split())

        # Question keyword overlap
        question_overlap = (
            len(query_words & question_words) / len(query_words) if query_words else 0
        )
        score += question_overlap * 0.4

        # Answer keyword overlap
        answer_overlap = len(query_words & answer_words) / len(query_words) if query_words else 0
        score += answer_overlap * 0.2

        # Question length preference (moderate length questions often better)
        question_length = len(doc.question.split())
        if 5 <= question_length <= 15:
            score += 0.1

        # Answer completeness (longer answers often more helpful)
        answer_length = len(doc.answer.split())
        if answer_length >= 20:
            score += 0.1
        elif answer_length >= 10:
            score += 0.05

        # Source confidence boost
        score += doc.confidence_score * 0.2

        return min(score, 1.0)  # Cap at 1.0

    def _generate_context(self, documents: list[RetrievedDocument]) -> str:
        """Generate formatted context text for LLM prompt"""
        if not documents:
            return ""

        context_parts = []
        context_parts.append(
            "Based on the following relevant information from Jupiter's help center:"
        )
        context_parts.append("")

        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[Context {i}]")
            context_parts.append(f"Q: {doc.question}")
            context_parts.append(f"A: {doc.answer}")
            context_parts.append(f"Category: {doc.category.title()}")
            context_parts.append("")

        return "\n".join(context_parts)

    def get_retrieval_stats(self) -> dict[str, Any]:
        """Get retrieval statistics"""
        try:
            chroma_stats = self.chroma_client.get_collection_stats()
            return {
                "total_documents": chroma_stats.get("total_documents", 0),
                "categories": chroma_stats.get("categories", []),
                "languages": chroma_stats.get("languages", []),
                "similarity_threshold": self.similarity_threshold,
                "top_k": self.top_k,
                "collection_name": chroma_stats.get("collection_name", "unknown"),
            }
        except Exception as e:
            log.error(f"Error getting retrieval stats: {e}")
            return {"error": str(e)}

    def health_check(self) -> bool:
        """Check if retrieval system is working"""
        try:
            # Test basic retrieval
            result = self.retrieve("test query", top_k=1)
            return result.total_found >= 0  # Should return at least empty result
        except Exception as e:
            log.error(f"Retrieval health check failed: {e}")
            return False
