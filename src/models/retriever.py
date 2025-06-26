"""
Retriever for Jupiter FAQ Bot
Production-ready retrieval without hardcoded patterns.
"""

from typing import Any

from src.database.chroma_client import ChromaClient
from src.database.data_models import CategoryEnum, FAQDocument, QueryResult
from src.utils.logger import get_logger

log = get_logger(__name__)


class Retriever:
    """Retrieves relevant FAQ documents using semantic search"""

    def __init__(self, chroma_client: ChromaClient):
        """Initialize retriever with database client"""
        self.chroma_client = chroma_client

    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.4,
        category_filter: CategoryEnum | None = None,
    ) -> QueryResult:
        """
        Search for relevant FAQ documents

        Args:
            query: User query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            category_filter: Optional category to filter by

        Returns:
            QueryResult with matched documents and metadata
        """
        try:
            log.info(f"Searching for: '{query}' (top_k={top_k}, threshold={similarity_threshold})")

            # Clean query
            cleaned_query = self._clean_query(query)
            if not cleaned_query:
                return QueryResult(
                    query=query,
                    matched_documents=[],
                    total_matches=0,
                    confidence=0.0,
                    search_metadata={
                        "cleaned_query": "",
                        "filters_applied": [],
                        "error": "Empty query after cleaning",
                    },
                )

            # Prepare search filters
            filters = self._prepare_filters(category_filter)

            # Search using ChromaDB
            results = self.chroma_client.search(
                query=cleaned_query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                filters=filters,
            )

            if not results:
                return QueryResult(
                    query=query,
                    matched_documents=[],
                    total_matches=0,
                    confidence=0.0,
                    search_metadata={
                        "cleaned_query": cleaned_query,
                        "filters_applied": list(filters.keys()) if filters else [],
                        "message": "No results found",
                    },
                )

            # Convert to FAQDocument objects
            matched_docs = self._convert_results_to_faq_docs(results)

            # Calculate overall confidence
            confidence = self._calculate_retrieval_confidence(results, similarity_threshold)

            return QueryResult(
                query=query,
                matched_documents=matched_docs,
                total_matches=len(matched_docs),
                confidence=confidence,
                search_metadata={
                    "cleaned_query": cleaned_query,
                    "filters_applied": list(filters.keys()) if filters else [],
                    "avg_similarity": sum(r["distance"] for r in results) / len(results),
                    "max_similarity": max(r["distance"] for r in results),
                },
            )

        except Exception as e:
            log.error(f"Search failed: {e}")
            return QueryResult(
                query=query,
                matched_documents=[],
                total_matches=0,
                confidence=0.0,
                search_metadata={"error": str(e), "search_failed": True},
            )

    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query"""
        if not query:
            return ""

        # Basic cleaning
        cleaned = query.strip()

        # Remove extra whitespace
        cleaned = " ".join(cleaned.split())

        return cleaned

    def _prepare_filters(self, category_filter: CategoryEnum | None) -> dict[str, Any]:
        """Prepare search filters"""
        filters = {}

        if category_filter:
            filters["category"] = category_filter.value

        return filters

    def _convert_results_to_faq_docs(self, results: list[dict]) -> list[FAQDocument]:
        """Convert search results to FAQDocument objects"""
        faq_docs = []

        for result in results:
            try:
                metadata = result.get("metadata", {})

                # Create FAQDocument from result
                from src.database.data_models import FAQMetadata, SourceTypeEnum
                
                faq_metadata = FAQMetadata(
                    source_url=metadata.get("source_url", ""),
                    source_type=SourceTypeEnum(metadata.get("source_type", SourceTypeEnum.UNKNOWN.value)),
                    confidence_score=metadata.get("confidence_score", 0.0)
                )
                
                faq_doc = FAQDocument(
                    question=metadata.get("question", ""),
                    answer=metadata.get("answer", ""),
                    category=CategoryEnum(metadata.get("category", CategoryEnum.GENERAL.value)),
                    metadata=faq_metadata
                )
                
                # Store similarity score separately for access later
                faq_doc.similarity_score = result.get("distance", 0.0)

                faq_docs.append(faq_doc)

            except Exception as e:
                log.warning(f"Failed to convert result to FAQDocument: {e}")
                continue

        return faq_docs

    def _calculate_retrieval_confidence(self, results: list[dict], threshold: float) -> float:
        """Calculate overall confidence for the retrieval"""
        if not results:
            return 0.0

        # Base confidence on top result similarity
        top_similarity = results[0].get("distance", 0.0)

        # Normalize to 0-1 scale
        confidence = min(max(top_similarity - threshold, 0.0) / (1.0 - threshold), 1.0)

        # Boost if we have multiple good results
        good_results = sum(1 for r in results if r.get("distance", 0.0) >= threshold)
        if good_results > 1:
            confidence = min(confidence * 1.1, 1.0)

        return confidence

    def get_retrieval_stats(self) -> dict[str, Any]:
        """Get retrieval statistics"""
        try:
            stats = self.chroma_client.get_collection_stats()
            return {
                "collection_stats": stats,
                "retriever_version": "2.0.0",
                "features": ["semantic_search", "category_filtering", "confidence_scoring"],
            }
        except Exception as e:
            log.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def health_check(self) -> bool:
        """Check if retriever is working properly"""
        try:
            # Try a simple search
            test_result = self.search("test", top_k=1)
            return test_result is not None
        except Exception as e:
            log.error(f"Retriever health check failed: {e}")
            return False
