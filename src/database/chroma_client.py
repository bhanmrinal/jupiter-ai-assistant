"""
ChromaDB client for Jupiter FAQ Bot
"""

import uuid
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config.settings import settings
from src.database.data_models import FAQDocument
from src.utils.logger import get_logger

log = get_logger(__name__)


class ChromaClient:
    """Client for managing ChromaDB vector database operations"""

    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        self._initialize_client()
        self._initialize_embedding_model()

    def _initialize_client(self):
        """Initialize ChromaDB client"""
        try:
            # Create persistent client with telemetry completely disabled
            self.client = chromadb.PersistentClient(
                path=settings.database.chromadb_path,
                settings=Settings(
                    anonymized_telemetry=False, 
                    allow_reset=True,
                    chroma_client_auth_provider=None,
                    chroma_client_auth_credentials=None
                ),
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=settings.database.collection_name,
                metadata={"description": "Jupiter FAQ embeddings"},
            )

            log.info(f"ChromaDB initialized with collection: {settings.database.collection_name}")
            log.info(f"Collection document count: {self.collection.count()}")

        except Exception as e:
            log.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _initialize_embedding_model(self):
        """Initialize sentence transformer model for embeddings"""
        try:
            self.embedding_model = SentenceTransformer(settings.model.embedding_model)
            log.info(f"Embedding model loaded: {settings.model.embedding_model}")
        except Exception as e:
            log.error(f"Failed to load embedding model: {e}")
            raise

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts"""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            log.error(f"Failed to generate embeddings: {e}")
            return []

    def add_documents(self, documents: list[FAQDocument]) -> bool:
        """Add FAQ documents to ChromaDB"""
        try:
            if not documents:
                log.warning("No documents to add")
                return False

            # Prepare data for ChromaDB
            ids = []
            texts = []
            metadatas = []

            for doc in documents:
                # Generate unique ID if not provided
                doc_id = doc.id or str(uuid.uuid4())
                ids.append(doc_id)

                # Combine question and answer for embedding
                combined_text = f"Q: {doc.question}\nA: {doc.answer}"
                texts.append(combined_text)

                # Prepare metadata
                metadata = {
                    "question": doc.question,
                    "answer": doc.answer,
                    "category": doc.category,
                    "language": doc.language,
                    "source_url": doc.metadata.source_url,
                    "source_type": doc.metadata.source_type.value,
                    "confidence_score": doc.metadata.confidence_score or 0.0,
                }
                metadatas.append(metadata)

            # Generate embeddings
            doc_embeddings = self.generate_embeddings(texts)
            if not doc_embeddings:
                log.error("Failed to generate embeddings for documents")
                return False

            # Add to ChromaDB
            self.collection.add(
                ids=ids, documents=texts, metadatas=metadatas, embeddings=doc_embeddings
            )

            log.info(f"Added {len(documents)} documents to ChromaDB")
            return True

        except Exception as e:
            log.error(f"Failed to add documents to ChromaDB: {e}")
            return False

    def search(
        self, query: str, top_k: int = 5, similarity_threshold: float = 0.4, filters: dict = None
    ) -> list[dict]:
        """Search for similar documents in ChromaDB"""
        try:
            # Generate embedding for query
            query_embedding = self.generate_embeddings([query])[0]

            # Prepare where clause for filtering
            where_clause = {}
            if filters:
                where_clause.update(filters)

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"],
            )

            # Format results for the new interface
            formatted_results = []

            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    distance = results["distances"][0][i]
                    
                    # Convert distance to similarity score (0-1 range)
                    import math
                    min_distance = 9.0  
                    max_distance = 45.0  
                    linear_sim = max(0.0, (max_distance - distance) / (max_distance - min_distance))
                    similarity_score = 1.0 / (1.0 + math.exp(-6 * (linear_sim - 0.5)))

                    # Only include results above threshold
                    if similarity_score >= similarity_threshold:
                        result = {
                            "id": results["ids"][0][i],
                            "distance": similarity_score,  # Use similarity score as distance
                            "metadata": results["metadatas"][0][i]
                        }
                        formatted_results.append(result)

            log.info(f"Found {len(formatted_results)} documents above threshold for: {query[:50]}...")
            return formatted_results

        except Exception as e:
            log.error(f"Failed to search ChromaDB: {e}")
            return {"query": query, "results": [], "total_found": 0}

    def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the ChromaDB collection"""
        try:
            count = self.collection.count()

            # Get sample documents to analyze
            sample_results = self.collection.peek(limit=10)

            categories = set()
            languages = set()
            source_types = set()

            if sample_results["metadatas"]:
                for metadata in sample_results["metadatas"]:
                    categories.add(metadata.get("category", "unknown"))
                    languages.add(metadata.get("language", "unknown"))
                    source_types.add(metadata.get("source_type", "unknown"))

            stats = {
                "total_documents": count,
                "categories": list(categories),
                "languages": list(languages),
                "source_types": list(source_types),
                "embedding_model": settings.model.embedding_model,
                "collection_name": settings.database.collection_name,
            }

            return stats

        except Exception as e:
            log.error(f"Failed to get collection stats: {e}")
            return {}

    def delete_documents(self, ids: list[str]) -> bool:
        """Delete documents from ChromaDB by IDs"""
        try:
            self.collection.delete(ids=ids)
            log.info(f"Deleted {len(ids)} documents from ChromaDB")
            return True
        except Exception as e:
            log.error(f"Failed to delete documents: {e}")
            return False

    def update_document(self, doc_id: str, document: FAQDocument) -> bool:
        """Update a document in ChromaDB"""
        try:
            # Delete existing document
            self.collection.delete(ids=[doc_id])

            # Add updated document
            document.id = doc_id
            return self.add_documents([document])

        except Exception as e:
            log.error(f"Failed to update document {doc_id}: {e}")
            return False

    def reset_collection(self) -> bool:
        """Reset the entire collection (delete all documents)"""
        try:
            # Delete the collection
            self.client.delete_collection(settings.database.collection_name)

            # Recreate the collection
            self.collection = self.client.create_collection(
                name=settings.database.collection_name,
                metadata={"description": "Jupiter FAQ embeddings"},
            )

            log.info("ChromaDB collection reset successfully")
            return True

        except Exception as e:
            log.error(f"Failed to reset collection: {e}")
            return False

    def health_check(self) -> bool:
        """Check if ChromaDB is healthy and accessible"""
        try:
            self.collection.count()
            return True
        except Exception as e:
            log.error(f"ChromaDB health check failed: {e}")
            return False
