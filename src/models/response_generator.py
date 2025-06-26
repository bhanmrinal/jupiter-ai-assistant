"""
Response Generator for Jupiter FAQ Bot

Production-ready response generation using Groq API.
Clean architecture without hardcoded patterns.
"""

from datetime import datetime
from typing import Any

from src.database.data_models import QueryResult
from src.models.llm_manager import LLMManager
from src.models.prompt_templates import PromptTemplates
from src.models.retriever import Retriever
from src.utils.logger import get_logger

log = get_logger(__name__)


class ResponseGenerator:
    """Generates responses using Groq API and semantic retrieval"""

    def __init__(self, retriever: Retriever, llm_manager: LLMManager):
        """
        Initialize response generator

        Args:
            retriever: Document retriever instance
            llm_manager: LLM manager instance
        """
        self.retriever = retriever
        self.llm_manager = llm_manager
        self.prompt_templates = PromptTemplates()

    def generate_response(self, query: str, max_tokens: int = 200) -> dict[str, Any]:
        """
        Generate response for user query

        Args:
            query: User query string
            max_tokens: Maximum tokens for response

        Returns:
            Response dictionary with answer and metadata
        """
        start_time = datetime.utcnow()

        try:
            log.info(f"Generating response for: {query[:50]}...")

            # Step 1: Retrieve relevant documents
            retrieval_result = self.retriever.search(query=query, top_k=3, similarity_threshold=0.4)

            # Step 2: Generate response based on retrieval results
            if retrieval_result.matched_documents and retrieval_result.confidence > 0.3:
                response = self._generate_contextual_response(query, retrieval_result, max_tokens)
            else:
                response = self._generate_fallback_response(query, max_tokens)

            # Step 3: Calculate response time
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds()

            # Step 4: Prepare final response
            return {
                "answer": response.get(
                    "answer", "I apologize, but I'm unable to provide a response at the moment."
                ),
                "confidence": response.get("confidence", 0.0),
                "source_documents": [
                    {
                        "question": doc.question,
                        "similarity": getattr(doc, "similarity_score", 0.0),
                        "source_url": doc.metadata.source_url,
                    }
                    for doc in retrieval_result.matched_documents[:2]
                ],
                "metadata": {
                    "query": query,
                    "response_time_seconds": round(response_time, 3),
                    "retrieval_confidence": retrieval_result.confidence,
                    "documents_found": retrieval_result.total_matches,
                    "generation_method": response.get("method", "unknown"),
                    "timestamp": start_time.isoformat(),
                },
            }

        except Exception as e:
            log.error(f"Response generation failed: {e}")
            return self._create_error_response(query, str(e))

    def _generate_contextual_response(
        self, query: str, retrieval_result: QueryResult, max_tokens: int
    ) -> dict[str, Any]:
        """Generate response using retrieved context"""
        try:
            # Prepare context from retrieved documents
            context = self._format_context(retrieval_result.matched_documents)

            # Detect language (simple detection)
            detected_language = self._detect_language(query)

            # Create system prompt
            system_prompt = self._create_system_prompt(context, detected_language)

            # PRIMARY: Try Groq conversation generation first
            if self.llm_manager.conversation_loaded:
                answer, confidence = self.llm_manager.generate_conversation(
                    system_prompt=system_prompt, user_query=query, max_tokens=max_tokens
                )

                if answer and len(answer.strip()) > 10:  # Valid response
                    return {"answer": answer, "confidence": confidence, "method": "groq_contextual"}

            # FALLBACK 1: DistilBERT Q&A extraction if Groq fails
            best_doc = retrieval_result.matched_documents[0]
            extracted_answer, extract_confidence = self.llm_manager.extract_answer(
                query, best_doc.answer
            )

            if extracted_answer and len(extracted_answer.strip()) > 5:
                return {
                    "answer": f"Based on our documentation: {extracted_answer}",
                    "confidence": extract_confidence * 0.8,  # Lower confidence for fallback
                    "method": "distilbert_fallback",
                }

            # FALLBACK 2: Direct document match as last resort
            return {
                "answer": f"Here's what I found: {best_doc.answer}",
                "confidence": retrieval_result.confidence * 0.6,  # Even lower confidence
                "method": "direct_match_fallback",
            }

        except Exception as e:
            log.error(f"Contextual response generation failed: {e}")
            return {"answer": "", "confidence": 0.0, "method": "error"}

    def _generate_fallback_response(self, query: str, max_tokens: int) -> dict[str, Any]:
        """Generate fallback response when no good context is found"""
        try:
            # Use Groq for general conversation
            if self.llm_manager.conversation_loaded:
                system_prompt = self._create_general_system_prompt()

                answer, confidence = self.llm_manager.generate_conversation(
                    system_prompt=system_prompt, user_query=query, max_tokens=max_tokens
                )

                if answer and len(answer.strip()) > 10:  # Valid response
                    return {
                        "answer": answer,
                        "confidence": confidence * 0.7,  # Lower confidence for no-context
                        "method": "groq_general",
                    }

            # Final fallback
            return {
                "answer": "I apologize, but I don't have specific information about that. Please contact Jupiter Money support for detailed assistance.",
                "confidence": 0.3,
                "method": "static_fallback",
            }

        except Exception as e:
            log.error(f"Fallback response generation failed: {e}")
            return {
                "answer": "I'm experiencing technical difficulties. Please try again or contact support.",
                "confidence": 0.1,
                "method": "error_fallback",
            }

    def _format_context(self, documents: list) -> str:
        """Format retrieved documents into context text"""
        if not documents:
            return ""

        context_parts = []
        for i, doc in enumerate(documents[:3], 1):  # Limit to top 3
            context_parts.append(f"Context {i}:")
            context_parts.append(f"Q: {doc.question}")
            context_parts.append(f"A: {doc.answer}")
            context_parts.append("")

        return "\n".join(context_parts)

    def _create_system_prompt(self, context: str, language: str) -> str:
        """Create system prompt for contextual response"""
        return f"""You are Jupiter Money's helpful customer service assistant for India's financial wellness community.

CONTEXT INFORMATION:
{context}

INSTRUCTIONS:
1. Answer based ONLY on the provided context above
2. Respond in {language} language naturally
3. Be helpful, concise, and accurate  
4. Include relevant steps when applicable
5. If context doesn't contain enough information, say so politely
6. Maintain a friendly, professional tone appropriate for Indian users
7. Don't mention that you're an AI or refer to the context directly
8. For financial advice, ensure regulatory compliance
9. Keep responses conversational and culturally appropriate for India

Provide a helpful response in {language}:"""

    def _create_general_system_prompt(self) -> str:
        """Create system prompt for general responses"""
        return """You are Jupiter Money's helpful customer service assistant for India's financial wellness community.

INSTRUCTIONS:
1. Provide general guidance about banking and financial services
2. Be helpful, friendly, and professional
3. Maintain a tone appropriate for Indian users
4. If you don't have specific information, suggest contacting Jupiter Money support
5. Keep responses concise and culturally appropriate
6. Focus on general financial wellness and banking concepts
7. Avoid giving specific financial advice without proper context

Provide a helpful general response:"""

    def _detect_language(self, query: str) -> str:
        """Simple language detection"""
        # Basic check for Hindi characters
        if any("\u0900" <= char <= "\u097f" for char in query):
            return "Hindi/English"
        return "English"

    def _create_error_response(self, query: str, error_message: str) -> dict[str, Any]:
        """Create error response"""
        return {
            "answer": "I apologize, but I'm experiencing technical difficulties. Please try again or contact Jupiter Money support.",
            "confidence": 0.0,
            "source_documents": [],
            "metadata": {
                "query": query,
                "error": error_message,
                "generation_method": "error",
                "timestamp": datetime.utcnow().isoformat(),
            },
        }

    def get_generation_stats(self) -> dict[str, Any]:
        """Get response generation statistics"""
        return {
            "generator_version": "2.0.0",
            "features": ["groq_primary", "distilbert_fallback", "semantic_retrieval"],
            "model_priority": ["Groq Llama-3.3-70B", "DistilBERT Q&A", "Direct Match"],
            "supported_languages": ["English", "Hindi", "Hinglish"],
            "llm_status": self.llm_manager.get_model_info(),
        }

    def health_check(self) -> bool:
        """Check if response generator is working"""
        try:
            # Test basic response generation
            test_response = self.generate_response("test", max_tokens=10)
            return test_response.get("answer", "") != ""
        except Exception as e:
            log.error(f"Response generator health check failed: {e}")
            return False
