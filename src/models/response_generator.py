"""
Response Generator for Jupiter FAQ Bot

Production-ready response generation with enhanced model support and strict guardrails.
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
    """Enhanced response generator with multiple models and strict guardrails"""

    def __init__(self, retriever: Retriever, llm_manager: LLMManager):
        """
        Initialize response generator

        Args:
            retriever: Document retriever instance
            llm_manager: Enhanced LLM manager instance
        """
        self.retriever = retriever
        self.llm_manager = llm_manager
        self.prompt_templates = PromptTemplates()

    def generate_response(self, query: str, max_tokens: int = 500, preferred_model: str = None) -> dict[str, Any]:
        """
        Generate comprehensive response using enhanced multi-model approach

        Args:
            query: User question
            max_tokens: Maximum tokens for response
            preferred_model: Preferred Groq model (optional)

        Returns:
            Comprehensive response dictionary with metadata
        """
        start_time = datetime.now()

        try:
            # Step 1: Retrieve relevant documents
            log.info(f"Processing query: {query[:100]}...")
            retrieval_result = self.retriever.search(query, top_k=5, similarity_threshold=0.4)

            # Step 2: Generate response based on retrieval quality
            if retrieval_result.confidence >= 0.4 and retrieval_result.matched_documents:
                response = self._generate_contextual_response(
                    query, retrieval_result, max_tokens, preferred_model
                )
            else:
                log.info("Low confidence retrieval, using fallback response")
                response = self._generate_fallback_response(query, max_tokens, preferred_model)

            # Step 3: Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()

            # Step 4: Prepare final response with enhanced metadata
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
                    "model_used": response.get("model_used", "unknown"),
                    "timestamp": start_time.isoformat(),
                    "guardrails_triggered": response.get("guardrails_triggered", False),
                },
            }

        except Exception as e:
            log.error(f"Response generation failed: {e}")
            return self._create_error_response(query, str(e))

    def _generate_contextual_response(
        self, query: str, retrieval_result: QueryResult, max_tokens: int, preferred_model: str = None
    ) -> dict[str, Any]:
        """Generate response using retrieved context with enhanced model selection"""
        try:
            # Prepare context from retrieved documents
            context = self._format_context(retrieval_result.matched_documents)

            # Detect language and category
            detected_language = self._detect_language(query)
            predicted_category = self._predict_category(query)

            # Check if we should trigger guardrails
            guardrails_triggered = False
            if retrieval_result.confidence < 0.4:
                guardrails_triggered = True
                log.warning(f"Low confidence ({retrieval_result.confidence}), triggering guardrails")

            # PRIMARY: Try enhanced Groq conversation generation
            if self.llm_manager.conversation_loaded:
                # Get best model for this query
                best_model = self.llm_manager.get_best_groq_model(
                    len(query), 
                    "complex" if len(query) > 200 else "medium"
                )
                
                if best_model:
                    # Use model-specific optimized prompt
                    system_prompt = self.prompt_templates.get_system_prompt_for_model(
                        best_model, context, detected_language, predicted_category, retrieval_result.confidence
                    )

                    answer, confidence = self.llm_manager.generate_conversation(
                        system_prompt=system_prompt, 
                        user_query=query, 
                        max_tokens=max_tokens,
                        model_preference=preferred_model or best_model
                    )

                    if answer and len(answer.strip()) > 10:
                        return {
                            "answer": answer,
                            "confidence": confidence,
                            "method": "groq_contextual_enhanced",
                            "model_used": preferred_model or best_model,
                            "guardrails_triggered": guardrails_triggered
                        }

            # FALLBACK 1: DistilBERT Q&A extraction
            if retrieval_result.matched_documents:
                best_doc = retrieval_result.matched_documents[0]
                extracted_answer, extract_confidence = self.llm_manager.extract_answer(
                    query, best_doc.answer
                )

                if extracted_answer and len(extracted_answer.strip()) > 5:
                    return {
                        "answer": f"Based on our documentation: {extracted_answer}",
                        "confidence": extract_confidence * 0.8,
                        "method": "distilbert_fallback",
                        "model_used": "distilbert-qa",
                        "guardrails_triggered": guardrails_triggered
                    }

            # FALLBACK 2: Direct document match as last resort
            best_doc = retrieval_result.matched_documents[0]
            return {
                "answer": f"Here's what I found: {best_doc.answer}",
                "confidence": retrieval_result.confidence * 0.6,
                "method": "direct_match_fallback",
                "model_used": "direct_retrieval",
                "guardrails_triggered": guardrails_triggered
            }

        except Exception as e:
            log.error(f"Contextual response generation failed: {e}")
            return {
                "answer": "", 
                "confidence": 0.0, 
                "method": "error",
                "model_used": "none",
                "guardrails_triggered": True
            }

    def _generate_fallback_response(self, query: str, max_tokens: int, preferred_model: str = None) -> dict[str, Any]:
        """Generate fallback response when no good context is found"""
        try:
            detected_language = self._detect_language(query)
            
            # Use enhanced no-context template
            no_context_prompt = self.prompt_templates.get_no_context_template().format(
                query=query,
                detected_language=detected_language
            )

            # Try Groq with no-context template
            if self.llm_manager.conversation_loaded:
                best_model = self.llm_manager.get_best_groq_model(len(query), "simple")
                
                if best_model:
                    answer, confidence = self.llm_manager.generate_conversation(
                        system_prompt=no_context_prompt,
                        user_query=query,
                        max_tokens=max_tokens,
                        model_preference=preferred_model or best_model
                    )

                    if answer and len(answer.strip()) > 10:
                        return {
                            "answer": answer,
                            "confidence": confidence * 0.7,  # Lower confidence for no-context
                            "method": "groq_no_context",
                            "model_used": preferred_model or best_model,
                            "guardrails_triggered": True  # No context always triggers guardrails
                        }

            # Try HuggingFace fallback models
            fallback_answer, fallback_confidence = self.llm_manager._fallback_conversation(query, max_tokens)
            
            if fallback_answer and len(fallback_answer.strip()) > 10:
                return {
                    "answer": fallback_answer,
                    "confidence": fallback_confidence,
                    "method": "hf_fallback",
                    "model_used": "blenderbot",
                    "guardrails_triggered": True
                }

            # Final static fallback
            return {
                "answer": "I apologize, but I don't have specific information about that. Please contact Jupiter Money support for detailed assistance.",
                "confidence": 0.3,
                "method": "static_fallback",
                "model_used": "template",
                "guardrails_triggered": True
            }

        except Exception as e:
            log.error(f"Fallback response generation failed: {e}")
            return {
                "answer": "I'm experiencing technical difficulties. Please try again or contact support.",
                "confidence": 0.1,
                "method": "error_fallback",
                "model_used": "none",
                "guardrails_triggered": True
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

    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        # Check for Hindi/Devanagari characters
        import re
        hindi_chars = re.findall(r"[\u0900-\u097F]", text)
        if hindi_chars:
            return "hindi"
        
        # Check for common Hindi/Hinglish words
        hinglish_words = ["kya", "hai", "hota", "kaise", "mein", "aur", "ka", "ki", "se", "ko"]
        if any(word in text.lower() for word in hinglish_words):
            return "hinglish"
        
        return "english"

    def _predict_category(self, query: str) -> str:
        """Simple category prediction based on keywords"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["card", "debit", "credit", "swipe"]):
            return "cards"
        elif any(word in query_lower for word in ["payment", "upi", "transfer", "send"]):
            return "payments"
        elif any(word in query_lower for word in ["account", "balance", "statement"]):
            return "accounts"
        elif any(word in query_lower for word in ["invest", "gold", "mutual", "fund"]):
            return "investments"
        elif any(word in query_lower for word in ["loan", "borrow", "emi"]):
            return "loans"
        elif any(word in query_lower for word in ["reward", "point", "cashback"]):
            return "rewards"
        elif any(word in query_lower for word in ["kyc", "verification", "document"]):
            return "kyc"
        elif any(word in query_lower for word in ["app", "login", "error", "problem"]):
            return "technical"
        else:
            return "general"

    def generate_follow_up_question(self, query: str, context_summary: str = "") -> str:
        """Generate a follow-up question using enhanced template"""
        try:
            category = self._predict_category(query)
            
            followup_prompt = self.prompt_templates.get_followup_generation_template().format(
                query=query,
                category=category,
                context_summary=context_summary
            )

            if self.llm_manager.conversation_loaded:
                best_model = self.llm_manager.get_best_groq_model(len(query), "simple")
                
                if best_model:
                    answer, _ = self.llm_manager.generate_conversation(
                        system_prompt=followup_prompt,
                        user_query="Generate follow-up question",
                        max_tokens=50
                    )
                    
                    if answer and len(answer.strip()) > 5:
                        return answer.strip()

            # Fallback follow-up questions by category
            fallback_questions = {
                "cards": "Would you like to know about setting up spending limits?",
                "payments": "Do you want to learn about UPI setup as well?",
                "accounts": "Should I explain account notifications too?",
                "investments": "Are you curious about minimum investment amounts?",
                "loans": "Would the application process help?",
                "rewards": "Want to know how to earn more rewards?",
                "kyc": "Need help with other documents?",
                "technical": "Is the app working fine otherwise?",
                "general": "Is there anything else I can help with?"
            }
            
            return fallback_questions.get(category, "Is there anything else I can help you with?")

        except Exception as e:
            log.error(f"Follow-up generation failed: {e}")
            return "Is there anything else I can help you with?"

    def _create_error_response(self, query: str, error: str) -> dict[str, Any]:
        """Create error response with metadata"""
        return {
            "answer": "I apologize, but I'm experiencing technical difficulties. Please try again or contact Jupiter Money support.",
            "confidence": 0.0,
            "source_documents": [],
            "metadata": {
                "query": query,
                "error": error,
                "response_time_seconds": 0.0,
                "retrieval_confidence": 0.0,
                "documents_found": 0,
                "generation_method": "error",
                "model_used": "none",
                "timestamp": datetime.now().isoformat(),
                "guardrails_triggered": True,
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
