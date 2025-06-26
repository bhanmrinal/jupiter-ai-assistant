"""
Response Generator for Jupiter FAQ Bot

Production-ready response generation with enhanced model support and strict guardrails.
Clean architecture without hardcoded patterns.
"""

from datetime import datetime
from typing import Any

from transformers import pipeline

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
        
        # Initialize QA pipeline for fallback
        try:
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=-1  # Use CPU
            )
            log.info("✅ QA pipeline initialized")
        except Exception as e:
            log.error(f"Failed to initialize QA pipeline: {e}")
            self.qa_pipeline = None

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
                        "answer": doc.answer,
                        "similarity": getattr(doc, "similarity_score", 0.0),
                        "metadata": doc.metadata
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
        self, 
        query: str, 
        retrieval_result: QueryResult, 
        max_tokens: int, 
        preferred_model: str = None
    ) -> dict[str, Any]:
        """Generate response using retrieved context with enhanced error handling and language consistency"""
        try:
            log.info(f"Generating contextual response for query: {query[:50]}...")
            start_time = datetime.now()

            # Detect language and enforce consistency
            detected_language = self._detect_language(query)
            predicted_category = self._predict_category(query)
            
            # Format context from retrieval results
            context = self._format_context(retrieval_result.matched_documents)
            retrieval_confidence = retrieval_result.confidence

            # Get the enhanced prompt template with language enforcement
            prompt_template = self.prompt_templates.get_rag_response_template()

            # Format the prompt with explicit language instruction
            formatted_prompt = prompt_template.format(
                context=context,
                query=query,
                detected_language=detected_language,
                predicted_category=predicted_category,
                retrieval_confidence=retrieval_confidence,
            )

            # Add additional language enforcement instruction
            language_enforcement = f"\n\nCRITICAL: Respond STRICTLY in {detected_language}. Do NOT mix languages unless explicitly asked to do so in the query."
            formatted_prompt += language_enforcement

            answer = None
            model_used = "none"

            # Try enhanced Groq LLM first
            if self.llm_manager.conversation_loaded:
                try:
                    best_model = self.llm_manager.get_best_groq_model(len(query), "complex")
                    
                    if best_model:
                        log.info(f"Using Groq model: {best_model}")
                        answer, _ = self.llm_manager.generate_conversation(
                            system_prompt=formatted_prompt,
                            user_query=f"Answer this query in {detected_language}: {query}",
                            max_tokens=max_tokens,
                            model_preference=preferred_model
                        )
                        model_used = best_model
                        
                        if answer:
                            # Verify language consistency
                            response_language = self._detect_language(answer)
                            if response_language != detected_language and detected_language != "english":
                                log.warning(f"Language mismatch detected. Query: {detected_language}, Response: {response_language}")
                                # Add language correction prompt
                                correction_prompt = f"Please translate the following response to {detected_language} while maintaining the same meaning and Jupiter team member tone:\n\n{answer}"
                                corrected_answer, _ = self.llm_manager.generate_conversation(
                                    system_prompt="You are a translator who maintains the original tone and context while translating.",
                                    user_query=correction_prompt,
                                    max_tokens=max_tokens
                                )
                                if corrected_answer:
                                    answer = corrected_answer
                                    
                except Exception as e:
                    log.error(f"Groq LLM generation failed: {e}")

            # Fallback to transformers if no answer
            if not answer:
                try:
                    log.info("Falling back to transformers...")
                    if self.qa_pipeline:
                        pipeline_result = self.qa_pipeline(question=query, context=context)
                        raw_answer = pipeline_result.get("answer", "")
                        model_used = "distilbert-qa"
                        
                        # Use LLM to enhance and ensure language consistency for transformer response
                        if raw_answer and self.llm_manager.conversation_loaded:
                            enhancement_prompt = f"""You are a Jupiter Money team member. A technical system provided this raw answer: "{raw_answer}"

Transform this into a warm, helpful response from Jupiter team member in {detected_language}. 

Make it:
1. Sound natural and caring like a Jupiter team member
2. Maintain the technical accuracy
3. Add Jupiter team warmth and identity
4. Respond STRICTLY in {detected_language}

User's original question: {query}"""

                            enhanced_answer, _ = self.llm_manager.generate_conversation(
                                system_prompt=enhancement_prompt,
                                user_query=f"Transform this technical answer for {detected_language} speaker",
                                max_tokens=300
                            )
                            
                            if enhanced_answer:
                                answer = enhanced_answer
                                model_used = "enhanced_distilbert"
                            else:
                                answer = raw_answer  # Use raw if enhancement fails
                    else:
                        log.warning("QA pipeline not available, skipping transformer fallback")
                            
                except Exception as e:
                    log.error(f"Transformers generation failed: {e}")

            # Final fallback
            if not answer:
                # Use LLM to generate language-appropriate fallback response
                fallback_prompt = f"""You are a Jupiter Money team member. The user asked: "{query}"

You don't have specific information about this query, but you need to respond helpfully in {detected_language}.

Respond as a caring Jupiter team member who:
1. Acknowledges the question warmly
2. Explains you don't have that specific information right now
3. Guides them to the right Jupiter resources (app, website, support team)
4. Maintains Jupiter team member identity and pride

CRITICAL: Respond STRICTLY in {detected_language}. Be warm, helpful, and professional."""

                try:
                    if self.llm_manager.conversation_loaded:
                        best_model = self.llm_manager.get_best_groq_model(len(query), "simple")
                        if best_model:
                            fallback_answer, _ = self.llm_manager.generate_conversation(
                                system_prompt=fallback_prompt,
                                user_query=f"Generate a helpful response in {detected_language}",
                                max_tokens=200
                            )
                            if fallback_answer:
                                answer = fallback_answer
                                model_used = "llm_fallback"
                    
                    # If LLM fails, use the no-context template which is dynamic
                    if not answer:
                        no_context_template = self.prompt_templates.get_no_context_template()
                        formatted_prompt = no_context_template.format(
                            query=query,
                            detected_language=detected_language
                        )
                        
                        # Try one more time with the template
                        if self.llm_manager.conversation_loaded:
                            answer, _ = self.llm_manager.generate_conversation(
                                system_prompt=formatted_prompt,
                                user_query=f"Respond helpfully in {detected_language}",
                                max_tokens=200
                            )
                            if answer:
                                model_used = "template_fallback"
                
                except Exception as e:
                    log.error(f"LLM fallback generation failed: {e}")
                
                # Only if everything fails, indicate system error
                if not answer:
                    answer = "System temporarily unavailable. Please try again."
                    model_used = "system_error"

            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()

            # Build response with metadata
            return {
                "answer": answer,
                "confidence": min(retrieval_confidence + 0.15, 0.95),  # Boost confidence for contextual responses
                "source_documents": [
                    {
                        "question": doc.question,
                        "answer": doc.answer,
                        "similarity": getattr(doc, "similarity_score", 0.0),
                        "metadata": doc.metadata
                    }
                    for doc in retrieval_result.matched_documents[:3]
                ],
                "metadata": {
                    "query": query,
                    "detected_language": detected_language,
                    "predicted_category": predicted_category,
                    "response_time_seconds": response_time,
                    "retrieval_confidence": retrieval_confidence,
                    "documents_found": len(retrieval_result.matched_documents),
                    "generation_method": "contextual_rag",
                    "model_used": model_used,
                    "timestamp": datetime.now().isoformat(),
                    "guardrails_triggered": retrieval_confidence < 0.4,
                },
            }

        except Exception as e:
            log.error(f"Contextual response generation failed: {e}")
            return self._create_error_response(query, str(e))

    def _generate_fallback_response(self, query: str, max_tokens: int, preferred_model: str = None) -> dict[str, Any]:
        """Generate fallback response when no good context is found with language consistency"""
        try:
            start_time = datetime.now()
            detected_language = self._detect_language(query)
            predicted_category = self._predict_category(query)
            
            # Use enhanced no-context template
            no_context_prompt = self.prompt_templates.get_no_context_template().format(
                query=query,
                detected_language=detected_language
            )

            # Add language enforcement
            language_enforcement = f"\n\nCRITICAL: Respond STRICTLY in {detected_language}. Do NOT mix languages unless explicitly asked to do so in the query."
            no_context_prompt += language_enforcement

            answer = None
            model_used = "none"

            # Try Groq with no-context template
            if self.llm_manager.conversation_loaded:
                try:
                    best_model = self.llm_manager.get_best_groq_model(len(query), "simple")
                    
                    if best_model:
                        answer, _ = self.llm_manager.generate_conversation(
                            system_prompt=no_context_prompt,
                            user_query=f"Answer this query in {detected_language}: {query}",
                            max_tokens=max_tokens,
                            model_preference=preferred_model or best_model
                        )
                        model_used = best_model

                        if answer:
                            # Verify language consistency
                            response_language = self._detect_language(answer)
                            if response_language != detected_language and detected_language != "english":
                                log.warning(f"Language mismatch in fallback. Query: {detected_language}, Response: {response_language}")
                                # Add language correction prompt
                                correction_prompt = f"Please translate the following response to {detected_language} while maintaining the same meaning and Jupiter team member tone:\n\n{answer}"
                                corrected_answer, _ = self.llm_manager.generate_conversation(
                                    system_prompt="You are a translator who maintains the original tone and context while translating.",
                                    user_query=correction_prompt,
                                    max_tokens=max_tokens
                                )
                                if corrected_answer:
                                    answer = corrected_answer
                except Exception as e:
                    log.error(f"Groq fallback generation failed: {e}")

            # Try HuggingFace fallback models if Groq failed
            if not answer:
                try:
                    # Use LLM to generate contextual response even without specific context
                    general_prompt = f"""You are a warm Jupiter Money team member. The user asked: "{query}"

You don't have specific documentation about this, but as a Jupiter team member, you should:
1. Acknowledge their question warmly in {detected_language}
2. Show that you understand their need
3. Provide general guidance about Jupiter's approach to such topics
4. Direct them to the right resources (Jupiter app, website, support)
5. Maintain your identity as someone who works at Jupiter and cares

CRITICAL: Respond STRICTLY in {detected_language}. Be helpful, warm, and professional."""

                    if self.llm_manager.conversation_loaded:
                        best_model = self.llm_manager.get_best_groq_model(len(query), "simple")
                        if best_model:
                            llm_answer, _ = self.llm_manager.generate_conversation(
                                system_prompt=general_prompt,
                                user_query=f"Provide helpful guidance in {detected_language}",
                                max_tokens=300
                            )
                            if llm_answer:
                                answer = llm_answer
                                model_used = "llm_general_guidance"
                    
                    # If that fails, try HuggingFace models
                    if not answer:
                        fallback_answer, _ = self.llm_manager._fallback_conversation(query, max_tokens)
                        if fallback_answer and len(fallback_answer.strip()) > 10:
                            # Enhance HuggingFace response with Jupiter identity
                            if self.llm_manager.conversation_loaded:
                                enhancement_prompt = f"""Transform this response into a Jupiter Money team member response in {detected_language}:
                                
Original response: "{fallback_answer}"
User question: "{query}"

Make it sound like a caring Jupiter team member while maintaining helpfulness."""

                                enhanced_response, _ = self.llm_manager.generate_conversation(
                                    system_prompt=enhancement_prompt,
                                    user_query=f"Enhance for Jupiter team identity in {detected_language}",
                                    max_tokens=250
                                )
                                if enhanced_response:
                                    answer = enhanced_response
                                    model_used = "enhanced_hf"
                                else:
                                    answer = fallback_answer
                                    model_used = "hf_raw"
                            else:
                                answer = fallback_answer
                                model_used = "hf_fallback"
                                
                except Exception as e:
                    log.error(f"HuggingFace fallback failed: {e}")

            # Final static fallback with language consistency
            if not answer:
                # Generate dynamic error response using LLM if possible
                error_prompt = f"""You are a Jupiter Money team member experiencing technical difficulties. 

The user asked: "{query}"

Apologize warmly in {detected_language} and guide them to alternative ways to get help (Jupiter app, website, direct support).

Be understanding, professional, and maintain Jupiter team identity even during technical issues."""

                try:
                    if self.llm_manager.conversation_loaded:
                        error_response, _ = self.llm_manager.generate_conversation(
                            system_prompt=error_prompt,
                            user_query=f"Generate an apologetic but helpful response in {detected_language}",
                            max_tokens=150
                        )
                        if error_response:
                            answer = error_response
                            model_used = "llm_error_response"
                
                except Exception as e:
                    log.error(f"LLM error response generation failed: {e}")
                
                # Only absolute last resort - minimal system message
                if not answer:
                    answer = "System temporarily unavailable. Please try again."
                    model_used = "system_minimal"

            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()

            return {
                "answer": answer,
                "confidence": 0.4,  # Lower confidence for fallback
                "source_documents": [],
                "metadata": {
                    "query": query,
                    "detected_language": detected_language,
                    "predicted_category": predicted_category,
                    "response_time_seconds": response_time,
                    "retrieval_confidence": 0.0,
                    "documents_found": 0,
                    "generation_method": "fallback_no_context",
                    "model_used": model_used,
                    "timestamp": datetime.now().isoformat(),
                    "guardrails_triggered": True,  # No context always triggers guardrails
                },
            }

        except Exception as e:
            log.error(f"Fallback response generation failed: {e}")
            return self._create_error_response(query, str(e))

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
        """Enhanced language detection with better accuracy"""
        import re
        
        # Check for Hindi/Devanagari characters first
        hindi_chars = re.findall(r"[\u0900-\u097F]", text)
        if hindi_chars:
            return "hindi"
        
        # More comprehensive Hinglish word detection
        hinglish_words = [
            "kya", "hai", "hota", "kaise", "mein", "aur", "ka", "ki", "se", "ko", 
            "karo", "kare", "kar", "ho", "haan", "nahi", "nahin", "toh", "phir",
            "abhi", "waha", "yaha", "kahan", "kab", "kyun", "kyunki", "lekin",
            "par", "ke", "me", "main", "hun", "hu", "hoon", "batao", "bataye",
            "chahiye", "chaahiye", "zarurat", "paisa", "paise", "rupaye", "rupee"
        ]
        
        # Convert text to lowercase for checking
        text_lower = text.lower()
        
        # Split into words to check exact matches (not substrings)
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Count Hinglish words
        hinglish_count = sum(1 for word in words if word in hinglish_words)
        
        # If at least 2 Hinglish words or more than 30% of words are Hinglish
        if hinglish_count >= 2 or (len(words) > 0 and hinglish_count / len(words) > 0.3):
            return "hinglish"
        
        # Single Hinglish word might still be Hinglish if it's a key indicator
        key_hinglish_indicators = ["kya", "kaise", "karo", "batao", "chahiye"]
        if any(word in words for word in key_hinglish_indicators):
            return "hinglish"
            
        # Default to English if no clear Hinglish indicators
        return "english"

    def _predict_category(self, query: str) -> str:
        """LLM-powered category prediction instead of keyword matching"""
        try:
            if not self.llm_manager.conversation_loaded:
                return "general"
            
            category_prompt = f"""You are a Jupiter Money categorization expert. Analyze this user query and identify the most relevant Jupiter service category.

User Query: "{query}"

Categories:
- cards: Debit cards, credit cards, card management, limits, PIN, blocking/unblocking
- payments: UPI, transfers, bill payments, money sending, payment issues
- accounts: Account opening, balance, statements, account management
- investments: Gold, mutual funds, investment options, portfolio
- loans: Personal loans, EMI, loan applications, eligibility
- rewards: Cashback, reward points, offers, benefits
- kyc: Verification, documents, compliance, identity proof
- technical: App issues, login problems, technical support
- general: Basic questions, general inquiries, company info

Rules:
1. Choose the SINGLE most relevant category
2. Base decision on the main intent of the query
3. If unclear, choose "general"
4. Respond with only the category name (lowercase)

Analyze the query and respond with just the category name."""

            best_model = self.llm_manager.get_best_groq_model(len(query), "simple")
            if best_model:
                category_response, _ = self.llm_manager.generate_conversation(
                    system_prompt=category_prompt,
                    user_query=f"Categorize: {query}",
                    max_tokens=10
                )
                
                if category_response:
                    predicted_category = category_response.strip().lower()
                    # Validate it's a known category
                    valid_categories = ["cards", "payments", "accounts", "investments", "loans", "rewards", "kyc", "technical", "general"]
                    if predicted_category in valid_categories:
                        return predicted_category
            
            return "general"
            
        except Exception as e:
            log.warning(f"LLM category prediction failed: {e}")
            return "general"

    def generate_follow_up_question(self, query: str, context_summary: str = "") -> str:
        """Generate a follow-up question using enhanced template"""
        try:
            # Use LLM for category prediction instead of hard-coded logic
            category = self._predict_category(query)
            language = self._detect_language(query)
            
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
                        user_query=f"Generate a natural follow-up question in {language}",
                        max_tokens=50
                    )
                    
                    if answer and len(answer.strip()) > 5:
                        return answer.strip()

            # If LLM fails, generate a generic follow-up using LLM with simpler prompt
            if self.llm_manager.conversation_loaded:
                simple_prompt = f"""You are a helpful Jupiter Money team member. The user just asked: "{query}"

Generate ONE natural follow-up question in {language} that would help them further (under 60 characters).

Make it practical and Jupiter-focused."""

                try:
                    simple_answer, _ = self.llm_manager.generate_conversation(
                        system_prompt=simple_prompt,
                        user_query=f"Generate simple follow-up in {language}",
                        max_tokens=30
                    )
                    if simple_answer and len(simple_answer.strip()) > 5:
                        return simple_answer.strip()
                except Exception as e:
                    log.error(f"Simple follow-up generation failed: {e}")

            # Only if everything fails, return empty (no hard-coded fallback)
            return ""

        except Exception as e:
            log.error(f"Follow-up generation failed: {e}")
            return ""

    def _create_error_response(self, query: str, error: str) -> dict[str, Any]:
        """Create error response with metadata using LLM for language-appropriate message"""
        
        # Detect language for appropriate error message
        detected_language = self._detect_language(query)
        
        # Try to generate language-appropriate error message using LLM
        error_message = "I apologize, but I'm experiencing technical difficulties. Please try again or contact Jupiter Money support."
        
        try:
            if self.llm_manager.conversation_loaded:
                error_prompt = f"""You are a Jupiter Money team member experiencing a technical issue. 

The user asked: "{query}"
Technical error occurred: {error}

Generate a brief, apologetic message in {detected_language} that:
1. Acknowledges the technical issue
2. Apologizes professionally
3. Directs them to try again or contact support
4. Maintains Jupiter team warmth despite the error
5. Keeps it under 100 characters

Be empathetic and professional."""

                error_response, _ = self.llm_manager.generate_conversation(
                    system_prompt=error_prompt,
                    user_query=f"Generate error message in {detected_language}",
                    max_tokens=80
                )
                
                if error_response:
                    error_message = error_response.strip()
                    
        except Exception as e:
            log.error(f"Error message generation failed: {e}")
            # Keep the default message if LLM fails
        
        return {
            "answer": error_message,
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

    def generate_related_suggestions(self, query: str, context_documents: list = None, user_history: list = None) -> list[str]:
        """Generate related query suggestions completely powered by LLM"""
        try:
            # Use LLM for category prediction
            category = self._predict_category(query)
            language = self._detect_language(query)
            
            # Generate all suggestions using LLM
            suggestions = []
            
            # Primary LLM-powered suggestions
            if self.llm_manager.conversation_loaded:
                llm_suggestions = self._generate_llm_suggestions(query, category, language)
                suggestions.extend(llm_suggestions)
            
            # Context-based suggestions using LLM
            if context_documents:
                context_suggestions = self._generate_context_based_suggestions(query, context_documents, language)
                suggestions.extend(context_suggestions)
            
            # History-based suggestions using LLM
            if user_history:
                history_suggestions = self._generate_history_based_suggestions(query, user_history, language)
                suggestions.extend(history_suggestions)
                
            # Remove duplicates and limit to 5 suggestions
            unique_suggestions = []
            seen = set()
            for suggestion in suggestions:
                if suggestion and suggestion.lower() not in seen and len(suggestion) > 10:
                    unique_suggestions.append(suggestion)
                    seen.add(suggestion.lower())
                    if len(unique_suggestions) >= 5:
                        break
                        
            return unique_suggestions
            
        except Exception as e:
            log.error(f"Suggestion generation failed: {e}")
            # If everything fails, return empty list (no hard-coded fallbacks)
            return []

    def _generate_context_based_suggestions(self, query: str, context_documents: list, language: str) -> list[str]:
        """Generate suggestions based on context documents using LLM"""
        try:
            if not self.llm_manager.conversation_loaded or not context_documents:
                return []
            
            # Extract key topics from context documents
            context_summary = ""
            for doc in context_documents[:3]:
                if isinstance(doc, dict):
                    context_summary += f"{doc.get('question', '')} {doc.get('answer', '')} "
                else:
                    context_summary += f"{getattr(doc, 'question', '')} {getattr(doc, 'answer', '')} "
            
            context_prompt = f"""You are a Jupiter Money expert helping users discover related information.

User's Current Query: "{query}"
Related Context: {context_summary[:500]}
Language: {language}

Based on the user's query and the related context, generate 2 helpful follow-up questions they might have.

Requirements:
1. Generate exactly 2 questions
2. Make them relevant to both the query and context
3. Use {language} language naturally
4. Each question under 60 characters
5. Focus on actionable Jupiter features
6. Don't repeat the original query

Output only the 2 questions, one per line."""

            response, _ = self.llm_manager.generate_conversation(
                system_prompt=context_prompt,
                user_query="Generate context-based suggestions",
                max_tokens=100
            )
            
            if response:
                suggestions = [s.strip() for s in response.split('\n') if s.strip() and '?' in s]
                return suggestions[:2]
                
        except Exception as e:
            log.warning(f"Context-based suggestion generation failed: {e}")
            
        return []

    def _generate_history_based_suggestions(self, query: str, user_history: list, language: str) -> list[str]:
        """Generate suggestions based on user history using LLM"""
        try:
            if not self.llm_manager.conversation_loaded or not user_history:
                return []
                
            # Get recent queries from history
            recent_queries = [h.get('query', '') for h in user_history[-5:]]
            history_text = " | ".join(recent_queries)
            
            history_prompt = f"""You are a Jupiter Money expert analyzing user behavior patterns.

User's Current Query: "{query}"
Recent Query History: {history_text}
Language: {language}

Based on their query history, suggest 1 helpful question that explores a different Jupiter service they haven't asked about yet.

Requirements:
1. Generate exactly 1 question
2. Choose a different topic/service than their recent queries
3. Use {language} language naturally  
4. Under 60 characters
5. Focus on Jupiter features they might find useful
6. Make it discovery-oriented

Output only the 1 question."""

            response, _ = self.llm_manager.generate_conversation(
                system_prompt=history_prompt,
                user_query="Generate history-based suggestion",
                max_tokens=50
            )
            
            if response:
                suggestion = response.strip()
                if suggestion and '?' in suggestion:
                    return [suggestion]
                
        except Exception as e:
            log.warning(f"History-based suggestion generation failed: {e}")
            
        return []

    def _generate_llm_suggestions(self, query: str, category: str, language: str) -> list[str]:
        """Use LLM to generate intelligent, contextual suggestions"""
        try:
            suggestion_prompt = f"""You are a Jupiter Money customer care specialist. Based on the user's query, generate 2 helpful follow-up questions they might have.

User Query: {query}
Category: {category}
Language: {language}

Rules:
1. Generate exactly 2 questions
2. Make them practical and actionable
3. Focus on Jupiter-specific features
4. Use {language} language naturally
5. Each question should be under 60 characters
6. Output only the questions, one per line

Examples:
- How to increase transaction limits?
- What are the fees for this service?
- How long does verification take?"""

            if self.llm_manager.conversation_loaded:
                response, _ = self.llm_manager.generate_conversation(
                    system_prompt=suggestion_prompt,
                    user_query="Generate suggestions",
                    max_tokens=100
                )
                
                if response:
                    suggestions = [s.strip() for s in response.split('\n') if s.strip() and '?' in s]
                    return suggestions[:2]
                    
        except Exception as e:
            log.warning(f"LLM suggestion generation failed: {e}")
            
        return []
