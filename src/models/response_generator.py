"""
Response Generator for Jupiter FAQ Bot

Combines retrieval and LLM generation for RAG-powered responses:
- Context injection from retrieved documents
- Prompt engineering for financial domain
- Response validation and confidence scoring
"""

from dataclasses import dataclass
from datetime import datetime

from loguru import logger as log

from config.settings import settings
from src.database.data_models import LanguageEnum
from src.models.llm_manager import LLMManager, ModelType
from src.models.prompt_templates import PromptTemplates
from src.models.retriever import RetrievalResult, Retriever


@dataclass
class GenerationResult:
    """Complete response generation result"""

    response: str
    confidence_score: float
    sources_used: list[str]
    retrieved_docs_count: int
    model_used: str
    generation_time_ms: int
    retrieval_time_ms: int
    language: LanguageEnum
    suggested_followup: str | None = None


class ResponseGenerator:
    """Generates RAG-powered responses for Jupiter FAQ queries"""

    def __init__(self, llm_manager=None, retriever=None):
        # Use provided instances or create new ones
        self.llm_manager = llm_manager if llm_manager is not None else LLMManager()
        self.retriever = retriever if retriever is not None else Retriever()
        self.confidence_threshold = settings.model.confidence_threshold

        log.info("ResponseGenerator initialized")

    def generate_response(
        self, query: str, language: LanguageEnum = None, category_hint: str = None
    ) -> GenerationResult:
        """
        Generate a comprehensive response using RAG

        Args:
            query: User query text
            language: Query language for appropriate response
            category_hint: Optional category hint for better retrieval

        Returns:
            GenerationResult with response and metadata
        """
        start_time = datetime.now()

        try:
            # Step 1: Retrieve relevant documents
            log.info(f"Generating response for query: {query[:50]}...")

            retrieval_result = self.retriever.retrieve(
                query=query, language=language, category_filter=category_hint
            )

            # Step 2: Determine language  
            detected_language = (
                retrieval_result.language if retrieval_result.language else LanguageEnum.ENGLISH
            )

            # Step 3: Generate response with context
            if retrieval_result.total_found > 0:
                response, confidence = self._generate_with_context(
                    query=query,
                    retrieval_result=retrieval_result,
                    language=detected_language,
                )
            else:
                response, confidence = self._generate_without_context(
                    query=query, language=detected_language
                )

            # Step 4: Post-process response
            response = self._post_process_response(response, detected_language)

            # Step 5: Generate suggested follow-up
            followup = self._generate_followup(query, retrieval_result)

            # Calculate timing
            end_time = datetime.now()
            total_time_ms = int((end_time - start_time).total_seconds() * 1000)
            generation_time_ms = total_time_ms - retrieval_result.retrieval_time_ms

            # Collect sources
            sources_used = [doc.source_url for doc in retrieval_result.documents]

            result = GenerationResult(
                response=response,
                confidence_score=confidence,
                sources_used=sources_used,
                retrieved_docs_count=retrieval_result.total_found,
                model_used="TinyLlama-1.1B",
                generation_time_ms=generation_time_ms,
                retrieval_time_ms=retrieval_result.retrieval_time_ms,
                language=detected_language,
                suggested_followup=followup,
            )

            log.info(f"Response generated with confidence: {confidence:.2f}")
            return result

        except Exception as e:
            log.error(f"Error generating response: {e}")
            return self._generate_error_response(query, str(e))

    def _generate_with_context(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        language: LanguageEnum,
    ) -> tuple[str, float]:
        """Generate response using retrieved context"""

        # Select appropriate prompt template
        # Use centralized template selection
        template = PromptTemplates.get_template_by_language(language)

        # Build prompt with context
        prompt = template.format(
            context=retrieval_result.context_text,
            query=query,
            num_sources=retrieval_result.total_found,
            detected_language=language.value,
            predicted_category=retrieval_result.suggested_category or "general",
            retrieval_confidence=f"{len(retrieval_result.documents)}/{retrieval_result.total_found} matches",
        )

        # Generate response with context for hybrid model
        response, base_confidence = self.llm_manager.generate_response(
            prompt=query,  # Use original query instead of full prompt
            context=retrieval_result.context_text,  # Pass context for fast extraction
            language=language, 
            max_tokens=300, 
            temperature=0.7
        )

        return response, base_confidence

    def _generate_without_context(
        self, query: str, language: LanguageEnum
    ) -> tuple[str, float]:
        """Generate response without retrieved context"""

        prompt = PromptTemplates.get_no_context_template().format(
            query=query, detected_language=language.value
        )

        response, confidence = self.llm_manager.generate_response(
            prompt=query,  # Use original query 
            context=None,  # No context available
            language=language, 
            max_tokens=200, 
            temperature=0.8
        )

        return response, confidence

    def _post_process_response(self, response: str, language: LanguageEnum) -> str:
        """Clean the generated response"""
        response = response.strip()
        if response and response[-1] not in ".!?":
            response += "."
        return response

    def _generate_followup(self, query: str, retrieval_result: RetrievalResult) -> str | None:
        """Generate suggested follow-up question using LLM"""

        if not retrieval_result.documents:
            return None

        # Prepare context summary for follow-up generation
        context_summary = f"Found {len(retrieval_result.documents)} relevant documents about {retrieval_result.suggested_category}"
        if retrieval_result.documents:
            # Use first document's answer as context hint
            context_summary += f". Main topic: {retrieval_result.documents[0].question[:50]}..."

        # Use LLM to generate contextual follow-up question
        followup_prompt = PromptTemplates.get_followup_generation_template().format(
            query=query,
            category=retrieval_result.suggested_category or "general",
            context_summary=context_summary,
        )

        # Generate follow-up using fast mode for quick suggestions
        followup_response, _ = self.llm_manager.generate_response(
            prompt=followup_prompt,
            context=context_summary,  # Use context summary for follow-up
            language=LanguageEnum.ENGLISH,
            force_mode=ModelType.TINYLLAMA,  # Use TinyLlama for follow-ups
            max_tokens=50,  # Keep follow-ups short
            temperature=0.3,  # Low temperature for consistent suggestions
        )

        # Clean and validate the response
        followup = followup_response.strip()
        if followup and len(followup) > 10 and "?" in followup:
            # Extract just the question if LLM added extra text
            if "?" in followup:
                followup = followup.split("?")[0] + "?"
            return followup

        return None

    def _generate_error_response(self, query: str, error: str) -> GenerationResult:
        """Generate error response when something goes wrong"""

        return GenerationResult(
            response="I apologize, but I'm having trouble processing your request right now. Please try again later or contact Jupiter support for assistance.",
            confidence_score=0.0,
            sources_used=[],
            retrieved_docs_count=0,
            model_used="error",
            generation_time_ms=0,
            retrieval_time_ms=0,
            language=LanguageEnum.ENGLISH,
            suggested_followup=None,
        )

    def health_check(self) -> bool:
        """Check if response generation system is working"""
        try:
            test_result = self.generate_response("test query")
            return len(test_result.response) > 0
        except Exception as e:
            log.error(f"Response generator health check failed: {e}")
            return False
