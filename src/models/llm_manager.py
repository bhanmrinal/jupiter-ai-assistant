"""
LLM Manager for Jupiter FAQ Bot

Manages multiple language models based on query characteristics:
- Qwen2.5-72B-Instruct: Primary response generation
- DeepSeek-R1: Complex reasoning
- Multilingual models: Hindi/Hinglish support
"""

import re
from enum import Enum
from typing import Any

import torch
from loguru import logger as log
from transformers import pipeline

from src.database.data_models import LanguageEnum


class ModelType(Enum):
    """Available model types"""

    PRIMARY = "primary"  # Qwen2.5-72B-Instruct (Primary)
    REASONING = "reasoning"  # DeepSeek-R1 (Complex reasoning)
    MULTILINGUAL = "multilingual"  # Hindi/Hinglish support


class QueryComplexity(Enum):
    """Query complexity levels"""

    SIMPLE = "simple"  # Basic FAQ lookup
    MODERATE = "moderate"  # Multi-step or calculation
    COMPLEX = "complex"  # Deep reasoning required


class LLMManager:
    """Manages multiple LLM models for different query types"""

    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"LLMManager initialized with device: {self.device}")

        # Load models lazily for memory efficiency
        self._initialize_primary_model()

    def _initialize_primary_model(self):
        """Initialize the primary model for basic queries"""
        try:
            # Use Qwen2.5-1.5B for primary responses (good balance of speed/quality)
            model_name = "Qwen/Qwen2.5-1.5B-Instruct"

            log.info(f"Loading primary model: {model_name}")

            # Initialize pipeline for text generation
            self.pipelines[ModelType.PRIMARY] = pipeline(
                "text-generation",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
                return_full_text=False,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                trust_remote_code=True,
            )

            log.info("Primary model loaded successfully")

        except Exception as e:
            log.error(f"Failed to load primary model: {e}")
            # Fallback to a simple text generation
            self.pipelines[ModelType.PRIMARY] = None

    def _initialize_reasoning_model(self):
        """Initialize reasoning model for complex queries"""
        try:
            # Use DeepSeek-Coder for complex reasoning and calculations
            model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"

            log.info(f"Loading reasoning model: {model_name}")

            self.pipelines[ModelType.REASONING] = pipeline(
                "text-generation",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.3,  # Lower temperature for more focused reasoning
                top_p=0.8,
                trust_remote_code=True,
            )

            log.info("Reasoning model loaded successfully")

        except Exception as e:
            log.error(f"Failed to load reasoning model: {e}")
            self.pipelines[ModelType.REASONING] = None

    def _initialize_multilingual_model(self):
        """Initialize multilingual model for Hindi/Hinglish"""
        try:
            # Use Qwen2.5 which has good multilingual capabilities
            model_name = "Qwen/Qwen2.5-1.5B-Instruct"

            log.info(f"Loading multilingual model: {model_name}")

            self.pipelines[ModelType.MULTILINGUAL] = pipeline(
                "text-generation",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
                max_new_tokens=200,
                temperature=0.7,
                trust_remote_code=True,
            )

            log.info("Multilingual model loaded successfully")

        except Exception as e:
            log.error(f"Failed to load multilingual model: {e}")
            self.pipelines[ModelType.MULTILINGUAL] = None

    def detect_query_complexity(self, query: str) -> QueryComplexity:
        """Analyze query to determine complexity level"""
        query = query.lower().strip()

        # Complex indicators
        complex_patterns = [
            r"\b(calculate|computation|compare|difference|versus|vs|analysis)\b",
            r"\b(how much|what is the difference|which is better)\b",
            r"\b(multiple|several|different|various)\b.*\b(options|choices|ways)\b",
            r"\b(step by step|process|procedure|workflow)\b",
        ]

        # Moderate indicators
        moderate_patterns = [
            r"\b(how to|how can|what should|where to)\b",
            r"\b(enable|disable|activate|deactivate|configure)\b",
            r"\b(transfer|send|receive|payment|transaction)\b",
            r"\b(account|card|limit|balance|statement)\b",
        ]

        # Check for complex patterns
        for pattern in complex_patterns:
            if re.search(pattern, query):
                return QueryComplexity.COMPLEX

        # Check for moderate patterns
        for pattern in moderate_patterns:
            if re.search(pattern, query):
                return QueryComplexity.MODERATE

        # Default to simple
        return QueryComplexity.SIMPLE

    def detect_language(self, query: str) -> LanguageEnum:
        """Detect query language"""
        # Simple language detection based on script and common words
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

    def select_model(self, query: str, language: LanguageEnum = None) -> ModelType:
        """Select appropriate model based on query characteristics"""

        if language is None:
            language = self.detect_language(query)

        complexity = self.detect_query_complexity(query)

        # Model selection logic
        if language in [LanguageEnum.HINDI, LanguageEnum.HINGLISH]:
            # Initialize multilingual model if not loaded
            if ModelType.MULTILINGUAL not in self.pipelines:
                self._initialize_multilingual_model()
            return ModelType.MULTILINGUAL

        elif complexity == QueryComplexity.COMPLEX:
            # Initialize reasoning model if not loaded
            if ModelType.REASONING not in self.pipelines:
                self._initialize_reasoning_model()
            return ModelType.REASONING

        else:
            # Use primary model for simple/moderate English queries
            return ModelType.PRIMARY

    def generate_response(
        self,
        prompt: str,
        model_type: ModelType = None,
        max_length: int = 200,
        temperature: float = 0.7,
    ) -> tuple[str, float]:
        """
        Generate response using selected model

        Returns:
            Tuple of (response_text, confidence_score)
        """
        try:
            if model_type is None:
                model_type = self.select_model(prompt)

            # Get the appropriate pipeline
            pipeline_obj = self.pipelines.get(model_type)

            if pipeline_obj is None:
                log.warning(f"Model {model_type} not available, falling back to primary")
                pipeline_obj = self.pipelines.get(ModelType.PRIMARY)

                if pipeline_obj is None:
                    log.error("No models available")
                    return "I'm sorry, I'm unable to process your request right now.", 0.0

            # All models now use text generation
            result = pipeline_obj(
                prompt,
                max_new_tokens=max_length,  # Use max_new_tokens instead of max_length
                temperature=temperature,
                num_return_sequences=1,
                truncation=True,
                pad_token_id=(
                    pipeline_obj.tokenizer.eos_token_id
                    if hasattr(pipeline_obj.tokenizer, "eos_token_id")
                    else None
                ),
            )

            # Extract generated text
            if isinstance(result, list) and len(result) > 0:
                response = result[0]["generated_text"].replace(prompt, "").strip()
            else:
                response = str(result).replace(prompt, "").strip()

            # Set confidence based on model type
            if model_type == ModelType.REASONING:
                confidence = 0.7  # Lower confidence for reasoning models
            else:
                confidence = 0.8  # Higher confidence for primary/multilingual models

            # Clean up response
            response = self._clean_response(response)

            log.info(f"Generated response using {model_type.value} model")
            return response, confidence

        except Exception as e:
            log.error(f"Error generating response with {model_type}: {e}")
            return "I apologize, but I encountered an error processing your request.", 0.0

    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response"""
        # Remove common artifacts
        response = re.sub(r"<.*?>", "", response)  # Remove HTML tags
        response = re.sub(r"\n+", "\n", response)  # Normalize newlines
        response = response.strip()

        # Ensure response ends with punctuation
        if response and response[-1] not in ".!?":
            response += "."

        return response

    def get_model_info(self) -> dict[str, Any]:
        """Get information about loaded models"""
        info = {
            "device": self.device,
            "loaded_models": list(self.pipelines.keys()),
            "model_details": {},
        }

        for model_type, pipeline_obj in self.pipelines.items():
            if pipeline_obj:
                info["model_details"][model_type.value] = {
                    "model_name": getattr(pipeline_obj.model, "name_or_path", "unknown"),
                    "is_loaded": True,
                }

        return info

    def health_check(self) -> bool:
        """Check if at least one model is working"""
        try:
            test_prompt = "Hello, this is a test."
            response, confidence = self.generate_response(test_prompt)
            return len(response) > 0 and confidence > 0
        except Exception as e:
            log.error(f"Health check failed: {e}")
            return False
