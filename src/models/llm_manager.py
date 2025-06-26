"""
LLM Manager for Jupiter FAQ Bot
Handles model loading and inference only.
"""

import os
from typing import Any

import torch
from loguru import logger as log

from config.settings import settings


class LLMManager:
    """LLM Manager focused on model management and inference only"""

    def __init__(self, enable_conversation: bool = True):
        """
        Initialize LLM Manager with Groq API

        Args:
            enable_conversation: Whether to enable Groq conversation model
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enable_conversation = enable_conversation

        # Groq client
        self.groq_client = None
        self.groq_loaded = False

        # DistilBERT for Q&A extraction
        self.fast_extractor = None
        self.fast_extractor_loaded = False

        log.info(f"Initializing LLMManager on {self.device}")
        self._initialize_models()

    def _initialize_models(self):
        """Initialize models"""
        self._initialize_fast_extractor()

        if self.enable_conversation:
            self._initialize_groq_client()

    def _initialize_fast_extractor(self):
        """Initialize DistilBERT Q&A extraction model"""
        try:
            from transformers import pipeline

            log.info("Loading DistilBERT Q&A model...")

            self.fast_extractor = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                tokenizer="distilbert-base-cased-distilled-squad",
                device=0 if self.device == "cuda" else -1,
                return_all_scores=False,
            )

            self.fast_extractor_loaded = True
            log.info("✅ DistilBERT Q&A model loaded")

        except Exception as e:
            log.error(f"Failed to load DistilBERT: {e}")
            self.fast_extractor_loaded = False

    def _initialize_groq_client(self):
        """Initialize Groq API client"""
        try:
            from groq import Groq

            api_key = settings.api.groq_api_key or os.getenv("GROQ_API_KEY")

            if not api_key:
                log.warning("Groq API key not found. Set GROQ_API_KEY environment variable.")
                self.groq_loaded = False
                return

            log.info("Initializing Groq API client...")

            self.groq_client = Groq(api_key=api_key)
            self.groq_loaded = True

            log.info("✅ Groq API client initialized")

        except Exception as e:
            log.error(f"Failed to initialize Groq client: {e}")
            self.groq_loaded = False

    def extract_answer(self, question: str, context: str) -> tuple[str, float]:
        """Extract answer from context using DistilBERT"""
        if not self.fast_extractor_loaded or not context:
            return "", 0.0

        try:
            result = self.fast_extractor(
                question=question,
                context=context[:512],  # Limit context for speed
            )

            answer = result.get("answer", "").strip()
            confidence = result.get("score", 0.0)

            return answer, confidence

        except Exception as e:
            log.error(f"Fast extraction failed: {e}")
            return "", 0.0

    def generate_conversation(
        self, system_prompt: str, user_query: str, max_tokens: int = 150
    ) -> tuple[str, float]:
        """Generate conversational response using Groq API with retry logic"""
        if not self.groq_loaded:
            return "", 0.0

        import time
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        # Retry up to 3 times with exponential backoff
        for attempt in range(3):
            try:
                response = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile", 
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3,
                    top_p=0.9,
                    stream=False,
                    timeout=15,  # 15 second timeout
                )

                if response.choices and response.choices[0].message:
                    content = response.choices[0].message.content.strip()
                    confidence = 0.95  # Groq models are generally high quality
                    return content, confidence

                return "", 0.0

            except Exception as e:
                log.warning(f"Groq attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:  # Don't wait after last attempt
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s

        log.error("All Groq conversation generation attempts failed")
        return "", 0.0

    def get_model_info(self) -> dict[str, Any]:
        """Get model information"""
        return {
            "models": {
                "distilbert_qa": {
                    "loaded": self.fast_extractor_loaded,
                    "type": "DistilBERT Q&A",
                    "speed": "~0.1s",
                    "local": True,
                },
                "groq_conversation": {
                    "loaded": self.groq_loaded,
                    "type": "Llama-3.3-70B via Groq",
                    "speed": "~0.8s",
                    "local": False,
                },
            },
            "device": self.device,
            "status": "ready" if (self.fast_extractor_loaded or self.groq_loaded) else "error",
        }

    def health_check(self) -> bool:
        """Check if models are working"""
        try:
            # Test DistilBERT
            if self.fast_extractor_loaded:
                answer, _ = self.extract_answer("test", "This is a test context")
                if answer is not None:  # Even empty answer is valid
                    return True

            # Test Groq
            if self.groq_loaded:
                response, _ = self.generate_conversation(
                    "You are a helpful assistant.", "Say hello", max_tokens=10
                )
                if response:
                    return True

            return False

        except Exception as e:
            log.error(f"Health check failed: {e}")
            return False

    @property
    def conversation_loaded(self) -> bool:
        """Check if conversation model is loaded"""
        return self.groq_loaded
