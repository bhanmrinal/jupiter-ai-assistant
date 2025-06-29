"""
LLM Manager for Jupiter FAQ Bot
Handles model loading and inference with multiple models.
"""

import os
from typing import Any

import torch
from loguru import logger as log

from config.settings import settings


class LLMManager:
    """Enhanced LLM Manager with multiple Groq models and HuggingFace fallbacks"""

    # Available Groq models with their characteristics
    GROQ_MODELS = {
        "llama-3.3-70b-versatile": {
            "name": "Llama 3.3 70B",
            "speed": "~0.8s",
            "context": "128k",
            "best_for": "General conversation, complex reasoning"
        },
        "llama-3.1-8b-instant": {
            "name": "Llama 3.1 8B",
            "speed": "~0.3s", 
            "context": "128k",
            "best_for": "Fast responses, simple queries"
        },
        "gemma2-9b-it": {
            "name": "Gemma 2 9B",
            "speed": "~0.5s",
            "context": "8k",
            "best_for": "Balanced performance"
        }
    }

    # HuggingFace fallback models
    HF_MODELS = {
        "distilbert-base-cased-distilled-squad": {
            "name": "DistilBERT QA",
            "type": "question-answering",
            "speed": "~0.1s",
            "best_for": "Fast Q&A extraction"
        },
        "microsoft/DialoGPT-medium": {
            "name": "DialoGPT Medium",
            "type": "text-generation", 
            "speed": "~1.2s",
            "best_for": "Conversational responses"
        },
        "facebook/blenderbot-400M-distill": {
            "name": "BlenderBot 400M",
            "type": "text2text-generation",
            "speed": "~0.8s", 
            "best_for": "Chat responses"
        }
    }

    def __init__(self, enable_conversation: bool = True, device: str = "auto"):
        """Initialize LLM Manager with multiple model backends"""
        self.device = self._determine_device(device)
        self.enable_conversation = enable_conversation
        
        # Model state tracking
        self.qa_models_loaded = False
        self.groq_loaded = False
        
        # Model containers
        self.hf_models = {}
        self.hf_models_loaded = {}
        self.groq_client = None
        self.available_groq_models = []
        self.preferred_model = "auto"

        log.info(f"Initializing Enhanced LLMManager on {self.device}")

        # Initialize models
        self._initialize_models()

    def _determine_device(self, device: str) -> str:
        """Determine the best device to use for models"""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        return device

    def _initialize_models(self):
        """Initialize all available models"""
        self._initialize_hf_models()
        
        if self.enable_conversation:
            self._initialize_groq_client()

    def _initialize_hf_models(self):
        """Initialize HuggingFace models"""
        try:
            from transformers import pipeline
            
            # Always load DistilBERT for Q&A
            log.info("Loading DistilBERT Q&A model...")
            try:
                self.hf_models["distilbert_qa"] = pipeline(
                    "question-answering",
                    model="distilbert-base-cased-distilled-squad",
                    tokenizer="distilbert-base-cased-distilled-squad",
                    device=0 if self.device == "cuda" else -1,
                    return_all_scores=False,
                )
                self.hf_models_loaded["distilbert_qa"] = True
                log.info("✅ DistilBERT Q&A model loaded")
            except Exception as e:
                log.error(f"Failed to load DistilBERT: {e}")
                self.hf_models_loaded["distilbert_qa"] = False

            # Try to load additional fallback models
            try:
                log.info("Loading BlenderBot fallback model...")
                self.hf_models["blenderbot"] = pipeline(
                    "text2text-generation",
                    model="facebook/blenderbot-400M-distill",
                    device=0 if self.device == "cuda" else -1,
                    max_length=200,
                    truncation=True
                )
                self.hf_models_loaded["blenderbot"] = True
                log.info("✅ BlenderBot fallback model loaded")
            except Exception as e:
                log.warning(f"BlenderBot fallback not available: {e}")
                self.hf_models_loaded["blenderbot"] = False

        except Exception as e:
            log.error(f"Failed to initialize HuggingFace models: {e}")

    def _initialize_groq_client(self):
        """Initialize Groq API client and check available models"""
        try:
            from groq import Groq

            api_key = settings.api.groq_api_key or os.getenv("GROQ_API_KEY")

            if not api_key:
                log.warning("Groq API key not found. Set GROQ_API_KEY environment variable.")
                self.groq_loaded = False
                return

            log.info("Initializing Groq API client...")
            self.groq_client = Groq(api_key=api_key)
            
            # Test connection and get available models
            self._check_available_groq_models()
            
            if self.available_groq_models:
                self.groq_loaded = True
                log.info(f"✅ Groq API client initialized with {len(self.available_groq_models)} models")
            else:
                log.warning("No Groq models available")
                self.groq_loaded = False

        except Exception as e:
            log.error(f"Failed to initialize Groq client: {e}")
            self.groq_loaded = False

    def _check_available_groq_models(self):
        """Check which Groq models are available"""
        try:
            # Test with a simple call to see which models work
            for model_id in self.GROQ_MODELS.keys():
                try:
                    response = self.groq_client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=1,
                        timeout=5
                    )
                    if response:
                        self.available_groq_models.append(model_id)
                        log.info(f"✅ {self.GROQ_MODELS[model_id]['name']} available")
                except Exception as e:
                    log.warning(f"❌ {model_id} not available: {e}")
                    
        except Exception as e:
            log.error(f"Failed to check available Groq models: {e}")

    def get_best_groq_model(self, query_length: int = 0, complexity: str = "medium") -> str:
        """Select the best Groq model based on query characteristics"""
        if not self.available_groq_models:
            return None
            
        # Auto-select based on query characteristics
        if self.preferred_model != "auto" and self.preferred_model in self.available_groq_models:
            return self.preferred_model
            
        # Smart selection logic
        if query_length < 100 and complexity == "simple":
            # Prefer fast models for simple queries
            for model in ["llama-3.1-8b-instant", "gemma2-9b-it"]:
                if model in self.available_groq_models:
                    return model
                    
        elif complexity == "complex" or query_length > 500:
            # Prefer powerful models for complex queries
            for model in ["llama-3.3-70b-versatile"]:
                if model in self.available_groq_models:
                    return model
        
        # Default fallback to best available
        priority_order = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant", 
            "gemma2-9b-it"
        ]
        
        for model in priority_order:
            if model in self.available_groq_models:
                return model
                
        return self.available_groq_models[0] if self.available_groq_models else None

    def extract_answer(self, question: str, context: str) -> tuple[str, float]:
        """Extract answer from context using DistilBERT"""
        if not self.hf_models_loaded.get("distilbert_qa", False) or not context:
            return "", 0.0

        try:
            result = self.hf_models["distilbert_qa"](
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
        self, system_prompt: str, user_query: str, max_tokens: int = 500, model_preference: str = None
    ) -> tuple[str, float]:
        """Generate conversational response using best available Groq model"""
        if not self.groq_loaded:
            return self._fallback_conversation(user_query, max_tokens)

        # Select best model for this query
        query_complexity = "complex" if len(user_query) > 200 else "medium"
        selected_model = model_preference or self.get_best_groq_model(len(user_query), query_complexity)
        
        if not selected_model:
            return self._fallback_conversation(user_query, max_tokens)

        import time
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        # Retry up to 3 times with exponential backoff
        for attempt in range(3):
            try:
                response = self.groq_client.chat.completions.create(
                    model=selected_model, 
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3,
                    top_p=0.9,
                    stream=False,
                    timeout=15,
                )

                if response.choices and response.choices[0].message:
                    content = response.choices[0].message.content.strip()
                    confidence = 0.95  # Groq models are generally high quality
                    
                    log.info(f"Response generated using {self.GROQ_MODELS[selected_model]['name']}")
                    return content, confidence

            except Exception as e:
                log.warning(f"Groq attempt {attempt + 1}/3 with {selected_model} failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)

        log.error("All Groq conversation generation attempts failed")
        return self._fallback_conversation(user_query, max_tokens)

    def _fallback_conversation(self, user_query: str, max_tokens: int) -> tuple[str, float]:
        """Generate response using HuggingFace fallback models"""
        try:
            # Try BlenderBot first
            if self.hf_models_loaded.get("blenderbot", False):
                log.info("Using BlenderBot fallback for conversation")
                result = self.hf_models["blenderbot"](
                    user_query,
                    max_length=min(max_tokens, 200),
                    do_sample=True,
                    temperature=0.7
                )
                
                if result and len(result) > 0:
                    answer = result[0].get("generated_text", "").strip()
                    if answer and len(answer) > 10:
                        return answer, 0.75  # Medium confidence for fallback
                        
            # Final fallback - simple template response
            return self._template_fallback(user_query), 0.4
            
        except Exception as e:
            log.error(f"Fallback conversation failed: {e}")
            return self._template_fallback(user_query), 0.3

    def _template_fallback(self, user_query: str) -> str:
        """Generate a template-based response as final fallback"""
        return (
            "I understand you're asking about Jupiter Money services. "
            "While I'm having technical difficulties right now, I'd recommend "
            "checking the Jupiter app or contacting our support team for "
            "detailed assistance with your query."
        )

    def get_model_info(self) -> dict[str, Any]:
        """Get comprehensive model information"""
        groq_models_info = {}
        for model_id in self.available_groq_models:
            model_data = self.GROQ_MODELS.get(model_id, {})
            groq_models_info[model_id] = {
                "loaded": True,
                "name": model_data.get("name", model_id),
                "speed": model_data.get("speed", "~1s"),
                "context": model_data.get("context", "Unknown"),
                "best_for": model_data.get("best_for", "General use"),
                "type": "groq",
                "local": False,
            }

        hf_models_info = {}
        for model_key, loaded in self.hf_models_loaded.items():
            if loaded:
                model_data = None
                if model_key == "distilbert_qa":
                    model_data = self.HF_MODELS["distilbert-base-cased-distilled-squad"]
                elif model_key == "blenderbot":
                    model_data = self.HF_MODELS["facebook/blenderbot-400M-distill"]
                
                if model_data:
                    hf_models_info[model_key] = {
                        "loaded": True,
                        "name": model_data["name"],
                        "type": model_data["type"],
                        "speed": model_data["speed"],
                        "best_for": model_data["best_for"],
                        "local": True,
                    }

        # Combine all models into a single 'models' key for compatibility
        all_models = {}
        all_models.update(groq_models_info)
        all_models.update(hf_models_info)

        return {
            "models": all_models,
            "groq_models": groq_models_info,
            "hf_models": hf_models_info,
            "device": self.device,
            "status": "ready" if (self.groq_loaded or any(self.hf_models_loaded.values())) else "error",
            "preferred_model": self.preferred_model,
            "available_groq_count": len(self.available_groq_models),
        }

    def health_check(self) -> bool:
        """Check if models are working"""
        try:
            # Test HuggingFace models
            if self.hf_models_loaded.get("distilbert_qa", False):
                answer, _ = self.extract_answer("test", "This is a test context")
                if answer is not None:
                    return True

            # Test Groq models
            if self.groq_loaded and self.available_groq_models:
                response, _ = self.generate_conversation(
                    "You are a helpful assistant.", "Say hello", max_tokens=10
                )
                if response:
                    return True

            return False

        except Exception as e:
            log.error(f"Health check failed: {e}")
            return False

    def set_preferred_model(self, model_id: str):
        """Set preferred Groq model"""
        if model_id in self.GROQ_MODELS or model_id == "auto":
            self.preferred_model = model_id
            log.info(f"Preferred model set to: {model_id}")
        else:
            log.warning(f"Invalid model ID: {model_id}")

    @property
    def conversation_loaded(self) -> bool:
        """Check if conversation models are loaded"""
        return self.groq_loaded or self.hf_models_loaded.get("blenderbot", False)
