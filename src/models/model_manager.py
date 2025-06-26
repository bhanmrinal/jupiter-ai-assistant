"""
Singleton Model Manager for Jupiter FAQ Bot

Ensures models are loaded once at startup and reused for instant responses:
- No cold-start delays after initial load
- Memory-resident models for <2 second responses
- Thread-safe singleton implementation
"""

import threading
import time
from datetime import datetime
from typing import Optional

from loguru import logger as log

from src.models.llm_manager import LLMManager
from src.models.response_generator import ResponseGenerator
from src.models.retriever import Retriever


class ModelManagerSingleton:
    """Thread-safe singleton for managing pre-loaded models"""

    _instance: Optional["ModelManagerSingleton"] = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            log.info("🔧 Initializing ModelManagerSingleton...")
            self._startup_time = datetime.now()

            # Initialize components
            self._llm_manager: LLMManager | None = None
            self._retriever: Retriever | None = None
            self._response_generator: ResponseGenerator | None = None

            # Loading status
            self._is_loading = False
            self._is_ready = False
            self._load_error: str | None = None

            ModelManagerSingleton._initialized = True
            log.info("✅ ModelManagerSingleton created")

    def initialize_models(self) -> bool:
        """
        Initialize all models. Call this at application startup.

        Returns:
            bool: True if successful, False if failed
        """
        if self._is_ready:
            log.info("📦 Models already loaded and ready")
            return True

        if self._is_loading:
            log.info("⏳ Models are currently loading...")
            return False

        self._is_loading = True
        start_time = time.time()

        try:
            log.info("🚀 Starting model initialization...")

            # Step 1: Initialize LLM Manager
            log.info("📦 Loading LLM Manager...")
            llm_start = time.time()
            self._llm_manager = LLMManager()
            llm_time = time.time() - llm_start
            log.info(f"✅ LLM Manager loaded in {llm_time:.1f}s")

            # Step 2: Initialize Retriever (includes ChromaDB + Embeddings)
            log.info("📦 Loading Retriever and ChromaDB...")
            retriever_start = time.time()
            self._retriever = Retriever()
            retriever_time = time.time() - retriever_start
            log.info(f"✅ Retriever loaded in {retriever_time:.1f}s")

            # Step 3: Initialize Response Generator (lightweight - uses pre-loaded components)
            log.info("📦 Creating Response Generator...")
            generator_start = time.time()

            # Create response generator with pre-loaded components (no re-initialization)
            from src.models.response_generator import ResponseGenerator

            self._response_generator = ResponseGenerator(
                llm_manager=self._llm_manager, retriever=self._retriever
            )

            generator_time = time.time() - generator_start
            log.info(f"✅ Response Generator configured in {generator_time:.1f}s")

            # Test the system
            log.info("🧪 Testing system...")
            test_start = time.time()
            self._response_generator.generate_response("test")
            test_time = time.time() - test_start
            log.info(f"✅ System test completed in {test_time:.1f}s")

            total_time = time.time() - start_time
            self._is_ready = True
            self._is_loading = False

            log.info(f"🎉 All models loaded successfully in {total_time:.1f}s")
            log.info("💾 Models are now resident in memory for instant responses")

            return True

        except Exception as e:
            self._load_error = str(e)
            self._is_loading = False
            self._is_ready = False
            log.error(f"❌ Model initialization failed: {e}")
            return False

    def generate_response(self, query: str) -> dict:
        """
        Generate response using pre-loaded models.

        Args:
            query: User question

        Returns:
            dict: Response data with timing information
        """
        if not self._is_ready:
            return {
                "success": False,
                "error": "Models not ready. Please wait for initialization to complete.",
                "is_loading": self._is_loading,
                "load_error": self._load_error,
            }

        start_time = time.time()

        try:
            # Generate response using pre-loaded models
            result = self._response_generator.generate_response(query)

            response_time = time.time() - start_time

            return {
                "success": True,
                "response": result.response,
                "confidence": result.confidence_score,
                "sources": result.sources_used,
                "docs_count": result.retrieved_docs_count,
                "model_used": result.model_used,
                "language": result.language.value,
                "followup": result.suggested_followup,
                "response_time_ms": int(response_time * 1000),
                "generation_time_ms": result.generation_time_ms,
                "retrieval_time_ms": result.retrieval_time_ms,
            }

        except Exception as e:
            log.error(f"❌ Error generating response: {e}")
            return {
                "success": False,
                "error": str(e),
                "response_time_ms": int((time.time() - start_time) * 1000),
            }

    def health_check(self) -> dict:
        """Check system health and readiness"""
        return {
            "is_ready": self._is_ready,
            "is_loading": self._is_loading,
            "load_error": self._load_error,
            "startup_time": self._startup_time.isoformat(),
            "components": {
                "llm_manager": self._llm_manager is not None,
                "retriever": self._retriever is not None,
                "response_generator": self._response_generator is not None,
            },
        }

    def get_model_info(self) -> dict:
        """Get information about loaded models"""
        if not self._is_ready:
            return {"error": "Models not ready"}

        try:
            llm_info = self._llm_manager.get_model_info() if self._llm_manager else {}
            retriever_info = self._retriever.get_stats() if self._retriever else {}

            return {
                "llm": llm_info,
                "retriever": retriever_info,
                "memory_resident": True,
                "ready_for_instant_responses": True,
            }
        except Exception as e:
            return {"error": str(e)}


# Global singleton instance
model_manager = ModelManagerSingleton()


def get_model_manager() -> ModelManagerSingleton:
    """Get the global model manager instance"""
    return model_manager


def initialize_models_for_streamlit():
    """
    Initialize models for Streamlit app.
    Call this once at app startup.
    """
    manager = get_model_manager()
    return manager.initialize_models()
