"""
LLM Manager for Jupiter FAQ Bot

Hybrid approach:
- Fast DistilBERT-style extraction for instant responses
- TinyLlama-1.1B for full conversational generation  
- RAG context integration for both modes
"""

from enum import Enum
from typing import Any

import torch
from loguru import logger as log

from src.database.data_models import LanguageEnum


class ModelType(Enum):
    """Available model types"""
    FAST_EXTRACT = "fast_extract"    # DistilBERT-style instant responses
    TINYLLAMA = "tinyllama"          # TinyLlama-1.1B-Chat-v1.0 (Full generation)


class LLMManager:
    """
    Hybrid LLM Manager for Jupiter FAQ Bot
    
    Combines fast extraction with full generation:
    - Fast mode: DistilBERT-style extraction (~0.1s)
    - Full mode: TinyLlama conversational AI (~20s)
    - Intelligent fallback based on query complexity
    """

    def __init__(self, enable_tinyllama: bool = True):
        """
        Initialize Hybrid LLM Manager
        
        Args:
            enable_tinyllama: Whether to load TinyLlama (slower but better quality)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enable_tinyllama = enable_tinyllama
        
        # TinyLlama components
        self.tinyllama_model = None
        self.tinyllama_tokenizer = None
        self.tinyllama_loaded = False
        
        # Fast extraction components
        self.fast_extractor = None
        self.fast_extractor_loaded = False
        
        log.info(f"Initializing Hybrid LLMManager on {self.device}")
        self._initialize_models()

    def _initialize_models(self):
        """Initialize both fast extraction and TinyLlama models"""
        
        # Always initialize fast extraction first (lightweight)
        self._initialize_fast_extractor()
        
        # Initialize TinyLlama if enabled
        if self.enable_tinyllama:
            self._initialize_tinyllama()
        else:
            log.info("TinyLlama disabled - using fast extraction only")

    def _initialize_fast_extractor(self):
        """Initialize fast DistilBERT-style extraction model"""
        try:
            from transformers import pipeline
            
            log.info("Loading fast extraction model (DistilBERT-based)...")
            
            # Use a lightweight model for fast Q&A extraction
            model_name = "distilbert-base-cased-distilled-squad"
            
            self.fast_extractor = pipeline(
                "question-answering",
                model=model_name,
                tokenizer=model_name,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=False
            )
            
            self.fast_extractor_loaded = True
            log.info("✅ Fast extraction model loaded successfully")
            
        except Exception as e:
            log.warning(f"Failed to load fast extraction model: {e}")
            self.fast_extractor_loaded = False

    def _initialize_tinyllama(self):
        """Initialize TinyLlama model for full generation"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            log.info(f"Loading {model_name} for full generation...")
            
            # Load tokenizer
            self.tinyllama_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tinyllama_tokenizer.pad_token is None:
                self.tinyllama_tokenizer.pad_token = self.tinyllama_tokenizer.eos_token
            
            # Load model with optimizations
            self.tinyllama_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.tinyllama_model = self.tinyllama_model.to(self.device)
            
            self.tinyllama_model.eval()
            self.tinyllama_loaded = True
            
            log.info("✅ TinyLlama loaded successfully")
            
        except Exception as e:
            log.error(f"Failed to load TinyLlama: {e}")
            self.tinyllama_loaded = False

    def _should_use_fast_mode(self, query: str, context: str = None) -> bool:
        """
        Determine whether to use fast extraction or full generation
        
        Args:
            query: User query
            context: Retrieved context (if available)
            
        Returns:
            True for fast extraction, False for full generation
        """
        query_lower = query.lower()
        
        # Use fast mode for simple, direct questions
        fast_patterns = [
            "what is", "how to", "how can i", "where is", "when",
            "pin", "reset", "balance", "account", "transfer"
        ]
        
        # Use full mode for complex or conversational queries
        complex_patterns = [
            "explain", "why", "difference", "compare", "help me understand",
            "step by step", "problem", "issue", "not working"
        ]
        
        # Check for complex patterns first
        if any(pattern in query_lower for pattern in complex_patterns):
            return False
        
        # Check for fast patterns
        if any(pattern in query_lower for pattern in fast_patterns):
            return True
        
        # Default: use fast mode if we have good context, full mode otherwise
        return context is not None and len(context.strip()) > 50

    def _extract_fast_response(self, query: str, context: str) -> tuple[str, float]:
        """Extract answer using fast DistilBERT-style model"""
        
        if not self.fast_extractor_loaded or not context:
            # Fall back to TinyLlama if no fast extractor or context
            return self._generate_tinyllama_response(query, context)
        
        try:
            # Use Q&A extraction on the context
            result = self.fast_extractor(
                question=query,
                context=context[:512]  # Limit context for speed
            )
            
            answer = result.get('answer', '').strip()
            confidence = result.get('score', 0.0)
            
            if answer and len(answer) > 10:
                # Enhance the extracted answer with Jupiter branding
                enhanced_answer = self._enhance_extracted_answer(answer, query)
                return enhanced_answer, min(confidence + 0.1, 0.95)  # Slight confidence boost
            else:
                # Fall back to TinyLlama if extraction quality is poor
                return self._generate_tinyllama_response(query, context)
                
        except Exception as e:
            log.error(f"Fast extraction failed: {e}")
            # Fall back to TinyLlama on extraction failure
            return self._generate_tinyllama_response(query, context)

    def _enhance_extracted_answer(self, answer: str, query: str) -> str:
        """Enhance extracted answer with Jupiter context and formatting"""
        
        # Clean up the answer
        answer = answer.strip()
        if not answer.endswith('.'):
            answer += '.'
        
        # Add Jupiter context for specific topics
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["pin", "reset"]):
            answer = f"{answer} You can do this securely through your Jupiter app."
        elif any(word in query_lower for word in ["transfer", "money", "send"]):
            answer = f"{answer} Jupiter offers UPI, NEFT, and IMPS transfer options."
        elif any(word in query_lower for word in ["account", "balance"]):
            answer = f"{answer} Check your Jupiter app for real-time updates."
        
        return answer

    def _generate_tinyllama_response(self, query: str, context: str = None, language: LanguageEnum = LanguageEnum.ENGLISH) -> tuple[str, float]:
        """Generate full response using TinyLlama"""
        
        if not self.tinyllama_loaded:
            # If TinyLlama failed to load, return a simple error message
            return ("I'm sorry, I'm currently unable to process your request. Please try again later.", 0.1)
        
        try:
            # Format prompt for TinyLlama
            system_message = (
                "You are Jupiter's helpful AI assistant specializing in banking and financial services. "
                "Provide accurate, concise, and friendly responses about account management, transfers, "
                "PIN resets, and general banking queries. Keep responses under 100 words and be specific."
            )
            
            if language == LanguageEnum.HINDI:
                system_message += " Respond in Hindi when appropriate."
            elif language == LanguageEnum.HINGLISH:
                system_message += " You can mix Hindi and English (Hinglish) in your responses."
            
            # Build prompt with context if available
            if context:
                user_message = f"Based on this information: {context[:300]}\n\nQuestion: {query}"
            else:
                user_message = query
            
            prompt = f"<|system|>\n{system_message}</s>\n<|user|>\n{user_message}</s>\n<|assistant|>\n"
            
            # Tokenize with proper attention mask
            inputs = self.tinyllama_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.tinyllama_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tinyllama_tokenizer.eos_token_id,
                    eos_token_id=self.tinyllama_tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3
                )
            
            # Decode and clean response
            response = self.tinyllama_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].strip()
            else:
                response = response[len(prompt):].strip()
            
            response = response.replace("</s>", "").strip()
            
            if len(response) > 300:
                response = response[:300] + "..."
            
            confidence = 0.9 if len(response) > 20 else 0.7
            
            log.debug(f"TinyLlama generated: {response[:50]}...")
            return response, confidence
            
        except Exception as e:
            log.error(f"TinyLlama generation failed: {e}")
            # Return error message if TinyLlama fails
            return ("I encountered an error processing your request. Please try again.", 0.1)



    def generate_response(
        self,
        prompt: str,
        context: str = None,
        language: LanguageEnum = LanguageEnum.ENGLISH,
        force_mode: ModelType = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
    ) -> tuple[str, float]:
        """
        Generate response using hybrid approach
        
        Args:
            prompt: User query
            context: Retrieved context from RAG
            language: Target response language
            force_mode: Force specific model type (optional)
            max_tokens: Maximum response length
            temperature: Response randomness
            
        Returns:
            tuple: (response_text, confidence_score)
        """
        
        # Determine which mode to use
        if force_mode == ModelType.FAST_EXTRACT:
            use_fast = True
        elif force_mode == ModelType.TINYLLAMA:
            use_fast = False
        else:
            # Intelligent mode selection
            use_fast = self._should_use_fast_mode(prompt, context)
        
        # Log mode selection
        mode = "Fast Extract" if use_fast else "TinyLlama"
        log.debug(f"Using {mode} mode for query: {prompt[:30]}...")
        
        # Generate response
        if use_fast and context:
            return self._extract_fast_response(prompt, context)
        else:
            # Use TinyLlama for all other cases (no context, complex queries, etc.)
            return self._generate_tinyllama_response(prompt, context, language)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about loaded models"""
        return {
            "models": {
                "fast_extractor": {
                    "loaded": self.fast_extractor_loaded,
                    "type": "DistilBERT Q&A",
                    "speed": "~0.1s"
                },
                "tinyllama": {
                    "loaded": self.tinyllama_loaded,
                    "type": "TinyLlama-1.1B-Chat",
                    "speed": "~20s"
                }
            },
            "device": self.device,
            "hybrid_mode": True,
            "status": "ready" if (self.fast_extractor_loaded or self.tinyllama_loaded) else "error",
            "capabilities": [
                "instant_extraction",
                "full_generation", 
                "financial_qa",
                "conversational",
                "multilingual_basic"
            ]
        }

    def health_check(self) -> bool:
        """Check if at least one model is working"""
        try:
            test_response, confidence = self.generate_response("Hello")
            return len(test_response) > 0 and confidence > 0
        except Exception as e:
            log.error(f"Health check failed: {e}")
            return False
