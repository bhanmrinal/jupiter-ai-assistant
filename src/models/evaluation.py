"""
Evaluation System for Jupiter FAQ Bot

Evaluates semantic similarity, answer relevance, and system performance.
Provides comprehensive metrics for retrieval-based vs LLM-based approaches.
"""

import os
import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple
import numpy as np
from dataclasses import dataclass

from src.database.chroma_client import ChromaClient
from src.models.llm_manager import LLMManager
from src.models.response_generator import ResponseGenerator
from src.models.retriever import Retriever
from src.utils.logger import get_logger
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results of evaluation run"""
    query: str
    expected_answer: str
    actual_answer: str
    method: str
    confidence: float
    response_time: float
    semantic_similarity: float
    relevance_score: float
    retrieval_accuracy: float
    bleu_score: float
    model_used: str


@dataclass
class EvaluationSummary:
    """Summary of evaluation across multiple queries"""
    total_queries: int
    avg_semantic_similarity: float
    avg_relevance_score: float
    avg_response_time: float
    avg_confidence: float
    method_breakdown: Dict[str, int]
    model_breakdown: Dict[str, int]
    accuracy_by_category: Dict[str, float]


class AnswerEvaluator:
    """Comprehensive evaluation system for FAQ bot responses"""

    def __init__(self, chroma_client: ChromaClient, llm_manager: LLMManager):
        """Initialize evaluator with database and LLM components"""
        self.chroma_client = chroma_client
        self.llm_manager = llm_manager
        self.retriever = Retriever(chroma_client)
        self.response_generator = ResponseGenerator(self.retriever, llm_manager)
        
        # Load evaluation models
        self._initialize_evaluation_models()

    def _initialize_evaluation_models(self):
        """Initialize evaluation models with proper error handling"""
        try:
            # Initialize sentence transformers for semantic similarity
            self.semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            log.info("✅ Semantic similarity model loaded")
            
            # Try to initialize NLTK for BLEU scores
            try:
                import nltk
                # Download required NLTK data if not available
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    log.info("Downloading NLTK punkt tokenizer...")
                    nltk.download('punkt', quiet=True)
                
                try:
                    nltk.data.find('tokenizers/punkt_tab')
                except LookupError:
                    log.info("Downloading NLTK punkt_tab tokenizer...")
                    nltk.download('punkt_tab', quiet=True)
                    
                self.nltk_available = True
                log.info("✅ NLTK initialized for BLEU scoring")
            except Exception as e:
                log.warning(f"NLTK not available for BLEU scoring: {e}")
                self.nltk_available = False
            
            self.models_initialized = True
            
        except Exception as e:
            log.error(f"Failed to initialize evaluation models: {e}")
            self.models_initialized = False
            self.nltk_available = False

    def evaluate_single_query(self, query: str, expected_answer: str = None, 
                            category: str = None, method: str = "auto") -> EvaluationResult:
        """Evaluate a single query against expected answer"""
        
        start_time = time.time()
        
        try:
            # Generate response
            if method == "retrieval_only":
                # Force retrieval-only approach
                original_groq_status = self.llm_manager.groq_loaded
                self.llm_manager.groq_loaded = False
                response = self.response_generator.generate_response(query, max_tokens=300)
                self.llm_manager.groq_loaded = original_groq_status
                
            elif method == "llm_only":
                # Force LLM-only approach with minimal context
                response = self.response_generator._generate_fallback_response(query, 300)
                
            else:
                # Standard auto approach
                response = self.response_generator.generate_response(query, max_tokens=300)
                
            response_time = time.time() - start_time
            
            # Extract response details
            actual_answer = response.get("answer", "")
            confidence = response.get("confidence", 0.0)
            method_used = response.get("metadata", {}).get("generation_method", "unknown")
            model_used = response.get("metadata", {}).get("model_used", "unknown")
            
            # Calculate evaluation metrics
            semantic_similarity = self._calculate_semantic_similarity(actual_answer, expected_answer) if expected_answer else 0.0
            relevance_score = self._calculate_relevance_score(query, actual_answer)
            retrieval_accuracy = self._calculate_retrieval_accuracy(query, response.get("source_documents", []))
            bleu_score = self._calculate_bleu_score(expected_answer, actual_answer) if expected_answer else 0.0
            
            return EvaluationResult(
                query=query,
                expected_answer=expected_answer or "N/A",
                actual_answer=actual_answer,
                method=method_used,
                confidence=confidence,
                response_time=response_time,
                semantic_similarity=semantic_similarity,
                relevance_score=relevance_score,
                retrieval_accuracy=retrieval_accuracy,
                bleu_score=bleu_score,
                model_used=model_used
            )
            
        except Exception as e:
            log.error(f"Evaluation failed for query '{query}': {e}")
            return EvaluationResult(
                query=query,
                expected_answer=expected_answer or "N/A",
                actual_answer=f"Error: {str(e)}",
                method="error",
                confidence=0.0,
                response_time=time.time() - start_time,
                semantic_similarity=0.0,
                relevance_score=0.0,
                retrieval_accuracy=0.0,
                bleu_score=0.0,
                model_used="none"
            )

    def evaluate_test_set(self, test_queries: List[Dict[str, str]], 
                         methods: List[str] = ["auto", "retrieval_only", "llm_only"]) -> Dict[str, List[EvaluationResult]]:
        """Evaluate multiple queries across different methods"""
        
        results = {method: [] for method in methods}
        
        log.info(f"Starting evaluation of {len(test_queries)} queries across {len(methods)} methods")
        
        for i, test_case in enumerate(test_queries):
            query = test_case.get("query", "")
            expected = test_case.get("expected_answer", "")
            category = test_case.get("category", "general")
            
            log.info(f"Evaluating query {i+1}/{len(test_queries)}: {query[:50]}...")
            
            for method in methods:
                try:
                    result = self.evaluate_single_query(query, expected, category, method)
                    results[method].append(result)
                    
                    log.info(f"  {method}: confidence={result.confidence:.2f}, "
                           f"similarity={result.semantic_similarity:.2f}, "
                           f"time={result.response_time:.2f}s")
                    
                except Exception as e:
                    log.error(f"Failed to evaluate {method} for query {i+1}: {e}")
                    
        return results

    def generate_evaluation_summary(self, results: Dict[str, List[EvaluationResult]]) -> Dict[str, EvaluationSummary]:
        """Generate comprehensive summary of evaluation results"""
        
        summaries = {}
        
        for method, method_results in results.items():
            if not method_results:
                continue
                
            # Calculate averages
            avg_similarity = np.mean([r.semantic_similarity for r in method_results])
            avg_relevance = np.mean([r.relevance_score for r in method_results])
            avg_response_time = np.mean([r.response_time for r in method_results])
            avg_confidence = np.mean([r.confidence for r in method_results])
            
            # Method breakdown
            method_breakdown = {}
            for result in method_results:
                method_breakdown[result.method] = method_breakdown.get(result.method, 0) + 1
                
            # Model breakdown
            model_breakdown = {}
            for result in method_results:
                model_breakdown[result.model_used] = model_breakdown.get(result.model_used, 0) + 1
            
            summaries[method] = EvaluationSummary(
                total_queries=len(method_results),
                avg_semantic_similarity=avg_similarity,
                avg_relevance_score=avg_relevance,
                avg_response_time=avg_response_time,
                avg_confidence=avg_confidence,
                method_breakdown=method_breakdown,
                model_breakdown=model_breakdown,
                accuracy_by_category={}  # Could be extended with category analysis
            )
            
        return summaries

    def _calculate_semantic_similarity(self, answer1: str, answer2: str) -> float:
        """Calculate semantic similarity between two answers"""
        if not self.models_initialized or not answer1 or not answer2:
            return 0.0
            
        try:
            # Encode both answers
            embeddings = self.semantic_model.encode([answer1, answer2])
            
            # Calculate cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return float(similarity)
            
        except Exception as e:
            log.error(f"Semantic similarity calculation failed: {e}")
            return 0.0

    def _calculate_relevance_score(self, query: str, answer: str) -> float:
        """Calculate how relevant the answer is to the query"""
        if not self.models_initialized or not query or not answer:
            return 0.0
            
        try:
            # Calculate similarity between query and answer
            similarity = self._calculate_semantic_similarity(query, answer)
            
            # Bonus for answer length (not too short, not too long)
            length_score = 1.0
            if len(answer) < 50:
                length_score = 0.5  # Too short
            elif len(answer) > 1000:
                length_score = 0.8  # Might be too verbose
                
            # Bonus for Jupiter-specific content
            jupiter_keywords = ["jupiter", "app", "upi", "card", "account", "payment"]
            jupiter_score = 1.0
            if any(keyword in answer.lower() for keyword in jupiter_keywords):
                jupiter_score = 1.2
                
            return min(similarity * length_score * jupiter_score, 1.0)
            
        except Exception as e:
            log.error(f"Relevance score calculation failed: {e}")
            return 0.0

    def _calculate_retrieval_accuracy(self, query: str, source_documents: List[Dict]) -> float:
        """Calculate accuracy of document retrieval"""
        if not source_documents:
            return 0.0
            
        try:
            # Simple relevance check based on similarity scores
            similarities = [doc.get("similarity", 0.0) for doc in source_documents]
            
            if not similarities:
                return 0.0
                
            # High accuracy if top result has good similarity
            top_similarity = max(similarities)
            
            if top_similarity > 0.8:
                return 1.0
            elif top_similarity > 0.6:
                return 0.8
            elif top_similarity > 0.4:
                return 0.6
            else:
                return 0.3
                
        except Exception as e:
            log.error(f"Retrieval accuracy calculation failed: {e}")
            return 0.0

    def _calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score between reference and candidate text"""
        if not self.nltk_available or not reference or not candidate:
            return 0.0
            
        try:
            from nltk.translate.bleu_score import sentence_bleu
            from nltk.tokenize import word_tokenize
            
            # Tokenize the sentences
            reference_tokens = [word_tokenize(reference.lower())]
            candidate_tokens = word_tokenize(candidate.lower())
            
            # Calculate BLEU score
            bleu_score = sentence_bleu(reference_tokens, candidate_tokens)
            return bleu_score
            
        except Exception as e:
            log.error(f"BLEU score calculation failed: {e}")
            return 0.0

    def create_test_dataset(self) -> List[Dict[str, str]]:
        """Create a test dataset from existing knowledge base"""
        test_cases = [
            {
                "query": "How do I set up UPI in Jupiter?",
                "expected_answer": "To set up UPI in Jupiter, open the app, go to Payments section, select UPI, and follow the setup process with your mobile number and create a UPI PIN.",
                "category": "payments"
            },
            {
                "query": "How to increase debit card limit?",
                "expected_answer": "You can increase your debit card limit by going to Cards section in Jupiter app, selecting your card, and adjusting the spending limits based on your needs.",
                "category": "cards"
            },
            {
                "query": "What documents are needed for KYC?",
                "expected_answer": "For KYC verification, you need a valid government ID (Aadhaar, PAN card), address proof, and a recent photograph. Video KYC can also be completed through the app.",
                "category": "kyc"
            },
            {
                "query": "How to invest in gold through Jupiter?",
                "expected_answer": "Jupiter offers digital gold investment. You can buy gold through the Investments section in the app, starting with small amounts and track your gold holdings.",
                "category": "investments"
            },
            {
                "query": "How to contact Jupiter customer support?",
                "expected_answer": "You can contact Jupiter support through the Help section in the app, live chat, email support, or phone support during business hours.",
                "category": "technical"
            },
            {
                "query": "Jupiter app login kaise kare?",
                "expected_answer": "Jupiter app mein login karne ke liye apna registered mobile number aur MPIN/biometric use kare. App download karne ke baad setup process follow kare.",
                "category": "technical"
            },
            {
                "query": "UPI payment limits kya hai?",
                "expected_answer": "UPI payment limits usually range from ₹1 lakh per day for individual transactions. Specific limits depend on your bank and account type.",
                "category": "payments"
            },
            {
                "query": "How to redeem Jupiter rewards?",
                "expected_answer": "Jupiter rewards can be redeemed through the Rewards section in the app. You can use points for cashback, bill payments, or other offers available.",
                "category": "rewards"
            }
        ]
        
        return test_cases

    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run a comprehensive evaluation of the system"""
        log.info("Starting comprehensive evaluation of Jupiter FAQ Bot")
        
        # Create test dataset
        test_dataset = self.create_test_dataset()
        
        # Evaluate across different methods
        methods = ["auto", "retrieval_only", "llm_only"]
        results = self.evaluate_test_set(test_dataset, methods)
        
        # Generate summaries
        summaries = self.generate_evaluation_summary(results)
        
        # Create comparison report
        report = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "test_dataset_size": len(test_dataset),
            "methods_evaluated": methods,
            "detailed_results": results,
            "summaries": summaries,
            "recommendations": self._generate_recommendations(summaries)
        }
        
        log.info("Comprehensive evaluation completed")
        return report

    def _generate_recommendations(self, summaries: Dict[str, EvaluationSummary]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        if not summaries:
            return ["No evaluation data available for recommendations"]
        
        # Compare methods
        method_performances = {}
        for method, summary in summaries.items():
            score = (summary.avg_semantic_similarity + summary.avg_relevance_score + summary.avg_confidence) / 3
            method_performances[method] = score
            
        best_method = max(method_performances.keys(), key=lambda x: method_performances[x])
        worst_method = min(method_performances.keys(), key=lambda x: method_performances[x])
        
        recommendations.append(f"🏆 Best performing method: {best_method} (score: {method_performances[best_method]:.3f})")
        
        # Speed recommendations
        fastest_method = min(summaries.keys(), key=lambda x: summaries[x].avg_response_time)
        recommendations.append(f"⚡ Fastest method: {fastest_method} ({summaries[fastest_method].avg_response_time:.2f}s avg)")
        
        # Quality recommendations
        if summaries.get("auto", None):
            auto_summary = summaries["auto"]
            if auto_summary.avg_confidence < 0.7:
                recommendations.append("⚠️ Consider improving retrieval quality - low confidence scores detected")
            if auto_summary.avg_response_time > 3.0:
                recommendations.append("⚠️ Consider optimizing response time - currently above 3 seconds")
                
        # Model recommendations
        for method, summary in summaries.items():
            if "groq" in str(summary.model_breakdown):
                recommendations.append(f"✅ {method} successfully uses Groq models for enhanced responses")
                
        return recommendations 