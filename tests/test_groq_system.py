#!/usr/bin/env python3
"""
Test script for Jupiter FAQ Bot with Groq API integration
Production-ready testing without hardcoded patterns.
"""

import os
import time
from datetime import datetime


def test_groq_system():
    """Test the complete Groq-based system"""

    print("🚀 Testing Jupiter FAQ Bot with Groq API")
    print("=" * 50)

    try:
        # Check Groq API key
        if not os.getenv("GROQ_API_KEY"):
            print("⚠️  Warning: GROQ_API_KEY not set. Some features may not work.")
            print("   Set it with: export GROQ_API_KEY=your_key_here")
        else:
            print("✅ Groq API key configured")

        # Import components
        print("\n📦 Importing components...")
        from src.database.chroma_client import ChromaClient
        from src.models.llm_manager import LLMManager
        from src.models.response_generator import ResponseGenerator
        from src.models.retriever import Retriever

        # Initialize ChromaDB
        print("🗄️  Initializing ChromaDB...")
        chroma_client = ChromaClient()

        # Initialize LLM Manager
        print("🤖 Initializing LLM Manager...")
        llm_manager = LLMManager(enable_conversation=True)

        # Test model loading
        print("🔍 Testing models...")
        model_info = llm_manager.get_model_info()
        print(f"   DistilBERT loaded: {model_info['models']['distilbert_qa']['loaded']}")
        print(f"   Groq loaded: {model_info['models']['groq_conversation']['loaded']}")

        # Initialize components
        print("🔗 Initializing retriever and response generator...")
        retriever = Retriever(chroma_client)
        response_generator = ResponseGenerator(retriever, llm_manager)

        # Test health checks
        print("🏥 Running health checks...")
        print(f"   LLM Manager: {'✅' if llm_manager.health_check() else '❌'}")
        print(f"   Retriever: {'✅' if retriever.health_check() else '❌'}")
        print(f"   Response Generator: {'✅' if response_generator.health_check() else '❌'}")

        # Test queries
        test_queries = [
            "How to reset my PIN?",
            "UPI payment not working",
            "Check account balance",
            "Investment options",
        ]

        print("\n🧪 Testing response generation...")
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Testing: '{query}'")
            start_time = time.time()

            try:
                result = response_generator.generate_response(query, max_tokens=100)
                elapsed = time.time() - start_time

                print(f"   ⏱️  Response time: {elapsed:.2f}s")
                print(f"   🎯 Confidence: {result.get('confidence', 0):.1%}")
                print(
                    f"   📊 Method: {result.get('metadata', {}).get('generation_method', 'unknown')}"
                )
                print(f"   📚 Documents: {result.get('metadata', {}).get('documents_found', 0)}")

                answer = result.get("answer", "No response")[:100]
                print(f"   💬 Answer: {answer}{'...' if len(answer) == 100 else ''}")

                if elapsed < 2.0:
                    print("   ✅ Performance: Good")
                elif elapsed < 5.0:
                    print("   ⚠️  Performance: Acceptable")
                else:
                    print("   ❌ Performance: Slow")

            except Exception as e:
                print(f"   ❌ Error: {e}")

        # Test statistics
        print("\n📊 System Statistics:")
        try:
            generation_stats = response_generator.get_generation_stats()
            print(f"   Version: {generation_stats.get('generator_version', 'unknown')}")
            print(f"   Features: {', '.join(generation_stats.get('features', []))}")
            print(f"   Model Priority: {' → '.join(generation_stats.get('model_priority', []))}")

            retrieval_stats = retriever.get_retrieval_stats()
            print(f"   Retriever: {retrieval_stats.get('retriever_version', 'unknown')}")

        except Exception as e:
            print(f"   ⚠️  Could not get stats: {e}")

        print("\n🎉 System test completed!")
        return True

    except Exception as e:
        print(f"\n❌ System test failed: {e}")
        return False


def test_individual_components():
    """Test individual components separately"""

    print("\n🔧 Testing Individual Components")
    print("=" * 50)

    try:
        # Test LLM Manager only
        print("1. Testing LLM Manager...")
        from src.models.llm_manager import LLMManager

        llm = LLMManager(enable_conversation=True)

        # Test DistilBERT
        if llm.fast_extractor_loaded:
            answer, confidence = llm.extract_answer(
                "How to reset PIN?",
                "To reset your PIN, go to card settings and select reset PIN option.",
            )
            print(f"   DistilBERT test: {answer[:50]}... (confidence: {confidence:.2f})")

        # Test Groq
        if llm.groq_loaded:
            response, confidence = llm.generate_conversation(
                "You are a helpful assistant.", "Say hello", max_tokens=20
            )
            print(f"   Groq test: {response[:50]}... (confidence: {confidence:.2f})")

        print("   ✅ LLM Manager test passed")

    except Exception as e:
        print(f"   ❌ LLM Manager test failed: {e}")


if __name__ == "__main__":
    print(f"🕐 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Test complete system
    success = test_groq_system()

    # Test individual components if main test fails
    if not success:
        test_individual_components()

    print(f"\n🕐 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
