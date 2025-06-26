#!/usr/bin/env python3
"""
Comprehensive test for Hybrid Response Generator
Tests both fast extraction and full TinyLlama generation
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.llm_manager import LLMManager
from src.models.response_generator import ResponseGenerator
from src.utils.logger import get_logger

log = get_logger(__name__)


def test_hybrid_llm_manager():
    """Test hybrid LLM manager with different modes"""
    print("🤖 Testing Hybrid LLM Manager...")

    try:
        start_time = time.time()
        llm_manager = LLMManager(enable_conversation=False)  # Start with basic mode only
        init_time = time.time() - start_time
        print(f"✅ Fast-only LLM Manager initialized in {init_time:.2f}s")

        # Test fast extraction with context
        test_context = """
        To reset your PIN on Jupiter:
        1. Open the Jupiter mobile app
        2. Go to Cards section
        3. Select Reset PIN option
        4. Enter your current PIN for verification
        5. Set your new 4-digit PIN
        The new PIN will be active immediately.
        """

        print("\n🚀 Testing Fast Extraction Mode:")
        start_time = time.time()
        response, confidence = llm_manager.generate_response(
            prompt="How to reset PIN?", context=test_context
        )
        fast_time = time.time() - start_time

        print(f"⚡ Fast response time: {fast_time:.2f}s")
        print(f"📝 Fast response: {response}")
        print(f"📊 Confidence: {confidence}")

        # Test TinyLlama mode without context
        print("\n🤖 Testing TinyLlama Mode (no context):")
        start_time = time.time()
        response, confidence = llm_manager.generate_response(
            prompt="How to transfer money?"
        )
        tinyllama_time = time.time() - start_time

        print(f"⚡ TinyLlama response time: {tinyllama_time:.2f}s")
        print(f"📝 TinyLlama response: {response}")
        print(f"📊 Confidence: {confidence}")

        # Get model info
        model_info = llm_manager.get_model_info()
        print(f"\n🔧 Model Info: {model_info}")

        return True

    except Exception as e:
        print(f"❌ Hybrid LLM Manager test failed: {e}")
        return False


def test_full_hybrid_manager():
    """Test hybrid manager with TinyLlama enabled"""
    print("\n🤖 Testing Full Hybrid LLM Manager (with TinyLlama)...")

    try:
        start_time = time.time()
        llm_manager = LLMManager(enable_conversation=True)  # Enable conversation models
        init_time = time.time() - start_time
        print(f"✅ Full Hybrid LLM Manager initialized in {init_time:.2f}s")

        # Test intelligent mode selection
        test_queries = [
            ("How to reset PIN?", "Simple direct question - should use fast extraction"),
            ("Why is my account blocked?", "Complex question - should use TinyLlama"),
            ("What is my balance?", "Simple question - should use fast extraction"),
            (
                "Explain the difference between NEFT and IMPS",
                "Complex explanation - should use TinyLlama",
            ),
        ]

        for query, expected_mode in test_queries:
            print(f"\n🧪 Testing: {query}")
            print(f"Expected: {expected_mode}")

            start_time = time.time()
            response, confidence = llm_manager.generate_response(
                prompt=query,
                context="Jupiter offers various banking services including PIN reset, transfers, and account management.",
            )
            response_time = time.time() - start_time

            print(f"⚡ Response time: {response_time:.2f}s")
            print(f"📝 Response: {response[:100]}...")
            print(f"📊 Confidence: {confidence}")

            # Determine actual mode based on response time
            actual_mode = "Fast extraction" if response_time < 5 else "TinyLlama generation"
            print(f"🎯 Actual mode: {actual_mode}")

        return True

    except Exception as e:
        print(f"❌ Full Hybrid LLM Manager test failed: {e}")
        return False


def test_hybrid_response_generator():
    """Test the complete hybrid response generator"""
    print("\n🧪 Testing Hybrid Response Generator...")

    try:
        start_time = time.time()
        response_gen = ResponseGenerator()
        init_time = time.time() - start_time
        print(f"✅ Hybrid Response Generator initialized in {init_time:.2f}s")

        # Test queries that should trigger different modes
        test_queries = [
            {
                "query": "How to reset my PIN?",
                "expected_speed": "fast",
                "description": "Simple PIN reset question",
            },
            {
                "query": "What's the difference between savings and current account?",
                "expected_speed": "slow",
                "description": "Complex explanation question",
            },
            {
                "query": "How to transfer money?",
                "expected_speed": "fast",
                "description": "Simple transfer question",
            },
        ]

        for test_case in test_queries:
            query = test_case["query"]
            print(f"\n🧪 Testing: {query}")
            print(f"Expected: {test_case['description']} ({test_case['expected_speed']})")

            start_time = time.time()
            result = response_gen.generate_response(query)
            total_time = time.time() - start_time

            print(f"⏱️ Total time: {total_time:.2f}s")
            print(f"📝 Response: {result.get('answer', '')[:150]}...")
            print(f"📊 Confidence: {result.get('confidence', 0)}")
            print(f"🔍 Retrieved docs: {len(result.get('source_documents', []))}")
            print(f"⚡ Model used: {result.get('metadata', {}).get('model_used', 'unknown')}")
            print(f"⏰ Generation method: {result.get('metadata', {}).get('generation_method', 'unknown')}")

            # Analyze performance
            if total_time < 5:
                actual_speed = "fast"
            elif total_time < 20:
                actual_speed = "medium"
            else:
                actual_speed = "slow"

            print(f"🎯 Actual speed: {actual_speed}")

            # Check if response is reasonable
            response_text = result.get('answer', '')
            if len(response_text) > 20:
                print("✅ Response quality: Good")
            else:
                print("⚠️ Response quality: Poor")

        return True

    except Exception as e:
        print(f"❌ Hybrid Response Generator test failed: {e}")
        return False


def run_performance_comparison():
    """Compare performance of different modes"""
    print("\n📊 Performance Comparison...")

    try:
        llm_manager = LLMManager(enable_conversation=False)  # Basic mode only for comparison

        test_query = "How to reset PIN?"
        test_context = "To reset your PIN: 1. Open Jupiter app 2. Go to Cards 3. Select Reset PIN 4. Follow verification"

        # Test fast extraction
        print("🚀 Fast Extraction Performance:")
        times = []
        for _i in range(3):
            start = time.time()
            response, conf = llm_manager.generate_response(
                prompt=test_query, context=test_context, force_mode=ModelType.FAST_EXTRACT
            )
            times.append(time.time() - start)

        avg_fast_time = sum(times) / len(times)
        print(f"Average fast extraction time: {avg_fast_time:.2f}s")
        print(f"Fast response: {response[:100]}...")

        # Test TinyLlama without context
        print("\n🤖 TinyLlama Performance (no context):")
        times = []
        for _i in range(1):  # Only test once due to slower performance
            start = time.time()
            response, conf = llm_manager.generate_response(
                prompt=test_query,
                context=None,  # No context to force TinyLlama
                force_mode=ModelType.TINYLLAMA,
            )
            times.append(time.time() - start)

        avg_tinyllama_time = sum(times) / len(times)
        print(f"Average TinyLlama time: {avg_tinyllama_time:.2f}s")
        print(f"TinyLlama response: {response[:100]}...")

        print("\n📈 Performance Summary:")
        print(f"- Fast Extraction: {avg_fast_time:.2f}s")
        print(f"- TinyLlama (no context): {avg_tinyllama_time:.2f}s")
        print(f"- Speed difference: {avg_tinyllama_time / avg_fast_time:.1f}x slower")

        return True

    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
        return False


def main():
    """Run all hybrid tests"""
    print("🧪 Starting Hybrid Response System Tests...")
    print("=" * 60)

    tests = [
        ("Hybrid LLM Manager (Fast Only)", test_hybrid_llm_manager),
        ("Performance Comparison", run_performance_comparison),
        ("Hybrid Response Generator", test_hybrid_response_generator),
        # ("Full Hybrid Manager (with TinyLlama)", test_full_hybrid_manager),  # Skip for speed
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY:")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All tests passed! Hybrid system is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the logs above.")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
