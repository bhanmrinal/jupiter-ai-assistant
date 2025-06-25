#!/usr/bin/env python3
"""
Test script specifically for Response Generator performance and functionality
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.data_models import LanguageEnum
from src.models.llm_manager import LLMManager
from src.models.response_generator import ResponseGenerator
from src.models.retriever import Retriever
from src.utils.logger import get_logger

log = get_logger(__name__)


def test_llm_manager():
    """Test LLM Manager initialization and model loading"""
    print("🤖 Testing LLM Manager...")
    
    try:
        start_time = time.time()
        llm_manager = LLMManager()
        init_time = time.time() - start_time
        print(f"   ✅ LLM Manager initialized in {init_time:.2f}s")
        
        # Test hybrid mode (automatic model selection)
        print("   📋 Hybrid mode: Automatic model selection based on query complexity")
        
        # Test simple generation with timeout
        print("   🔄 Testing simple generation...")
        start_time = time.time()
        
        try:
            response, confidence = llm_manager.generate_response(
                prompt="Answer in one word: What is 2+2?",
                max_tokens=10,
                temperature=0.1
            )
            gen_time = time.time() - start_time
            print(f"   ✅ Simple generation completed in {gen_time:.2f}s")
            print(f"   📝 Response: '{response}' (confidence: {confidence:.2f})")
            
            if gen_time > 30:
                print(f"   ⚠️ Generation took {gen_time:.2f}s - this is too slow!")
                
        except Exception as e:
            gen_time = time.time() - start_time
            print(f"   ❌ Generation failed after {gen_time:.2f}s: {e}")
            return False
            
    except Exception as e:
        print(f"   ❌ LLM Manager initialization failed: {e}")
        return False
        
    return True


def test_retriever():
    """Test Retriever functionality"""
    print("\n🔍 Testing Retriever...")
    
    try:
        retriever = Retriever()
        print("   ✅ Retriever initialized")
        
        # Test retrieval
        start_time = time.time()
        result = retriever.retrieve("PIN reset", language=LanguageEnum.ENGLISH)
        retrieval_time = time.time() - start_time
        
        print(f"   🔍 Retrieval completed in {retrieval_time:.2f}s")
        print(f"   📊 Found {result.total_found} documents")
        print(f"   📝 Context length: {len(result.context_text)} chars")
        
        if result.total_found > 0:
            print(f"   📄 First document: {result.documents[0].question[:50]}...")
        
    except Exception as e:
        print(f"   ❌ Retriever test failed: {e}")
        return False
        
    return True


def test_response_generation_with_timeout():
    """Test Response Generator with timeout monitoring"""
    print("\n🎯 Testing Response Generator with timeout monitoring...")
    
    try:
        # Initialize components separately to isolate issues
        print("   🔄 Initializing LLM Manager...")
        llm_manager = LLMManager()
        
        print("   🔄 Initializing Retriever...")
        retriever = Retriever()
        
        print("   🔄 Initializing Response Generator...")
        response_generator = ResponseGenerator(llm_manager=llm_manager, retriever=retriever)
        
        print("   ✅ All components initialized")
        
        # Test queries with different complexity
        test_queries = [
            ("Hi", "Simple greeting"),
            ("PIN", "Short query"),
            ("How to reset PIN?", "Standard question"),
        ]
        
        for query, description in test_queries:
            print(f"\n   🧪 Testing: {description} - '{query}'")
            
            start_time = time.time()
            
            try:
                # Set a reasonable timeout for testing
                result = response_generator.generate_response(
                    query=query,
                    language=LanguageEnum.ENGLISH
                )
                
                end_time = time.time()
                total_time = end_time - start_time
                
                print(f"   ✅ Response generated in {total_time:.2f}s")
                print(f"   📝 Response: '{result.response[:100]}...'")
                print(f"   📊 Confidence: {result.confidence_score:.2f}")
                print(f"   🔍 Retrieved docs: {result.retrieved_docs_count}")
                print(f"   ⚡ Generation time: {result.generation_time_ms}ms")
                print(f"   🔍 Retrieval time: {result.retrieval_time_ms}ms")
                
                if total_time > 60:
                    print(f"   ⚠️ Response took {total_time:.2f}s - too slow!")
                    
            except Exception as e:
                end_time = time.time()
                total_time = end_time - start_time
                print(f"   ❌ Response generation failed after {total_time:.2f}s: {e}")
                return False
                
    except Exception as e:
        print(f"   ❌ Response Generator setup failed: {e}")
        return False
        
    return True


def test_response_generation_components():
    """Test individual components of response generation"""
    print("\n🔧 Testing Response Generation Components...")
    
    try:
        response_generator = ResponseGenerator()
        
        # Test health check
        print("   🏥 Testing health check...")
        is_healthy = response_generator.health_check()
        print(f"   {'✅' if is_healthy else '❌'} Health check: {is_healthy}")
        
        if not is_healthy:
            print("   ⚠️ System is not healthy - investigating...")
            
        # Test individual methods
        print("   🧪 Testing individual methods...")
        
        # Test without context
        start_time = time.time()
        response, confidence = response_generator._generate_without_context(
            query="Hello",
            language=LanguageEnum.ENGLISH
        )
        no_context_time = time.time() - start_time
        
        print(f"   ✅ No-context generation: {no_context_time:.2f}s")
        print(f"   📝 Response: '{response[:50]}...'")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Component test failed: {e}")
        return False


def monitor_system_resources():
    """Monitor system resources during testing"""
    print("\n💻 System Resource Check...")
    
    try:
        import psutil
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"   🖥️ CPU Usage: {cpu_percent}%")
        
        # Memory usage
        memory = psutil.virtual_memory()
        print(f"   🧠 Memory Usage: {memory.percent}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")
        
        # Check if system is under heavy load
        if cpu_percent > 80:
            print("   ⚠️ High CPU usage detected!")
        if memory.percent > 80:
            print("   ⚠️ High memory usage detected!")
            
    except ImportError:
        print("   ℹ️ psutil not available - skipping resource monitoring")
    except Exception as e:
        print(f"   ⚠️ Resource monitoring failed: {e}")


def main():
    """Run all response generator tests"""
    print("=" * 70)
    print("🧪 JUPITER FAQ BOT - RESPONSE GENERATOR DIAGNOSTIC TESTS")
    print("=" * 70)
    
    # Monitor system resources first
    monitor_system_resources()
    
    tests = [
        test_llm_manager,
        test_retriever,
        test_response_generation_components,
        test_response_generation_with_timeout,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\n{'='*50}")
            if test():
                passed += 1
                print(f"✅ {test.__name__} PASSED")
            else:
                failed += 1
                print(f"❌ {test.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} FAILED with exception: {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 DIAGNOSTIC RESULTS: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("🔧 Issues detected in response generation system!")
        print("💡 Recommendations:")
        print("   - Check model loading and initialization")
        print("   - Verify LLM inference performance")
        print("   - Monitor system resources during generation")
    else:
        print("🎉 All response generator tests passed!")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main()) 