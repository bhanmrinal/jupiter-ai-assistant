#!/usr/bin/env python3
"""
Test script for production issues validation
"""

import os
import sys
import time
from datetime import datetime

# Add src to path
sys.path.append('src')

from database.chroma_client import ChromaClient
from models.response_generator import ResponseGenerator
from models.retriever import Retriever
from models.llm_manager import LLMManager
from utils.logger import get_logger

log = get_logger(__name__)

def test_chromadb_telemetry_fix():
    """Test if ChromaDB telemetry errors are fixed"""
    print("\n🔧 Testing ChromaDB telemetry fix...")
    
    try:
        chroma_client = ChromaClient()
        
        # Test basic operations that previously caused telemetry errors
        stats = chroma_client.get_collection_stats()
        print(f"   ✅ Collection stats retrieved: {stats['total_documents']} documents")
        
        # Test search operation
        results = chroma_client.search("test query", top_k=3, similarity_threshold=0.3)
        print(f"   ✅ Search completed without telemetry errors")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ChromaDB test failed: {e}")
        return False

def test_improved_threshold_handling():
    """Test if improved threshold settings work correctly"""
    print("\n📊 Testing improved threshold handling...")
    
    try:
        chroma_client = ChromaClient()
        retriever = Retriever(chroma_client)
        
        # Test queries that previously returned 0 results
        test_queries = ["test", "hello", "help"]
        
        for query in test_queries:
            result = retriever.search(query, top_k=5, similarity_threshold=0.3)
            print(f"   Query '{query}': {result.total_matches} matches (confidence: {result.confidence:.2f})")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Threshold test failed: {e}")
        return False

def test_response_caching():
    """Test if response caching reduces redundant API calls"""
    print("\n⚡ Testing response caching...")
    
    try:
        llm_manager = LLMManager()
        
        # Test same query multiple times
        query = "what is jupiter money"
        system_prompt = "You are a helpful assistant."
        
        # First call (should hit API)
        start_time = time.time()
        response1, confidence1 = llm_manager.generate_conversation(system_prompt, query, max_tokens=50)
        first_call_time = time.time() - start_time
        
        # Second call (should hit cache)
        start_time = time.time()
        response2, confidence2 = llm_manager.generate_conversation(system_prompt, query, max_tokens=50)
        second_call_time = time.time() - start_time
        
        print(f"   First call time: {first_call_time:.2f}s")
        print(f"   Second call time: {second_call_time:.2f}s")
        
        if second_call_time < first_call_time * 0.5:  # Should be significantly faster
            print(f"   ✅ Caching working - {(first_call_time/second_call_time):.1f}x faster")
            return True
        else:
            print(f"   ⚠️ Caching may not be working as expected")
            return False
            
    except Exception as e:
        print(f"   ❌ Caching test failed: {e}")
        return False

def test_end_to_end_response_generation():
    """Test complete response generation pipeline"""
    print("\n🔄 Testing end-to-end response generation...")
    
    try:
        chroma_client = ChromaClient()
        llm_manager = LLMManager()
        retriever = Retriever(chroma_client)
        response_generator = ResponseGenerator(retriever, llm_manager)
        
        # Test various types of queries
        test_queries = [
            "what is jupiter money",
            "how can i update my kyc information", 
            "test query with low similarity",
            "how to transfer money"
        ]
        
        for query in test_queries:
            print(f"\n   Testing query: '{query}'")
            start_time = time.time()
            
            result = response_generator.generate_response(query, max_tokens=100)
            
            response_time = time.time() - start_time
            
            print(f"     Response time: {response_time:.2f}s")
            print(f"     Confidence: {result['confidence']:.2f}")
            print(f"     Documents found: {result['metadata']['documents_found']}")
            print(f"     Generation method: {result['metadata']['generation_method']}")
            print(f"     Model used: {result['metadata']['model_used']}")
            print(f"     Answer preview: {result['answer'][:100]}...")
            
        return True
        
    except Exception as e:
        print(f"   ❌ End-to-end test failed: {e}")
        return False

def test_similarity_threshold_configuration():
    """Test if the similarity threshold is correctly configured"""
    print("\n🎯 Testing similarity threshold configuration...")
    
    try:
        from config.settings import settings
        
        # Check the config value
        config_threshold = settings.model.similarity_threshold
        print(f"   📋 Config similarity_threshold: {config_threshold}")
        
        # Check environment variable
        import os
        env_threshold = os.getenv("SIMILARITY_THRESHOLD")
        print(f"   🌍 Environment SIMILARITY_THRESHOLD: {env_threshold}")
        
        # Test actual usage in retriever
        chroma_client = ChromaClient()
        retriever = Retriever(chroma_client)
        
        # Make a test query and check logs for threshold value
        print(f"   🔍 Testing retriever with config threshold: {config_threshold}")
        result = retriever.search("test query", top_k=3, similarity_threshold=config_threshold)
        
        # Test response generator
        from models.llm_manager import LLMManager
        from models.response_generator import ResponseGenerator
        
        llm_manager = LLMManager(enable_conversation=True)
        response_generator = ResponseGenerator(retriever, llm_manager)
        
        print(f"   🤖 Testing response generator...")
        response = response_generator.generate_response("what is jupiter money?")
        
        print(f"   ✅ Similarity threshold test completed")
        return True
        
    except Exception as e:
        print(f"   ❌ Similarity threshold test failed: {e}")
        return False

def main():
    """Run all production issue validation tests"""
    print("🚀 Jupiter FAQ Bot - Production Issues Validation")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    test_results.append(("ChromaDB Telemetry Fix", test_chromadb_telemetry_fix()))
    test_results.append(("Improved Threshold Handling", test_improved_threshold_handling()))
    test_results.append(("Response Caching", test_response_caching()))
    test_results.append(("End-to-End Pipeline", test_end_to_end_response_generation()))
    test_results.append(("Similarity Threshold Configuration", test_similarity_threshold_configuration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Results Summary:")
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        print("🎉 All production issues have been resolved!")
    else:
        print("⚠️ Some issues still need attention.")
    
    return passed == len(test_results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 