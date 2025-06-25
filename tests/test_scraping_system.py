#!/usr/bin/env python3
"""
Test script for Jupiter FAQ Bot scraping system
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_processing.processor import DataProcessor
from src.database.chroma_client import ChromaDBClient
from src.scraper.blog_scraper import BlogScraper
from src.scraper.community_scraper import CommunityScraper
from src.scraper.jupiter_scraper import JupiterScraper
from src.utils.logger import get_logger

log = get_logger(__name__)

def test_scrapers():
    """Test all scrapers individually"""
    print("🧪 Testing Scrapers...")
    
    # Test Jupiter Help Center Scraper
    print("1️⃣ Testing Help Center Scraper...")
    try:
        help_scraper = JupiterScraper()
        help_urls = help_scraper.get_all_urls()
        print(f"   ✅ Found {len(help_urls)} help center URLs")
        
        if help_urls:
            sample_url = help_urls[0]
            content = help_scraper.scrape_url(sample_url)
            if content:
                print(f"   ✅ Successfully scraped sample URL: {sample_url}")
                print(f"   📄 Title: {content.title[:50]}...")
                print(f"   📝 Content length: {len(content.content)} chars")
            else:
                print(f"   ⚠️ Failed to scrape content from {sample_url}")
    except Exception as e:
        print(f"   ❌ Help center scraper failed: {e}")
    
    # Test Community Scraper
    print("\n2️⃣ Testing Community Scraper...")
    try:
        community_scraper = CommunityScraper()
        community_urls = community_scraper.get_all_urls()
        print(f"   ✅ Found {len(community_urls)} community URLs")
        
        if community_urls:
            sample_url = community_urls[0]
            content = community_scraper.scrape_url(sample_url)
            if content:
                print(f"   ✅ Successfully scraped sample URL: {sample_url}")
                print(f"   📄 Title: {content.title[:50]}...")
                print(f"   📝 Content length: {len(content.content)} chars")
            else:
                print(f"   ⚠️ Failed to scrape content from {sample_url}")
    except Exception as e:
        print(f"   ❌ Community scraper failed: {e}")
    
    # Test Blog Scraper
    print("\n3️⃣ Testing Blog Scraper...")
    try:
        blog_scraper = BlogScraper()
        blog_urls = blog_scraper.get_all_urls()
        print(f"   ✅ Found {len(blog_urls)} blog URLs")
        
        if blog_urls:
            sample_url = blog_urls[0]
            content = blog_scraper.scrape_url(sample_url)
            if content:
                print(f"   ✅ Successfully scraped sample URL: {sample_url}")
                print(f"   📄 Title: {content.title[:50]}...")
                print(f"   📝 Content length: {len(content.content)} chars")
            else:
                print(f"   ⚠️ Failed to scrape content from {sample_url}")
    except Exception as e:
        print(f"   ❌ Blog scraper failed: {e}")

def test_data_processing():
    """Test data processing pipeline"""
    print("\n🔄 Testing Data Processing...")
    
    try:
        from src.database.data_models import ScrapedContent, SourceTypeEnum
        
        # Create sample scraped content
        sample_content = ScrapedContent(
            url="https://help.jupiter.money/test",
            title="How to reset your PIN?",
            content="To reset your PIN, follow these steps: 1. Open the Jupiter app 2. Go to Card settings 3. Select 'Reset PIN' 4. Follow the on-screen instructions",
            raw_html="<html>test</html>",
            source_type=SourceTypeEnum.HELP_CENTER
        )
        
        processor = DataProcessor()
        faq_documents = processor.process_scraped_content([sample_content])
        
        print(f"   ✅ Processed 1 scraped item into {len(faq_documents)} FAQ documents")
        
        if faq_documents:
            faq = faq_documents[0]
            print("   📄 Sample FAQ:")
            print(f"      Q: {faq.question}")
            print(f"      A: {faq.answer}")
            print(f"      Category: {faq.category}")
            print(f"      Language: {faq.language}")
    
    except Exception as e:
        print(f"   ❌ Data processing failed: {e}")

def test_chromadb():
    """Test ChromaDB integration"""
    print("\n💾 Testing ChromaDB...")
    
    try:
        chroma_client = ChromaDBClient()
        
        # Health check
        if chroma_client.health_check():
            print("   ✅ ChromaDB connection successful")
            
            # Get stats
            stats = chroma_client.get_collection_stats()
            print(f"   📊 Collection stats: {stats['total_documents']} documents")
            
            # Test search (if there are documents)
            if stats['total_documents'] > 0:
                results = chroma_client.search_similar("PIN reset", n_results=1)
                print(f"   🔍 Sample search found {results['total_found']} results")
            else:
                print("   ℹ️ No documents in collection yet")
        else:
            print("   ❌ ChromaDB health check failed")
    
    except Exception as e:
        print(f"   ❌ ChromaDB test failed: {e}")

def test_full_pipeline_small():
    """Test the full pipeline with a small dataset"""
    print("\n🚀 Testing Full Pipeline (Limited)...")
    
    try:
        # Import the main pipeline
        sys.path.append('scripts')
        from scripts.run_scraper import ScrapingPipeline
        
        pipeline = ScrapingPipeline()
        
        # Run with very limited scope for testing
        results = pipeline.run_full_pipeline(
            sources=["help"],  # Only help center
            limit=2           # Only 2 URLs
        )
        
        print("   ✅ Pipeline completed!")
        print(f"   📊 URLs scraped: {results['total_urls_scraped']}")
        print(f"   📋 FAQs generated: {results['total_faqs_generated']}")
        print(f"   💾 FAQs stored: {results['total_faqs_stored']}")
        
        if results['errors']:
            print(f"   ⚠️ Errors encountered: {len(results['errors'])}")
        
        # Test search
        if results['total_faqs_stored'] > 0:
            search_results = pipeline.test_chromadb_search("PIN")
            print(f"   🔍 Search test found {search_results.get('total_found', 0)} results")
    
    except Exception as e:
        print(f"   ❌ Full pipeline test failed: {e}")

def main():
    """Run all tests"""
    print("="*60)
    print("🧪 JUPITER FAQ BOT - SYSTEM TESTS")
    print("="*60)
    
    test_scrapers()
    test_data_processing()
    test_chromadb()
    test_full_pipeline_small()
    
    print("\n" + "="*60)
    print("✅ TESTING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main() 