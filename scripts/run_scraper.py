#!/usr/bin/env python3
"""
Main automation script for Jupiter FAQ Bot data collection
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from src.data_processing.processor import DataProcessor
from src.database.chroma_client import ChromaDBClient
from src.database.data_models import FAQDocument, ScrapedContent
from src.scraper.blog_scraper import BlogScraper
from src.scraper.community_scraper import CommunityScraper
from src.scraper.jupiter_scraper import JupiterScraper
from src.utils.logger import get_logger

log = get_logger(__name__)

class ScrapingPipeline:
    """Main pipeline for scraping, processing, and storing FAQ data"""
    
    def __init__(self):
        self.help_scraper = JupiterScraper()
        self.community_scraper = CommunityScraper()
        self.blog_scraper = BlogScraper()
        self.processor = DataProcessor()
        self.chroma_client = ChromaDBClient()
        
        self.scraped_data = []
        self.processed_faqs = []
        
    def run_full_pipeline(self, sources: list[str] = None, limit: int = None) -> dict[str, Any]:
        """Run the complete scraping and processing pipeline"""
        if sources is None:
            sources = ["help", "community", "blog"]
        
        results = {
            "started_at": datetime.now().isoformat(),
            "sources_processed": [],
            "total_urls_scraped": 0,
            "total_faqs_generated": 0,
            "total_faqs_stored": 0,
            "errors": [],
            "stats": {}
        }
        
        try:
            log.info("🚀 Starting Jupiter FAQ Bot data pipeline")
            
            # Step 1: Scrape content from all sources
            all_scraped_content = []
            
            if "help" in sources:
                log.info("📖 Scraping Help Center...")
                help_content = self._scrape_help_center(limit)
                all_scraped_content.extend(help_content)
                results["sources_processed"].append("help_center")
            
            if "community" in sources:
                log.info("💬 Scraping Community...")
                community_content = self._scrape_community(limit)
                all_scraped_content.extend(community_content)
                results["sources_processed"].append("community")
            
            if "blog" in sources:
                log.info("📝 Scraping Blog...")
                blog_content = self._scrape_blog(limit)
                all_scraped_content.extend(blog_content)
                results["sources_processed"].append("blog")
            
            results["total_urls_scraped"] = len(all_scraped_content)
            log.info(f"✅ Scraped {len(all_scraped_content)} pieces of content")
            
            # Step 2: Process content into FAQ documents
            log.info("🔄 Processing content into FAQ documents...")
            faq_documents = self.processor.process_scraped_content(all_scraped_content)
            results["total_faqs_generated"] = len(faq_documents)
            log.info(f"✅ Generated {len(faq_documents)} FAQ documents")
            
            # Step 3: Store in ChromaDB
            log.info("💾 Storing FAQs in ChromaDB...")
            if faq_documents:
                success = self.chroma_client.add_documents(faq_documents)
                if success:
                    results["total_faqs_stored"] = len(faq_documents)
                    log.info(f"✅ Stored {len(faq_documents)} FAQs in ChromaDB")
                else:
                    results["errors"].append("Failed to store FAQs in ChromaDB")
            
            # Step 4: Save processed data
            self._save_data(all_scraped_content, faq_documents)
            
            # Step 5: Generate statistics
            results["stats"] = self._generate_stats(all_scraped_content, faq_documents)
            
            results["completed_at"] = datetime.now().isoformat()
            log.info("🎉 Pipeline completed successfully!")
            
        except Exception as e:
            log.error(f"❌ Pipeline failed: {e}")
            results["errors"].append(str(e))
            raise
        
        return results
    
    def _scrape_help_center(self, limit: int = None) -> list[ScrapedContent]:
        """Scrape Jupiter Help Center"""
        try:
            urls = self.help_scraper.get_all_urls()
            if limit:
                urls = urls[:limit]
            
            scraped_content = []
            
            for url in tqdm(urls, desc="Scraping Help Center"):
                content = self.help_scraper.scrape_url(url)
                if content:
                    scraped_content.append(content)
            
            log.info(f"Help Center: {len(scraped_content)} articles scraped")
            return scraped_content
            
        except Exception as e:
            log.error(f"Failed to scrape help center: {e}")
            return []
    
    def _scrape_community(self, limit: int = None) -> list[ScrapedContent]:
        """Scrape Jupiter Community"""
        try:
            urls = self.community_scraper.get_all_urls()
            if limit:
                urls = urls[:limit]
            
            scraped_content = []
            
            for url in tqdm(urls, desc="Scraping Community"):
                content = self.community_scraper.scrape_url(url)
                if content:
                    scraped_content.append(content)
            
            log.info(f"Community: {len(scraped_content)} posts scraped")
            return scraped_content
            
        except Exception as e:
            log.error(f"Failed to scrape community: {e}")
            return []
    
    def _scrape_blog(self, limit: int = None) -> list[ScrapedContent]:
        """Scrape Jupiter Blog"""
        try:
            urls = self.blog_scraper.get_all_urls()
            if limit:
                urls = urls[:limit]
            
            scraped_content = []
            
            for url in tqdm(urls, desc="Scraping Blog"):
                content = self.blog_scraper.scrape_url(url)
                if content:
                    scraped_content.append(content)
            
            log.info(f"Blog: {len(scraped_content)} articles scraped")
            return scraped_content
            
        except Exception as e:
            log.error(f"Failed to scrape blog: {e}")
            return []
    
    def _save_data(self, scraped_content: list[ScrapedContent], faq_documents: list[FAQDocument]):
        """Save scraped and processed data to files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save scraped content
            processed_dir = Path(settings.application.data_processed_path)
            scraped_file = processed_dir / f"scraped_content_{timestamp}.json"
            scraped_data = [content.model_dump() for content in scraped_content]
            
            with open(scraped_file, 'w', encoding='utf-8') as f:
                json.dump(scraped_data, f, indent=2, ensure_ascii=False, default=str)
            
            # Save FAQ documents
            faq_file = processed_dir / f"faq_documents_{timestamp}.json"
            faq_data = [faq.model_dump() for faq in faq_documents]
            
            with open(faq_file, 'w', encoding='utf-8') as f:
                json.dump(faq_data, f, indent=2, ensure_ascii=False, default=str)
            
            log.info(f"Data saved to {scraped_file} and {faq_file}")
            
        except Exception as e:
            log.error(f"Failed to save data: {e}")
    
    def _generate_stats(self, scraped_content: list[ScrapedContent], faq_documents: list[FAQDocument]) -> dict[str, Any]:
        """Generate statistics about the scraped and processed data"""
        stats = {
            "scraped_content": {
                "total": len(scraped_content),
                "by_source": {},
                "avg_content_length": 0
            },
            "faq_documents": {
                "total": len(faq_documents),
                "by_category": {},
                "by_language": {},
                "avg_question_length": 0,
                "avg_answer_length": 0
            },
            "chromadb_stats": {}
        }
        
        # Scraped content stats
        if scraped_content:
            source_counts = {}
            total_length = 0
            
            for content in scraped_content:
                source = content.source_type.value
                source_counts[source] = source_counts.get(source, 0) + 1
                total_length += len(content.content or "")
            
            stats["scraped_content"]["by_source"] = source_counts
            stats["scraped_content"]["avg_content_length"] = total_length // len(scraped_content)
        
        # FAQ documents stats
        if faq_documents:
            category_counts = {}
            language_counts = {}
            total_q_length = 0
            total_a_length = 0
            
            for faq in faq_documents:
                category = faq.category.value if hasattr(faq.category, 'value') else faq.category
                language = faq.language.value if hasattr(faq.language, 'value') else faq.language
                
                category_counts[category] = category_counts.get(category, 0) + 1
                language_counts[language] = language_counts.get(language, 0) + 1
                
                total_q_length += len(faq.question)
                total_a_length += len(faq.answer)
            
            stats["faq_documents"]["by_category"] = category_counts
            stats["faq_documents"]["by_language"] = language_counts
            stats["faq_documents"]["avg_question_length"] = total_q_length // len(faq_documents)
            stats["faq_documents"]["avg_answer_length"] = total_a_length // len(faq_documents)
        
        # ChromaDB stats
        try:
            stats["chromadb_stats"] = self.chroma_client.get_collection_stats()
        except Exception as e:
            log.error(f"Failed to get ChromaDB stats: {e}")
            stats["chromadb_stats"] = {"error": str(e)}
        
        return stats
    
    def test_chromadb_search(self, query: str = "How to reset my PIN?") -> dict[str, Any]:
        """Test ChromaDB search functionality"""
        log.info(f"🔍 Testing ChromaDB search with query: '{query}'")
        
        try:
            results = self.chroma_client.search_similar(query, n_results=3)
            
            log.info(f"Found {results['total_found']} similar results")
            for i, result in enumerate(results['results'], 1):
                log.info(f"{i}. Q: {result['question'][:100]}...")
                log.info(f"   A: {result['answer'][:150]}...")
                log.info(f"   Similarity: {result['similarity_score']:.3f}")
                log.info(f"   Category: {result['category']}")
                log.info("---")
            
            return results
            
        except Exception as e:
            log.error(f"ChromaDB search test failed: {e}")
            return {"error": str(e)}

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Jupiter FAQ Bot Data Pipeline")
    parser.add_argument(
        "--sources", 
        nargs="+", 
        choices=["help", "community", "blog"], 
        default=["help", "community", "blog"],
        help="Sources to scrape"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit number of URLs to scrape per source (for testing)"
    )
    parser.add_argument(
        "--test-search", 
        action="store_true",
        help="Test ChromaDB search after pipeline completion"
    )
    parser.add_argument(
        "--search-query", 
        type=str, 
        default="How to reset my PIN?",
        help="Query to test search functionality"
    )
    
    args = parser.parse_args()
    
    try:
        pipeline = ScrapingPipeline()
        
        # Run the pipeline
        results = pipeline.run_full_pipeline(args.sources, args.limit)
        
        # Print results
        print("\n" + "="*60)
        print("📊 PIPELINE RESULTS")
        print("="*60)
        print(f"Sources processed: {', '.join(results['sources_processed'])}")
        print(f"URLs scraped: {results['total_urls_scraped']}")
        print(f"FAQs generated: {results['total_faqs_generated']}")
        print(f"FAQs stored: {results['total_faqs_stored']}")
        
        if results['errors']:
            print(f"Errors: {len(results['errors'])}")
            for error in results['errors']:
                print(f"  - {error}")
        
        # Print detailed stats
        if results['stats']:
            print("\n📈 DETAILED STATISTICS")
            print("-" * 40)
            
            scraped_stats = results['stats']['scraped_content']
            print("Scraped content by source:")
            for source, count in scraped_stats['by_source'].items():
                print(f"  {source}: {count}")
            
            faq_stats = results['stats']['faq_documents']
            print("FAQs by category:")
            for category, count in faq_stats['by_category'].items():
                print(f"  {category}: {count}")
            
            print("FAQs by language:")
            for language, count in faq_stats['by_language'].items():
                print(f"  {language}: {count}")
        
        # Test search if requested
        if args.test_search:
            print("\n🔍 TESTING SEARCH FUNCTIONALITY")
            print("-" * 40)
            search_results = pipeline.test_chromadb_search(args.search_query)
            
            if "error" not in search_results:
                print(f"Search query: '{args.search_query}'")
                print(f"Results found: {search_results['total_found']}")
                
                for i, result in enumerate(search_results['results'][:3], 1):
                    print(f"\n{i}. Q: {result['question']}")
                    print(f"   A: {result['answer'][:200]}...")
                    print(f"   Similarity: {result['similarity_score']:.3f}")
                    print(f"   Category: {result['category']}")
        
        print("\n✅ Pipeline completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        log.error(f"Pipeline failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main()) 