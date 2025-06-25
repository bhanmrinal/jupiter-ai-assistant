"""
Base scraper class for Jupiter FAQ Bot
"""
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from config.settings import settings
from src.database.data_models import ScrapedContent, SourceTypeEnum
from src.utils.logger import get_logger
from src.utils.validators import DataValidator

log = get_logger(__name__)

class BaseScraper(ABC):
    """Abstract base class for all scrapers"""
    
    def __init__(self, source_type: SourceTypeEnum):
        self.source_type = source_type
        self.session = self._create_session()
        self.scraped_urls = set()
        self.failed_urls = set()
        
    def _create_session(self) -> requests.Session:
        """Create a requests session with proper headers"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': settings.scraping.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        return session
    
    def _make_request(self, url: str, retries: int = None) -> requests.Response | None:
        """Make HTTP request with retry logic"""
        if retries is None:
            retries = settings.scraping.max_retries
            
        for attempt in range(retries + 1):
            try:
                log.info(f"Fetching URL: {url} (attempt {attempt + 1})")
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                # Add delay between requests
                time.sleep(settings.scraping.scraping_delay)
                return response
                
            except requests.exceptions.RequestException as e:
                log.warning(f"Request failed for {url} (attempt {attempt + 1}): {e}")
                if attempt < retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    log.error(f"Failed to fetch {url} after {retries + 1} attempts")
                    self.failed_urls.add(url)
                    return None
    
    def _parse_html(self, html_content: str) -> BeautifulSoup:
        """Parse HTML content with BeautifulSoup"""
        return BeautifulSoup(html_content, 'html.parser')
    
    def _clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Sanitize using validator
        return DataValidator.sanitize_text(text)
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and belongs to Jupiter domain"""
        return DataValidator.validate_jupiter_url(url)
    
    def _save_content(self, content: ScrapedContent) -> bool:
        """Save scraped content to file"""
        try:
            # Create filename based on URL and timestamp
            parsed_url = urlparse(content.url)
            filename = f"{self.source_type.value}_{parsed_url.path.replace('/', '_')}_{int(datetime.now().timestamp())}.json"
            filepath = settings.application.data_raw_path + "/" + filename
            
            # Convert to dict for JSON serialization
            content_dict = content.dict()
            content_dict['scraped_at'] = content_dict['scraped_at'].isoformat()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(content_dict, f, indent=2, ensure_ascii=False)
            
            log.info(f"Saved content to {filepath}")
            return True
            
        except Exception as e:
            log.error(f"Failed to save content for {content.url}: {e}")
            return False
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """Extract all relevant links from the page"""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            if self._is_valid_url(full_url) and full_url not in self.scraped_urls:
                links.append(full_url)
        
        return links
    
    def get_scraping_stats(self) -> dict[str, Any]:
        """Get scraping statistics"""
        return {
            'scraped_urls_count': len(self.scraped_urls),
            'failed_urls_count': len(self.failed_urls),
            'success_rate': len(self.scraped_urls) / (len(self.scraped_urls) + len(self.failed_urls)) if (len(self.scraped_urls) + len(self.failed_urls)) > 0 else 0,
            'source_type': self.source_type.value
        }
    
    @abstractmethod
    def scrape_url(self, url: str) -> ScrapedContent | None:
        """Scrape content from a specific URL"""
        pass
    
    @abstractmethod
    def get_all_urls(self) -> list[str]:
        """Get all URLs to scrape for this source type"""
        pass
    
    @abstractmethod
    def extract_content(self, soup: BeautifulSoup, url: str) -> dict[str, Any]:
        """Extract structured content from parsed HTML"""
        pass
    
    def scrape_all(self) -> list[ScrapedContent]:
        """Scrape all content for this source type"""
        log.info(f"Starting scraping for {self.source_type.value}")
        
        urls = self.get_all_urls()
        scraped_content = []
        
        log.info(f"Found {len(urls)} URLs to scrape")
        
        for url in urls:
            if url in self.scraped_urls:
                continue
                
            content = self.scrape_url(url)
            if content:
                scraped_content.append(content)
                self._save_content(content)
                self.scraped_urls.add(url)
            
            # Log progress
            if len(scraped_content) % 10 == 0:
                log.info(f"Scraped {len(scraped_content)} pages so far...")
        
        stats = self.get_scraping_stats()
        log.info(f"Scraping completed. Stats: {stats}")
        
        return scraped_content 