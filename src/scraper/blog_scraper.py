"""
Jupiter Blog scraper for FAQ Bot
"""
import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from src.database.data_models import ScrapedContent, SourceTypeEnum
from src.scraper.base_scraper import BaseScraper
from src.utils.logger import get_logger

log = get_logger(__name__)

class BlogScraper(BaseScraper):
    """Scraper for Jupiter Blog (jupiter.money/blog)"""
    
    def __init__(self):
        super().__init__(SourceTypeEnum.BLOG)
        self.base_url = "https://jupiter.money/blog"
        self.scraped_articles = set()
        
    def get_all_urls(self) -> list[str]:
        """Get all blog URLs to scrape"""
        urls = []
        
        # Start with main blog page
        main_url = self.base_url
        response = self._make_request(main_url)
        
        if not response:
            log.error("Failed to fetch main blog page")
            return urls
        
        soup = self._parse_html(response.text)
        
        # Get article URLs from main page
        article_urls = self._extract_article_urls(soup, main_url)
        urls.extend(article_urls)
        log.info(f"Found {len(article_urls)} articles on main blog page")
        
        # Get category/tag pages
        category_urls = self._extract_category_urls(soup, main_url)
        log.info(f"Found {len(category_urls)} blog categories")
        
        # Get articles from each category
        for category_url in category_urls:
            category_articles = self._get_category_articles(category_url)
            urls.extend(category_articles)
            log.info(f"Found {len(category_articles)} articles in category {category_url}")
        
        return list(set(urls))  # Remove duplicates
    
    def _extract_article_urls(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """Extract blog article URLs from the page"""
        article_urls = []
        
        # Common selectors for blog articles
        selectors = [
            'a[href*="/blog/"]',
            '.blog-post a',
            '.article-link',
            '.post-title a',
            '.entry-title a',
            '.blog-item a',
            'article a',
            '.post-link'
        ]
        
        for selector in selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href:
                    full_url = urljoin(base_url, href)
                    if self._is_valid_url(full_url) and self._is_article_url(full_url):
                        article_urls.append(full_url)
        
        return list(set(article_urls))
    
    def _extract_category_urls(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """Extract blog category/tag URLs"""
        category_urls = []
        
        selectors = [
            'a[href*="/category/"]',
            'a[href*="/tag/"]',
            '.category-link',
            '.tag-link',
            '.blog-category a',
            '.categories a'
        ]
        
        for selector in selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href:
                    full_url = urljoin(base_url, href)
                    if self._is_valid_url(full_url):
                        category_urls.append(full_url)
        
        return list(set(category_urls))
    
    def _get_category_articles(self, category_url: str) -> list[str]:
        """Get all article URLs from a category page"""
        response = self._make_request(category_url)
        if not response:
            return []
        
        soup = self._parse_html(response.text)
        return self._extract_article_urls(soup, category_url)
    
    def _is_article_url(self, url: str) -> bool:
        """Check if URL is likely a blog article"""
        # Blog articles usually have specific patterns
        article_patterns = [
            r'/blog/[^/]+/$',  # /blog/article-name/
            r'/blog/\d{4}/',   # /blog/2023/...
            r'/\d{4}/\d{2}/',  # /2023/01/...
            r'/blog/.+\.html', # /blog/article.html
        ]
        
        for pattern in article_patterns:
            if re.search(pattern, url):
                return True
        
        # Exclude navigation and category pages
        exclude_patterns = [
            '/category/', '/tag/', '/page/', '/author/', 
            '/archive/', '/search/', '/feed'
        ]
        
        return not any(pattern in url for pattern in exclude_patterns)
    
    def scrape_url(self, url: str) -> ScrapedContent | None:
        """Scrape content from a specific blog URL"""
        response = self._make_request(url)
        if not response:
            return None
        
        soup = self._parse_html(response.text)
        content_data = self.extract_content(soup, url)
        
        if not content_data.get('content'):
            log.warning(f"No content extracted from {url}")
            return None
        
        return ScrapedContent(
            url=url,
            title=content_data.get('title', ''),
            content=content_data['content'],
            raw_html=response.text[:10000],  # Truncate for storage
            source_type=self.source_type
        )
    
    def extract_content(self, soup: BeautifulSoup, url: str) -> dict[str, Any]:
        """Extract structured content from blog article"""
        content_data = {
            'title': '',
            'content': '',
            'author': '',
            'date': '',
            'category': '',
            'tags': [],
            'questions': [],
            'answers': []
        }
        
        # Extract title
        title_selectors = [
            'h1.entry-title',
            'h1.post-title',
            'h1.article-title',
            '.blog-title h1',
            'h1',
            '.page-title',
            'title'
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                content_data['title'] = self._clean_text(title_elem.get_text())
                break
        
        # Extract author
        author_selectors = [
            '.author-name',
            '.by-author',
            '.post-author',
            '.entry-author',
            '[rel="author"]'
        ]
        
        for selector in author_selectors:
            author_elem = soup.select_one(selector)
            if author_elem:
                content_data['author'] = self._clean_text(author_elem.get_text())
                break
        
        # Extract publication date
        date_selectors = [
            '.post-date',
            '.entry-date',
            '.published',
            'time[datetime]',
            '.date'
        ]
        
        for selector in date_selectors:
            date_elem = soup.select_one(selector)
            if date_elem:
                date_text = date_elem.get('datetime') or date_elem.get_text()
                content_data['date'] = self._clean_text(date_text)
                break
        
        # Extract main content
        content_selectors = [
            '.entry-content',
            '.post-content',
            '.article-content',
            '.blog-content',
            'article .content',
            '.post-body',
            'main'
        ]
        
        main_content = ""
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                # Remove unwanted elements
                for unwanted in content_elem(['script', 'style', 'nav', 'aside', '.social-share']):
                    unwanted.decompose()
                main_content = self._clean_text(content_elem.get_text())
                break
        
        content_data['content'] = main_content
        
        # Extract categories and tags
        content_data['category'] = self._extract_category(soup, url)
        content_data['tags'] = self._extract_tags(soup)
        
        # Extract Q&A patterns from blog content
        self._extract_blog_qa(main_content, content_data)
        
        return content_data
    
    def _extract_category(self, soup: BeautifulSoup, url: str) -> str:
        """Extract category from blog post"""
        category_selectors = [
            '.post-category',
            '.entry-category',
            '.blog-category',
            '.category a',
            '.categories a'
        ]
        
        for selector in category_selectors:
            category_elem = soup.select_one(selector)
            if category_elem:
                return self._clean_text(category_elem.get_text())
        
        # Extract from URL
        if '/category/' in url:
            match = re.search(r'/category/([^/]+)', url)
            if match:
                return match.group(1).replace('-', ' ').title()
        
        return 'blog'
    
    def _extract_tags(self, soup: BeautifulSoup) -> list[str]:
        """Extract tags from blog post"""
        tags = []
        
        tag_selectors = [
            '.post-tags a',
            '.entry-tags a',
            '.tags a',
            '.tag-links a'
        ]
        
        for selector in tag_selectors:
            tag_elements = soup.select(selector)
            for tag_elem in tag_elements:
                tag_text = self._clean_text(tag_elem.get_text())
                if tag_text:
                    tags.append(tag_text)
        
        return tags
    
    def _extract_blog_qa(self, content: str, content_data: dict[str, Any]):
        """Extract Q&A patterns from blog content"""
        if not content:
            return
        
        # Extract headings as potential questions
        self._extract_heading_qa(content, content_data)
    
    def _extract_heading_qa(self, content: str, content_data: dict[str, Any]):
        """Extract Q&A from headings and following content"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check if line looks like a heading and a question
            if (len(line) > 10 and 
                ('?' in line or 
                 any(word in line.lower() for word in ['how', 'what', 'why', 'when', 'where']))):
                
                # Collect following paragraphs as potential answer
                answer_lines = []
                for j in range(i + 1, min(i + 5, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and len(next_line) > 20:
                        answer_lines.append(next_line)
                    elif not next_line:
                        continue
                    else:
                        break
                
                if answer_lines:
                    answer = ' '.join(answer_lines)
                    content_data['questions'].append(line)
                    content_data['answers'].append(answer) 