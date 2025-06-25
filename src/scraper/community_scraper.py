"""
Jupiter Community scraper for FAQ Bot
"""
import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from src.database.data_models import ScrapedContent, SourceTypeEnum
from src.scraper.base_scraper import BaseScraper
from src.utils.logger import get_logger

log = get_logger(__name__)

class CommunityScraper(BaseScraper):
    """Scraper for Jupiter Community (community.jupiter.money)"""
    
    def __init__(self):
        super().__init__(SourceTypeEnum.COMMUNITY)
        self.base_url = "https://community.jupiter.money"
        self.scraped_posts = set()
        
    def get_all_urls(self) -> list[str]:
        """Get all community URLs to scrape"""
        urls = []
        
        # Start with main community page
        main_url = self.base_url
        response = self._make_request(main_url)
        
        if not response:
            log.error("Failed to fetch main community page")
            return urls
        
        soup = self._parse_html(response.text)
        
        # Get category URLs
        category_urls = self._extract_category_urls(soup, main_url)
        log.info(f"Found {len(category_urls)} community categories")
        
        # Get post URLs from each category
        for category_url in category_urls:
            post_urls = self._get_category_posts(category_url)
            urls.extend(post_urls)
            log.info(f"Found {len(post_urls)} posts in category {category_url}")
        
        # Get featured/popular posts
        featured_urls = self._get_featured_posts(soup, main_url)
        urls.extend(featured_urls)
        
        return list(set(urls))  # Remove duplicates
    
    def _extract_category_urls(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """Extract community category URLs"""
        category_urls = []
        
        # Common selectors for community platforms
        selectors = [
            'a[href*="/c/"]',  # Discourse-style categories
            'a[href*="/category/"]',
            'a[href*="/forum/"]',
            '.category-link',
            '.forum-category a',
            '.topic-category a'
        ]
        
        for selector in selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href:
                    full_url = urljoin(base_url, href)
                    if self._is_valid_url(full_url):
                        category_urls.append(full_url)
        
        # If no structured categories, look for FAQ-related links
        if not category_urls:
            all_links = soup.find_all('a', href=True)
            for link in all_links:
                href = link['href']
                text = link.get_text().lower()
                
                if any(keyword in text for keyword in ['faq', 'help', 'support', 'question', 'guide']):
                    full_url = urljoin(base_url, href)
                    if self._is_valid_url(full_url):
                        category_urls.append(full_url)
        
        return list(set(category_urls))
    
    def _get_category_posts(self, category_url: str) -> list[str]:
        """Get all post URLs from a category page"""
        response = self._make_request(category_url)
        if not response:
            return []
        
        soup = self._parse_html(response.text)
        post_urls = []
        
        # Common selectors for community post links
        selectors = [
            'a[href*="/t/"]',  # Discourse-style topics
            'a[href*="/topic/"]',
            'a[href*="/post/"]',
            'a[href*="/discussion/"]',
            '.topic-title a',
            '.post-title a',
            '.discussion-title a'
        ]
        
        for selector in selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href:
                    full_url = urljoin(category_url, href)
                    if self._is_valid_url(full_url) and '/t/' in full_url:
                        post_urls.append(full_url)
        
        return list(set(post_urls))
    
    def _get_featured_posts(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """Get featured/popular posts from main page"""
        featured_urls = []
        
        selectors = [
            '.featured-posts a',
            '.popular-posts a',
            '.recent-posts a',
            '.pinned-posts a'
        ]
        
        for selector in selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href:
                    full_url = urljoin(base_url, href)
                    if self._is_valid_url(full_url):
                        featured_urls.append(full_url)
        
        return list(set(featured_urls))
    
    def scrape_url(self, url: str) -> ScrapedContent | None:
        """Scrape content from a specific community URL"""
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
        """Extract structured content from community post"""
        content_data = {
            'title': '',
            'content': '',
            'author': '',
            'category': '',
            'questions': [],
            'answers': [],
            'replies': []
        }
        
        # Extract title
        title_selectors = [
            'h1.topic-title',
            'h1',
            '.post-title',
            '.discussion-title',
            '.topic-header h1',
            'title'
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                content_data['title'] = self._clean_text(title_elem.get_text())
                break
        
        # Extract author
        author_selectors = [
            '.topic-author',
            '.post-author',
            '.username',
            '.author-name',
            '[data-username]'
        ]
        
        for selector in author_selectors:
            author_elem = soup.select_one(selector)
            if author_elem:
                content_data['author'] = self._clean_text(author_elem.get_text())
                break
        
        # Extract main post content
        content_selectors = [
            '.topic-body',
            '.post-content',
            '.cooked',  # Discourse content container
            '.post-text',
            '.discussion-content',
            '.topic-post'
        ]
        
        main_content = ""
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                main_content = self._clean_text(content_elem.get_text())
                break
        
        # If no structured content found, get text from main area
        if not main_content:
            main = soup.find('main') or soup.find('article')
            if main:
                # Remove navigation and sidebar elements
                for elem in main(["nav", "aside", "footer", "header", "script", "style"]):
                    elem.decompose()
                main_content = self._clean_text(main.get_text())
        
        content_data['content'] = main_content
        
        # Extract replies/answers
        self._extract_replies(soup, content_data)
        
        # Extract Q&A patterns
        self._extract_community_qa(soup, content_data)
        
        # Extract category
        content_data['category'] = self._extract_category(soup, url)
        
        return content_data
    
    def _extract_replies(self, soup: BeautifulSoup, content_data: dict[str, Any]):
        """Extract replies/comments from the post"""
        reply_selectors = [
            '.topic-post',
            '.post-stream .post',
            '.reply',
            '.comment',
            '.response'
        ]
        
        for selector in reply_selectors:
            replies = soup.select(selector)
            for reply in replies[1:]:  # Skip the first post (original)
                reply_content = self._clean_text(reply.get_text())
                if len(reply_content) > 20:  # Filter out very short replies
                    content_data['replies'].append(reply_content)
    
    def _extract_community_qa(self, soup: BeautifulSoup, content_data: dict[str, Any]):
        """Extract Q&A patterns from community content"""
        # Check if the title is a question
        title = content_data.get('title', '')
        main_content = content_data.get('content', '')
        
        if self._is_question(title):
            content_data['questions'].append(title)
            
            # If there are replies, consider the first substantive reply as an answer
            if content_data['replies']:
                best_reply = self._find_best_answer(content_data['replies'])
                if best_reply:
                    content_data['answers'].append(best_reply)
            elif main_content:
                # If no replies, the main content might contain a self-answered question
                content_data['answers'].append(main_content)
        
        # Look for embedded Q&A in the content
        self._extract_embedded_qa(main_content, content_data)
    
    def _is_question(self, text: str) -> bool:
        """Check if text appears to be a question"""
        if not text:
            return False
        
        question_indicators = [
            '?', 'how', 'what', 'why', 'when', 'where', 'can i', 'do i', 
            'should i', 'is it', 'are there', 'does', 'help', 'problem'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in question_indicators)
    
    def _find_best_answer(self, replies: list[str]) -> str | None:
        """Find the most likely answer from replies"""
        # Look for replies that seem like answers (longer, helpful content)
        best_reply = None
        best_score = 0
        
        for reply in replies:
            score = len(reply)  # Longer replies often contain more helpful info
            
            # Boost score for answer-like indicators
            reply_lower = reply.lower()
            if any(indicator in reply_lower for indicator in ['try', 'use', 'go to', 'click', 'follow']):
                score += 50
            
            if score > best_score and len(reply) > 50:
                best_score = score
                best_reply = reply
        
        return best_reply
    
    def _extract_embedded_qa(self, content: str, content_data: dict[str, Any]):
        """Extract Q&A pairs embedded in content"""
        if not content:
            return
        
        # Split content into sentences/paragraphs
        paragraphs = content.split('\n')
        
        for i, paragraph in enumerate(paragraphs):
            if self._is_question(paragraph) and len(paragraph) > 10:
                # Look for answer in next paragraph
                if i + 1 < len(paragraphs):
                    next_paragraph = paragraphs[i + 1].strip()
                    if len(next_paragraph) > 20 and not self._is_question(next_paragraph):
                        content_data['questions'].append(paragraph.strip())
                        content_data['answers'].append(next_paragraph)
    
    def _extract_category(self, soup: BeautifulSoup, url: str) -> str:
        """Extract category from community post"""
        # Try breadcrumbs
        breadcrumb_selectors = [
            '.breadcrumb a',
            '.category-breadcrumb a',
            '.topic-category',
            '.category-name'
        ]
        
        for selector in breadcrumb_selectors:
            elements = soup.select(selector)
            if elements:
                # Get the last meaningful breadcrumb
                for elem in reversed(elements):
                    text = self._clean_text(elem.get_text())
                    if text and text.lower() not in ['home', 'community', 'forum']:
                        return text
        
        # Extract from URL
        if '/c/' in url:
            # Discourse-style category URLs
            match = re.search(r'/c/([^/]+)', url)
            if match:
                return match.group(1).replace('-', ' ').title()
        
        return 'community' 