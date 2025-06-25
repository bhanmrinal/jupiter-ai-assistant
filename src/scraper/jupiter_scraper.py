"""
Jupiter Help Center scraper for FAQ Bot
"""

from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from src.database.data_models import ScrapedContent, SourceTypeEnum
from src.scraper.base_scraper import BaseScraper
from src.utils.logger import get_logger

log = get_logger(__name__)


class JupiterScraper(BaseScraper):
    """Scraper for Jupiter Help Center (support.jupiter.money)"""

    def __init__(self):
        super().__init__(SourceTypeEnum.HELP_CENTER)
        self.base_url = "https://community.jupiter.money/c/help"
        self.main_site_url = "https://jupiter.money"
        self.community_base = "https://community.jupiter.money"
        self.faq_categories = []

    def get_all_urls(self) -> list[str]:
        """Get all FAQ URLs from the help center"""
        urls = []

        # Start with community help section
        main_url = self.base_url  # https://community.jupiter.money/c/help
        response = self._make_request(main_url)

        if not response:
            log.error("Failed to fetch main help center page")
            return urls

        soup = self._parse_html(response.text)

        # Find all category links
        category_links = self._extract_category_links(soup, main_url)
        log.info(f"Found {len(category_links)} category links")

        # For each category, get all FAQ article links
        for category_url in category_links:
            article_links = self._get_category_articles(category_url)
            urls.extend(article_links)
            log.info(f"Found {len(article_links)} articles in category {category_url}")

        # Also add the main FAQ pages
        urls.extend(category_links)

        return list(set(urls))  # Remove duplicates

    def _extract_category_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """Extract FAQ category links from main page"""
        category_links = []

        # Look for community help topic patterns
        selectors = [
            'a[href*="/t/"]',  # Community topics
            ".topic-list-item a",
            ".topic-title a",
            ".discourse-topic a",
            "table tbody tr a",  # Help table links
        ]

        for selector in selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get("href")
                if href:
                    full_url = urljoin(base_url, href)
                    if self._is_valid_url(full_url):
                        category_links.append(full_url)

        # If no structured topics found, look for any help-related links
        if not category_links:
            all_links = soup.find_all("a", href=True)
            for link in all_links:
                href = link["href"]
                if any(
                    keyword in href.lower()
                    for keyword in ["/t/", "help", "support", "bug", "guide"]
                ):
                    full_url = urljoin(base_url, href)
                    if self._is_valid_url(full_url) and "/t/" in full_url:
                        category_links.append(full_url)

        return list(set(category_links))

    def _get_category_articles(self, category_url: str) -> list[str]:
        """Get all article links from a category page"""
        response = self._make_request(category_url)
        if not response:
            return []

        soup = self._parse_html(response.text)
        article_links = []

        # Look for topic links in community help pages
        selectors = ['a[href*="/t/"]', ".topic-list-item a", ".topic-title a", ".discourse-topic a"]

        for selector in selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get("href")
                if href:
                    full_url = urljoin(category_url, href)
                    if self._is_valid_url(full_url) and "/t/" in full_url:
                        article_links.append(full_url)

        return list(set(article_links))

    def scrape_url(self, url: str) -> ScrapedContent | None:
        """Scrape content from a specific Jupiter help center URL"""
        response = self._make_request(url)
        if not response:
            return None

        soup = self._parse_html(response.text)
        content_data = self.extract_content(soup, url)

        if not content_data.get("content"):
            log.warning(f"No content extracted from {url}")
            return None

        return ScrapedContent(
            url=url,
            title=content_data.get("title", ""),
            content=content_data["content"],
            raw_html=response.text[:10000],  # Truncate for storage
            source_type=self.source_type,
        )

    def extract_content(self, soup: BeautifulSoup, url: str) -> dict[str, Any]:
        """Extract structured content from Jupiter help center page"""
        content_data = {"title": "", "content": "", "category": "", "questions": [], "answers": []}

        # Extract title from community topic
        title_selectors = [
            "h1.topic-title",
            "h1[data-topic-id]",
            ".fancy-title",
            "h1",
            ".topic-title a",
            "title",
        ]

        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                content_data["title"] = self._clean_text(title_elem.get_text())
                break

        # Extract main content from community topic
        content_selectors = [
            ".topic-body",
            ".post-stream",
            ".regular.contents",
            ".cooked",
            ".topic-post",
            "main",
            ".content",
        ]

        main_content = ""
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                main_content = self._clean_text(content_elem.get_text())
                break

        # If no structured content found, get text from body
        if not main_content:
            body = soup.find("body")
            if body:
                # Remove script and style elements
                for script in body(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                main_content = self._clean_text(body.get_text())

        content_data["content"] = main_content

        # Extract FAQ-style Q&A pairs
        self._extract_qa_pairs(soup, content_data)

        # Extract category information
        content_data["category"] = self._extract_category(soup, url)

        return content_data

    def _extract_qa_pairs(self, soup: BeautifulSoup, content_data: dict[str, Any]):
        """Extract question-answer pairs from the page"""
        # Look for FAQ-style structures
        faq_patterns = [
            (".faq-item", "h3, h4, .question", ".answer, p"),
            (".question-answer", ".question", ".answer"),
            (".accordion-item", ".accordion-header", ".accordion-body"),
            ("dt", "dt", "dd"),  # Definition list pattern
        ]

        for container_selector, question_selector, answer_selector in faq_patterns:
            containers = soup.select(container_selector)

            for container in containers:
                question_elem = container.select_one(question_selector)
                answer_elem = container.select_one(answer_selector)

                if question_elem and answer_elem:
                    question = self._clean_text(question_elem.get_text())
                    answer = self._clean_text(answer_elem.get_text())

                    if len(question) > 5 and len(answer) > 10:
                        content_data["questions"].append(question)
                        content_data["answers"].append(answer)

        # If no structured Q&A found, try to extract from headings and following paragraphs
        if not content_data["questions"]:
            self._extract_heading_based_qa(soup, content_data)

    def _extract_heading_based_qa(self, soup: BeautifulSoup, content_data: dict[str, Any]):
        """Extract Q&A based on heading patterns"""
        headings = soup.find_all(["h2", "h3", "h4"])

        for heading in headings:
            heading_text = self._clean_text(heading.get_text())

            # Check if heading looks like a question
            if any(
                indicator in heading_text.lower()
                for indicator in ["?", "how", "what", "why", "when", "where", "can i", "do i"]
            ):
                # Get following content until next heading
                answer_parts = []
                current = heading.find_next_sibling()

                while current and current.name not in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    if current.name in ["p", "div", "span"] and current.get_text().strip():
                        answer_parts.append(self._clean_text(current.get_text()))
                    current = current.find_next_sibling()

                answer = " ".join(answer_parts)
                if len(answer) > 10:
                    content_data["questions"].append(heading_text)
                    content_data["answers"].append(answer)

    def _extract_category(self, soup: BeautifulSoup, url: str) -> str:
        """Extract category information from the page"""
        # Try to extract from breadcrumbs
        breadcrumb_selectors = [
            ".breadcrumb a",
            ".breadcrumbs a",
            ".nav-breadcrumb a",
            ".category-breadcrumb a",
        ]

        for selector in breadcrumb_selectors:
            breadcrumbs = soup.select(selector)
            if len(breadcrumbs) > 1:  # Skip "Home" breadcrumb
                return self._clean_text(breadcrumbs[-1].get_text())

        # Try to extract from URL path
        path_parts = url.split("/")
        for part in reversed(path_parts):
            if part and part not in ["hc", "en-us", "articles", "categories", "sections"]:
                return part.replace("-", " ").title()

        return "general"
