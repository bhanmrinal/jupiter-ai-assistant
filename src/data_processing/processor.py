"""
Data processing pipeline for Jupiter FAQ Bot
Production-ready processor without hardcoded patterns.
"""

import re
from datetime import datetime
from typing import Any

from src.database.data_models import (
    CategoryEnum,
    FAQDocument,
    FAQMetadata,
    LanguageEnum,
    ScrapedContent,
    SourceTypeEnum,
)
from src.utils.logger import get_logger
from src.utils.validators import DataValidator

log = get_logger(__name__)


class DataProcessor:
    """Process scraped content into structured FAQ documents"""

    def __init__(self):
        self.validator = DataValidator()
        self.processed_count = 0

    def process_scraped_content(self, scraped_data: list[ScrapedContent]) -> list[FAQDocument]:
        """Process scraped content into FAQ documents"""
        log.info(f"Processing {len(scraped_data)} scraped items...")
        faq_documents = []

        for item in scraped_data:
            try:
                documents = self._process_single_item(item)
                faq_documents.extend(documents)
            except Exception as e:
                log.error(f"Error processing item {item.url}: {e}")
                continue

        log.info(f"Generated {len(faq_documents)} FAQ documents from {len(scraped_data)} items")
        return faq_documents

    def _process_single_item(self, item: ScrapedContent) -> list[FAQDocument]:
        """Process a single scraped item into FAQ documents"""
        documents = []

        # Process based on content type
        if item.source_type == SourceTypeEnum.FAQ:
            documents.extend(self._process_faq_content(item))
        elif item.source_type == SourceTypeEnum.BLOG:
            documents.extend(self._process_blog_content(item))
        elif item.source_type == SourceTypeEnum.COMMUNITY:
            documents.extend(self._process_community_content(item))
        else:
            # Generic processing
            documents.extend(self._process_generic_content(item))

        self.processed_count += len(documents)
        return documents

    def _process_faq_content(self, item: ScrapedContent) -> list[FAQDocument]:
        """Process FAQ-specific content"""
        documents = []

        # Clean content
        content = self._clean_content(item.content)

        if not content:
            return documents

        # Extract Q&A pairs from structured FAQ
        qa_pairs = self._extract_structured_qa(content)

        for question, answer in qa_pairs:
            if self._is_valid_qa_pair(question, answer):
                doc = self._create_faq_document(item, question, answer)
                if doc:
                    documents.append(doc)

        # If no structured Q&A found, treat as single document
        if not documents and item.title:
            doc = self._create_faq_document(item, item.title, content)
            if doc:
                documents.append(doc)

        return documents

    def _process_blog_content(self, item: ScrapedContent) -> list[FAQDocument]:
        """Process blog content"""
        documents = []

        if not item.title or not item.content:
            return documents

        # Clean content
        content = self._clean_content(item.content)

        # Create document from blog post
        doc = self._create_faq_document(item, item.title, content)
        if doc:
            documents.append(doc)

        return documents

    def _process_community_content(self, item: ScrapedContent) -> list[FAQDocument]:
        """Process community discussion content"""
        documents = []

        if not item.title or not item.content:
            return documents

        # Clean content
        content = self._clean_content(item.content)

        # Create document from community post
        doc = self._create_faq_document(item, item.title, content)
        if doc:
            documents.append(doc)

        return documents

    def _process_generic_content(self, item: ScrapedContent) -> list[FAQDocument]:
        """Process generic content"""
        documents = []

        if not item.content:
            return documents

        # Clean content
        content = self._clean_content(item.content)
        title = item.title or "General Information"

        # Create document
        doc = self._create_faq_document(item, title, content)
        if doc:
            documents.append(doc)

        return documents

    def _clean_content(self, content: str) -> str:
        """Clean and normalize content text"""
        if not content:
            return ""

        # Remove extra whitespace
        content = re.sub(r"\s+", " ", content)

        # Remove excessive punctuation
        content = re.sub(r"[.]{3,}", "...", content)
        content = re.sub(r"[!]{2,}", "!", content)
        content = re.sub(r"[?]{2,}", "?", content)

        return content.strip()

    def _extract_structured_qa(self, text: str) -> list[tuple]:
        """Extract Q&A from structured FAQ sections using patterns"""
        qa_pairs = []

        # Pattern 1: Q: ... A: ...
        pattern1 = r"Q[:\.]?\s*(.+?)\s*A[:\.]?\s*(.+?)(?=\s*Q[:\.]|$)"
        matches = re.findall(pattern1, text, re.IGNORECASE | re.DOTALL)
        for q, a in matches:
            qa_pairs.append((q.strip(), a.strip()))

        # Pattern 2: Question: ... Answer: ...
        pattern2 = r"Question[:\.]?\s*(.+?)\s*Answer[:\.]?\s*(.+?)(?=\s*Question|$)"
        matches = re.findall(pattern2, text, re.IGNORECASE | re.DOTALL)
        for q, a in matches:
            qa_pairs.append((q.strip(), a.strip()))

        # Pattern 3: FAQ numbered format
        pattern3 = r"\d+\.\s*(.+?)\s*(?:Answer|Solution)?[:\.]?\s*(.+?)(?=\d+\.|$)"
        matches = re.findall(pattern3, text, re.IGNORECASE | re.DOTALL)
        for q, a in matches:
            qa_pairs.append((q.strip(), a.strip()))

        return qa_pairs

    def _is_valid_qa_pair(self, question: str, answer: str) -> bool:
        """Validate Q&A pair quality"""
        if not question or not answer:
            return False

        # Basic length checks
        if len(question) < 10 or len(answer) < 20:
            return False

        # Avoid too long content
        if len(question) > 500 or len(answer) > 2000:
            return False

        return True

    def _create_faq_document(
        self, item: ScrapedContent, question: str, answer: str
    ) -> FAQDocument | None:
        """Create FAQ document from processed content"""
        try:
            # Truncate question and answer to fit model constraints
            question = question[:500] if len(question) > 500 else question
            answer = answer[:4500] if len(answer) > 4500 else answer  # Leave room for safety
            
            # Skip if too short after truncation
            if len(question) < 10 or len(answer) < 20:
                return None

            # Detect language
            language = self._detect_language(question, answer)

            # Calculate confidence score
            confidence = self._calculate_confidence_score(question, answer)

            # Create metadata with required fields
            metadata = FAQMetadata(
                source_url=item.url,
                source_type=item.source_type,
                confidence_score=confidence,
                last_updated=datetime.utcnow(),
                page_title=item.title,
            )

            # Create FAQ document
            faq_doc = FAQDocument(
                question=question,
                answer=answer,
                category=CategoryEnum.GENERAL,  # Default, can be enhanced later
                language=language,
                metadata=metadata,
            )

            return faq_doc

        except Exception as e:
            log.error(f"Error creating FAQ document: {e}")
            return None

    def _detect_language(self, question: str, answer: str) -> LanguageEnum:
        """Detect language of the content"""
        text = f"{question} {answer}"

        # Simple heuristic - check for Hindi/Devanagari characters
        hindi_chars = re.findall(r"[\u0900-\u097F]", text)
        if hindi_chars:
            return LanguageEnum.HINDI

        # Default to English
        return LanguageEnum.ENGLISH

    def _calculate_confidence_score(self, question: str, answer: str) -> float:
        """Calculate confidence score for the Q&A pair"""
        score = 0.5  # Base score

        # Question quality
        if "?" in question:
            score += 0.1

        # Length indicators
        if 20 <= len(question) <= 200:
            score += 0.1
        if 50 <= len(answer) <= 1000:
            score += 0.1

        # Completeness indicators
        if answer.endswith(".") or answer.endswith("!"):
            score += 0.05

        return min(score, 1.0)

    def _assess_source_quality(self, item: ScrapedContent) -> float:
        """Assess quality of the source"""
        score = 0.5

        # Official sources are higher quality
        if "support.jupiter.money" in item.url:
            score += 0.3
        elif "jupiter.money" in item.url:
            score += 0.2

        # FAQ sections are typically high quality
        if item.source_type == SourceTypeEnum.FAQ:
            score += 0.2

        return min(score, 1.0)

    def get_processing_stats(self) -> dict[str, Any]:
        """Get processing statistics"""
        return {
            "total_processed": self.processed_count,
            "processor_version": "2.0.0",
            "features": ["structured_qa_extraction", "language_detection", "quality_scoring"],
        }
