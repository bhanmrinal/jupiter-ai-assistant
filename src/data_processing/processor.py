"""
Data processing pipeline for Jupiter FAQ Bot
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
        self.qa_patterns = self._load_qa_patterns()
    
    def _load_qa_patterns(self) -> dict[str, Any]:
        """Load Q&A extraction patterns"""
        return {
            "question_indicators": [
                "?", "how", "what", "why", "when", "where", "which", "who",
                "can i", "do i", "should i", "is it", "are there", "does",
                "help", "problem", "issue", "error", "fail", "unable"
            ],
            "answer_indicators": [
                "try", "use", "go to", "click", "follow", "steps", "solution",
                "you can", "to do this", "first", "next", "then", "finally"
            ],
            "faq_section_headers": [
                "faq", "frequently asked questions", "common questions",
                "help", "support", "troubleshooting", "questions and answers"
            ]
        }
    
    def process_scraped_content(self, scraped_content: list[ScrapedContent]) -> list[FAQDocument]:
        """Process list of scraped content into FAQ documents"""
        all_faqs = []
        
        for content in scraped_content:
            try:
                faqs = self._process_single_content(content)
                all_faqs.extend(faqs)
                self.processed_count += 1
                
                if self.processed_count % 10 == 0:
                    log.info(f"Processed {self.processed_count} content items, generated {len(all_faqs)} FAQs")
                    
            except Exception as e:
                log.error(f"Failed to process content from {content.url}: {e}")
                continue
        
        log.info(f"Processing complete: {len(all_faqs)} FAQ documents generated from {len(scraped_content)} scraped items")
        return all_faqs
    
    def _process_single_content(self, content: ScrapedContent) -> list[FAQDocument]:
        """Process single scraped content item into FAQ documents"""
        faqs = []
        
        # Clean and validate content
        cleaned_content = self._clean_content(content.content)
        if not cleaned_content or len(cleaned_content) < 50:
            return faqs
        
        # Extract Q&A pairs using different strategies
        qa_pairs = self._extract_qa_pairs(content, cleaned_content)
        
        # Convert to FAQ documents
        for question, answer in qa_pairs:
            faq = self._create_faq_document(content, question, answer)
            if faq:
                faqs.append(faq)
        
        return faqs
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content text"""
        if not content:
            return ""
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove navigation text
        nav_patterns = [
            r'(home|back|next|previous|menu|navigation|breadcrumb)',
            r'(sign in|sign up|login|logout|register)',
            r'(copyright|all rights reserved|privacy policy|terms)',
            r'(follow us|social media|share|like|tweet)'
        ]
        
        for pattern in nav_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # Remove excessive punctuation
        content = re.sub(r'[.]{3,}', '...', content)
        content = re.sub(r'[!]{2,}', '!', content)
        content = re.sub(r'[?]{2,}', '?', content)
        
        return content.strip()
    
    def _extract_qa_pairs(self, content: ScrapedContent, cleaned_text: str) -> list[tuple]:
        """Extract Q&A pairs from content using multiple strategies"""
        qa_pairs = []
        
        # Strategy 1: Extract from structured Q&A sections
        structured_pairs = self._extract_structured_qa(cleaned_text)
        qa_pairs.extend(structured_pairs)
        
        # Strategy 2: Extract from headings and paragraphs
        heading_pairs = self._extract_heading_qa(cleaned_text)
        qa_pairs.extend(heading_pairs)
        
        # Strategy 3: Extract from title and content (for help articles)
        if content.source_type == SourceTypeEnum.HELP_CENTER:
            title_pairs = self._extract_title_content_qa(content.title, cleaned_text)
            qa_pairs.extend(title_pairs)
        
        # Strategy 4: Extract from community discussions
        if content.source_type == SourceTypeEnum.COMMUNITY:
            discussion_pairs = self._extract_discussion_qa(content.title, cleaned_text)
            qa_pairs.extend(discussion_pairs)
        
        # Remove duplicates and filter quality
        qa_pairs = self._filter_and_deduplicate_pairs(qa_pairs)
        
        return qa_pairs
    
    def _extract_structured_qa(self, text: str) -> list[tuple]:
        """Extract Q&A from structured FAQ sections"""
        qa_pairs = []
        
        # Pattern 1: Q: ... A: ...
        pattern1 = r'Q[:\.]?\s*(.+?)\s*A[:\.]?\s*(.+?)(?=\s*Q[:\.]|$)'
        matches = re.findall(pattern1, text, re.IGNORECASE | re.DOTALL)
        for q, a in matches:
            qa_pairs.append((q.strip(), a.strip()))
        
        # Pattern 2: Question: ... Answer: ...
        pattern2 = r'Question[:\.]?\s*(.+?)\s*Answer[:\.]?\s*(.+?)(?=\s*Question|$)'
        matches = re.findall(pattern2, text, re.IGNORECASE | re.DOTALL)
        for q, a in matches:
            qa_pairs.append((q.strip(), a.strip()))
        
        # Pattern 3: FAQ numbered format
        pattern3 = r'\d+\.\s*(.+?)\s*(?:Answer|Solution)?[:\.]?\s*(.+?)(?=\d+\.|$)'
        matches = re.findall(pattern3, text, re.IGNORECASE | re.DOTALL)
        for q, a in matches:
            if self._is_question(q):
                qa_pairs.append((q.strip(), a.strip()))
        
        return qa_pairs
    
    def _extract_heading_qa(self, text: str) -> list[tuple]:
        """Extract Q&A from headings followed by content"""
        qa_pairs = []
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            # Check if paragraph looks like a question heading
            if (len(paragraph) < 200 and 
                self._is_question(paragraph) and 
                i + 1 < len(paragraphs)):
                
                # Collect following paragraphs as answer
                answer_parts = []
                for j in range(i + 1, min(i + 4, len(paragraphs))):
                    next_para = paragraphs[j]
                    
                    # Stop if we hit another question
                    if self._is_question(next_para) and len(next_para) < 200:
                        break
                    
                    if len(next_para) > 20:
                        answer_parts.append(next_para)
                
                if answer_parts:
                    answer = ' '.join(answer_parts)
                    qa_pairs.append((paragraph, answer))
        
        return qa_pairs
    
    def _extract_title_content_qa(self, title: str, content: str) -> list[tuple]:
        """Extract Q&A from title and content for help articles"""
        qa_pairs = []
        
        if not title or not content:
            return qa_pairs
        
        # If title is a question, use content as answer
        if self._is_question(title):
            # Take first meaningful paragraph as answer
            paragraphs = [p.strip() for p in content.split('\n') if p.strip() and len(p) > 30]
            if paragraphs:
                answer = paragraphs[0]
                qa_pairs.append((title, answer))
        
        # Look for "How to" patterns
        elif any(phrase in title.lower() for phrase in ['how to', 'guide to', 'steps to']):
            question = f"How to {title.lower().replace('how to', '').strip()}?"
            # Use first substantial content as answer
            content_lines = [line.strip() for line in content.split('\n') if len(line.strip()) > 30]
            if content_lines:
                answer = ' '.join(content_lines[:3])  # First 3 meaningful lines
                qa_pairs.append((question, answer))
        
        return qa_pairs
    
    def _extract_discussion_qa(self, title: str, content: str) -> list[tuple]:
        """Extract Q&A from community discussions"""
        qa_pairs = []
        
        if not title or not content:
            return qa_pairs
        
        # If title is a question, find answer in content
        if self._is_question(title):
            # Look for answer patterns in content
            content_parts = content.split('\n')
            best_answer = self._find_best_answer_in_content(content_parts)
            
            if best_answer:
                qa_pairs.append((title, best_answer))
        
        return qa_pairs
    
    def _find_best_answer_in_content(self, content_parts: list[str]) -> str | None:
        """Find the best answer from content parts"""
        candidates = []
        
        for part in content_parts:
            part = part.strip()
            if len(part) < 30:
                continue
            
            # Score based on answer indicators
            score = 0
            part_lower = part.lower()
            
            for indicator in self.qa_patterns["answer_indicators"]:
                if indicator in part_lower:
                    score += 10
            
            # Longer answers often better
            score += min(len(part) // 10, 50)
            
            candidates.append((score, part))
        
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
        
        return None
    
    def _is_question(self, text: str) -> bool:
        """Check if text appears to be a question"""
        if not text:
            return False
        
        text_lower = text.lower().strip()
        
        # Direct question indicators
        if '?' in text:
            return True
        
        # Question words at start
        question_starters = ['how', 'what', 'why', 'when', 'where', 'which', 'who', 'can', 'do', 'should', 'is', 'are', 'does']
        words = text_lower.split()
        if words and words[0] in question_starters:
            return True
        
        # Problem/help indicators
        help_indicators = ['help', 'problem', 'issue', 'error', 'unable', 'trouble', 'fail']
        if any(indicator in text_lower for indicator in help_indicators):
            return True
        
        return False
    
    def _filter_and_deduplicate_pairs(self, qa_pairs: list[tuple]) -> list[tuple]:
        """Filter and deduplicate Q&A pairs"""
        if not qa_pairs:
            return []
        
        # Filter by quality
        quality_pairs = []
        for question, answer in qa_pairs:
            if (len(question) >= 10 and len(answer) >= 20 and
                len(question) <= 500 and len(answer) <= 2000):
                quality_pairs.append((question.strip(), answer.strip()))
        
        # Deduplicate
        seen_questions = set()
        unique_pairs = []
        
        for question, answer in quality_pairs:
            # Create normalized version for comparison
            normalized_q = re.sub(r'[^\w\s]', '', question.lower())
            
            if normalized_q not in seen_questions:
                seen_questions.add(normalized_q)
                unique_pairs.append((question, answer))
        
        return unique_pairs
    
    def _create_faq_document(self, content: ScrapedContent, question: str, answer: str) -> FAQDocument | None:
        """Create FAQ document from Q&A pair"""
        try:
            # Validate content
            if not self.validator.validate_text(question) or not self.validator.validate_text(answer):
                return None
            
            # Determine category
            category = self._categorize_content(question, answer, content.url)
            
            # Determine language
            language = self._detect_language(question, answer)
            
            # Create metadata
            metadata = FAQMetadata(
                source_url=content.url,
                source_type=content.source_type,
                extracted_at=datetime.now(),
                confidence_score=self._calculate_confidence_score(question, answer),
                processing_notes=""
            )
            
            return FAQDocument(
                question=question[:500],  # Truncate if too long
                answer=answer[:2000],     # Truncate if too long
                category=category,
                language=language,
                metadata=metadata
            )
            
        except Exception as e:
            log.error(f"Failed to create FAQ document: {e}")
            return None
    
    def _categorize_content(self, question: str, answer: str, url: str) -> CategoryEnum:
        """Categorize content based on question, answer and URL"""
        text = f"{question} {answer}".lower()
        
        # Category keywords mapping
        category_keywords = {
            # Core Banking
            CategoryEnum.ACCOUNTS: ['account', 'savings', 'salary', 'corporate', 'balance', 'statement', 'open account', 'close account'],
            CategoryEnum.PAYMENTS: ['payment', 'upi', 'transfer', 'send money', 'pay', 'transaction', 'bill', 'recharge'],
            CategoryEnum.CARDS: ['card', 'credit', 'debit', 'atm', 'pin', 'limit', 'block', 'unblock', 'edge', 'rupay', 'visa'],
            CategoryEnum.LOANS: ['loan', 'personal loan', 'mutual fund loan', 'credit', 'borrow', 'emi', 'interest', 'repay'],
            CategoryEnum.INVESTMENTS: ['invest', 'mutual fund', 'sip', 'portfolio', 'returns', 'gold', 'fd', 'recurring deposit'],
            CategoryEnum.REWARDS: ['reward', 'cashback', 'points', 'benefits', 'jupiter rewards'],
            CategoryEnum.TRACK: ['track', 'money', 'spend', 'insight', 'budget', 'analytics'],
            
            # Support Categories
            CategoryEnum.KYC: ['kyc', 'verification', 'document', 'identity', 'verify', 'upload', 'aadhar', 'pan'],
            CategoryEnum.TECHNICAL: ['app', 'login', 'error', 'bug', 'technical', 'support', 'issue'],
            CategoryEnum.HELP: ['help', 'support', 'how to', 'guide', 'tutorial'],
            
            # Community Categories
            CategoryEnum.FEATURES: ['feature', 'request', 'suggestion', 'enhancement'],
            CategoryEnum.FEEDBACK: ['feedback', 'idea', 'opinion', 'review'],
            CategoryEnum.BUG_REPORTS: ['bug', 'issue', 'problem', 'error', 'crash'],
            
            # Product Specific
            CategoryEnum.POTS: ['pot', 'save', 'auto-save', 'goal', 'target'],
            CategoryEnum.UPI: ['upi', 'unified payment', 'instant payment', 'qr code'],
            CategoryEnum.DIGIQUIZ: ['digiquiz', 'gold', 'digital gold', 'investment quiz']
        }
        
        # Score each category
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                category_scores[category] = score
        
        # Return highest scoring category, default to GENERAL
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return CategoryEnum.GENERAL
    
    def _detect_language(self, question: str, answer: str) -> LanguageEnum:
        """Detect language of the content"""
        text = f"{question} {answer}"
        
        # Simple heuristic - check for Hindi/Devanagari characters
        hindi_chars = re.findall(r'[\u0900-\u097F]', text)
        if hindi_chars:
            return LanguageEnum.HI
        
        # Check for Hinglish patterns (English mixed with Hindi transliteration)
        hinglish_words = ['kaise', 'kya', 'hai', 'hain', 'kar', 'ke', 'ki', 'ko', 'se', 'mein']
        text_lower = text.lower()
        hinglish_count = sum(1 for word in hinglish_words if word in text_lower)
        
        if hinglish_count >= 2:
            return LanguageEnum.HINGLISH
        
        return LanguageEnum.ENGLISH
    
    def _calculate_confidence_score(self, question: str, answer: str) -> float:
        """Calculate confidence score for the Q&A pair"""
        score = 0.5  # Base score
        
        # Question quality indicators
        if '?' in question:
            score += 0.1
        if any(word in question.lower() for word in self.qa_patterns["question_indicators"]):
            score += 0.1
        
        # Answer quality indicators
        if any(word in answer.lower() for word in self.qa_patterns["answer_indicators"]):
            score += 0.1
        
        # Length indicators
        if 20 <= len(question) <= 200:
            score += 0.1
        if 50 <= len(answer) <= 1000:
            score += 0.1
        
        # Completeness indicators
        if answer.endswith('.') or answer.endswith('!'):
            score += 0.05
        
        return min(score, 1.0) 