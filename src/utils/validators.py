"""
Validation utilities for Jupiter FAQ Bot
"""

import re
from typing import Any
from urllib.parse import urlparse

import validators

from src.database.data_models import CategoryEnum, LanguageEnum, SourceTypeEnum
from src.utils.logger import get_logger

log = get_logger(__name__)


class ValidationError(Exception):
    """Custom validation error"""

    pass


class DataValidator:
    """Data validation utilities"""

    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate if URL is properly formatted and accessible"""
        try:
            if not validators.url(url):
                return False

            parsed = urlparse(url)
            return parsed.scheme in ["http", "https"] and bool(parsed.netloc)
        except Exception as e:
            log.error(f"URL validation error: {e}")
            return False

    @staticmethod
    def validate_jupiter_url(url: str) -> bool:
        """Validate if URL belongs to Jupiter domain"""
        if not DataValidator.validate_url(url):
            return False

        parsed = urlparse(url)
        allowed_domains = [
            "jupiter.money",
            "support.jupiter.money",
            "community.jupiter.money",
            "blog.jupiter.money",
        ]

        return any(domain in parsed.netloc for domain in allowed_domains)

    @staticmethod
    def validate_question(question: str) -> bool:
        """Validate question format and content"""
        if not question or len(question.strip()) < 5:
            return False

        if len(question) > 500:
            return False

        # Check for meaningful content (not just punctuation/numbers)
        meaningful_chars = re.sub(r"[^\w\s]", "", question)
        return len(meaningful_chars.strip()) >= 3

    @staticmethod
    def validate_answer(answer: str) -> bool:
        """Validate answer format and content"""
        if not answer or len(answer.strip()) < 10:
            return False

        if len(answer) > 5000:
            return False

        # Check for meaningful content
        meaningful_chars = re.sub(r"[^\w\s]", "", answer)
        return len(meaningful_chars.strip()) >= 5

    @staticmethod
    def validate_category(category: str) -> bool:
        """Validate category against allowed values"""
        try:
            CategoryEnum(category)
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_language(language: str) -> bool:
        """Validate language against supported values"""
        try:
            LanguageEnum(language)
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_source_type(source_type: str) -> bool:
        """Validate source type against allowed values"""
        try:
            SourceTypeEnum(source_type)
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_confidence_score(score: float) -> bool:
        """Validate confidence score is between 0 and 1"""
        return isinstance(score, int | float) and 0.0 <= score <= 1.0

    @staticmethod
    def validate_embedding_vector(embedding: list[float]) -> bool:
        """Validate embedding vector format"""
        if not isinstance(embedding, list):
            return False

        if len(embedding) == 0:
            return False

        # Check all elements are numbers
        return all(isinstance(x, int | float) for x in embedding)

    @staticmethod
    def validate_text(text: str, min_length: int = 5, max_length: int = 10000) -> bool:
        """Validate text content"""
        if not text or not isinstance(text, str):
            return False

        text = text.strip()
        if len(text) < min_length or len(text) > max_length:
            return False

        # Check for meaningful content (not just punctuation/numbers)
        meaningful_chars = re.sub(r"[^\w\s]", "", text)
        return len(meaningful_chars.strip()) >= min_length // 2

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize text input by removing dangerous characters"""
        if not text:
            return ""

        # Remove null bytes and control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]", "", text)

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    @staticmethod
    def detect_language(text: str) -> str:
        """Basic language detection for Hindi/English content"""
        if not text:
            return LanguageEnum.ENGLISH.value

        # Count Hindi (Devanagari) characters
        hindi_chars = len(re.findall(r"[\u0900-\u097F]", text))
        english_chars = len(re.findall(r"[a-zA-Z]", text))

        total_chars = hindi_chars + english_chars
        if total_chars == 0:
            return LanguageEnum.ENGLISH.value

        hindi_ratio = hindi_chars / total_chars

        if hindi_ratio > 0.7:
            return LanguageEnum.HINDI.value
        elif hindi_ratio > 0.1:
            return LanguageEnum.HINGLISH.value
        else:
            return LanguageEnum.ENGLISH.value

    @staticmethod
    def validate_faq_data(data: dict[str, Any]) -> dict[str, list[str]]:
        """Validate complete FAQ data and return validation errors"""
        errors = {}

        # Validate required fields
        required_fields = ["question", "answer", "metadata"]
        for field in required_fields:
            if field not in data:
                errors.setdefault("missing_fields", []).append(field)

        # Validate question
        if "question" in data:
            if not DataValidator.validate_question(data["question"]):
                errors.setdefault("question", []).append("Invalid question format or length")

        # Validate answer
        if "answer" in data:
            if not DataValidator.validate_answer(data["answer"]):
                errors.setdefault("answer", []).append("Invalid answer format or length")

        # Validate category
        if "category" in data:
            if not DataValidator.validate_category(data["category"]):
                errors.setdefault("category", []).append("Invalid category value")

        # Validate language
        if "language" in data:
            if not DataValidator.validate_language(data["language"]):
                errors.setdefault("language", []).append("Invalid language value")

        # Validate metadata
        if "metadata" in data:
            metadata = data["metadata"]
            if not isinstance(metadata, dict):
                errors.setdefault("metadata", []).append("Metadata must be a dictionary")
            else:
                if "source_url" not in metadata:
                    errors.setdefault("metadata", []).append("Missing source_url in metadata")
                elif not DataValidator.validate_jupiter_url(metadata["source_url"]):
                    errors.setdefault("metadata", []).append("Invalid source URL")

                if "source_type" in metadata:
                    if not DataValidator.validate_source_type(metadata["source_type"]):
                        errors.setdefault("metadata", []).append("Invalid source type")

        return errors
