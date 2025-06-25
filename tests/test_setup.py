#!/usr/bin/env python3
"""
Test script to verify Jupiter FAQ Bot setup
"""

import sys
from pathlib import Path


def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")

    try:
        from config.settings import settings

        print("✓ Configuration loaded successfully")
        print(f"  - Data raw path: {settings.application.data_raw_path}")
        print(f"  - ChromaDB path: {settings.database.chromadb_path}")
        print(f"  - Embedding model: {settings.model.embedding_model}")
    except Exception as e:
        print(f"✗ Configuration import failed: {e}")
        return False

    try:
        from src.utils.logger import log

        print("✓ Logger imported successfully")
        log.info("Logger test message")
    except Exception as e:
        print(f"✗ Logger import failed: {e}")
        return False

    try:
        print("✓ Data models imported successfully")
    except Exception as e:
        print(f"✗ Data models import failed: {e}")
        return False

    try:
        print("✓ Validators imported successfully")
    except Exception as e:
        print(f"✗ Validators import failed: {e}")
        return False

    try:
        print("✓ Scraper imported successfully")
    except Exception as e:
        print(f"✗ Scraper import failed: {e}")
        return False

    return True


def test_directories():
    """Test if required directories exist"""
    print("\nTesting directory structure...")

    from config.settings import settings

    required_dirs = [
        settings.application.data_raw_path,
        settings.application.data_processed_path,
        settings.application.data_embeddings_path,
        settings.database.chromadb_path,
        "logs",
    ]

    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Directory missing: {dir_path}")
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"  → Created directory: {dir_path}")
            except Exception as e:
                print(f"  → Failed to create directory: {e}")
                return False

    return True


def test_data_models():
    """Test data model validation"""
    print("\nTesting data models...")

    from src.database.data_models import (
        CategoryEnum,
        FAQDocument,
        FAQMetadata,
        LanguageEnum,
        SourceTypeEnum,
    )

    try:
        # Test valid FAQ document
        metadata = FAQMetadata(
            source_url="https://support.jupiter.money/test", source_type=SourceTypeEnum.HELP_CENTER
        )

        faq = FAQDocument(
            question="How do I create an account?",
            answer="To create an account, visit our signup page and follow the instructions.",
            category=CategoryEnum.GENERAL,
            language=LanguageEnum.ENGLISH,
            metadata=metadata,
        )

        print("✓ Valid FAQ document created successfully")
        print(f"  - Question: {faq.question[:50]}...")
        print(f"  - Category: {faq.category}")
        print(f"  - Language: {faq.language}")

    except Exception as e:
        print(f"✗ Data model validation failed: {e}")
        return False

    return True


def test_validators():
    """Test validation utilities"""
    print("\nTesting validators...")

    from src.utils.validators import DataValidator

    # Test URL validation
    valid_urls = [
        "https://support.jupiter.money/hc/en-us/",
        "https://community.jupiter.money/posts",
    ]

    invalid_urls = ["not-a-url", "https://example.com", ""]

    for url in valid_urls:
        if DataValidator.validate_jupiter_url(url):
            print(f"✓ Valid Jupiter URL: {url}")
        else:
            print(f"✗ Failed to validate Jupiter URL: {url}")

    for url in invalid_urls:
        if not DataValidator.validate_jupiter_url(url):
            print(f"✓ Correctly rejected invalid URL: {url}")
        else:
            print(f"✗ Incorrectly accepted invalid URL: {url}")

    # Test text validation
    valid_question = "How do I reset my password?"
    invalid_question = "Hi"

    if DataValidator.validate_question(valid_question):
        print(f"✓ Valid question accepted: {valid_question}")
    else:
        print(f"✗ Valid question rejected: {valid_question}")

    if not DataValidator.validate_question(invalid_question):
        print(f"✓ Invalid question rejected: {invalid_question}")
    else:
        print(f"✗ Invalid question accepted: {invalid_question}")

    return True


def test_scraper_basic():
    """Test basic scraper functionality"""
    print("\nTesting scraper (basic setup only)...")

    try:
        from src.scraper.jupiter_scraper import JupiterScraper

        scraper = JupiterScraper()
        print("✓ Jupiter scraper initialized successfully")
        print(f"  - Base URL: {scraper.base_url}")
        print(f"  - Source type: {scraper.source_type}")

        # Test session creation
        if scraper.session:
            print("✓ HTTP session created successfully")
            user_agent = scraper.session.headers.get("User-Agent")
            print(f"  - User Agent: {user_agent}")
        else:
            print("✗ Failed to create HTTP session")
            return False

    except Exception as e:
        print(f"✗ Scraper initialization failed: {e}")
        return False

    return True


def main():
    """Run all tests"""
    print("Jupiter FAQ Bot Setup Test")
    print("=" * 40)

    tests = [test_imports, test_directories, test_data_models, test_validators, test_scraper_basic]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            failed += 1

    print("\n" + "=" * 40)
    print(f"Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed! Setup is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
