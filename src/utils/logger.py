"""
Logging utility for Jupiter FAQ Bot
"""

import sys
from pathlib import Path

from loguru import logger

from config.settings import settings


def setup_logger():
    """Setup loguru logger with appropriate configuration"""

    # Remove default handler
    logger.remove()

    # Console handler with colored output
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.application.log_level,
        colorize=True,
    )

    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # File handler for all logs
    logger.add(
        logs_dir / "jupiter_bot.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="100 MB",
        retention="30 days",
        compression="zip",
    )

    # Separate file for errors
    logger.add(
        logs_dir / "errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="50 MB",
        retention="90 days",
        compression="zip",
    )

    # Scraping-specific logs
    logger.add(
        logs_dir / "scraping.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        rotation="50 MB",
        retention="15 days",
        filter=lambda record: "scraper" in record["name"].lower()
        or "scraping" in record["message"].lower(),
    )

    return logger


# Initialize logger
log = setup_logger()


def get_logger(name: str):
    """Get a logger instance for a specific module"""
    return logger.bind(name=name)
