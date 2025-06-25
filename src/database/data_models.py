"""
Data models for Jupiter FAQ Bot
"""
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class CategoryEnum(str, Enum):
    """FAQ categories for Jupiter services"""
    # Core Banking Categories
    ACCOUNTS = "accounts"
    PAYMENTS = "payments"
    CARDS = "cards"
    LOANS = "loans"
    INVESTMENTS = "investments"
    REWARDS = "rewards"
    TRACK = "track"
    
    # Support Categories
    KYC = "kyc"
    TECHNICAL = "technical"
    HELP = "help"
    
    # Community Categories
    GENERAL = "general"
    COMMUNITY = "community"
    BLOG = "blog"
    FEATURES = "features"
    FEEDBACK = "feedback"
    BUG_REPORTS = "bug_reports"
    
    # Product Specific
    POTS = "pots"
    UPI = "upi"
    DIGIQUIZ = "digiquiz"

class LanguageEnum(str, Enum):
    """Supported languages"""
    ENGLISH = "en"
    HINDI = "hi"
    HINGLISH = "hinglish"

class SourceTypeEnum(str, Enum):
    """Source types for content"""
    HELP_CENTER = "help_center"
    COMMUNITY = "community"
    BLOG = "blog"

class FAQMetadata(BaseModel):
    """Metadata for FAQ documents"""
    source_url: str
    source_type: SourceTypeEnum
    confidence_score: float | None = None
    last_updated: datetime = Field(default_factory=datetime.now)
    scraped_at: datetime = Field(default_factory=datetime.now)
    page_title: str | None = None
    author: str | None = None
    tags: list[str] = Field(default_factory=list)

class FAQDocument(BaseModel):
    """Main FAQ document model"""
    id: str | None = None
    question: str = Field(..., min_length=5, max_length=500)
    answer: str = Field(..., min_length=10, max_length=5000)
    category: CategoryEnum = CategoryEnum.GENERAL
    language: LanguageEnum = LanguageEnum.ENGLISH
    embeddings: list[float] | None = None
    metadata: FAQMetadata
    
    class Config:
        use_enum_values = True

class UserQuery(BaseModel):
    """User query model"""
    query_id: str | None = None
    user_id: str | None = None
    question: str = Field(..., min_length=1, max_length=500)
    language: LanguageEnum = LanguageEnum.ENGLISH
    category_hint: CategoryEnum | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: str | None = None

class QueryResponse(BaseModel):
    """Response to user query"""
    response_id: str | None = None
    query_id: str
    answer: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    sources_used: list[str] = Field(default_factory=list)
    response_time_ms: int | None = None
    model_used: str
    retrieved_faqs: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)

class UserFeedback(BaseModel):
    """User feedback on responses"""
    feedback_id: str | None = None
    response_id: str
    user_id: str | None = None
    rating: int = Field(..., ge=1, le=5)
    helpful: bool
    comments: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ConversationMemory(BaseModel):
    """Long-term conversation memory"""
    memory_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    query: str
    response: str
    context_used: list[str] = Field(default_factory=list)
    feedback_received: UserFeedback | None = None
    learning_value: float = Field(default=0.0, ge=0.0, le=1.0)
    retention_until: datetime
    created_at: datetime = Field(default_factory=datetime.now)
    accessed_count: int = Field(default=1)
    last_accessed: datetime = Field(default_factory=datetime.now)

class ProcessingStatus(BaseModel):
    """Status tracking for data processing"""
    status_id: str | None = None
    operation_type: str  # "scraping", "preprocessing", "embedding", etc.
    status: str  # "pending", "running", "completed", "failed"
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    items_processed: int = Field(default=0)
    items_total: int | None = None
    error_message: str | None = None
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None

class ScrapedContent(BaseModel):
    """Raw scraped content before processing"""
    content_id: str | None = None
    url: str
    title: str | None = None
    content: str
    raw_html: str | None = None
    source_type: SourceTypeEnum
    scraped_at: datetime = Field(default_factory=datetime.now)
    processing_status: str = "pending"  # "pending", "processed", "failed" 