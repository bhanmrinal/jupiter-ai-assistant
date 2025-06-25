"""
Configuration settings for Jupiter FAQ Bot
"""
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()

class DatabaseConfig(BaseSettings):
    """Database configuration settings"""
    mongodb_uri: str = Field(default="mongodb://localhost:27017/", env="MONGODB_URI")
    chromadb_path: str = Field(default="./data/embeddings/chroma_db", env="CHROMADB_PATH")
    collection_name: str = "jupiter_faqs"
    
class ModelConfig(BaseSettings):
    """AI model configuration settings"""
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        env="EMBEDDING_MODEL"
    )
    llm_model: str = Field(default="microsoft/DialoGPT-medium", env="LLM_MODEL")
    max_context_length: int = Field(default=2048, env="MAX_CONTEXT_LENGTH")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    top_k_results: int = Field(default=5, env="TOP_K_RESULTS")
    confidence_threshold: float = Field(default=0.8, env="CONFIDENCE_THRESHOLD")

class ScrapingConfig(BaseSettings):
    """Web scraping configuration settings"""
    scraping_delay: int = Field(default=1, env="SCRAPING_DELAY")
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    user_agent: str = Field(
        default="Mozilla/5.0 (compatible; JupiterBot/1.0)",
        env="USER_AGENT"
    )
    target_urls: list[str] = [
        "https://support.jupiter.money/hc/en-us/",
        "https://community.jupiter.money/",
        "https://jupiter.money/blog/"
    ]

class ApplicationConfig(BaseSettings):
    """General application configuration"""
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    data_raw_path: str = Field(default="./data/raw", env="DATA_RAW_PATH")
    data_processed_path: str = Field(default="./data/processed", env="DATA_PROCESSED_PATH")
    data_embeddings_path: str = Field(default="./data/embeddings", env="DATA_EMBEDDINGS_PATH")
    
class MemoryConfig(BaseSettings):
    """Long-term memory configuration"""
    memory_retention_days: int = Field(default=365, env="MEMORY_RETENTION_DAYS")
    feedback_learning_enabled: bool = Field(default=True, env="FEEDBACK_LEARNING_ENABLED")

class APIConfig(BaseSettings):
    """API keys and external service configuration"""
    huggingface_api_token: str | None = Field(default=None, env="HUGGINGFACE_API_TOKEN")
    openai_api_key: str | None = Field(default=None, env="OPENAI_API_KEY")

class Settings:
    """Main settings class that combines all configurations"""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.model = ModelConfig()
        self.scraping = ScrapingConfig()
        self.application = ApplicationConfig()
        self.memory = MemoryConfig()
        self.api = APIConfig()
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.application.data_raw_path,
            self.application.data_processed_path,
            self.application.data_embeddings_path,
            self.database.chromadb_path,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = Settings() 