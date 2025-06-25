# Jupiter FAQ Bot

A multilingual AI-powered FAQ bot for Jupiter that uses RAG (Retrieval-Augmented Generation) to provide intelligent responses to user queries.

## 🎯 Features

- **Comprehensive Scraping**: Scrapes FAQs from Jupiter Help Center, Community, and Blog
- **Multilingual Support**: English, Hindi, and Hinglish language detection and processing
- **RAG Architecture**: Combines vector similarity search with LLM generation
- **Long-term Memory**: Stores conversation history and learns from feedback
- **Streamlit UI**: Clean, user-friendly chat interface
- **Modular Design**: Extensible architecture with clear separation of concerns

## 🏗️ Architecture

```
User Query → Embedding → Vector Search → Context Retrieval → LLM Generation → Response
                ↓
           Long-term Memory Storage ← Feedback Collection
```

## 📁 Project Structure

```
jupiter_faq_bot/
├── config/                 # Configuration and settings
│   ├── __init__.py
│   └── settings.py
├── src/                    # Source code
│   ├── scraper/           # Web scraping modules
│   ├── preprocessing/     # Data cleaning and processing
│   ├── models/           # AI models and embeddings
│   ├── database/         # Data models and storage
│   ├── api/              # API handlers and logic
│   └── utils/            # Utilities and helpers
├── app/                   # Streamlit application
├── data/                  # Data storage
│   ├── raw/              # Raw scraped content
│   ├── processed/        # Cleaned and structured data
│   └── embeddings/       # Vector embeddings and ChromaDB
├── tests/                 # Test suite
├── scripts/              # Automation scripts
└── logs/                 # Application logs
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configuration

Copy the environment template and configure:

```bash
cp env.example .env
# Edit .env with your API keys and settings
```

### 3. Test Setup

```bash
python test_setup.py
```

### 4. Run Initial Scraping

```bash
python scripts/run_scraper.py
```

### 5. Start the Application

```bash
streamlit run app/main.py
```

## 🔧 Configuration

Key configuration options in `config/settings.py`:

- **Models**: Embedding and LLM model selection
- **Database**: ChromaDB and MongoDB settings
- **Scraping**: Rate limiting and retry policies
- **Memory**: Long-term memory retention settings

## 📊 Data Models

### FAQ Document
```python
{
    "question": "How do I reset my password?",
    "answer": "To reset your password, go to...",
    "category": "technical",
    "language": "en",
    "embeddings": [0.1, 0.2, ...],
    "metadata": {
        "source_url": "https://support.jupiter.money/...",
        "source_type": "help_center",
        "confidence_score": 0.95
    }
}
```

### Conversation Memory
```python
{
    "query": "User question",
    "response": "Bot response",
    "context_used": ["faq_id_1", "faq_id_2"],
    "feedback_received": {...},
    "learning_value": 0.8
}
```

## 🤖 AI Models

### Primary Models
1. **Embedding Model**: `paraphrase-multilingual-MiniLM-L12-v2`
   - 384-dimensional embeddings
   - Supports 50+ languages
   - Used for semantic similarity search

2. **LLM Model**: Hugging Face Transformers
   - Fallback to OpenAI if needed
   - Context-aware response generation
   - Financial domain fine-tuning

### Model Selection Logic
- **English queries**: Primary model
- **Hindi/Hinglish**: Language-specific routing
- **Complex reasoning**: Route to specialized models

## 🕷️ Scraping Strategy

### Target Sources
1. **Help Center**: `support.jupiter.money`
   - FAQ articles and categories
   - Step-by-step guides
   - Policy documentation

2. **Community**: `community.jupiter.money`
   - User discussions
   - Common questions and answers
   - Community-driven solutions

3. **Blog**: `jupiter.money/blog`
   - Product updates
   - Educational content
   - Feature announcements

### Extraction Patterns
- Q&A pairs from structured content
- Heading-based question extraction
- Category classification from breadcrumbs
- Automatic language detection

## 🔍 RAG Pipeline

### 1. Query Processing
- Text normalization and cleaning
- Language detection
- Embedding generation

### 2. Similarity Search
- ChromaDB vector search
- Top-k retrieval (configurable)
- Metadata filtering by category

### 3. Context Preparation
- Relevance ranking
- Context window optimization
- Prompt template formatting

### 4. Response Generation
- LLM prompt construction
- Response generation with context
- Confidence scoring and validation

### 5. Memory Storage
- Query-response pair storage
- Context tracking for learning
- Feedback integration

## 📝 Logging

Comprehensive logging with loguru:

- **Console**: Colored output for development
- **File Logs**: Rotating logs with compression
- **Error Logs**: Separate error tracking
- **Scraping Logs**: Dedicated scraping activity logs

## 🧪 Testing

Run the test suite:

```bash
# Basic setup test
python test_setup.py

# Full test suite
python -m pytest tests/

# Specific module tests
python -m pytest tests/test_scraper.py
```

## 🔄 Development Workflow

### 1. Scraping Phase
```bash
python scripts/run_scraper.py --source help_center
```

### 2. Preprocessing Phase
```bash
python scripts/preprocess_data.py
```

### 3. Embedding Generation
```bash
python scripts/generate_embeddings.py
```

### 4. Testing and Evaluation
```bash
python scripts/evaluate_rag.py
```

## 📈 Monitoring and Metrics

### Scraping Metrics
- Success/failure rates
- Response times
- Content quality scores

### RAG Performance
- Retrieval accuracy
- Response relevance
- User satisfaction ratings

### System Health
- Memory usage
- Database performance
- Model inference times

## 🚨 Error Handling

- **Graceful Degradation**: Fallback responses when models fail
- **Retry Logic**: Exponential backoff for network requests
- **Validation**: Input validation at all entry points
- **Monitoring**: Automatic error detection and alerting

## 🔐 Security Considerations

- **Input Sanitization**: Prevent injection attacks
- **Rate Limiting**: Protect against abuse
- **Data Privacy**: Anonymize user interactions
- **API Security**: Secure API key management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the logs in the `logs/` directory
2. Run `python test_setup.py` to verify configuration
3. Review the troubleshooting section in the documentation

---

**Current Status**: ✅ Initial setup complete - Ready for development 