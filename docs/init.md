Overview
This document provides detailed technical specifications for the Jupiter FAQ Bot system flow, complementing the visual flow diagrams.

1. Data Ingestion Layer
Data Sources
Primary Source: jupiter.money (main website FAQs)

Secondary Source: community.jupiter.money (community forums)

Tertiary Source: Jupiter blog content and help articles

Scraping Components
text
src/scraper/
├── base_scraper.py      # Abstract scraper interface
├── jupiter_scraper.py   # Main website scraper
├── community_scraper.py # Community forum scraper
└── utils.py            # Common scraping utilities
Data Flow Process
Scheduled Scraping: scripts/run_scraper.py triggers scraping jobs

Content Extraction: Each scraper extracts FAQ content, metadata, and context

Raw Storage: Data stored in data/raw/ as JSON files

Validation: Input validation using src/utils/validators.py

2. Data Preprocessing Pipeline
LLM-Based Preprocessing Components
text
src/preprocessing/
├── text_cleaner.py      # LLM-based text cleaning
├── deduplicator.py      # Semantic deduplication
├── categorizer.py       # ML-based categorization
└── language_detector.py # Hindi/English detection
Processing Workflow
Text Cleaning: Remove HTML, normalize formatting using smaller LLMs

Language Detection: Identify Hindi, English, or Hinglish content

Semantic Deduplication: Use embeddings to identify similar questions

Categorization: ML-based assignment to Jupiter service categories

Quality Validation: Confidence scoring and manual review flags

3. Storage Architecture
Dual Database Strategy
NoSQL: MongoDB for unstructured data json files

Vector: ChromaDB for semantic search

Data Models (src/database/data_models.py)
python
# FAQ Document Schema
{
  "_id": "uuid",
  "question": "string",
  "answer": "string", 
  "category": "enum[kyc,payments,cards,rewards,investments,technical,general]",
  "language": "enum[en,hi,hinglish]",
  "embeddings": "vector[384]",
  "metadata": {
    "source_url": "string",
    "confidence_score": "float",
    "last_updated": "datetime"
  }
}
4. AI Model Integration Layer
Hybrid AI Architecture (3 Models Total)
1. DistilBERT-base-cased-distilled-squad (Fast Extraction)
Role: Instant Q&A extraction from retrieved documents

Strengths: Lightweight, fast inference (~0.1s), accurate extraction

Usage: Simple queries with available context (60% of queries)

Implementation: Hugging Face pipeline for question-answering

2. TinyLlama-1.1B-Chat-v1.0 (Conversational Generation)
Role: Full conversational AI and complex reasoning

Strengths: Chat-optimized, contextual understanding, banking domain

Usage: Complex queries, no context scenarios, conversations (40% of queries)

Implementation: src/models/llm_manager.py with attention mask optimization

3. paraphrase-multilingual-MiniLM-L12-v2 (Embeddings)
Role: Semantic similarity and retrieval

Strengths: 50+ languages, cross-lingual search

Usage: All query processing for semantic matching

Output: 384-dimensional embeddings

5. RAG (Retrieval-Augmented Generation) Pipeline
Retrieval Process (src/models/retriever.py)
Query Embedding: Convert user query to vector using multilingual model

Similarity Search: ChromaDB cosine similarity search (top-k=5)

Metadata Filtering: Category-based filtering for relevance

Context Ranking: Relevance scoring and reranking

Context Injection: Format retrieved content for LLM prompt

Generation Process (src/models/response_generator.py)
Hybrid Model Selection: Intelligent routing between fast extraction and full generation

Fast Extraction: DistilBERT Q&A extraction for simple queries with context

Full Generation: TinyLlama conversational AI for complex queries

Context Integration: RAG-based context injection for both modes

Quality Assurance: Confidence scoring and automatic fallback between models

6. API and Handler Layer
Request Handling Components
text
src/api/
├── chat_handler.py      # Conversational logic and session management
├── search_handler.py    # FAQ search functionality
├── feedback_handler.py  # User feedback collection
└── admin_handler.py     # Administrative functions
API Flow
Request Reception: Streamlit frontend sends user query

Session Management: Maintain conversation context and history

Intent Classification: Determine query type (chat, search, feedback)

Handler Routing: Route to appropriate handler based on intent

Response Formatting: Structure response for frontend consumption

7. User Interface Layer (Streamlit)
Frontend Components
text
app/
├── main.py                 # Main Streamlit application
├── components/
│   ├── chat_interface.py   # Chat UI components
│   ├── search_interface.py # Search functionality UI
│   └── sidebar.py          # Navigation and settings
User Interaction Flow
Query Input: User enters question in chat interface

Language Detection: Automatic Hindi/English detection

Processing Indicator: Real-time status updates

Response Display: Formatted response with sources

Feedback Collection: User satisfaction rating

Conversation History: Session-based chat history

8. Performance and Reliability Implementation
Hybrid Response Strategy
Fast Mode: DistilBERT extraction for instant responses (~0.1s)

Intelligent Routing: Automatic selection based on query complexity and context availability

Full Mode: TinyLlama generation for comprehensive responses (~20s)

Graceful Degradation: Automatic fallback between models on failure

Performance Characteristics
Fast Extraction: 0.02-0.3s average response time for simple queries

Full Generation: 15-30s for complex conversational responses

High Availability: No rule-based fallbacks, pure ML approach

Context-Aware: RAG integration for both fast and full modes