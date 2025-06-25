## 📌 Jupiter AI FAQ Bot: Coding Agent Instructions

### 🎯 Objective:
Build a human-friendly FAQ bot for Jupiter that:
- Scrapes FAQs from the Jupiter Help Centre.
- Cleans, deduplicates, and categorizes them.
- Stores vector embeddings in a vector DB (like Chroma).
- Uses retrieval-augmented generation (RAG) for semantic search.
- Integrates with an LLM (OpenAI or others) to answer queries naturally.
- Supports caching and long-term memory with a fallback strategy.

---

## 🗂️ Project Structure to Follow:
Use the provided directory exactly. Create modular and reusable code.

jupiter_faq_bot/
├── config/ # config/settings.py manages keys and thresholds
├── data/ # raw, processed, embeddings
├── src/
│ ├── scraper/ # Jupiter FAQ scraper
│ ├── preprocessing/ # Cleaning, deduplication, categorization
│ ├── models/ # LLMs, embeddings, retrieval
│ ├── database/ # ChromaDB client + data models
│ ├── api/ # Core logic for user interactions
│ └── utils/ # Logging, validation
├── app/ # Streamlit UI
├── scripts/ # For automation: scraping, embedding, deploying
├── tests/ # Test each module
├── notebooks/ # Demos and evaluations
├── docker/ # Containerization

markdown
Copy
Edit

---

## 🧠 Architecture Reference:
Refer to the attached image (VectorDB-LLM design.png):

### Flow:
1. User sends a query
2. Embedding is generated and checked against a **cache**
3. If **cache miss**, search Chroma vector DB for top-k similar questions
4. Construct a prompt with relevant Q&A and send to the LLM
5. Response is generated, **stored in long-term memory** and/or cache
6. Send friendly, natural-language response back

---

## ✅ Steps to Implement:

### 1. Scraping (src/scraper/)
- Target: https://support.jupiter.money/hc/en-us/
- Extract:
  - Category
  - Question
  - Answer
  - URL
- Output JSON: stored in `data/raw/`

### 2. Preprocessing (src/preprocessing/)
- `text_cleaner.py`: Strip HTML, normalize, remove emojis
- `deduplicator.py`: Use embeddings + cosine similarity to remove duplicates
- `categorizer.py`: Cluster questions (e.g., using zero-shot classification or keyword matching)
- Output JSON: stored in `data/processed/`

### 3. Hybrid AI Models and Storage (src/models/)
- `llm_manager.py`: Hybrid architecture with DistilBERT + TinyLlama
- `retriever.py`: Build and save vector index using ChromaDB
- `embeddings.py`: Use `paraphrase-multilingual-MiniLM-L12-v2` for multilingual support
- Store in `data/embeddings/chroma_db/`

### 4. Query Handling (src/models/)
- `response_generator.py`:
  - Take user query
  - Generate embedding using multilingual model
  - Search ChromaDB for relevant context
  - Intelligent routing: Fast extraction vs. Full generation
  - DistilBERT: Instant Q&A extraction from context (~0.1s)
  - TinyLlama: Full conversational AI for complex queries (~20s)
  - Automatic fallback between models
  - Context-aware response generation
- `retriever.py`: Semantic search and context ranking
- Streamlit integration: Direct response generation in UI

### 5. Streamlit App (streamlit_app.py)
- `streamlit_app.py`: Main application with:
  - Real-time chat interface
  - Automatic response generation
  - Performance indicators (response time, model used)
  - Retrieved document display
  - Confidence scoring display
  - Hybrid model status indicators

### 6. Configuration (config/)
- Use `.env.example` and `settings.py` for:
  - API keys
  - Thresholds
  - Paths
  - Vector DB type
  - Prompt templates

---

## ⚙️ Tools & Packages
- Core: `torch`, `transformers`, `sentence-transformers`, `chromadb`, `streamlit`
- Scraping: `beautifulsoup4`, `requests`, `pydantic`
- Utils: `loguru`, `python-dotenv`, `pytest`
- Models: `DistilBERT` (fast extraction), `TinyLlama` (conversation), `MiniLM` (embeddings)

---

## ✅ Deliverables
- Hybrid RAG-based chatbot with fast extraction + full generation.
- Streamlit UI with performance indicators and model status.
- Comprehensive test suite for all components.
- ChromaDB vector database with 185+ financial FAQ documents.
- Intelligent model routing with automatic fallback.
- No rule-based responses - pure ML approach.

---