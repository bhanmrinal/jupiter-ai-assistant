#!/usr/bin/env python3
"""
Jupiter FAQ Bot - Production Streamlit Application
Advanced AI-powered customer support for Jupiter Money
"""

import os
import sys
import time
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.express as px

import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Fix for ChromaDB SQLite version compatibility on Streamlit Cloud
try:
    import sqlite3
    # Check SQLite version and fix if needed
    if sqlite3.sqlite_version_info < (3, 35, 0):
        # Try to use pysqlite3 as replacement
        try:
            import pysqlite3
            sys.modules['sqlite3'] = pysqlite3
        except ImportError:
            # If pysqlite3 not available, we'll handle this gracefully
            pass
except ImportError:
    pass

# Set page config first
st.set_page_config(
    page_title="Jupiter AI FAQ-Bot",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://support.jupiter.money',
        'Report a bug': "https://community.jupiter.money",
        'About': "Jupiter FAQ Bot - AI-powered customer support"
    }
)

# Import our modules after page config and path setup
try:
    from src.database.chroma_client import ChromaClient
    from src.database.data_models import CategoryEnum, SourceTypeEnum
    from src.models.llm_manager import LLMManager
    from src.models.response_generator import ResponseGenerator
    from src.models.retriever import Retriever
except ImportError as e:
    st.error(f"Failed to import modules: {e}")
    st.error("Make sure you're running from the correct directory and all dependencies are installed.")
    st.stop()


class JupiterFAQApp:
    """Main Jupiter FAQ Bot Streamlit Application"""

    def __init__(self):
        self.initialize_session_state()
        self.setup_system()
        self.apply_custom_css()

    def initialize_session_state(self):
        """Initialize all session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
            
        if "system_initialized" not in st.session_state:
            st.session_state.system_initialized = False
            
        if "system_components" not in st.session_state:
            st.session_state.system_components = {}
        
        # Initialize model preferences
        if "preferred_groq_model" not in st.session_state:
            st.session_state.preferred_groq_model = "auto"
            
        if "max_tokens" not in st.session_state:
            st.session_state.max_tokens = 500
            
        if "show_analytics" not in st.session_state:
            st.session_state.show_analytics = False
            
        # Analytics and feedback variables
        if "query_analytics" not in st.session_state:
            st.session_state.query_analytics = []
            
        if "user_feedback" not in st.session_state:
            st.session_state.user_feedback = []
        
        if "suggestion_clicked" not in st.session_state:
            st.session_state.suggestion_clicked = None

    def apply_custom_css(self):
        """Apply custom CSS for better UX"""
        st.markdown("""
        <style>
        /* Fix chat input staying at top */
        .stTextInput > div > div > input {
            font-size: 16px;
            padding: 12px;
            border-radius: 8px;
        }
        
        /* Improve button styling */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
        }
        
        /* Better spacing for expanders */
        .streamlit-expanderHeader {
            font-size: 16px;
            font-weight: 600;
        }
        
        /* Improve info/success boxes */
        .stAlert {
            border-radius: 8px;
            margin: 8px 0;
        }
        
        /* Custom styling for chat containers */
        .chat-container {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 12px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
        }
        
        /* Feedback button styling */
        .feedback-buttons {
            margin-top: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

    @st.cache_resource
    def setup_system(_self):
        """Initialize the FAQ system with caching"""
        try:
            with st.spinner("🚀 Initializing Jupiter FAQ Bot..."):
                # Set ChromaDB path for Streamlit deployment
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                chromadb_path = os.path.join(current_dir, "data", "embeddings", "chroma_db")
                
                # Override settings for Streamlit
                from config.settings import settings
                original_path = settings.database.chromadb_path
                settings.database.chromadb_path = chromadb_path
                                
                # Initialize components
                chroma_client = ChromaClient()
                llm_manager = LLMManager(enable_conversation=True)
                retriever = Retriever(chroma_client)
                response_generator = ResponseGenerator(retriever, llm_manager)
                
                # Initialize Answer Evaluator for evaluation functionality
                try:
                    from src.models.evaluation import AnswerEvaluator
                    answer_evaluator = AnswerEvaluator(chroma_client, llm_manager)
                    evaluation_available = True
                except Exception as e:
                    st.warning(f"Evaluation system not available: {e}")
                    answer_evaluator = None
                    evaluation_available = False
                
                return {
                    'chroma_client': chroma_client,
                    'llm_manager': llm_manager,
                    'retriever': retriever,
                    'response_generator': response_generator,
                    'answer_evaluator': answer_evaluator,
                    'evaluation_available': evaluation_available
                }
        except Exception as e:
            st.error(f"Failed to initialize system: {e}")
            return None

    def render_sidebar(self):
        """Render clean, simplified sidebar"""
        with st.sidebar:
            st.markdown("# 🪐 Jupiter FAQ Bot")
            st.markdown("*AI-powered customer support*")
            st.divider()
            
            # AI Models section  
            st.markdown("### 🤖 AI Models")
            
            # Try to get system info, but show basic options if not available
            try:
                if st.session_state.system_initialized and st.session_state.system_components:
                    system = st.session_state.system_components
                    model_info = system['llm_manager'].get_model_info()
                    
                    # Model selection - Handle both old and new structure
                    groq_models = model_info.get('groq_models', {})
                    if groq_models and len(groq_models) > 0:
                        model_names = ["Auto (Smart Selection)"]
                        for model_id, model_data in groq_models.items():
                            if isinstance(model_data, dict) and model_data.get('loaded', False):
                                model_names.append(model_data.get('name', model_id))
                        
                        selected_model = st.selectbox(
                            "Choose Model:", 
                            model_names,
                            index=0,
                            help="Auto mode intelligently selects the best model for each query"
                        )
                        
                        # Store preference 
                        if selected_model == "Auto (Smart Selection)":
                            st.session_state.preferred_groq_model = "auto"
                        else:
                            # Find the model ID for the selected name
                            for model_id, model_data in groq_models.items():
                                if isinstance(model_data, dict) and model_data.get('name') == selected_model:
                                    st.session_state.preferred_groq_model = model_id
                                    break
                    else:
                        st.selectbox("Choose Model:", ["Auto (Smart Selection)"], index=0, disabled=True)
                        st.session_state.preferred_groq_model = "auto"
                        
                    # Show model status
                    st.caption(f"✅ {len(groq_models)} Groq models loaded")
                else:
                    # Show basic selection while system initializes
                    st.selectbox("Choose Model:", ["Auto (Smart Selection)"], index=0, disabled=True)
                    st.session_state.preferred_groq_model = "auto"
                    st.caption("🔄 Initializing models...")
            except Exception as e:
                # Fallback if there's any error
                st.selectbox("Choose Model:", ["Auto (Smart Selection)"], index=0, disabled=True)
                st.session_state.preferred_groq_model = "auto"
                st.caption(f"⚠️ Model info error: {str(e)[:50]}...")
            
            # Show guardrails info
            with st.expander("🛡️ Safety Guardrails", expanded=False):
                st.markdown("""
                **Active Protections:**
                - ✅ Anti-hallucination checks
                - ✅ Confidence-based responses  
                - ✅ Jupiter team identity consistency
                - ✅ Fallback model redundancy
                """)
            
            # Enhanced System Status
            st.markdown("### 🔧 System Status")
            try:
                if st.session_state.system_initialized and st.session_state.system_components:
                    st.success("🟢 All systems operational")
                    
                    # API Key status
                    if os.getenv("GROQ_API_KEY"):
                        st.text("✅ Groq API Key configured")
                    else:
                        st.text("⚠️ Groq API Key missing")
                        
                    # Database status
                    try:
                        system = st.session_state.system_components
                        stats = system['chroma_client'].get_collection_stats()
                        doc_count = stats.get('total_documents', 0)
                        st.text(f"✅ {doc_count} documents loaded")
                    except:
                        st.text("⚠️ Database status unknown")
                else:
                    st.info("🔄 System ready")
            except Exception:
                st.info("🔄 System ready")
            
            # Clear chat button
            st.divider()
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_history = []
                st.rerun()

    def render_quick_stats(self):
        """Render quick statistics"""
        st.markdown("### 📈 Session Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", len(st.session_state.conversation_history))
        with col2:
            avg_time = self.calculate_average_response_time()
            st.metric("Avg Time", f"{avg_time:.1f}s")
        
        if st.session_state.conversation_history:
            latest_confidence = st.session_state.conversation_history[-1].get('confidence', 0)
            st.metric("Last Confidence", f"{latest_confidence:.0%}")

    def render_system_status(self):
        """Render system status indicators"""
        st.markdown("### 🔧 System Status")
        
        try:
            system = self.setup_system()
            if system:
                # Check model status
                model_info = system['llm_manager'].get_model_info()
                
                groq_status = "🟢" if model_info['models']['groq_conversation']['loaded'] else "🔴"
                distilbert_status = "🟢" if model_info['models']['distilbert_qa']['loaded'] else "🔴"
                
                st.markdown(f"""
                - {groq_status} **Groq API**: {'Online' if groq_status == '🟢' else 'Offline'}
                - {distilbert_status} **DistilBERT**: {'Ready' if distilbert_status == '🟢' else 'Error'}
                - 🟢 **Database**: 185 docs loaded
                """)
                
                # API Key status
                if os.getenv("GROQ_API_KEY"):
                    st.success("🔑 API Key configured")
                else:
                    st.warning("⚠️ API Key missing")
            
        except Exception as e:
            st.error(f"System check failed: {e}")

    def render_main_chat(self):
        """Render main chat interface with input at top"""
        # Initialize system if needed
        if not st.session_state.system_initialized:
            with st.spinner("🚀 Initializing Jupiter FAQ Bot..."):
                system = self.setup_system()
                if system:
                    st.session_state.system_components = system
                    st.session_state.system_initialized = True
                else:
                    st.error("❌ Failed to initialize system")
                    return

        st.markdown("# 🤖 Jupiter Money AI Assistant")
        st.markdown("*Your AI Help-Desk for Jupiter Money*")
        
        # Check if a suggestion was clicked
        suggestion_text = st.session_state.get('suggestion_clicked', '')
        if suggestion_text:
            # Clear the suggestion after displaying
            st.session_state.suggestion_clicked = None
            
            # Show the suggestion in a text input for the user to edit/send
            with st.form("suggestion_form", clear_on_submit=True):
                edited_prompt = st.text_area(
                    "✨ Suggestion selected - Edit if needed and click Send:",
                    value=suggestion_text,
                    height=100,
                    help="You can edit this suggestion or send it as is"
                )
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    send_clicked = st.form_submit_button("📤 Send", type="primary", use_container_width=True)
                with col2:
                    cancel_clicked = st.form_submit_button("❌ Cancel", use_container_width=True)
                
                # Handle form submission
                if send_clicked and edited_prompt.strip():
                    self.handle_user_input(edited_prompt.strip())
                elif cancel_clicked:
                    st.rerun()
        else:
            # Regular chat input
            if prompt := st.chat_input("Ask me anything about Jupiter Money..."):
                self.handle_user_input(prompt)
        
        # Display chat history (newest first)
        self.display_chat_history_newest_first()

    def display_chat_history_newest_first(self):
        """Display chat history with newest conversations at the top"""
        if not st.session_state.messages:
            st.info("👋 Welcome! Ask your first question about Jupiter Money to get started.")
            return
        
        # Group messages in pairs (user + assistant)
        message_pairs = []
        for i in range(0, len(st.session_state.messages), 2):
            if i + 1 < len(st.session_state.messages):
                message_pairs.append((
                    st.session_state.messages[i],      # user message
                    st.session_state.messages[i + 1],  # assistant message
                    i + 1  # index for feedback
                ))
        
        # Display newest conversations first (reverse order)
        for user_msg, assistant_msg, feedback_idx in reversed(message_pairs):
            # User message
            with st.chat_message("user"):
                st.markdown(user_msg['content'])
            
            # Assistant message
            with st.chat_message("assistant"):
                st.markdown(assistant_msg['content'])
                
                # Only show feedback buttons and metadata for the most recent message
                if feedback_idx == len(st.session_state.messages) - 1:
                    # Show response metadata if available
                    if hasattr(st.session_state, 'latest_response_metadata'):
                        metadata_info = st.session_state.latest_response_metadata
                        self.display_response_metadata(
                            metadata_info["response"], 
                            metadata_info["response_time"]
                        )
                        
                        # Generate and display related suggestions
                        self.display_related_suggestions(
                            metadata_info["query"], 
                            metadata_info["response"]
                        )
                    
                    # Feedback section removed as per user request
            
            # Add a subtle separator between conversations
            if feedback_idx < len(st.session_state.messages) - 1:
                st.markdown("---")

    def display_simple_chat_history(self):
        """Display clean, simple chat history - newest first"""
        # Group messages in pairs and reverse for newest first
        message_pairs = []
        for i in range(0, len(st.session_state.messages), 2):
            if i + 1 < len(st.session_state.messages):
                message_pairs.append((
                    st.session_state.messages[i],      # user message
                    st.session_state.messages[i + 1],  # assistant message
                    i + 1  # index for feedback
                ))
        
        # Display newest conversations first
        for user_msg, assistant_msg, feedback_idx in reversed(message_pairs):
            # User question
            with st.container():
                st.markdown(f"**You asked:** {user_msg['content']}")
                
                # Bot response
                st.markdown("**Jupiter Bot:**")
                st.markdown(assistant_msg['content'])
                
                # Simple feedback
                col1, col2, col3 = st.columns([1, 1, 6])
                with col1:
                    if st.button("👍", key=f"up_{feedback_idx}", help="Helpful"):
                        self.handle_feedback(feedback_idx, "positive", "thumbs_up")
                with col2:
                    if st.button("👎", key=f"down_{feedback_idx}", help="Not helpful"):
                        self.handle_feedback(feedback_idx, "negative", "thumbs_down")
                
                st.markdown("---")

    def display_chat_history_improved(self):
        """Display chat history with improved UX - newest first (legacy)"""
        # Group messages in pairs (user question + bot answer)
        message_pairs = []
        for i in range(0, len(st.session_state.messages), 2):
            if i + 1 < len(st.session_state.messages):
                message_pairs.append((
                    st.session_state.messages[i],      # user message
                    st.session_state.messages[i + 1],  # assistant message
                    i + 1  # index for feedback
                ))
        
        # Display newest conversations first
        for user_msg, assistant_msg, feedback_idx in reversed(message_pairs):
            # Create expandable container for each conversation
            with st.expander(f"🔍 {user_msg['content'][:60]}{'...' if len(user_msg['content']) > 60 else ''}", expanded=True):
                # User question
                st.markdown("**Your Question:**")
                st.info(user_msg['content'])
                
                # Bot response  
                st.markdown("**Jupiter Bot Response:**")
                st.success(assistant_msg['content'])
                
                # Feedback section
                st.markdown("**Was this helpful?**")
                col1, col2, col3, col4 = st.columns([1, 1, 1, 5])
                
                with col1:
                    if st.button("👍", key=f"thumb_up_{feedback_idx}"):
                        self.handle_feedback(feedback_idx, "positive", "thumbs_up")
                
                with col2:
                    if st.button("👎", key=f"thumb_down_{feedback_idx}"):
                        self.handle_feedback(feedback_idx, "negative", "thumbs_down")
                
                with col3:
                    if st.button("⭐", key=f"star_{feedback_idx}"):
                        self.handle_feedback(feedback_idx, "excellent", "star")

    def display_chat_history(self):
        """Display the chat message history (legacy method)"""
        for i, message_data in enumerate(st.session_state.messages):
            with st.chat_message(message_data["role"]):
                st.markdown(message_data["content"])
                
                # Add feedback buttons for assistant messages
                if message_data["role"] == "assistant" and i == len(st.session_state.messages) - 1:
                    self.render_feedback_buttons(i)

    def render_feedback_buttons(self, message_index: int):
        """Render feedback buttons for the last assistant message"""
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 4])
        
        with col1:
            if st.button("👍", key=f"thumb_up_{message_index}"):
                self.handle_feedback(message_index, "positive", "thumbs_up")
        
        with col2:
            if st.button("👎", key=f"thumb_down_{message_index}"):
                self.handle_feedback(message_index, "negative", "thumbs_down")
        
        with col3:
            if st.button("⭐", key=f"star_{message_index}"):
                self.handle_feedback(message_index, "excellent", "star")
        
        with col4:
            if st.button("📋", key=f"copy_{message_index}"):
                st.code(st.session_state.messages[message_index]["content"])
                st.toast("✅ Copied to clipboard format!")

    def handle_user_input(self, prompt: str):
        """Handle user input and generate response"""
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()
            response = self.generate_response(prompt)
            end_time = time.time()
        
        # Add response to messages
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response["answer"]
        })
        
        # Store analytics
        analytics_data = {
            "timestamp": datetime.now(),
            "query": prompt,
            "response_time": end_time - start_time,
            "confidence": response.get("confidence", 0),
            "method": response.get("metadata", {}).get("generation_method", "unknown"),
            "documents_found": response.get("metadata", {}).get("documents_found", 0)
        }
        
        st.session_state.conversation_history.append(analytics_data)
        st.session_state.query_analytics.append(analytics_data)
        
        # Store response metadata and suggestions for display
        st.session_state.latest_response_metadata = {
            "response": response,
            "response_time": end_time - start_time,
            "query": prompt
        }
        
        # Force rerun to display the new messages
        st.rerun()

    def generate_response(self, query: str) -> dict[str, Any]:
        """Generate response using the enhanced model configuration"""
        try:
            system = st.session_state.system_components
            
            # Get user preferences with defaults
            max_tokens = 500  # Default value since we removed the slider
            preferred_model = getattr(st.session_state, 'preferred_groq_model', 'auto')
            
            # Convert "auto" to None for the LLM manager
            model_preference = None if preferred_model == "auto" else preferred_model
            
            # Generate response with enhanced options
            response = system['response_generator'].generate_response(
                query, 
                max_tokens=max_tokens,
                preferred_model=model_preference
            )
            
            return response
            
        except Exception as e:
            return {
                "answer": f"I apologize, but I encountered an error: {str(e)}",
                "confidence": 0.0,
                "metadata": {
                    "error": str(e), 
                    "generation_method": "error",
                    "model_used": "none",
                    "guardrails_triggered": True
                }
            }

    def display_response_metadata(self, response: dict[str, Any], response_time: float):
        """Display enhanced response metadata and analytics as inline info"""
        metadata = response.get("metadata", {})
        
        # Create inline metrics display
        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
        
        with col1:
            confidence = response.get('confidence', 0)
            confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.5 else "red"
            st.markdown(f"**Confidence:** :{confidence_color}[{confidence:.0%}]")
        
        with col2:
            time_color = "green" if response_time < 1.0 else "orange" if response_time < 3.0 else "red" 
            st.markdown(f"**Time:** :{time_color}[{response_time:.1f}s]")
        
        with col3:
            docs_found = metadata.get("documents_found", 0)
            docs_color = "green" if docs_found > 2 else "orange" if docs_found > 0 else "red"
            st.markdown(f"**Sources:** :{docs_color}[{docs_found} docs]")
        
        with col4:
            model_used = metadata.get("model_used", "unknown")
            # If it's already a friendly name (contains spaces), use as is
            if " " in model_used:
                model_short = model_used
            else:
                # Otherwise, clean up technical model names
                model_short = model_used.replace("llama-", "").replace("-versatile", "").replace("-instant", "").replace("gemma2-", "gemma-")
            st.markdown(f"**Model:** {model_short}")
            
        with col5:
            # Analytics icon with detailed analytics in popover
            if st.button("📊", key=f"analytics_{hash(str(metadata))}", help="View detailed analytics"):
                st.session_state.show_detailed_analytics = True
        
        # Show detailed analytics if requested
        if st.session_state.get('show_detailed_analytics', False):
            with st.expander("📊 Detailed Analytics", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Confidence", f"{confidence:.0%}")
                    method = metadata.get("generation_method", "unknown")
                    st.caption(f"**Method:** {method}")
                
                with col2:
                    st.metric("Response Time", f"{response_time:.2f}s")
                    retrieval_conf = metadata.get("retrieval_confidence", 0)
                    st.metric("Retrieval Score", f"{retrieval_conf:.0%}")
                
                with col3:
                    st.metric("Documents Used", docs_found)
                    guardrails = metadata.get("guardrails_triggered", False)
                    status = "🛡️ Active" if guardrails else "✅ Clear"
                    st.caption(f"**Guardrails:** {status}")
                
                with col4:
                    timestamp = metadata.get("timestamp", "")
                    if timestamp:
                        time_str = timestamp.split("T")[1][:8] if "T" in timestamp else "N/A"
                        st.caption(f"**Time:** {time_str}")
                    st.caption(f"**Model:** {model_used}")
                
                # Source documents if available
                source_docs = response.get("source_documents", [])
                if source_docs:
                    st.markdown("**📚 Source Documents:**")
                    for i, doc in enumerate(source_docs[:3], 1):
                        if isinstance(doc, dict):
                            # Handle dict format from API response
                            similarity = doc.get("similarity", doc.get("distance", 0))
                            question = doc.get("question", "")[:80]
                            st.caption(f"{i}. {question}... (score: {similarity:.0%})")
                        else:
                            # Handle object format from direct database query
                            similarity = getattr(doc, 'similarity_score', 0)
                            question = getattr(doc, 'question', '')[:80]
                            st.caption(f"{i}. {question}... (score: {similarity:.0%})")
                            
                # Performance insights
                if response_time > 3.0:
                    st.warning("⚠️ Slow response detected. Consider using a faster model.")
                elif confidence < 0.4:
                    st.info("💡 Low confidence response. Consider rephrasing your question.")
                elif metadata.get("guardrails_triggered", False):
                    st.info("🛡️ Safety guardrails were triggered to ensure response quality.")
                    
                # Close button
                if st.button("❌ Close", key="close_analytics"):
                    st.session_state.show_detailed_analytics = False
                    st.rerun()

    def display_related_suggestions(self, query: str, response: dict):
        """Display AI-generated related query suggestions"""
        try:
            system = st.session_state.system_components
            response_generator = system['response_generator']
            
            # Generate suggestions
            context_docs = response.get("source_documents", [])
            user_history = st.session_state.conversation_history[-5:] if st.session_state.conversation_history else []
            
            suggestions = response_generator.generate_related_suggestions(
                query=query,
                context_documents=context_docs,
                user_history=user_history
            )
            
            if suggestions:
                st.markdown("### 💡 Related Questions")
                st.caption("Here are some questions you might also be interested in:")
                
                # Display suggestions as clickable buttons
                cols = st.columns(min(len(suggestions), 3))
                for i, suggestion in enumerate(suggestions[:3]):
                    with cols[i % 3]:
                        suggestion_key = f"sugg_{i}_{hash(suggestion)%1000}_{len(st.session_state.conversation_history)}"
                        if st.button(suggestion, key=suggestion_key, use_container_width=True):
                            # User clicked a suggestion - set it as the next query to process
                            st.session_state.suggestion_clicked = suggestion
                
                # Show additional suggestions if more than 3
                if len(suggestions) > 3:
                    with st.expander("🔍 More suggestions", expanded=False):
                        for j, suggestion in enumerate(suggestions[3:], 4):
                            suggestion_key = f"sugg_more_{j}_{hash(suggestion)%1000}_{len(st.session_state.conversation_history)}"
                            if st.button(suggestion, key=suggestion_key, use_container_width=True):
                                # User clicked a suggestion - set it as the next query to process
                                st.session_state.suggestion_clicked = suggestion
                                
        except Exception:
            # Show fallback suggestions
            fallback_suggestions = [
                "How can I contact Jupiter support?",
                "What are Jupiter's key features?",
                "How do I update my account information?"
            ]
            
            st.markdown("### 💡 Common Questions")
            for i, suggestion in enumerate(fallback_suggestions):
                fallback_key = f"fallback_{i}_{hash(suggestion)%1000}_{len(st.session_state.conversation_history)}"
                if st.button(suggestion, key=fallback_key, use_container_width=True):
                    # User clicked a suggestion - set it as the next query to process
                    st.session_state.suggestion_clicked = suggestion

    def handle_feedback(self, message_index: int, sentiment: str, type: str):
        """Handle user feedback"""
        feedback_data = {
            "timestamp": datetime.now(),
            "message_index": message_index,
            "sentiment": sentiment,
            "type": type,
            "query": st.session_state.conversation_history[-1]["query"] if st.session_state.conversation_history else "unknown"
        }
        
        st.session_state.user_feedback.append(feedback_data)
        st.toast("✅ Thank you for your feedback!")

    def render_analytics_dashboard(self):
        """Render analytics dashboard"""
        st.markdown("# 📊 Analytics Dashboard")
        
        if not st.session_state.query_analytics:
            st.info("No analytics data available yet. Start chatting to see insights!")
            return
        
        # Create DataFrame for analysis
        df = pd.DataFrame(st.session_state.query_analytics)
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", len(df))
        
        with col2:
            avg_response_time = df['response_time'].mean()
            st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
        
        with col3:
            avg_confidence = df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.0%}")
        
        with col4:
            avg_docs = df['documents_found'].mean()
            st.metric("Avg Documents Used", f"{avg_docs:.1f}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Response time over time
            fig_time = px.line(
                df, 
                x='timestamp', 
                y='response_time',
                title="Response Time Over Time",
                labels={'response_time': 'Response Time (s)', 'timestamp': 'Time'}
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            # Confidence distribution
            fig_conf = px.histogram(
                df, 
                x='confidence',
                title="Confidence Score Distribution",
                labels={'confidence': 'Confidence Score', 'count': 'Frequency'}
            )
            st.plotly_chart(fig_conf, use_container_width=True)
        
        # Method usage
        method_counts = df['method'].value_counts()
        fig_methods = px.pie(
            values=method_counts.values,
            names=method_counts.index,
            title="AI Method Usage Distribution"
        )
        st.plotly_chart(fig_methods, use_container_width=True)
        
        # Recent queries table
        st.markdown("### 📝 Recent Queries")
        recent_df = df.tail(10)[['timestamp', 'query', 'response_time', 'confidence', 'method']]
        st.dataframe(recent_df, use_container_width=True)

    def render_evaluation_dashboard(self):
        """Render evaluation dashboard with semantic similarity and approach comparison"""
        st.markdown("# 🧪 Evaluation Dashboard")
        st.markdown("*Comprehensive evaluation of semantic similarity, relevance, and approach comparison*")
        
        if not st.session_state.system_initialized:
            st.warning("⚠️ System not initialized. Please go to the Chat tab first.")
            return
        
        system = st.session_state.system_components
        
        # Check if evaluation is available
        if not system.get('evaluation_available', False):
            st.error("🔬 Evaluation system not available. Missing dependencies or initialization failed.")
            st.info("Required: sentence-transformers, nltk")
            return
        
        st.markdown("## 📊 Evaluation Overview")
        
        # Quick evaluation section
        with st.expander("🚀 Quick Single Query Evaluation", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                test_query = st.text_input(
                    "Enter a test query:",
                    placeholder="e.g., How do I set up UPI in Jupiter?",
                    help="Enter a question to evaluate across different methods"
                )
                
                expected_answer = st.text_area(
                    "Expected answer (optional):",
                    placeholder="Enter the expected answer for semantic similarity comparison...",
                    help="Optional: Provide expected answer for similarity scoring"
                )
                
            with col2:
                st.markdown("**Evaluation Methods:**")
                st.caption("🤖 **Auto**: Smart model selection")
                st.caption("🔍 **Retrieval Only**: Database search only")
                st.caption("🧠 **LLM Only**: Pure language model")
                
                if st.button("🧪 Run Evaluation", use_container_width=True, type="primary"):
                    if test_query:
                        self.run_single_evaluation(test_query, expected_answer)
                    else:
                        st.warning("Please enter a test query")
        
        # Comprehensive evaluation section
        st.markdown("## 🔬 Comprehensive Evaluation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Run Standard Test Suite", use_container_width=True):
                self.run_comprehensive_evaluation()
        
        with col2:
            if st.button("⚡ Quick Performance Test", use_container_width=True):
                self.run_quick_performance_test()
        
        # Results display section
        if hasattr(st.session_state, 'evaluation_results'):
            self.display_evaluation_results()
        
        # Methodology documentation
        with st.expander("📚 Evaluation Methodology", expanded=False):
            st.markdown("""
            ### 🔍 Semantic Similarity
            - Uses paraphrase-multilingual-MiniLM-L12-v2 model
            - Cosine similarity between embeddings
            - Supports English, Hindi, and Hinglish
            
            ### 📈 Relevance Scoring
            - Query-answer semantic similarity
            - Length appropriateness (50-1000 chars optimal)
            - Jupiter-specific content bonus
            - Confidence-based weighting
            
            ### ⚡ Performance Metrics
            - **Response Time**: End-to-end latency
            - **Retrieval Accuracy**: Document relevance scoring
            - **Confidence Score**: Model certainty
            - **BLEU Score**: Text similarity metric
            
            ### 🔬 Approach Comparison
            - **Auto Mode**: Intelligent model selection with fallbacks
            - **Retrieval Only**: Pure vector search + template responses
            - **LLM Only**: Language model without knowledge base
            
            ### 🎯 Quality Measures
            - Hallucination detection and prevention
            - Multi-language support evaluation
            - Context relevance assessment
            - Response consistency analysis
            """)

    def run_single_evaluation(self, query: str, expected_answer: str = None):
        """Run evaluation for a single query across all methods"""
        try:
            system = st.session_state.system_components
            evaluator = system['answer_evaluator']
            
            with st.spinner("🧪 Running evaluation across all methods..."):
                # Evaluate across different methods
                methods = ["auto", "retrieval_only", "llm_only"]
                results = {}
                
                progress_bar = st.progress(0)
                
                for i, method in enumerate(methods):
                    st.caption(f"Testing {method}...")
                    result = evaluator.evaluate_single_query(
                        query=query,
                        expected_answer=expected_answer,
                        method=method
                    )
                    results[method] = result
                    progress_bar.progress((i + 1) / len(methods))
                
                # Display results
                st.success("✅ Evaluation completed!")
                
                # Comparison table
                self.display_single_evaluation_results(results, query)
                
        except Exception as e:
            st.error(f"❌ Evaluation failed: {str(e)}")
            st.error(f"Single evaluation failed: {e}")

    def display_single_evaluation_results(self, results: dict, query: str):
        """Display results from single query evaluation"""
        st.markdown("### 📊 Evaluation Results")
        
        # Create comparison dataframe
        comparison_data = []
        for method, result in results.items():
            comparison_data.append({
                "Method": method.replace("_", " ").title(),
                "Confidence": f"{result.confidence:.1%}",
                "Response Time": f"{result.response_time:.2f}s",
                "Semantic Similarity": f"{result.semantic_similarity:.3f}",
                "Relevance Score": f"{result.relevance_score:.3f}",
                "Model Used": result.model_used
            })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # Detailed responses
        st.markdown("### 💬 Generated Responses")
        
        for method, result in results.items():
            with st.expander(f"📝 {method.replace('_', ' ').title()} Response", expanded=False):
                st.markdown(f"**Answer:** {result.actual_answer}")
                st.caption(f"Confidence: {result.confidence:.1%} | Time: {result.response_time:.2f}s | Model: {result.model_used}")
        
        # Performance insights
        st.markdown("### 🎯 Performance Insights")
        
        # Best method analysis
        best_confidence = max(results.values(), key=lambda x: x.confidence)
        fastest = min(results.values(), key=lambda x: x.response_time)
        most_relevant = max(results.values(), key=lambda x: x.relevance_score)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🏆 Highest Confidence", best_confidence.method, f"{best_confidence.confidence:.1%}")
        
        with col2:
            st.metric("⚡ Fastest Response", fastest.method, f"{fastest.response_time:.2f}s")
        
        with col3:
            st.metric("🎯 Most Relevant", most_relevant.method, f"{most_relevant.relevance_score:.3f}")

    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation on test dataset"""
        try:
            system = st.session_state.system_components
            evaluator = system['answer_evaluator']
            
            with st.spinner("🔬 Running comprehensive evaluation..."):
                # Run evaluation
                evaluation_report = evaluator.run_comprehensive_evaluation()
                
                # Store results
                st.session_state.evaluation_results = evaluation_report
                
                st.success("✅ Comprehensive evaluation completed!")
                st.balloons()
                
        except Exception as e:
            st.error(f"❌ Comprehensive evaluation failed: {str(e)}")

    def run_quick_performance_test(self):
        """Run quick performance test with sample queries"""
        try:
            system = st.session_state.system_components
            response_generator = system['response_generator']
            
            test_queries = [
                "How do I set up UPI?",
                "Card limit kaise badhaye?",
                "Contact support",
                "Investment options in Jupiter"
            ]
            
            with st.spinner("⚡ Running performance test..."):
                results = []
                
                for query in test_queries:
                    start_time = time.time()
                    response = response_generator.generate_response(query, max_tokens=200)
                    end_time = time.time()
                    
                    results.append({
                        "Query": query,
                        "Response Time": f"{end_time - start_time:.2f}s",
                        "Confidence": f"{response.get('confidence', 0):.1%}",
                        "Method": response.get('metadata', {}).get('generation_method', 'unknown'),
                        "Model": response.get('metadata', {}).get('model_used', 'unknown')
                    })
                
                # Display results
                st.success("✅ Performance test completed!")
                
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                # Performance summary
                avg_time = sum(float(r['Response Time'].replace('s', '')) for r in results) / len(results)
                st.info(f"📊 Average response time: {avg_time:.2f}s")
                
        except Exception as e:
            st.error(f"❌ Performance test failed: {str(e)}")

    def display_evaluation_results(self):
        """Display comprehensive evaluation results"""
        results = st.session_state.evaluation_results
        
        st.markdown("### 📈 Comprehensive Evaluation Results")
        
        # Summary metrics
        summaries = results.get('summaries', {})
        
        if summaries:
            st.markdown("#### 🎯 Method Comparison")
            
            comparison_data = []
            for method, summary in summaries.items():
                comparison_data.append({
                    "Method": method.replace("_", " ").title(),
                    "Avg Confidence": f"{summary.avg_confidence:.1%}",
                    "Avg Response Time": f"{summary.avg_response_time:.2f}s",
                    "Avg Semantic Similarity": f"{summary.avg_semantic_similarity:.3f}",
                    "Avg Relevance": f"{summary.avg_relevance_score:.3f}",
                    "Total Queries": summary.total_queries
                })
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Response time comparison
                methods = list(summaries.keys())
                times = [summaries[m].avg_response_time for m in methods]
                
                fig_time = px.bar(
                    x=[m.replace("_", " ").title() for m in methods],
                    y=times,
                    title="Average Response Time by Method",
                    labels={'x': 'Method', 'y': 'Response Time (s)'}
                )
                st.plotly_chart(fig_time, use_container_width=True)
            
            with col2:
                # Confidence comparison
                confidences = [summaries[m].avg_confidence for m in methods]
                
                fig_conf = px.bar(
                    x=[m.replace("_", " ").title() for m in methods],
                    y=confidences,
                    title="Average Confidence by Method",
                    labels={'x': 'Method', 'y': 'Confidence Score'}
                )
                st.plotly_chart(fig_conf, use_container_width=True)
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            st.markdown("#### 💡 Recommendations")
            for rec in recommendations:
                st.info(rec)

    def render_settings_panel(self):
        """Render simplified settings panel"""
        st.markdown("# ⚙️ Settings")
        
        if not st.session_state.system_initialized:
            st.warning("⚠️ System not initialized. Please go to the Chat tab first.")
            return
        
        # System Status
        st.markdown("## 🔧 System Status")
        
        system = st.session_state.system_components
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Database status
            try:
                doc_count = len(system['chroma_client'].collection.get()['ids'])
                st.metric("📄 Documents", doc_count)
                if doc_count > 0:
                    st.success("✅ Database Online")
                else:
                    st.warning("⚠️ No documents loaded")
            except Exception:
                st.error("❌ Database Error")
        
        with col2:
            # API status
            api_key_status = "✅ Connected" if os.getenv("GROQ_API_KEY") else "❌ Not Set"
            st.metric("🔑 Groq API", api_key_status)
            
        with col3:
            # Model status
            try:
                model_info = system['llm_manager'].get_model_info()
                model_count = sum(1 for model in model_info['models'].values() if model['loaded'])
                groq_count = len([m for m in model_info['models'].values() if m['loaded'] and m['type'] == 'groq'])
                st.metric("🤖 Models", f"{model_count} loaded")
                if groq_count >= 1:
                    st.success("✅ Models Ready")
                else:
                    st.warning("⚠️ Limited Models")
            except Exception as e:
                st.error("❌ Models Error")
                st.caption(f"Error: {str(e)}")
        
        st.divider()
        
        # Configuration
        st.markdown("## ⚙️ Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎛️ Response Settings")
            st.slider("Max Response Length", 50, 1000, 500, 50)
            st.slider("Search Precision", 0.1, 0.9, 0.4, 0.05)
            
        with col2:
            st.markdown("### 🔧 System Info")
            if st.session_state.system_initialized:
                try:
                    response_gen = system['response_generator']
                    health_status = response_gen.health_check()
                    if health_status:
                        st.success("✅ System Healthy")
                    else:
                        st.warning("⚠️ System Issues Detected")
                except Exception:
                    st.error("❌ Health Check Failed")
            else:
                st.info("🔄 System Not Initialized")
        
        st.divider()
        
        # Data Management
        st.markdown("## 📊 Data Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh System", use_container_width=True):
                st.cache_resource.clear()
                st.session_state.system_initialized = False
                st.success("✅ System will refresh on next chat")
        
        with col2:
            if st.session_state.messages:
                if st.button("🗑️ Clear Chat History", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.conversation_history = []
                    st.success("✅ Chat history cleared")
        
        with col3:
            # Export data
            if st.session_state.conversation_history:
                import json
                data = {
                    'total_queries': len(st.session_state.conversation_history),
                    'conversations': st.session_state.conversation_history[-10:],  # Last 10 only
                    'export_time': datetime.now().isoformat()
                }
                
                st.download_button(
                    "📥 Export Data",
                    data=json.dumps(data, indent=2, default=str),
                    file_name=f"chat_export_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )

    def calculate_average_response_time(self) -> float:
        """Calculate average response time"""
        if not st.session_state.conversation_history:
            return 0.0
        
        times = [conv['response_time'] for conv in st.session_state.conversation_history]
        return sum(times) / len(times)

    def run(self):
        """Run the main application"""
        self.render_sidebar()
        
        # Main content area with tabs
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Analytics", "🧪 Evaluation", "⚙️ Settings"])
        
        with tab1:
            self.render_main_chat()
        
        with tab2:
            self.render_analytics_dashboard()
        
        with tab3:
            self.render_evaluation_dashboard()
        
        with tab4:
            self.render_settings_panel()


def main():
    """Main application entry point"""
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    
    .stAlert > div {
        border-radius: 8px;
    }
    
    .stButton > button {
        border-radius: 6px;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check for required environment variables
    if not os.getenv("GROQ_API_KEY"):
        st.error("🔑 GROQ_API_KEY environment variable is not set!")
        st.info("Please set your Groq API key to use the chat functionality.")
        st.code("export GROQ_API_KEY='your-api-key-here'")
        
        # Still allow viewing other tabs
        st.warning("You can still view Analytics and Admin panels.")
    
    # Run the application
    app = JupiterFAQApp()
    app.run()


if __name__ == "__main__":
    main()
