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
        """Initialize session state variables"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
            
        if 'query_analytics' not in st.session_state:
            st.session_state.query_analytics = []
            
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
            
        if 'current_model' not in st.session_state:
            st.session_state.current_model = "Auto (Groq + DistilBERT)"
            
        if 'user_feedback' not in st.session_state:
            st.session_state.user_feedback = []

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
                # Initialize components
                chroma_client = ChromaClient()
                llm_manager = LLMManager(enable_conversation=True)
                retriever = Retriever(chroma_client)
                response_generator = ResponseGenerator(retriever, llm_manager)
                
                return {
                    'chroma_client': chroma_client,
                    'llm_manager': llm_manager,
                    'retriever': retriever,
                    'response_generator': response_generator
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
            
            # Model Selection - simplified
            st.markdown("### 🤖 AI Model")
            model_options = [
                "Auto (Recommended)",
                "Groq Only", 
                "DistilBERT Only"
            ]
            
            selected_model = st.selectbox(
                "Choose Model:",
                model_options,
                index=0,
                help="Auto mode uses the best available model"
            )
            
            if selected_model != st.session_state.current_model:
                st.session_state.current_model = selected_model
                st.rerun()
            
            # Quick Stats (only if system is running)
            if st.session_state.system_initialized and st.session_state.conversation_history:
                st.markdown("### 📊 Session Stats")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Queries", len(st.session_state.conversation_history))
                with col2:
                    avg_time = self.calculate_average_response_time()
                    st.metric("Avg Time", f"{avg_time:.1f}s")
            
            # System Status - simplified
            st.markdown("### 🔧 System Status")
            if st.session_state.system_initialized:
                st.success("🟢 All systems online")
            else:
                st.warning("🟡 Initializing...")
                
            # Data Info - minimal
            st.markdown("### 📚 Knowledge Base")
            st.info("185 Jupiter Money documents loaded")
            
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
        """Render clean, simple chat interface"""
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

        # Clean header
        st.markdown("# 🪐 Jupiter AI Assistant")
        st.markdown("*Your AI Help-Desk for Jupiter Money *")
        
        # Simple chat input
        with st.form("chat_form", clear_on_submit=True):
            prompt = st.text_input(
                "💬 Ask your question:",
                placeholder="e.g., How to reset my PIN? What are Jupiter's investment options?",
                help="Ask anything about Jupiter Money - banking, payments, investments, cards, UPI, and more!"
            )
            submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)
        
        # Handle form submission
        if submitted and prompt.strip():
            self.handle_user_input(prompt.strip())
        
        # Chat history - simple display
        if st.session_state.messages:
            st.markdown("---")
            self.display_simple_chat_history()
        else:
            st.info("👋 Welcome! Ask your first question about Jupiter Money to get started.")

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
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            with st.spinner("🤔 Thinking..."):
                start_time = time.time()
                response = self.generate_response(prompt)
                end_time = time.time()
            
            # Display response
            response_placeholder.markdown(response["answer"])
            
            # Store response
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
            
            # Show response metadata
            self.display_response_metadata(response, end_time - start_time)

    def generate_response(self, query: str) -> dict[str, Any]:
        """Generate response using the selected model configuration"""
        try:
            system = st.session_state.system_components
            
            # Configure model based on selection
            if st.session_state.current_model == "DistilBERT Q&A Only":
                # Disable Groq temporarily
                original_groq_status = system['llm_manager'].groq_loaded
                system['llm_manager'].groq_loaded = False
                
                response = system['response_generator'].generate_response(query, max_tokens=500)
                
                # Restore original status
                system['llm_manager'].groq_loaded = original_groq_status
                
            elif st.session_state.current_model == "Groq Llama-3.3-70B Only":
                # Force Groq usage with higher confidence threshold
                response = system['response_generator'].generate_response(query, max_tokens=500)
                
            else:  # Auto mode or fallback
                response = system['response_generator'].generate_response(query, max_tokens=500)
            
            return response
            
        except Exception as e:
            return {
                "answer": f"I apologize, but I encountered an error: {str(e)}",
                "confidence": 0.0,
                "metadata": {"error": str(e), "generation_method": "error"}
            }

    def display_response_metadata(self, response: dict[str, Any], response_time: float):
        """Display response metadata"""
        metadata = response.get("metadata", {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.caption(f"⏱️ {response_time:.2f}s")
        
        with col2:
            confidence = response.get("confidence", 0)
            st.caption(f"🎯 {confidence:.0%}")
        
        with col3:
            method = metadata.get("generation_method", "unknown")
            st.caption(f"🤖 {method}")
        
        with col4:
            docs = metadata.get("documents_found", 0)
            st.caption(f"📚 {docs} docs")

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
                st.metric("🤖 Models", f"{model_count}/2")
                if model_count >= 1:
                    st.success("✅ Models Ready")
                else:
                    st.error("❌ Models Error")
            except:
                st.error("❌ Models Error")
        
        st.divider()
        
        # Configuration
        st.markdown("## ⚙️ Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎛️ Response Settings")
            max_tokens = st.slider("Max Response Length", 50, 1000, 500, 50)
            similarity_threshold = st.slider("Search Precision", 0.1, 0.9, 0.4, 0.05)
            
        with col2:
            st.markdown("### 🔍 Test Search")
            test_query = st.text_input("Test query:", placeholder="e.g., How to reset PIN?")
            
            if st.button("🔍 Test") and test_query:
                try:
                    with st.spinner("Searching..."):
                        results = system['retriever'].search(test_query, top_k=2)
                    st.success(f"✅ Found {len(results)} relevant documents")
                    for i, doc in enumerate(results[:2]):
                        st.text(f"{i+1}. {doc.question[:80]}... (Score: {doc.similarity_score:.2f})")
                except Exception as e:
                    st.error(f"❌ Search failed: {str(e)}")
        
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
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analytics", "⚙️ Settings"])
        
        with tab1:
            self.render_main_chat()
        
        with tab2:
            self.render_analytics_dashboard()
        
        with tab3:
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
