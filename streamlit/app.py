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

# Set page config first
st.set_page_config(
    page_title="Jupiter FAQ Bot",
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
        """Render the sidebar with controls and information"""
        with st.sidebar:
            st.markdown("# 🪐 Jupiter FAQ Bot")
            st.markdown("*AI-powered customer support*")
            st.divider()
            
            # Model Selection
            st.markdown("### 🤖 AI Model Configuration")
            model_options = [
                "Auto (Groq + DistilBERT)",
                "Groq Llama-3.3-70B Only",
                "DistilBERT Q&A Only",
                "Fallback Mode"
            ]
            
            selected_model = st.selectbox(
                "Choose AI Model:",
                model_options,
                index=0,
                help="Select the AI model for generating responses"
            )
            
            if selected_model != st.session_state.current_model:
                st.session_state.current_model = selected_model
                st.rerun()
            
            # Response Settings
            st.markdown("### ⚙️ Response Settings")
            max_tokens = st.slider("Max Response Length", 50, 500, 200, 25)
            
            similarity_threshold = st.slider(
                "Search Similarity Threshold", 
                0.1, 0.9, 0.4, 0.05,
                help="Higher values = more precise matches"
            )
            
            # Quick Stats
            if st.session_state.system_initialized:
                self.render_quick_stats()
            
            # Data Sources Info
            st.markdown("### 📊 Data Sources")
            with st.expander("Content Information", expanded=False):
                st.markdown("""
                **185 Documents** from:
                - 🏛️ Help Center (Official docs)
                - 👥 Community (User discussions)
                - 📝 Blog (Financial articles)
                - ❓ FAQ (Direct Q&A)
                
                **Categories:**
                - Banking & Payments
                - Investments & Loans
                - Cards & UPI
                - Technical Support
                - Community Features
                """)
            
            # System Status
            self.render_system_status()

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
        """Render the main chat interface"""
        st.markdown("# 💬 Chat with Jupiter FAQ Bot")
        st.markdown("Ask me anything about Jupiter Money - banking, investments, cards, UPI, and more!")
        
        # Initialize system if needed
        if not st.session_state.system_initialized:
            system = self.setup_system()
            if system:
                st.session_state.system_components = system
                st.session_state.system_initialized = True
                st.success("✅ System initialized successfully!")
            else:
                st.error("❌ Failed to initialize system")
                return

        # Display chat messages
        self.display_chat_history()
        
        # Chat input
        if prompt := st.chat_input("Ask about Jupiter Money... (e.g., 'How to reset my PIN?')"):
            self.handle_user_input(prompt)

    def display_chat_history(self):
        """Display the chat message history"""
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
                
                response = system['response_generator'].generate_response(query, max_tokens=200)
                
                # Restore original status
                system['llm_manager'].groq_loaded = original_groq_status
                
            elif st.session_state.current_model == "Groq Llama-3.3-70B Only":
                # Force Groq usage with higher confidence threshold
                response = system['response_generator'].generate_response(query, max_tokens=200)
                
            else:  # Auto mode or fallback
                response = system['response_generator'].generate_response(query, max_tokens=200)
            
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

    def render_admin_panel(self):
        """Render admin panel for system management"""
        st.markdown("# 🔧 Admin Panel")
        
        # System Information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🖥️ System Information")
            if st.session_state.system_initialized:
                system = st.session_state.system_components
                model_info = system['llm_manager'].get_model_info()
                
                st.json({
                    "Status": model_info.get("status", "unknown"),
                    "Device": model_info.get("device", "unknown"),
                    "Models": {
                        "Groq": model_info["models"]["groq_conversation"]["loaded"],
                        "DistilBERT": model_info["models"]["distilbert_qa"]["loaded"]
                    }
                })
            
            # Clear data buttons
            st.markdown("### 🗑️ Data Management")
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("Clear Chat History", type="secondary"):
                    st.session_state.messages = []
                    st.session_state.conversation_history = []
                    st.toast("✅ Chat history cleared!")
            
            with col_b:
                if st.button("Clear Analytics", type="secondary"):
                    st.session_state.query_analytics = []
                    st.session_state.user_feedback = []
                    st.toast("✅ Analytics cleared!")
        
        with col2:
            st.markdown("### 📈 Real-time Metrics")
            
            # System health check
            if st.button("🔍 Run Health Check"):
                with st.spinner("Running health check..."):
                    try:
                        system = st.session_state.system_components
                        health_status = {
                            "LLM Manager": system['llm_manager'].health_check(),
                            "Retriever": system['retriever'].health_check(),
                            "Response Generator": system['response_generator'].health_check()
                        }
                        
                        for component, status in health_status.items():
                            if status:
                                st.success(f"✅ {component}: Healthy")
                            else:
                                st.error(f"❌ {component}: Error")
                                
                    except Exception as e:
                        st.error(f"Health check failed: {e}")
            
            # Environment variables
            st.markdown("### 🔐 Environment")
            env_status = {
                "GROQ_API_KEY": "✅ Set" if os.getenv("GROQ_API_KEY") else "❌ Missing",
                "Environment": "Production" if os.getenv("ENVIRONMENT") == "prod" else "Development"
            }
            
            for key, value in env_status.items():
                st.text(f"{key}: {value}")

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
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analytics", "🔧 Admin"])
        
        with tab1:
            self.render_main_chat()
        
        with tab2:
            self.render_analytics_dashboard()
        
        with tab3:
            self.render_admin_panel()


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
