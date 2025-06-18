"""
European Civil Code RAG System - Simplified Streamlit Interface
Using only Gemini 2.0 Flash and free translation services
"""

import streamlit as st
import json
from typing import Dict, Any
import logging

# Import our simplified RAG system components
from config import config, validate_config
from rag_engine import RAGEngine, load_rag_engine, format_gemini_response
from document_parser import DocumentProcessor, get_processing_status
from country_classifier import CountryClassifier
from translator import TranslationService, load_translation_service, format_translation_info

# Configure page
st.set_page_config(
    page_title="European Civil Code RAG (Simplified)",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("🏛️ European Civil Code RAG System")
    st.markdown("""
    **Simplified Version:** Ask questions about civil law across European countries. 
    This system uses **Gemini 2.0 Flash** and free translation services to automatically detect 
    the relevant country and language, translate your query if needed, and provide legal insights.
    """)
    
    # Sidebar for configuration and status
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # Configuration validation
        config_valid = validate_config()
        if config_valid:
            st.success("✅ Configuration Valid")
        else:
            st.error("❌ Configuration Issues")
            st.info("Please configure your Gemini API key in config.py")
            st.stop()
        
        # Vector store status
        st.subheader("📚 Document Processing")
        processing_status = get_processing_status()
        
        for country, status in processing_status.items():
            if status:
                st.success(f"✅ {country}")
            else:
                st.warning(f"⏳ {country} - Not processed")
        
        # Initialize vector stores button
        if st.button("🔄 Initialize Vector Stores"):
            with st.spinner("Initializing vector stores..."):
                try:
                    doc_processor = DocumentProcessor()
                    doc_processor.initialize_all_vector_stores()
                    st.success("✅ Vector stores initialized!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        st.subheader("🌍 Supported Countries")
        for country, info in config.COUNTRIES.items():
            st.write(f"🇪🇺 {country} ({info['lang_code']})")
        
        st.subheader("🤖 AI Model")
        st.info("**Gemini 2.0 Flash** - Fast, multilingual responses")
    
    # Main query interface
    st.header("💬 Ask Your Legal Question")
    
    # Query input
    query = st.text_area(
        "Enter your question in any supported language:",
        placeholder="Example: What is property law? / Qu'est-ce que le droit de propriété?",
        height=100
    )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            specific_country = st.selectbox(
                "Force specific country (optional):",
                ["Auto-detect"] + list(config.COUNTRIES.keys())
            )
            
            enable_translation = st.checkbox("Enable translation", value=True)
        
        with col2:
            show_debug_info = st.checkbox("Show debug information", value=False)
            show_original_response = st.checkbox("Show original response", value=False)
    
    # Query processing
    if st.button("🔍 Search Civil Codes", type="primary"):
        if not query.strip():
            st.warning("Please enter a question.")
            return
        
        with st.spinner("Processing your query with Gemini 2.0 Flash..."):
            try:
                # Load RAG engine
                rag_engine = load_rag_engine()
                
                # Process query
                if specific_country != "Auto-detect":
                    result = rag_engine.query_single_country(
                        query, specific_country, 
                        translate_query=enable_translation,
                        translate_response=enable_translation
                    )
                else:
                    result = rag_engine.query_with_country_detection(query)
                
                # Display results
                display_results(result, show_debug_info, show_original_response)
                
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")
                logger.error(f"Query processing error: {e}", exc_info=True)

def display_results(result: Dict[str, Any], show_debug: bool = False, show_original: bool = False):
    """Display query results"""
    
    # Country detection results
    if "country_detection" in result:
        detection = result["country_detection"]
        
        st.subheader("🎯 Country Detection")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Country", detection["predicted_country"])
        with col2:
            st.metric("Confidence", f"{detection['confidence']:.1%}")
        with col3:
            st.metric("Strategy", detection["search_strategy"])
    
    # Handle multiple country results
    if "results_by_country" in result:
        st.subheader("🏛️ Results by Country")
        
        for country, country_result in result["results_by_country"].items():
            with st.expander(f"📋 {country} Results", expanded=True):
                display_single_country_result(country_result, show_debug, show_original)
    else:
        # Single country result
        display_single_country_result(result, show_debug, show_original)

def display_single_country_result(result: Dict[str, Any], show_debug: bool = False, show_original: bool = False):
    """Display results for a single country"""
    
    if "error" in result:
        st.error(f"❌ {result['error']}")
        return
    
    # Translation information
    if result.get("translated_query") and result.get("translated_query") != result.get("original_query"):
        st.subheader("🌐 Translation")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Original:** {result['original_query']}")
        with col2:
            st.info(f"**Translated:** {result['translated_query']}")
    
    # Gemini Response
    st.subheader("🤖 Gemini 2.0 Flash Response")
    
    if "gemini_response" in result:
        response = result["gemini_response"]
        
        if response.get("error"):
            st.error(f"❌ {response['error']}")
        else:
            # Show processing time
            st.caption(f"⏱️ Processing time: {response['processing_time']:.2f}s")
            
            # Show translated response
            if response.get("translated_response"):
                st.markdown(response["translated_response"])
            
            # Show original response if requested and different
            if (show_original and 
                response.get("original_response") and 
                response.get("original_response") != response.get("translated_response")):
                
                with st.expander("📝 Original Response (before translation)"):
                    st.markdown(response["original_response"])
    
    # Context information
    if result.get("source_info"):
        st.subheader("📚 Source Information")
        st.info(f"Found {len(result['source_info'])} relevant sections from {result.get('country', 'Unknown')} civil code")
        
        if show_debug:
            with st.expander("🔍 Source Details"):
                for i, source in enumerate(result["source_info"]):
                    st.json(source)
    
    # Debug information
    if show_debug:
        st.subheader("🐛 Debug Information")
        st.json(result)
    
    # Export options
    st.subheader("💾 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Copy as JSON", key=f"copy_{result.get('country', 'result')}"):
            st.code(json.dumps(result, indent=2, ensure_ascii=False), language="json")
    
    with col2:
        json_str = json.dumps(result, indent=2, ensure_ascii=False)
        timestamp = result.get('processing_timestamp', 'unknown').replace(':', '-')
        country = result.get('country', 'query')
        filename = f"civil_code_{country}_{timestamp}.json"
        
        st.download_button(
            label="📁 Download JSON",
            data=json_str,
            file_name=filename,
            mime="application/json",
            key=f"download_{result.get('country', 'result')}"
        )

def setup_page():
    """Setup and configuration page"""
    st.header("⚙️ System Setup")
    
    st.subheader("📋 Configuration Check")
    
    # Check API key
    if config.GOOGLE_API_KEY.startswith("your-"):
        st.error("❌ Gemini API Key Not Configured")
        st.code("""
# Add this to your config.py:
GOOGLE_API_KEY = "your-actual-gemini-api-key-here"

# Or set as environment variable:
export GOOGLE_API_KEY="your-actual-gemini-api-key-here"
        """)
    else:
        st.success("✅ Gemini API Key Configured")
    
    st.subheader("📚 Civil Code Files")
    
    # Check file existence
    for country, info in config.COUNTRIES.items():
        file_path = f"{config.CIVIL_CODES_DIR}/{info['file']}"
        # Note: In actual implementation, you'd check if file exists
        st.info(f"📄 {country}: {info['file']}")
    
    st.subheader("🔧 System Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Initialize All Vector Stores"):
            with st.spinner("Initializing vector stores..."):
                try:
                    doc_processor = DocumentProcessor()
                    doc_processor.initialize_all_vector_stores()
                    st.success("✅ Vector stores initialized!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    with col2:
        if st.button("🧪 Test Translation"):
            with st.spinner("Testing translation..."):
                try:
                    translator = TranslationService()
                    test_text = "What is property law?"
                    translated = translator.translate_text(test_text, "fr")
                    st.success(f"✅ Translation test successful!")
                    st.info(f"EN: {test_text}")
                    st.info(f"FR: {translated}")
                except Exception as e:
                    st.error(f"❌ Translation test failed: {e}")

def help_page():
    """Help and documentation page"""
    st.header("❓ Help & Documentation")
    
    st.subheader("🚀 Getting Started")
    st.markdown("""
    1. **Setup**: Configure your Gemini API key in the config.py file
    2. **Initialize**: Run vector store initialization for all countries
    3. **Query**: Ask questions in any supported language
    4. **Review**: Get responses from Gemini 2.0 Flash with automatic translation
    """)
    
    st.subheader("🌍 Supported Languages & Countries")
    for country, info in config.COUNTRIES.items():
        st.markdown(f"- **{country}** ({info['lang_code']}): {info['file']}")
    
    st.subheader("🤖 AI Model")
    st.markdown("""
    - **Gemini 2.0 Flash**: Google's latest fast, multilingual model
    - **Features**: Advanced reasoning, legal analysis, multilingual understanding
    - **Translation**: Free Google Translate service via googletrans library
    """)
    
    st.subheader("💡 Query Tips")
    st.markdown("""
    - Ask questions in any supported language
    - Be specific about the legal concept you're interested in
    - Include country names for better targeting
    - Use legal terminology when possible
    - The system will automatically translate your query and response as needed
    """)
    
    st.subheader("🔧 Troubleshooting")
    with st.expander("Common Issues"):
        st.markdown("""
        **API Key Errors**: Ensure your Gemini API key is properly configured in config.py
        
        **Missing Files**: Check that all civil code PDF files are in the "./Civil Codes" directory
        
        **Translation Issues**: The googletrans library is free but may have rate limits
        
        **Vector Store Errors**: Try reinitializing vector stores from the setup page
        
        **Slow Responses**: Gemini 2.0 Flash is optimized for speed, but complex queries may take time
        """)

def analytics_page():
    """Analytics and statistics page"""
    st.header("📊 Analytics Dashboard")
    st.info("🚧 Analytics dashboard would be implemented here")
    
    # Placeholder metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", "1,234", "+12%")
    with col2:
        st.metric("Avg Response Time", "1.8s", "-0.3s")
    with col3:
        st.metric("Most Queried Country", "France", "")
    with col4:
        st.metric("Translation Success Rate", "97%", "+1%")
    
    st.subheader("🔥 Popular Queries")
    sample_queries = [
        "What is property law?",
        "Contract formation requirements",
        "Inheritance rights",
        "Tort liability",
        "Consumer protection"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        st.write(f"{i}. {query}")

# Sidebar navigation
def render_sidebar():
    """Render sidebar navigation"""
    st.sidebar.title("Navigation")
    
    pages = {
        "🏠 Home": "home",
        "💬 Query Interface": "query", 
        "⚙️ Setup": "setup",
        "📊 Analytics": "analytics",
        "❓ Help": "help"
    }
    
    selected_page = st.sidebar.radio("Go to:", list(pages.keys()))
    return pages[selected_page]

# Main app logic
if __name__ == "__main__":
    # Render sidebar navigation
    current_page = render_sidebar()
    
    # Route to appropriate page
    if current_page == "home" or current_page == "query":
        main()
    elif current_page == "setup":
        setup_page()
    elif current_page == "analytics":
        analytics_page()
    elif current_page == "help":
        help_page()
    else:
        main()  # Default to main page