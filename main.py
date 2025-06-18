#!/usr/bin/env python3
"""
European Civil Code RAG System - Simplified Version
Main CLI interface using only Gemini 2.0 Flash and free translation
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any
import logging

from config import config, validate_config
from rag_engine import RAGEngine
from document_parser import DocumentProcessor, get_processing_status
from country_classifier import CountryClassifier
from translator import TranslationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('civil_code_rag.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_system():
    """Initialize and setup the RAG system"""
    print("🏛️  European Civil Code RAG System (Simplified)")
    print("=" * 50)
    
    # Validate configuration
    if not validate_config():
        print("❌ Configuration validation failed. Please check config.py")
        return False
    
    print("✅ Configuration validated")
    
    # Check if civil code files exist
    missing_files = []
    for country, info in config.COUNTRIES.items():
        file_path = os.path.join(config.CIVIL_CODES_DIR, info["file"])
        if not os.path.exists(file_path):
            missing_files.append(f"{country}: {file_path}")
    
    if missing_files:
        print("⚠️  Missing civil code files:")
        for file in missing_files:
            print(f"   - {file}")
        print("Please ensure all PDF files are in the correct directory.")
        return False
    
    print("✅ All civil code files found")
    return True

def initialize_vector_stores(force_recreate: bool = False):
    """Initialize vector stores for all countries"""
    print("\n📚 Initializing document processing...")
    
    doc_processor = DocumentProcessor()
    
    # Check current status
    status = get_processing_status()
    
    if not force_recreate and all(status.values()):
        print("✅ All vector stores already exist")
        return doc_processor
    
    print("🔄 Creating vector stores (this may take a while)...")
    
    try:
        doc_processor.initialize_all_vector_stores(force_recreate)
        print("✅ Vector stores initialized successfully")
        return doc_processor
    except Exception as e:
        print(f"❌ Error initializing vector stores: {e}")
        return None

def test_translation():
    """Test translation functionality"""
    print("\n🌐 Testing translation service...")
    
    service = TranslationService()
    
    test_queries = [
        ("What is property law?", "France"),
        ("Qu'est-ce que le droit de propriété?", "Germany"),
        ("¿Qué es el derecho de propiedad?", "Italy"),
        ("Was ist Eigentumsrecht?", "Spain")
    ]
    
    for query, target_country in test_queries:
        try:
            # Test the actual methods available in TranslationService
            result = service.translate_query_and_get_country(query, target_country)
            
            print(f"✅ Query: '{query}' -> Target: {target_country}")
            print(f"   Original language: {result['original_language']}")
            print(f"   Translated query: '{result['translated_query']}'")
            print(f"   Translation needed: {result['translation_needed']}")
            
            # Test response translation back
            sample_response = "Property law governs ownership rights."
            if result['original_language'] and result['original_language'] != 'en':
                translated_back = service.translate_response_to_original(
                    sample_response, result['original_language']
                )
                print(f"   Response translated back: '{translated_back}'")
            
            print()
            
        except Exception as e:
            print(f"❌ Translation failed for '{query}': {e}")
            
    # Test basic language detection
    print("🔍 Testing language detection...")
    test_detection_queries = [
        "Qu'est-ce que le droit de propriété?",  # French
        "Was ist Eigentumsrecht?",               # German
        "What is property law?",                 # English
        "¿Qué es el derecho de propiedad?",      # Spanish
        "Qual è il diritto di proprietà?"       # Italian
    ]
    
    for query in test_detection_queries:
        try:
            detected_lang = service.detect_language(query)
            country, confidence = service.detect_intended_country(query)
            print(f"✅ '{query}' -> Language: {detected_lang}, Country: {country} ({confidence:.2f})")
        except Exception as e:
            print(f"❌ Detection failed for '{query}': {e}")

def test_country_classification():
    """Test country classification functionality"""
    print("\n🎯 Testing country classification...")
    
    classifier = CountryClassifier()
    
    test_queries = [
        "Qu'est-ce que le droit de propriété en France?",
        "Was ist das deutsche Eigentumsrecht?",
        "Qual è il diritto di proprietà in Italia?",
        "¿Cuál es el derecho de propiedad en España?",
        "Qual é o direito de propriedade em Portugal?"
    ]
    
    for query in test_queries:
        try:
            country, confidence, scores = classifier.classify_country(query)
            print(f"✅ Query: '{query}'")
            print(f"   Predicted: {country} (confidence: {confidence:.2f})")
            print(f"   All scores: {scores}")
        except Exception as e:
            print(f"❌ Classification failed: {e}")

def interactive_query():
    """Interactive query interface"""
    print("\n💬 Interactive Query Mode")
    print("Type 'quit' to exit, 'help' for commands")
    print("-" * 40)
    
    rag_engine = RAGEngine()
    
    while True:
        try:
            query = input("\n🤔 Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'help':
                print_help()
                continue
            
            if not query:
                continue
            
            print("🔍 Processing your query with Gemini 2.0 Flash...")
            
            # Process query
            result = rag_engine.query_with_country_detection(query)
            
            # Display results
            display_query_results(result)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            logger.error(f"Query processing error: {e}", exc_info=True)

def display_query_results(result: Dict[str, Any]):
    """Display query results in a formatted way"""
    print("\n" + "=" * 60)
    print("📋 QUERY RESULTS")
    print("=" * 60)
    
    # Country detection info
    if "country_detection" in result:
        detection = result["country_detection"]
        print(f"🎯 Country Detection:")
        print(f"   Predicted: {detection['predicted_country']}")
        print(f"   Confidence: {detection['confidence']:.2f}")
        print(f"   Strategy: {detection['search_strategy']}")
        
        if detection['search_strategy'] == 'multiple_countries':
            print(f"   Countries searched: {', '.join(detection['searched_countries'])}")
    
    # Results
    if "results_by_country" in result:
        # Multiple countries
        for country, country_result in result["results_by_country"].items():
            print(f"\n🏛️  {country.upper()} RESULTS:")
            display_single_country_result(country_result)
    else:
        # Single country
        display_single_country_result(result)

def display_single_country_result(result: Dict[str, Any]):
    """Display results for a single country"""
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    # Translation info
    if result.get("translated_query") and result.get("translated_query") != result.get("original_query"):
        print(f"🌐 Query translated: {result['original_query']} -> {result['translated_query']}")
    
    # Gemini response
    if "gemini_response" in result:
        response = result["gemini_response"]
        print(f"\n🤖 Gemini 2.0 Flash Response:")
        
        if response.get("error"):
            print(f"❌ Error: {response['error']}")
        else:
            print(f"⏱️  Processing time: {response['processing_time']:.2f}s")
            
            # Show translated response if different from original
            if response.get("translated_response") != response.get("original_response"):
                print(f"📝 Translated Response:\n{response['translated_response']}")
                print(f"\n📝 Original Response:\n{response['original_response']}")
            else:
                print(f"📝 Response:\n{response['translated_response']}")
    
    # Context info
    if result.get("source_info"):
        print(f"\n📚 Sources used: {len(result['source_info'])} document chunks")

def print_help():
    """Print help information"""
    print("""
Available commands:
- Just type your legal question in any supported language
- 'quit' or 'exit' - Exit the program
- 'help' - Show this help

Supported languages:
- French (France)
- German (Germany)  
- Italian (Italy)
- Spanish (Spain)
- Portuguese (Portugal)
- English (queries will be translated to appropriate language)

Example queries:
- "What is property law?"
- "Qu'est-ce que le droit de propriété?"
- "Was ist Eigentumsrecht?"
- "Qual è il diritto di proprietà?"
- "¿Cuál es el derecho de propiedad?"

Note: This simplified version uses only Gemini 2.0 Flash and free translation services.
""")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="European Civil Code RAG System (Simplified)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --setup                 # Initial setup
  python main.py --init-vectors          # Initialize vector stores
  python main.py --test                  # Run all tests
  python main.py --query "What is property law?"  # Single query
  python main.py --interactive           # Interactive mode
        """
    )
    
    parser.add_argument('--setup', action='store_true', 
                       help='Setup and validate system configuration')
    parser.add_argument('--init-vectors', action='store_true',
                       help='Initialize vector stores for all countries')
    parser.add_argument('--force-recreate', action='store_true',
                       help='Force recreate vector stores even if they exist')
    parser.add_argument('--test', action='store_true',
                       help='Run system tests')
    parser.add_argument('--test-translation', action='store_true',
                       help='Test translation functionality')
    parser.add_argument('--test-classification', action='store_true',
                       help='Test country classification')
    parser.add_argument('--query', type=str,
                       help='Single query to process')
    parser.add_argument('--country', type=str,
                       help='Specific country to query (use with --query)')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive query mode')
    parser.add_argument('--output', type=str,
                       help='Output file for results (JSON format)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle different modes
    if args.setup:
        if not setup_system():
            sys.exit(1)
        return
    
    if args.init_vectors:
        if not setup_system():
            sys.exit(1)
        doc_processor = initialize_vector_stores(args.force_recreate)
        if not doc_processor:
            sys.exit(1)
        return
    
    if args.test:
        if not setup_system():
            sys.exit(1)
        test_translation()
        test_country_classification()
        return
    
    if args.test_translation:
        test_translation()
        return
    
    if args.test_classification:
        test_country_classification()
        return
    
    if args.query:
        if not setup_system():
            sys.exit(1)
        
        # Initialize RAG engine
        rag_engine = RAGEngine()
        
        # Process query
        if args.country:
            if args.country not in config.COUNTRIES:
                print(f"❌ Unknown country: {args.country}")
                print(f"Available countries: {', '.join(config.COUNTRIES.keys())}")
                sys.exit(1)
            result = rag_engine.query_single_country(args.query, args.country)
        else:
            result = rag_engine.query_with_country_detection(args.query)
        
        # Output results
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                # Convert complex objects to JSON-serializable format
                serializable_result = convert_to_serializable(result)
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)
            print(f"✅ Results saved to {args.output}")
        else:
            display_query_results(result)
        return
    
    if args.interactive:
        if not setup_system():
            sys.exit(1)
        interactive_query()
        return
    
    # Default: show help
    parser.print_help()

def convert_to_serializable(obj):
    """Convert complex objects to JSON-serializable format"""
    if hasattr(obj, '__dict__'):
        # Convert objects with __dict__ to dictionary
        result = {}
        for key, value in obj.__dict__.items():
            if key.startswith('_'):
                continue  # Skip private attributes
            if callable(value):
                continue  # Skip methods
            result[key] = convert_to_serializable(value)
        return result
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

if __name__ == "__main__":
    main()