import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import time
from datetime import datetime

import google.generativeai as genai
import streamlit as st

from config import config
from document_parser import DocumentProcessor
from country_classifier import CountryClassifier
from translator import TranslationService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiResponse:
    def __init__(self, response: str, processing_time: float, 
                 context_used: str, error: Optional[str] = None):
        self.response = response
        self.processing_time = processing_time
        self.context_used = context_used
        self.error = error
        self.timestamp = datetime.now()

class RAGEngine:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.country_classifier = CountryClassifier()
        self.translator = TranslationService()
        
        # Initialize Gemini client
        self.gemini_client = None
        self._initialize_gemini_client()
    
    def _initialize_gemini_client(self):
        """Initialize Gemini API client"""
        try:
            if config.GOOGLE_API_KEY and not config.GOOGLE_API_KEY.startswith("your-"):
                genai.configure(api_key=config.GOOGLE_API_KEY)
                self.gemini_client = genai.GenerativeModel(config.GEMINI_MODEL)
                logger.info("Gemini 2.0 Flash client initialized")
            else:
                logger.error("Gemini API key not configured")
                
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {e}")
    
    def _create_system_prompt(self, country: str, context: str) -> str:
        """Create a system prompt for Gemini"""
        return f"""You are an expert legal assistant specializing in {country}'s civil code. 
You have been provided with relevant sections from the civil code to answer the user's question.

CONTEXT FROM {country.upper()} CIVIL CODE:
{context}

INSTRUCTIONS:
1. Answer the question based ONLY on the provided context from {country}'s civil code
2. Be precise and cite specific articles or sections when possible
3. If the context doesn't contain enough information to answer the question, say so clearly
4. Provide clear, practical explanations that a non-lawyer could understand
5. When referencing legal articles, use the exact format from the civil code
6. If multiple interpretations are possible, explain the different perspectives

Answer in a clear, structured manner. Focus on accuracy and practical applicability."""
    
    def _query_gemini(self, system_prompt: str, user_query: str) -> GeminiResponse:
        """Query Gemini 2.0 Flash model"""
        start_time = time.time()
        
        try:
            if not self.gemini_client:
                return GeminiResponse(
                    "", 0, "", 
                    "Gemini client not initialized - check API key"
                )
            
            # Combine system prompt and user query for Gemini
            full_prompt = f"{system_prompt}\n\nUSER QUESTION: {user_query}"
            
            response = self.gemini_client.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=config.MAX_TOKENS,
                    temperature=config.TEMPERATURE
                )
            )
            
            processing_time = time.time() - start_time
            answer = response.text
            
            return GeminiResponse(answer, processing_time, system_prompt)
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Gemini query failed: {e}")
            return GeminiResponse("", processing_time, system_prompt, str(e))
    
    def _ensure_vector_stores_ready(self):
        """Ensure all vector stores are initialized"""
        if not self.doc_processor.vector_stores:
            logger.info("Initializing vector stores...")
            self.doc_processor.initialize_all_vector_stores()
    
    def query_single_country(self, query: str, country: str, 
                           translate_query: bool = True,
                           translate_response: bool = True) -> Dict[str, Any]:
        """
        Query a specific country's civil code using Gemini 2.0 Flash
        
        Returns comprehensive results including translations
        """
        self._ensure_vector_stores_ready()
        
        original_query = query
        original_language = self.translator.detect_language(query)
        
        # Step 1: Translate query if needed
        if translate_query:
            translated_query, detected_lang = self.translator.translate_query_for_country(query, country)
            query_for_search = translated_query
            original_language = detected_lang
        else:
            query_for_search = query
        
        # Step 2: Get relevant context from civil code
        # Replace the existing try-catch block in rag_engine.py (around line 98-108) with this enhanced version:

# Step 2: Get relevant context from civil code
        try:
            context, source_info = self.doc_processor.get_relevant_context(country, query_for_search)
        except FileNotFoundError as e:
            logger.error(f"Civil code file not found for {country}: {e}")
            return {
                "country": country,
                "error": f"Civil code file for {country} not found",
                "user_message": f"The PDF file for {country}'s civil code is missing. Please ensure the file is in the './Civil Codes/' directory.",
                "technical_error": str(e),
                "original_query": original_query,
                "translated_query": query_for_search if translate_query else None
            }
        except ValueError as e:
            logger.error(f"Document processing error for {country}: {e}")
            return {
                "country": country,
                "error": f"Document processing failed for {country}",
                "user_message": f"There was an issue processing {country}'s civil code. The document may be corrupted or in an unsupported format.",
                "technical_error": str(e),
                "original_query": original_query,
                "translated_query": query_for_search if translate_query else None
            }
        except Exception as e:
            logger.error(f"Unexpected error getting context for {country}: {e}")
            return {
                "country": country,
                "error": f"Could not retrieve context from {country}'s civil code",
                "user_message": f"An unexpected error occurred while searching {country}'s civil code. Please try again or contact support if the issue persists.",
                "technical_error": str(e),
                "original_query": original_query,
                "translated_query": query_for_search if translate_query else None
            }

        # Keep the existing check for empty context, but enhance it:
        if not context:
            return {
                "country": country,
                "error": "No relevant context found in civil code",
                "user_message": f"No relevant sections were found in {country}'s civil code for your question. Try rephrasing your query with more specific legal terms or check if this topic is covered in {country}'s civil law.",
                "suggestion": f"Consider asking about: property law, contract law, tort law, family law, or inheritance law in {country}",
                "original_query": original_query,
                "translated_query": query_for_search if translate_query else None
            }
        
        # Step 3: Create system prompt
        system_prompt = self._create_system_prompt(country, context)
        
        # Step 4: Query Gemini
        gemini_response = self._query_gemini(system_prompt, query_for_search)
        
        # Check if Gemini query was successful
        if gemini_response.error:
            return {
                "country": country,
                "error": f"Gemini query failed: {gemini_response.error}",
                "original_query": original_query,
                "translated_query": query_for_search if translate_query else None
            }
        
        # Step 5: Translate response back if needed
        translated_response = gemini_response.response
        if translate_response and original_language and original_language != 'unknown':
            translated_response = self.translator.translate_response_to_original(
                gemini_response.response, original_language
            )
        
        # Step 6: Compile results
        result = {
            "country": country,
            "original_query": original_query,
            "translated_query": query_for_search if translate_query else None,
            "original_language": original_language,
            "context": context,
            "source_info": source_info,
            "gemini_response": {
                "original_response": gemini_response.response,
                "translated_response": translated_response,
                "processing_time": gemini_response.processing_time,
                "error": gemini_response.error
            },
            "processing_timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def query_with_country_detection(self, query: str) -> Dict[str, Any]:
        """
        Main query function with automatic country detection
        """
        # Step 1: Classify country
        predicted_country, confidence, all_scores = self.country_classifier.classify_country(query)
        
        logger.info(f"Country detection: {predicted_country} (confidence: {confidence:.2f})")
        
        # Step 2: Decide whether to query single or multiple countries
        if confidence >= config.COUNTRY_DETECTION_THRESHOLD:
            # High confidence - query single country
            result = self.query_single_country(query, predicted_country)
            result["country_detection"] = {
                "predicted_country": predicted_country,
                "confidence": confidence,
                "all_scores": all_scores,
                "search_strategy": "single_country"
            }
        else:
            # Low confidence - query top countries
            relevant_countries = self.country_classifier.get_relevant_countries(query)[:2]
            
            results = {}
            for country, score in relevant_countries:
                country_result = self.query_single_country(query, country)
                results[country] = country_result
            
            result = {
                "original_query": query,
                "country_detection": {
                    "predicted_country": predicted_country,
                    "confidence": confidence,
                    "all_scores": all_scores,
                    "search_strategy": "multiple_countries",
                    "searched_countries": [c for c, _ in relevant_countries]
                },
                "results_by_country": results,
                "processing_timestamp": datetime.now().isoformat()
            }
        
        return result

# Utility functions for Streamlit integration
@st.cache_resource
def load_rag_engine():
    """Cached RAG engine for Streamlit"""
    return RAGEngine()

def format_gemini_response(response: Dict[str, Any]) -> str:
    """Format Gemini response for display"""
    if response.get("error"):
        return f"**Gemini 2.0 Flash** ❌\n*Error: {response['error']}*\n"
    
    processing_time = response.get("processing_time", 0)
    translated_response = response.get("translated_response", "")
    original_response = response.get("original_response", "")
    
    output = f"**Gemini 2.0 Flash** ✅\n"
    output += f"*Processing time: {processing_time:.2f}s*\n\n"
    
    if translated_response != original_response:
        output += f"**Translated Response:**\n{translated_response}\n\n"
        output += f"**Original Response:**\n{original_response}\n"
    else:
        output += f"{translated_response}\n"
    
    return output + "\n---\n"