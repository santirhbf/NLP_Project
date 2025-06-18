# Updated translator.py using deep-translator (replacing googletrans)
from deep_translator import GoogleTranslator
from langdetect import detect
import time
import logging
from typing import Tuple, Optional
import streamlit as st

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Translator:
    def __init__(self, target_lang='en', max_retries=3):
        self.target_lang = target_lang
        self.max_retries = max_retries

    def detect_language(self, text):
        try:
            return detect(text)
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return None

    def translate_text(self, text, src_lang=None, dest_lang=None):
        if not text.strip():
            return text  # skip empty input

        if dest_lang is None:
            dest_lang = self.target_lang

        retries = 0
        while retries < self.max_retries:
            try:
                if not src_lang:
                    src_lang = self.detect_language(text)

                if src_lang == dest_lang:
                    return text  # No need to translate

                # Use deep-translator
                translator = GoogleTranslator(source=src_lang, target=dest_lang)
                translation = translator.translate(text)
                return translation
                
            except Exception as e:
                wait = 2 ** retries
                logger.warning(f"Translation failed (attempt {retries+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
                retries += 1

        logger.error("Max retries reached. Returning original text.")
        return text

    def translate_list(self, texts, src_lang=None, dest_lang=None):
        return [self.translate_text(text, src_lang, dest_lang) for text in texts]


class TranslationService:
    """Enhanced translation service for the RAG system"""
    
    def __init__(self):
        self.translator = Translator()
        
        # Language mappings
        self.country_to_lang = {
            "France": "fr",
            "Germany": "de", 
            "Italy": "it",
            "Spain": "es",
            "Portugal": "pt"
        }
        
        self.lang_to_country = {
            "fr": "France",
            "de": "Germany",
            "it": "Italy", 
            "es": "Spain",
            "pt": "Portugal"
        }
    
    def detect_language(self, text: str) -> Optional[str]:
        """Detect the language of input text"""
        return self.translator.detect_language(text)
    
    def detect_intended_country(self, text: str) -> Tuple[Optional[str], float]:
        """
        Detect intended country based on language of the query
        
        Returns:
            (country_name, confidence) or (None, 0.0) if cannot determine
        """
        detected_lang = self.detect_language(text)
        
        if not detected_lang:
            return None, 0.0
        
        country = self.lang_to_country.get(detected_lang)
        confidence = 0.8 if country else 0.0  # High confidence for direct language mapping
        
        logger.info(f"Language-based country detection: {detected_lang} -> {country} (confidence: {confidence})")
        return country, confidence
    
    def get_country_language(self, country: str) -> str:
        """Get the language code for a country"""
        return self.country_to_lang.get(country, "en")
    
    def translate_query_for_country(self, query: str, country: str) -> Tuple[str, str]:
        """
        Translate query to the language of the target country's civil code
        
        Args:
            query: Original user query
            country: Target country for civil code search
            
        Returns:
            (translated_query, detected_language)
        """
        detected_lang = self.detect_language(query)
        target_lang = self.get_country_language(country)
        
        logger.info(f"Query language: {detected_lang}, Target country: {country}, Target language: {target_lang}")
        
        if detected_lang == target_lang:
            logger.info("No translation needed - languages match")
            return query, detected_lang
        
        # Translate query to target language
        translated_query = self.translator.translate_text(
            query, 
            src_lang=detected_lang, 
            dest_lang=target_lang
        )
        
        logger.info(f"Query translated: '{query}' -> '{translated_query}'")
        return translated_query, detected_lang
    
    def translate_response_to_original(self, response: str, original_language: str) -> str:
        """
        Translate the RAG response back to the original query language
        
        Args:
            response: Response from RAG system
            original_language: Language code to translate back to
            
        Returns:
            Translated response
        """
        if not original_language or original_language == 'unknown':
            logger.info("Cannot translate response - original language unknown")
            return response
        
        # Detect current language of response
        response_lang = self.detect_language(response)
        
        if response_lang == original_language:
            logger.info("No translation needed - response already in original language")
            return response
        
        # Translate response back to original language
        translated_response = self.translator.translate_text(
            response,
            src_lang=response_lang,
            dest_lang=original_language
        )
        
        logger.info(f"Response translated back to {original_language}")
        return translated_response
    
    def translate_query_and_get_country(self, query: str, force_country: Optional[str] = None) -> dict:
        """
        Complete translation workflow: detect language/country, translate query if needed
        
        Args:
            query: Original user query
            force_country: Force a specific country (optional)
            
        Returns:
            Dictionary with translation information
        """
        original_language = self.detect_language(query)
        
        # Determine target country
        if force_country:
            target_country = force_country
            country_confidence = 1.0
        else:
            target_country, country_confidence = self.detect_intended_country(query)
            if not target_country:
                # Default to France if cannot determine
                target_country = "France"
                country_confidence = 0.1
        
        # Translate query if needed
        translated_query, detected_lang = self.translate_query_for_country(query, target_country)
        
        return {
            "original_query": query,
            "translated_query": translated_query,
            "original_language": original_language,
            "target_country": target_country,
            "target_language": self.get_country_language(target_country),
            "country_confidence": country_confidence,
            "translation_needed": translated_query != query
        }


# Utility functions for Streamlit integration
@st.cache_resource
def load_translation_service():
    """Cached translation service for Streamlit"""
    return TranslationService()

def format_translation_info(translation_info: dict) -> str:
    """Format translation information for display"""
    output = "**Translation Information:**\n"
    output += f"- Original Language: {translation_info['original_language']}\n"
    output += f"- Target Country: {translation_info['target_country']}\n"
    output += f"- Target Language: {translation_info['target_language']}\n"
    output += f"- Country Confidence: {translation_info['country_confidence']:.1%}\n"
    
    if translation_info['translation_needed']:
        output += f"- Query Translation: '{translation_info['original_query']}' → '{translation_info['translated_query']}'\n"
    else:
        output += "- No translation needed\n"
    
    return output


# Example usage
if __name__ == "__main__":
    # Test the enhanced translation service
    service = TranslationService()
    
    test_queries = [
        "Qu'est-ce que le droit de propriété?",  # French
        "Was ist Eigentumsrecht?",               # German
        "What is property law?",                 # English
        "¿Qué es el derecho de propiedad?",      # Spanish
        "Qual è il diritto di proprietà?"       # Italian
    ]
    
    for query in test_queries:
        print(f"\n--- Testing: '{query}' ---")
        
        # Test complete workflow
        result = service.translate_query_and_get_country(query)
        print(f"Original: {result['original_query']}")
        print(f"Language: {result['original_language']}")
        print(f"Country: {result['target_country']} (confidence: {result['country_confidence']:.2f})")
        print(f"Translated: {result['translated_query']}")
        
        # Test response translation back
        sample_response = "Property law governs the ownership and use of real estate and personal property."
        translated_back = service.translate_response_to_original(sample_response, result['original_language'])
        print(f"Response translated back: {translated_back}")