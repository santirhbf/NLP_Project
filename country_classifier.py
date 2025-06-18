import logging
from typing import Dict, List, Tuple
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st

from config import config
from translator import TranslationService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CountryClassifier:
    def __init__(self):
        self.translator = TranslationService()
        
        # Keywords and phrases associated with each country
        self.country_keywords = {
            "France": [
                # French language indicators
                "france", "français", "française", "droit français", "code civil français",
                "article", "alinéa", "paragraphe", "section", "chapitre",
                "qu'est-ce que", "comment", "pourquoi", "où", "quand",
                "propriété", "contrat", "responsabilité", "obligation", "succession",
                "mariage", "divorce", "famille", "patrimoine", "bien", "meuble", "immeuble",
                # French legal terms
                "usufruit", "servitude", "hypothèque", "gage", "nantissement",
                "tutelle", "curatelle", "émancipation", "filiation", "adoption"
            ],
            
            "Germany": [
                # German language indicators
                "deutschland", "deutsch", "deutsche", "deutsches", "bürgerliches gesetzbuch",
                "bgb", "paragraph", "absatz", "satz", "nummer", "ziffer",
                "was ist", "wie", "warum", "wo", "wann",
                "eigentum", "vertrag", "haftung", "verpflichtung", "erbschaft",
                "ehe", "scheidung", "familie", "vermögen", "sache", "beweglich", "unbeweglich",
                # German legal terms
                "nießbrauch", "dienstbarkeit", "hypothek", "pfandrecht", "sicherung",
                "vormundschaft", "betreuung", "volljährigkeit", "abstammung", "adoption"
            ],
            
            "Italy": [
                # Italian language indicators
                "italia", "italiano", "italiana", "diritto italiano", "codice civile italiano",
                "articolo", "comma", "paragrafo", "sezione", "capitolo",
                "che cosa è", "come", "perché", "dove", "quando",
                "proprietà", "contratto", "responsabilità", "obbligo", "successione",
                "matrimonio", "divorzio", "famiglia", "patrimonio", "bene", "mobile", "immobile",
                # Italian legal terms
                "usufrutto", "servitù", "ipoteca", "pegno", "garanzia",
                "tutela", "curatela", "emancipazione", "filiazione", "adozione"
            ],
            
            "Spain": [
                # Spanish language indicators
                "españa", "español", "española", "derecho español", "código civil español",
                "artículo", "párrafo", "apartado", "sección", "capítulo",
                "qué es", "cómo", "por qué", "dónde", "cuándo",
                "propiedad", "contrato", "responsabilidad", "obligación", "sucesión",
                "matrimonio", "divorcio", "familia", "patrimonio", "bien", "mueble", "inmueble",
                # Spanish legal terms
                "usufructo", "servidumbre", "hipoteca", "prenda", "garantía",
                "tutela", "curatela", "emancipación", "filiación", "adopción"
            ],
            
            "Portugal": [
                # Portuguese language indicators
                "portugal", "português", "portuguesa", "direito português", "código civil português",
                "artigo", "parágrafo", "alínea", "secção", "capítulo",
                "o que é", "como", "por que", "onde", "quando",
                "propriedade", "contrato", "responsabilidade", "obrigação", "sucessão",
                "casamento", "divórcio", "família", "património", "bem", "móvel", "imóvel",
                # Portuguese legal terms
                "usufruto", "servidão", "hipoteca", "penhor", "garantia",
                "tutela", "curatela", "emancipação", "filiação", "adoção"
            ]
        }
        
        # Compile country name patterns for direct detection
        self.country_patterns = {
            "France": [r'\bfrance\b', r'\bfrancês\b', r'\bfrançais\b', r'\bfrancesa\b'],
            "Germany": [r'\bgermany\b', r'\bdeutschland\b', r'\balemanha\b', r'\ballemagne\b', r'\bgerman\b'],
            "Italy": [r'\bitaly\b', r'\bitalia\b', r'\bitalian\b', r'\bitaliano\b'],
            "Spain": [r'\bspain\b', r'\bespaña\b', r'\bespanha\b', r'\bespagne\b', r'\bspanish\b'],
            "Portugal": [r'\bportugal\b', r'\bportuguês\b', r'\bportuguese\b']
        }
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=None,  # We'll handle stopwords manually for multilingual support
            ngram_range=(1, 2),  # Include bigrams
            max_features=5000
        )
        
        # Build country profiles
        self._build_country_profiles()
    
    def _build_country_profiles(self):
        """Build TF-IDF profiles for each country"""
        # Create documents from country keywords
        country_documents = []
        self.country_names = []
        
        for country, keywords in self.country_keywords.items():
            # Join keywords to create a document
            document = " ".join(keywords)
            country_documents.append(document)
            self.country_names.append(country)
        
        # Fit vectorizer on country documents
        self.country_vectors = self.vectorizer.fit_transform(country_documents)
        
        logger.info("Country classification profiles built successfully")
    
    def _direct_country_detection(self, query: str) -> Tuple[str, float]:
        """
        Check for direct country mentions in the query
        
        Returns:
            (country_name, confidence) or (None, 0.0) if no direct mention
        """
        query_lower = query.lower()
        
        for country, patterns in self.country_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    logger.info(f"Direct country detection: {country} found in query")
                    return country, 1.0
        
        return None, 0.0
    
    def _language_based_detection(self, query: str) -> Tuple[str, float]:
        """
        Detect country based on language of the query
        
        Returns:
            (country_name, confidence)
        """
        detected_lang = self.translator.detect_language(query)
        
        if not detected_lang:
            return None, 0.0
        
        # Map languages to countries
        lang_to_country = {
            'fr': 'France',
            'de': 'Germany', 
            'it': 'Italy',
            'es': 'Spain',
            'pt': 'Portugal',
            'en': None  # English doesn't map to a specific country
        }
        
        country = lang_to_country.get(detected_lang)
        if country:
            confidence = 0.7  # High confidence for language-based detection
            logger.info(f"Language-based detection: {country} (language: {detected_lang})")
            return country, confidence
        
        return None, 0.0
    
    def _content_based_detection(self, query: str) -> Tuple[str, float]:
        """
        Detect country based on query content using TF-IDF similarity
        
        Returns:
            (country_name, confidence)
        """
        try:
            # Transform query to vector
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities with each country
            similarities = cosine_similarity(query_vector, self.country_vectors)[0]
            
            # Find best match
            best_idx = np.argmax(similarities)
            best_country = self.country_names[best_idx]
            best_score = similarities[best_idx]
            
            logger.info(f"Content-based detection: {best_country} (score: {best_score:.3f})")
            return best_country, best_score
            
        except Exception as e:
            logger.error(f"Content-based detection failed: {e}")
            return None, 0.0
    
    def classify_country(self, query: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify which country's civil code is most relevant to the query
        
        Args:
            query: User's legal query
            
        Returns:
            (predicted_country, confidence, all_country_scores)
        """
        if not query or len(query.strip()) < 3:
            return "France", 0.0, {}  # Default fallback
        
        # Try different detection methods in order of reliability
        
        # 1. Direct country mention (highest priority)
        country, confidence = self._direct_country_detection(query)
        if country and confidence > 0.8:
            all_scores = {country: confidence}
            for c in self.country_names:
                if c != country:
                    all_scores[c] = 0.0
            return country, confidence, all_scores
        
        # 2. Language-based detection
        lang_country, lang_confidence = self._language_based_detection(query)
        
        # 3. Content-based detection
        content_country, content_confidence = self._content_based_detection(query)
        
        # Combine scores (weighted)
        final_scores = {}
        
        for country_name in self.country_names:
            score = 0.0
            
            # Language-based score (weight: 0.6)
            if lang_country == country_name:
                score += lang_confidence * 0.6
            
            # Content-based score (weight: 0.4)
            if content_country == country_name:
                score += content_confidence * 0.4
            
            final_scores[country_name] = score
        
        # Find best overall match
        if final_scores:
            best_country = max(final_scores, key=final_scores.get)
            best_confidence = final_scores[best_country]
        else:
            best_country = "France"  # Default fallback
            best_confidence = 0.0
            final_scores = {c: 0.0 for c in self.country_names}
        
        logger.info(f"Final classification: {best_country} (confidence: {best_confidence:.3f})")
        return best_country, best_confidence, final_scores
    
    def get_relevant_countries(self, query: str, threshold: float = 0.1) -> List[Tuple[str, float]]:
        """
        Get all countries that might be relevant to the query
        
        Args:
            query: User's legal query
            threshold: Minimum confidence threshold
            
        Returns:
            List of (country, confidence) tuples, sorted by confidence
        """
        _, _, all_scores = self.classify_country(query)
        
        # Filter and sort by confidence
        relevant = [(country, score) for country, score in all_scores.items() 
                   if score >= threshold]
        relevant.sort(key=lambda x: x[1], reverse=True)
        
        # Ensure we always return at least one country
        if not relevant:
            predicted_country, confidence, _ = self.classify_country(query)
            relevant = [(predicted_country, confidence)]
        
        return relevant
    
    def explain_classification(self, query: str) -> Dict[str, str]:
        """
        Provide explanation for why a particular country was chosen
        
        Returns:
            Dictionary with explanation details
        """
        country, confidence, all_scores = self.classify_country(query)
        
        explanation = {
            "predicted_country": country,
            "confidence": f"{confidence:.3f}",
            "reasoning": []
        }
        
        # Check for direct mentions
        direct_country, direct_conf = self._direct_country_detection(query)
        if direct_country:
            explanation["reasoning"].append(f"Direct mention of {direct_country} found in query")
        
        # Check language
        detected_lang = self.translator.detect_language(query)
        if detected_lang:
            explanation["reasoning"].append(f"Query language detected as: {detected_lang}")
        
        # Content analysis
        explanation["reasoning"].append(f"Content analysis favored: {country}")
        
        explanation["all_scores"] = {k: f"{v:.3f}" for k, v in all_scores.items()}
        
        return explanation

# Utility functions for Streamlit integration
@st.cache_resource
def load_country_classifier():
    """Cached country classifier for Streamlit"""
    return CountryClassifier()

def format_classification_results(country: str, confidence: float, 
                                all_scores: Dict[str, float]) -> str:
    """Format classification results for display"""
    result = f"**Predicted Country:** {country}\n"
    result += f"**Confidence:** {confidence:.1%}\n\n"
    result += "**All Country Scores:**\n"
    
    # Sort by score
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    
    for country_name, score in sorted_scores:
        result += f"- {country_name}: {score:.1%}\n"
    
    return result