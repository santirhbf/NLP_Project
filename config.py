import os
from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path

@dataclass
class Config:
    # API Keys - Only Gemini 2.0 Flash needed
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "AIzaSyA9aIFmIQrov6ls9FrAS0YvGG7wev5OMIY")
    
    # File paths
    CIVIL_CODES_DIR: str = "./Civil Codes"
    VECTOR_DB_PATH: str = "./vector_db"
    
    # Countries and their language codes
    COUNTRIES: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "France": {"lang_code": "fr", "file": "Civil Code - France.pdf"},
        "Germany": {"lang_code": "de", "file": "Civil Code - Germany.pdf"},
        "Italy": {"lang_code": "it", "file": "Civil Code - Italy.pdf"},
        "Spain": {"lang_code": "es", "file": "Civil Code - Spain.pdf"},
        "Portugal": {"lang_code": "pt", "file": "Civil Code - Portugal.pdf"}
    })
    
    # Model configurations
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # Gemini Model
    GEMINI_MODEL: str = "gemini-2.0-flash-exp"  # Updated to 2.0 Flash
    
    # Processing parameters
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_TOKENS: int = 4000
    TEMPERATURE: float = 0.1
    
    # Vector DB settings
    SIMILARITY_THRESHOLD: float = 0.7
    MAX_RESULTS: int = 5
    
    # Language detection confidence threshold
    COUNTRY_DETECTION_THRESHOLD: float = 0.6

# Global config instance
config = Config()

def validate_config() -> bool:
    """Validate that required configurations are set"""
    if config.GOOGLE_API_KEY.startswith("your-"):
        print("⚠️  Warning: Gemini API key is not configured")
        return False
    
    # Check if civil codes directory exists
    if not Path(config.CIVIL_CODES_DIR).exists():
        print(f"⚠️  Warning: Civil codes directory '{config.CIVIL_CODES_DIR}' not found")
        return False
    
    return True

def get_country_file_path(country: str) -> str:
    """Get the full file path for a country's civil code"""
    if country not in config.COUNTRIES:
        raise ValueError(f"Country '{country}' not supported")
    
    filename = config.COUNTRIES[country]["file"]
    return os.path.join(config.CIVIL_CODES_DIR, filename)

def get_language_code(country: str) -> str:
    """Get the language code for a country"""
    if country not in config.COUNTRIES:
        raise ValueError(f"Country '{country}' not supported")
    
    return config.COUNTRIES[country]["lang_code"]