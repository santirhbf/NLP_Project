import os
import pickle
from typing import List, Dict, Tuple
from pathlib import Path
import logging

import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Use sklearn for stable, memory-efficient embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import streamlit as st

from config import config, get_country_file_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TfidfEmbeddings:
    """Memory-efficient TF-IDF based embeddings"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True,
            max_df=0.95,
            min_df=2
        )
        self.fitted = False
        self.document_vectors = None
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit the vectorizer and transform texts"""
        logger.info(f"Fitting TF-IDF on {len(texts)} documents...")
        self.document_vectors = self.vectorizer.fit_transform(texts)
        self.fitted = True
        logger.info(f"TF-IDF fitted with {len(self.vectorizer.vocabulary_)} features")
        return self.document_vectors
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts using fitted vectorizer"""
        if not self.fitted:
            raise ValueError("Must fit vectorizer first")
        return self.vectorizer.transform(texts)
    
    def transform_query(self, query: str) -> np.ndarray:
        """Transform a single query"""
        return self.transform([query])

class SimpleVectorStore:
    """Simple, memory-efficient vector store using TF-IDF"""
    
    def __init__(self, documents: List[Document], embeddings: TfidfEmbeddings):
        self.documents = documents
        self.embeddings = embeddings
        
        # Extract texts and create embeddings
        texts = [doc.page_content for doc in documents]
        logger.info(f"Creating vector store with {len(texts)} documents...")
        
        # Fit and transform in one step for memory efficiency
        self.doc_vectors = embeddings.fit_transform(texts)
        
        logger.info("Vector store created successfully")
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents"""
        # Transform query
        query_vector = self.embeddings.transform_query(query)
        
        # Calculate similarities (use sparse matrix operations for efficiency)
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return top documents
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:  # Minimum similarity threshold
                results.append(self.documents[idx])
        
        return results if results else [self.documents[0]]  # Return at least one document

class DocumentProcessor:
    def __init__(self):
        # Always use TF-IDF for stability and memory efficiency
        self.embeddings = TfidfEmbeddings()
        logger.info("Using TF-IDF embeddings for stable, memory-efficient processing")
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        self.vector_stores: Dict[str, SimpleVectorStore] = {}
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                total_pages = len(pdf_reader.pages)
                logger.info(f"Extracting text from {total_pages} pages...")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                        
                        # Log progress every 50 pages
                        if (page_num + 1) % 50 == 0:
                            logger.info(f"Processed {page_num + 1}/{total_pages} pages")
                            
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1}: {e}")
                        continue
                        
                logger.info(f"Extracted {len(text)} characters total")
                return text.strip()
                
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            raise
    
    def process_civil_code(self, country: str) -> List[Document]:
        """Process a country's civil code into chunks"""
        pdf_path = get_country_file_path(country)
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Civil code file not found: {pdf_path}")
        
        logger.info(f"Processing civil code for {country}...")
        
        # Extract text
        raw_text = self.extract_text_from_pdf(pdf_path)
        
        if not raw_text or len(raw_text.strip()) < 100:
            raise ValueError(f"Insufficient text extracted from {pdf_path}")
        
        # Create documents with metadata
        documents = []
        chunks = self.text_splitter.split_text(raw_text)
        
        logger.info(f"Split text into {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # Skip very small chunks
                continue
                
            doc = Document(
                page_content=chunk,
                metadata={
                    "country": country,
                    "language": config.COUNTRIES[country]["lang_code"],
                    "source": pdf_path,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} valid chunks for {country}")
        return documents
    
    def create_vector_store(self, country: str, force_recreate: bool = False) -> SimpleVectorStore:
        """Create or load vector store for a country"""
        vector_store_path = f"{config.VECTOR_DB_PATH}/{country.lower()}_vectorstore"
        
        # Check if vector store already exists
        if not force_recreate and os.path.exists(f"{vector_store_path}.pkl"):
            logger.info(f"Loading existing vector store for {country}...")
            try:
                with open(f"{vector_store_path}.pkl", 'rb') as f:
                    vector_store = pickle.load(f)
                self.vector_stores[country] = vector_store
                return vector_store
            except Exception as e:
                logger.warning(f"Error loading vector store for {country}: {e}")
                logger.info("Creating new vector store...")
        
        # Create new vector store
        documents = self.process_civil_code(country)
        
        if not documents:
            raise ValueError(f"No documents created for {country}")
        
        logger.info(f"Creating vector store for {country}...")
        
        # Create a fresh embeddings instance for each country to avoid memory issues
        country_embeddings = TfidfEmbeddings()
        vector_store = SimpleVectorStore(documents, country_embeddings)
        
        # Save vector store
        os.makedirs(config.VECTOR_DB_PATH, exist_ok=True)
        with open(f"{vector_store_path}.pkl", 'wb') as f:
            pickle.dump(vector_store, f)
        
        self.vector_stores[country] = vector_store
        logger.info(f"Vector store created and saved for {country}")
        
        return vector_store
    
    def initialize_all_vector_stores(self, force_recreate: bool = False) -> Dict[str, SimpleVectorStore]:
        """Initialize vector stores for all countries"""
        logger.info("Initializing vector stores for all countries...")
        
        for country in config.COUNTRIES.keys():
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Processing {country}...")
                logger.info(f"{'='*50}")
                
                self.create_vector_store(country, force_recreate)
                
                logger.info(f"✅ {country} completed successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to create vector store for {country}: {e}")
                continue
        
        logger.info(f"\n✅ Vector store initialization complete!")
        logger.info(f"Successfully processed {len(self.vector_stores)} countries")
        return self.vector_stores
    
    def search_similar_documents(self, country: str, query: str, k: int = None) -> List[Document]:
        """Search for similar documents in a country's vector store"""
        if k is None:
            k = config.MAX_RESULTS
            
        if country not in self.vector_stores:
            logger.info(f"Vector store not loaded for {country}, creating...")
            self.create_vector_store(country)
        
        vector_store = self.vector_stores[country]
        similar_docs = vector_store.similarity_search(query, k=k)
        
        return similar_docs
    
    def get_relevant_context(self, country: str, query: str) -> Tuple[str, List[Dict]]:
        """Get relevant context for a query from a country's civil code"""
        similar_docs = self.search_similar_documents(country, query)
        
        context_parts = []
        source_info = []
        
        for doc in similar_docs:
            context_parts.append(doc.page_content)
            source_info.append({
                "country": doc.metadata.get("country"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "source": doc.metadata.get("source")
            })
        
        context = "\n\n---\n\n".join(context_parts)
        return context, source_info

# Utility functions for Streamlit integration
@st.cache_resource
def load_document_processor():
    """Cached document processor for Streamlit"""
    return DocumentProcessor()

def get_processing_status() -> Dict[str, bool]:
    """Check which countries have processed vector stores"""
    status = {}
    
    if not os.path.exists(config.VECTOR_DB_PATH):
        return {country: False for country in config.COUNTRIES.keys()}
    
    for country in config.COUNTRIES.keys():
        vector_store_path = f"{config.VECTOR_DB_PATH}/{country.lower()}_vectorstore.pkl"
        status[country] = os.path.exists(vector_store_path)
    
    return status