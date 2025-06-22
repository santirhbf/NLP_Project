#!/usr/bin/env python3
"""
Quick Civil Code Converter - Test Version
Extracts and structures the Spanish Civil Code in ~30 minutes
"""

import re
import json
import fitz  # pip install PyMuPDF
from datetime import datetime

def quick_convert_civil_code(pdf_path: str, output_path: str = "codigo_civil_spain.json"):
    """
    Quick conversion of Spanish Civil Code PDF to JSON
    Should complete in 30-60 minutes for 500 pages
    """
    
    print(f"🚀 Starting quick conversion of {pdf_path}")
    start_time = datetime.now()
    
    # Step 1: Extract text (2-5 minutes)
    print("📄 Extracting text from PDF...")
    doc = fitz.open(pdf_path)
    
    full_text = ""
    for page_num in range(len(doc)):
        if page_num % 50 == 0:
            print(f"   Processing page {page_num}/{len(doc)}")
        
        page = doc[page_num]
        text = page.get_text()
        
        # Skip if page doesn't contain legal content
        if page_num > 15 and ('Artículo' in text or 'TÍTULO' in text or 'CAPÍTULO' in text):
            full_text += text + "\n"
    
    doc.close()
    print(f"✅ Text extraction complete. Extracted {len(full_text)} characters")
    
    # Step 2: Quick clean (1-2 minutes)
    print("🧹 Quick text cleaning...")
    full_text = re.sub(r'\n\d+\n', '\n', full_text)  # Remove page numbers
    full_text = re.sub(r'Código Civil.*?\n', '', full_text)  # Remove headers
    full_text = full_text.replace('fliación', 'filiación')  # Common OCR fixes
    full_text = full_text.replace('califcación', 'calificación')
    
    # Step 3: Extract articles (10-20 minutes)  
    print("📋 Extracting articles...")
    articles = extract_articles_quick(full_text)
    print(f"✅ Found {len(articles)} articles")
    
    # Step 4: Basic structure (5-10 minutes)
    print("🏗️ Building structure...")
    structured_data = build_structure_quick(full_text, articles)
    
    # Step 5: Add domains (5-10 minutes)
    print("🏷️ Adding domain classifications...")
    add_domains_quick(structured_data)
    
    # Step 6: Save JSON (1 minute)
    print(f"💾 Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(structured_data, f, ensure_ascii=False, indent=2)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"🎉 Conversion complete in {duration}")
    print(f"📊 Results: {len(articles)} articles processed")
    print(f"📁 Output: {output_path}")
    
    return structured_data

def extract_articles_quick(text: str) -> list:
    """Extract all articles quickly using regex"""
    articles = []
    
    # Find all article patterns
    article_pattern = r'Artículo\s+(\d+)\.?\s*(.*?)(?=Artículo\s+\d+|TÍTULO|CAPÍTULO|LIBRO|$)'
    matches = re.finditer(article_pattern, text, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        article_num = match.group(1)
        content = match.group(2).strip()
        
        # Split content into subsections
        subsections = []
        paragraphs = []
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check for numbered subsections (1., 2., etc.)
            if num_match := re.match(r'(\d+)\.?\s+(.*)', line):
                subsections.append({
                    "number": num_match.group(1),
                    "content": num_match.group(2)
                })
            # Check for lettered subsections (a), b), etc.)
            elif letter_match := re.match(r'([a-z])\)\s+(.*)', line):
                subsections.append({
                    "letter": letter_match.group(1),
                    "content": letter_match.group(2)
                })
            else:
                paragraphs.append(line)
        
        articles.append({
            "number": article_num,
            "content": " ".join(paragraphs),
            "subsections": subsections,
            "domains": []  # Will be filled later
        })
    
    return articles

def build_structure_quick(text: str, articles: list) -> dict:
    """Build basic hierarchical structure"""
    
    structure = {
        "metadata": {
            "title": "Código Civil de España",
            "extraction_date": datetime.now().isoformat(),
            "total_articles": len(articles),
            "source": "Official BOE PDF"
        },
        "preliminary_title": {
            "articles": []
        },
        "books": []
    }
    
    # Extract main sections
    books_pattern = r'LIBRO\s+([A-Z]+)\.?\s*(.*?)(?=LIBRO|$)'
    book_matches = re.finditer(books_pattern, text, re.DOTALL | re.IGNORECASE)
    
    for book_match in book_matches:
        book_name = book_match.group(1)
        book_title = book_match.group(2).split('\n')[0].strip()
        
        book = {
            "name": book_name,
            "title": book_title,
            "articles": []
        }
        
        # Find articles in this book (simplified)
        book_content = book_match.group(2)
        book_article_nums = re.findall(r'Artículo\s+(\d+)', book_content)
        
        for article in articles:
            if article["number"] in book_article_nums:
                book["articles"].append(article)
        
        structure["books"].append(book)
    
    # Handle preliminary title articles (Articles 1-16)
    for article in articles:
        if int(article["number"]) <= 16:
            structure["preliminary_title"]["articles"].append(article)
    
    return structure

def add_domains_quick(structure: dict):
    """Add quick domain classification"""
    
    domain_keywords = {
        "Fuentes del Derecho": ["fuentes", "ley", "costumbre", "principios generales"],
        "Derecho de Familia": ["matrimonio", "cónyuge", "divorcio", "separación", "filiación", "adopción", "patria potestad", "familia"],
        "Derecho de Propiedad": ["propiedad", "posesión", "dominio", "usufructo", "servidumbre", "bienes"],
        "Contratos y Obligaciones": ["contrato", "obligación", "deuda", "acreedor", "deudor", "compraventa", "arrendamiento"],
        "Derecho Sucesorio": ["herencia", "testamento", "legado", "sucesión", "heredero"],
        "Personas y Capacidad": ["personalidad", "capacidad", "menor", "tutela", "curatela"],
        "Registro Civil": ["registro civil", "inscripción", "nacionalidad"],
        "Derecho Internacional Privado": ["internacional", "extranjero", "tratados"]
    }
    
    def classify_article(article):
        text = f"{article.get('content', '')} {' '.join([s.get('content', '') for s in article.get('subsections', [])])}"
        text = text.lower()
        
        domains = []
        for domain, keywords in domain_keywords.items():
            if any(keyword in text for keyword in keywords):
                domains.append(domain)
        
        return domains if domains else ["General"]
    
    # Classify preliminary title articles
    for article in structure["preliminary_title"]["articles"]:
        article["domains"] = classify_article(article)
    
    # Classify book articles
    for book in structure["books"]:
        for article in book["articles"]:
            article["domains"] = classify_article(article)

# Quick test function
def quick_test():
    """Test with a sample file"""
    import os
    
    # Look for PDF in current directory
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    
    if pdf_files:
        pdf_path = pdf_files[0]
        print(f"🔍 Found PDF: {pdf_path}")
        
        try:
            result = quick_convert_civil_code(pdf_path)
            print(f"✅ Success! Found {len(result.get('books', []))} books")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    else:
        print("❌ No PDF file found in current directory")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "codigo_civil_spain.json"
        quick_convert_civil_code(pdf_path, output_path)
    else:
        print("🧪 Running quick test...")
        quick_test()