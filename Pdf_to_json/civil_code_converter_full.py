#!/usr/bin/env python3
"""
Spanish Civil Code PDF to JSON Converter
Converts the Código Civil de España from PDF to structured JSON format
"""

import re
import json
import PyPDF2
from typing import Dict, List, Any, Optional
import fitz  # PyMuPDF - better for text extraction
from dataclasses import dataclass

@dataclass
class Article:
    """Represents a single article in the Civil Code"""
    number: str
    title: Optional[str]
    content: str
    subsections: List[Dict[str, str]]
    
class CivilCodeConverter:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.structure = {
            "metadata": {
                "title": "Código Civil de España",
                "last_updated": "2025-01-03",
                "source": "BOE - Boletín Oficial del Estado"
            },
            "books": []
        }
        
    def extract_text_from_pdf(self) -> str:
        """Extract text from PDF using PyMuPDF for better accuracy"""
        doc = fitz.open(self.pdf_path)
        full_text = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Skip cover pages and index pages
            if self._is_content_page(text, page_num):
                full_text += text + "\n"
                
        doc.close()
        return full_text
    
    def _is_content_page(self, text: str, page_num: int) -> bool:
        """Determine if page contains actual legal content"""
        # Skip first few pages (cover, index)
        if page_num < 10:
            return False
            
        # Look for article patterns
        has_articles = bool(re.search(r'Artículo \d+', text))
        has_legal_content = any(keyword in text for keyword in 
                               ['TÍTULO', 'CAPÍTULO', 'LIBRO', 'Artículo'])
        
        return has_articles or has_legal_content
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text from common PDF artifacts"""
        # Remove page numbers and headers
        text = re.sub(r'\n\d+\n', '\n', text)
        text = re.sub(r'CÓDIGO CIVIL.*?\n', '', text)
        
        # Fix common OCR issues
        text = text.replace('fliación', 'filiación')
        text = text.replace('fnalidad', 'finalidad') 
        text = text.replace('califcación', 'calificación')
        text = text.replace('ofcio', 'oficio')
        text = text.replace('Ofcial', 'Oficial')
        
        # Normalize spacing
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def parse_structure(self, text: str) -> None:
        """Parse the hierarchical structure of the Civil Code"""
        lines = text.split('\n')
        current_book = None
        current_title = None
        current_chapter = None
        current_section = None
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
                
            # Parse LIBRO (Book)
            if book_match := re.match(r'LIBRO\s+([A-Z]+)\.?\s*(.*)', line):
                current_book = {
                    "number": book_match.group(1),
                    "title": book_match.group(2).strip(),
                    "titles": []
                }
                self.structure["books"].append(current_book)
                current_title = None
                current_chapter = None
                
            # Parse TÍTULO (Title) 
            elif title_match := re.match(r'TÍTULO\s+([IVX]+)\.?\s*(.*)', line):
                if current_book:
                    current_title = {
                        "number": title_match.group(1),
                        "title": title_match.group(2).strip(),
                        "chapters": []
                    }
                    current_book["titles"].append(current_title)
                    current_chapter = None
                    
            # Parse CAPÍTULO (Chapter)
            elif chapter_match := re.match(r'CAPÍTULO\s+([IVX]+)\.?\s*(.*)', line):
                if current_title:
                    current_chapter = {
                        "number": chapter_match.group(1),
                        "title": chapter_match.group(2).strip(),
                        "sections": [],
                        "articles": []
                    }
                    current_title["chapters"].append(current_chapter)
                    
            # Parse Article
            elif article_match := re.match(r'Artículo\s+(\d+)\.?\s*(.*)', line):
                article = self._parse_article(lines, i)
                if current_chapter:
                    current_chapter["articles"].append(article)
                elif current_title:
                    # Sometimes articles are directly under titles
                    if "articles" not in current_title:
                        current_title["articles"] = []
                    current_title["articles"].append(article)
                    
                # Skip to end of article
                i = self._find_next_major_section(lines, i)
                continue
            
            i += 1
    
    def _parse_article(self, lines: List[str], start_idx: int) -> Dict[str, Any]:
        """Parse a single article and its content"""
        article_line = lines[start_idx].strip()
        article_match = re.match(r'Artículo\s+(\d+)\.?\s*(.*)', article_line)
        
        if not article_match:
            return {}
            
        article_num = article_match.group(1)
        article_title = article_match.group(2).strip() if article_match.group(2) else None
        
        # Collect article content
        content_lines = []
        subsections = []
        i = start_idx + 1
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Stop at next major section
            if re.match(r'(LIBRO|TÍTULO|CAPÍTULO|Artículo)\s+', line):
                break
                
            if line:
                # Check for numbered subsections
                if subsection_match := re.match(r'(\d+)\.?\s+(.*)', line):
                    subsections.append({
                        "number": subsection_match.group(1),
                        "content": subsection_match.group(2)
                    })
                # Check for lettered subsections  
                elif letter_match := re.match(r'([a-z])\)\s+(.*)', line):
                    subsections.append({
                        "letter": letter_match.group(1),
                        "content": letter_match.group(2)
                    })
                else:
                    content_lines.append(line)
                    
            i += 1
            
        return {
            "number": article_num,
            "title": article_title,
            "content": " ".join(content_lines).strip(),
            "subsections": subsections
        }
    
    def _find_next_major_section(self, lines: List[str], start_idx: int) -> int:
        """Find the index of the next major section (LIBRO, TÍTULO, etc.)"""
        for i in range(start_idx + 1, len(lines)):
            line = lines[i].strip()
            if re.match(r'(LIBRO|TÍTULO|CAPÍTULO|Artículo)\s+', line):
                return i - 1
        return len(lines) - 1
    
    def add_domains_classification(self) -> None:
        """Add legal domain classification to articles"""
        domain_keywords = {
            "Derecho de Familia": ["matrimonio", "cónyuge", "divorcio", "separación", "filiación", "adopción", "patria potestad"],
            "Derecho de Propiedad": ["propiedad", "posesión", "dominio", "usufructo", "servidumbre"],
            "Derecho de Obligaciones": ["obligación", "contrato", "deuda", "acreedor", "deudor"],
            "Derecho Sucesorio": ["herencia", "testamento", "legado", "sucesión", "heredero"],
            "Derecho Civil General": ["capacidad", "personalidad", "registro civil", "nacionalidad"]
        }
        
        for book in self.structure["books"]:
            for title in book.get("titles", []):
                for chapter in title.get("chapters", []):
                    for article in chapter.get("articles", []):
                        article["domains"] = self._classify_article(article, domain_keywords)
                        
                # Also check title-level articles
                for article in title.get("articles", []):
                    article["domains"] = self._classify_article(article, domain_keywords)
    
    def _classify_article(self, article: Dict[str, Any], domain_keywords: Dict[str, List[str]]) -> List[str]:
        """Classify an article into legal domains"""
        text_to_analyze = f"{article.get('title', '')} {article.get('content', '')}".lower()
        domains = []
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text_to_analyze for keyword in keywords):
                domains.append(domain)
                
        return domains if domains else ["General"]
    
    def convert_to_json(self, output_path: str) -> None:
        """Main conversion function"""
        print("🔄 Extracting text from PDF...")
        raw_text = self.extract_text_from_pdf()
        
        print("🧹 Cleaning extracted text...")  
        clean_text = self.clean_text(raw_text)
        
        print("📋 Parsing structure...")
        self.parse_structure(clean_text)
        
        print("🏷️ Adding domain classifications...")
        self.add_domains_classification()
        
        print(f"💾 Saving to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.structure, f, ensure_ascii=False, indent=2)
            
        print("✅ Conversion complete!")
        self._print_stats()
    
    def _print_stats(self) -> None:
        """Print conversion statistics"""
        total_articles = 0
        for book in self.structure["books"]:
            for title in book.get("titles", []):
                for chapter in title.get("chapters", []):
                    total_articles += len(chapter.get("articles", []))
                total_articles += len(title.get("articles", []))
                
        print(f"📊 Statistics:")
        print(f"   📚 Books: {len(self.structure['books'])}")
        print(f"   📄 Total Articles: {total_articles}")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Spanish Civil Code PDF to JSON')
    parser.add_argument('pdf_path', help='Path to the Civil Code PDF file')
    parser.add_argument('-o', '--output', default='codigo_civil_spain_full.json', 
                       help='Output JSON file path')
    
    args = parser.parse_args()
    
    converter = CivilCodeConverter(args.pdf_path)
    converter.convert_to_json(args.output)
    
    print(f"\n🎉 Conversion completed! JSON saved to: {args.output}")
    print(f"📁 You can now use this JSON file for your legal assistant RAG system.")


if __name__ == "__main__":
    main()